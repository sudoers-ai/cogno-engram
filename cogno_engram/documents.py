"""
cogno_engram.documents — a store of DOCUMENTS: owned, versioned, chunked and searched.

The third kind of knowledge this library keeps, beside the two it already had:

* **memories** (``MemoryStore``) — what one contact said, consolidated from their turns;
* **the graph** (``KnowledgeGraph``) — relations derived from conversations, audience-filtered
  because they carry other people's lives;
* **documents** (``DocumentStore``, this module) — text somebody WROTE to be read: a manual, a
  syllabus, a price list, uploaded as Markdown or PDF and searched on demand.

They share no table and no read, on purpose. A document is published to a GROUP of readers; a
memory belongs to one contact; a graph edge is somebody's private relation. One search over the
three would force a publisher to open the graph in order to publish a timetable.

**The owner is an OPAQUE key.** ``owner_key`` is to this store what ``scope`` is to the others:
the host composes it, this library never interprets it. It follows the SAME subtree convention
(``"a"`` owns ``"a"`` and every ``"a/…"``, never ``"ab"``), which is what lets a host purge
everything under one prefix without this library knowing what the prefix names.

**The readers are OPAQUE labels.** A document carries ``profiles`` — the labels allowed to read
it — and every read that can return a document's TEXT or TITLE takes ``profile`` as a REQUIRED
keyword. There is no wildcard reader and no default: a forgotten profile is a ``TypeError`` at
the call and a blank one is a ``ValueError``, never "every document". What the labels mean (a
role, a plan, a group) is the host's business.

**Embedding models are never mixed.** A vector is a point in ONE model's space at ONE width,
and a cosine between two models returns a plausible number that means nothing — nothing errors.
The deployment has ONE embedder, but it can be swapped globally, and a swap leaves every stored
chunk in the old model's space until it is re-indexed. So every version records the
``embed_model`` it was indexed with, a search is handed ONE query vector plus the label of the
model that made it, and a chunk is scored against that vector ONLY when its label is the same,
by equality of the key. When any readable chunk was indexed by another model (mid re-index), or
the caller has no vector (the embedder is down), the WHOLE search is scored LEXICALLY — one
scale per result — and says so with :data:`KB_EMBED_SPACE_UNAVAILABLE`; never with a cosine
across models. A chunk is never stored without its vector, so a served version is always fully
comparable under its own model.

**Nothing is served until it is whole.** A document has versions; a version is built in the
background (``processing``) while the previous one keeps answering, becomes ``ready`` in one
atomic swap, or ends in ``error`` with a reason from a closed alphabet — and the previous one
still answers. Deleting a document takes it out of every search at once, removes its stored
originals of EVERY version, and leaves a TOMBSTONE (what was removed, when, by whom — no content).

**Two steps when the cost must be confirmed first.** An upload can stop half-way on purpose: it
is extracted and chunked, its embedding cost is ESTIMATED from the chunks, and the version waits
in ``awaiting_confirmation`` — nothing embedded, nothing spent — until the uploader confirms it
(``cogno_engram.ingest.prepare`` then ``commit``). A draft nobody confirms EXPIRES (24 h by
default, on the caller's clock) and leaves a tombstone like any other removal.

**The original file is stored in the database, in its own table** (``kb_originals`` in
Postgres), so it lives outside any repository and inside every purge by construction — the row
that holds it cascades from the document. It is never read by a search: the search reads chunks
only, and the table the bytes live in is not joined by any read that returns text. **What a purge
cannot reach is the database's BACKUPS**: they keep an original until their own retention expires,
and that retention is a decision for whoever operates the database, recorded here, not taken.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

# ── states ───────────────────────────────────────────────────────────────────────────
KB_PROCESSING = "processing"
#: Extracted, chunked and costed, NOT embedded — waiting for the uploader to confirm the cost.
KB_AWAITING_CONFIRMATION = "awaiting_confirmation"
KB_READY = "ready"
KB_ERROR = "error"
VALID_KB_STATES: frozenset[str] = frozenset({KB_PROCESSING, KB_AWAITING_CONFIRMATION, KB_READY,
                                             KB_ERROR})

# ── why an extraction failed — the EXTRACTOR's closed alphabet ───────────────────────
#
# Five, and only five, because they are the only failures an extractor can know about and each
# needs a different action from the uploader: a scanned PDF needs OCR (not included), an
# oversized one needs cutting, a protected one needs its password removed, a broken one needs
# re-exporting, a pathological one needs a human. ``TextExtractor`` implementations (the PDF one
# is in ``cogno-vox``) raise with one of these strings.
REASON_NO_TEXT = "no_text"            # no text layer (a scanned PDF): OCR is not included
REASON_OVER_LIMIT = "over_limit"      # over a ceiling — bytes, pages, or chunks
REASON_ENCRYPTED = "encrypted"        # password-protected
REASON_INVALID = "invalid"            # corrupt, truncated, not what it says it is (or not UTF-8)
REASON_TIMEOUT = "timeout"            # the extraction did not finish in time
EXTRACTOR_REASONS: frozenset[str] = frozenset({
    REASON_NO_TEXT, REASON_OVER_LIMIT, REASON_ENCRYPTED, REASON_INVALID, REASON_TIMEOUT,
})

# ── why a VERSION ended in ``error`` — the extractor's five plus the ingestion's own ──
#
# Closed because each one is rendered to the person who uploaded the file. A free-text reason
# would carry whatever an exception said — a path, a byte of the file — into a column a UI
# renders; the detail goes to the log.
REASON_UNSUPPORTED_TYPE = "unsupported_type"        # not Markdown, not PDF
REASON_EXTRACTOR_UNAVAILABLE = "extractor_unavailable"   # no extractor for this type is wired
REASON_GATE_REFUSED = "gate_refused"                # the caller's pre-embedding gate said no
REASON_EMBED_FAILED = "embed_failed"                # the embedder raised or answered the wrong width
REASON_EXPIRED = "expired"                          # a draft nobody confirmed before `expires_at`
REASON_DISCARDED = "discarded"                      # a draft its uploader withdrew
REASON_INTERRUPTED = "interrupted"                  # `processing` outlived the process working it
REASON_INTERNAL = "internal"                        # anything else — the log has the detail
VALID_KB_REASONS: frozenset[str] = EXTRACTOR_REASONS | frozenset({
    REASON_UNSUPPORTED_TYPE, REASON_EXTRACTOR_UNAVAILABLE, REASON_GATE_REFUSED,
    REASON_EMBED_FAILED, REASON_EXPIRED, REASON_DISCARDED, REASON_INTERRUPTED, REASON_INTERNAL,
})

# ── media types ──────────────────────────────────────────────────────────────────────
MEDIA_MARKDOWN = "text/markdown"
MEDIA_PDF = "application/pdf"
VALID_MEDIA_TYPES: frozenset[str] = frozenset({MEDIA_MARKDOWN, MEDIA_PDF})

# ── what a search can report about itself ────────────────────────────────────────────
#: Some readable chunks were indexed by a model other than the one that made the query vector
#: (a global model swap not yet re-indexed), or the caller had no vector at all (the embedder is
#: down) — those chunks were scored by WORDS only. Not an error: the honest degraded answer,
#: instead of a cosine across two models. The name keeps "space" because a model at a width IS
#: an embedding space, and the host's traces already read this string.
KB_EMBED_SPACE_UNAVAILABLE = "kb_embed_space_unavailable"
VALID_KB_DEGRADATIONS: frozenset[str] = frozenset({KB_EMBED_SPACE_UNAVAILABLE})

# ── the outcome of committing a version ──────────────────────────────────────────────
COMMIT_READY = "ready"             # the version is now the one every search reads
COMMIT_DELETED = "deleted"         # the document is gone — nothing was written
COMMIT_SUPERSEDED = "superseded"   # a NEWER version exists; this one was discarded
VALID_COMMIT_OUTCOMES: frozenset[str] = frozenset({COMMIT_READY, COMMIT_DELETED, COMMIT_SUPERSEDED})

# ── tombstones ───────────────────────────────────────────────────────────────────────
TOMBSTONE_DELETED = "deleted"      # one document, removed on request
TOMBSTONE_PURGED = "purged"        # removed by a subtree purge
TOMBSTONE_EXPIRED = "expired"      # a draft nobody confirmed: its chunks and original removed
TOMBSTONE_DISCARDED = "discarded"  # a draft its uploader withdrew — at once, not in 24 h
TOMBSTONE_INTERRUPTED = "interrupted"   # a `processing` version whose worker is gone
VALID_TOMBSTONE_KINDS: frozenset[str] = frozenset({TOMBSTONE_DELETED, TOMBSTONE_PURGED,
                                                   TOMBSTONE_EXPIRED, TOMBSTONE_DISCARDED,
                                                   TOMBSTONE_INTERRUPTED})

# ── discarding a draft ───────────────────────────────────────────────────────────────
DISCARD_OK = "discarded"           # the draft is gone (or already was — a repeat is free)
DISCARD_NOT_A_DRAFT = "not_a_draft"   # the version exists and is not awaiting confirmation
DISCARD_MISSING = "missing"        # no such version of a document of this owner
VALID_DISCARD_OUTCOMES: frozenset[str] = frozenset({DISCARD_OK, DISCARD_NOT_A_DRAFT,
                                                    DISCARD_MISSING})

# ── claiming a draft for its commit ──────────────────────────────────────────────────
#: How long an unconfirmed draft lives by default. A mechanism default — a host passes its own.
DRAFT_TTL = timedelta(hours=24)
CLAIM_OK = "claimed"               # the draft is this caller's; the version is `processing` now
CLAIM_EXPIRED = "expired"          # past its `expires_at` on the caller's clock — not claimed
CLAIM_MISSING = "missing"          # no draft for that version (never prepared, or claimed already)
VALID_CLAIM_OUTCOMES: frozenset[str] = frozenset({CLAIM_OK, CLAIM_EXPIRED, CLAIM_MISSING})


def sanitize_reason(raw: object) -> str:
    """Any input → a reason from :data:`VALID_KB_REASONS`. Pure, never raises; unknown → internal."""
    try:
        text = str(raw or "").strip().lower()
    except Exception:                                   # noqa: BLE001 — a hostile __str__ is data
        return REASON_INTERNAL
    return text if text in VALID_KB_REASONS else REASON_INTERNAL


# ── keys ─────────────────────────────────────────────────────────────────────────────

def require_owner(owner_key: object) -> str:
    """A non-blank owner key, or ``ValueError``. A forgotten owner must fail LOUD — a blank key
    that matched "every owner" would be a cross-tenant read."""
    text = owner_key if isinstance(owner_key, str) else ""
    if not text.strip():
        raise ValueError("owner_key must be a non-empty string (every document is isolated by it)")
    return text


def require_profile(profile: object) -> str:
    """A non-blank reader profile, or ``ValueError``. Blank is not "any reader"."""
    text = profile.strip() if isinstance(profile, str) else ""
    if not text:
        raise ValueError("profile must be a non-empty string — there is no wildcard reader")
    return text


def sanitize_profiles(raw: object) -> tuple[str, ...]:
    """The profiles a document is published to — stripped, blanks dropped, deduplicated, SORTED.

    A bare string is ONE profile (iterating it would publish the document to its letters). Pure,
    never raises; garbage → ``()``, and an empty tuple means NOBODY may read the document — the
    safe direction, like an unclassified graph edge.
    """
    if raw is None:
        return ()
    items: Iterable[Any] = (raw,) if isinstance(raw, str) else raw if isinstance(
        raw, (list, tuple, set, frozenset)) else ()
    out = set()
    for item in items:
        if isinstance(item, str) and item.strip():
            out.add(item.strip())
    return tuple(sorted(out))


def profile_can_read(profile: str, profiles: Sequence[str]) -> bool:
    """May a reader holding ``profile`` read a document published to ``profiles``? PURE — the ONE
    rule both adapters apply (the Postgres one as ``%s = ANY(d.profiles)``), pinned against each
    other by the parity tests. Exact membership after stripping: no case folding, no hierarchy —
    the labels are the host's, and guessing their order here would be deciding who reads what."""
    p = profile.strip() if isinstance(profile, str) else ""
    return bool(p) and p in tuple(profiles or ())


def owner_in_subtree(owner_key: str, prefix: str) -> bool:
    """``owner_key`` IS ``prefix`` or a ``prefix/…`` descendant — the store-wide subtree rule
    (``admin_turns`` uses the same). ``"t1"`` covers ``"t1/p"`` and never ``"t10"``."""
    return owner_key == prefix or owner_key.startswith(prefix + "/")


# ── the embedding model label ────────────────────────────────────────────────────────

_MODEL_RE = re.compile(r"^(?P<model>\S+)@(?P<dim>[1-9][0-9]{0,5})$")


def embed_model_label(model_spec: str, dimensions: int) -> str:
    """The label a version records and a search presents: ``"<model spec>@<width>"``, e.g.
    ``embed_model_label("ollama:nomic-embed-text:latest", 768)``.

    Opaque to this library except for the width suffix, which the stores CHECK against their
    column: a label that declares a width the column does not have is refused before a row is
    written — the half of the model-swap problem a machine can see. The other half (same width,
    different model) is what the label EQUALITY in every search exists for."""
    spec = str(model_spec or "").strip()
    if not spec or any(ch.isspace() for ch in spec):
        raise ValueError(f"model spec must be a non-empty token without whitespace: {model_spec!r}")
    label = f"{spec}@{int(dimensions)}"
    model_dimensions(label)
    return label


def model_dimensions(label: object) -> int:
    """The width a model label declares, or ``ValueError`` for anything not ``<model>@<int>``."""
    match = _MODEL_RE.match(label if isinstance(label, str) else "")
    if match is None:
        raise ValueError(f"not an embedding model label (expected '<model>@<width>'): {label!r}")
    return int(match.group("dim"))


def require_model(label: object, embedding_dim: int) -> str:
    """``label`` if it is well formed AND declares the store's width, else ``ValueError``."""
    dim = model_dimensions(label)
    if dim != embedding_dim:
        raise ValueError(f"embedding model {label!r} declares width {dim}, but this store's "
                         f"vectors are {embedding_dim} wide")
    assert isinstance(label, str)
    return label


def require_vector(vector: object, embedding_dim: int, *, what: str = "vector") -> list[float]:
    """A list of ``embedding_dim`` finite floats, or ``ValueError``."""
    if not isinstance(vector, (list, tuple)) or len(vector) != embedding_dim:
        got = len(vector) if isinstance(vector, (list, tuple)) else type(vector).__name__
        raise ValueError(f"{what} must be {embedding_dim} floats wide, got {got}")
    out = [float(x) for x in vector]
    if not all(math.isfinite(x) for x in out):
        raise ValueError(f"{what} holds a non-finite value")
    return out


# ── records ──────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KbVersion:
    """One attempt to index a document's content, with one embedding model.

    The uploaded bytes are deliberately NOT a field: they are read by
    :meth:`DocumentStore.get_original` only, so listing a tenant's documents never drags
    megabytes through a page render. ``has_original`` says whether they were kept."""

    document_id: str
    version: int
    state: str
    sha256: str
    embed_model: str
    reason: str = ""
    size_bytes: int = 0
    pages: int = 0
    chunks: int = 0
    has_original: bool = False
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    #: The embedding cost estimated at PREPARE, over the exact chunks the commit will embed —
    #: 0 for a version that never waited for confirmation. Kept after the draft is gone, so
    #: "what was the uploader shown" stays answerable.
    estimated_tokens: int = 0
    #: When the draft stops being confirmable (``prepare``'s clock + ttl); ``None`` for a version
    #: that never waited for confirmation.
    expires_at: Optional[datetime] = None
    #: When this version last ENTERED ``processing`` — begun by a prepare, or claimed by a
    #: commit. The one clock a stale ``processing`` is judged by (``interrupt_stale``):
    #: ``created_at`` cannot tell a draft confirmed a minute ago from one abandoned an hour ago.
    claimed_at: Optional[datetime] = None


#: The most section headings one document reports (:attr:`KbDocument.sections`). A ceiling in
#: the STORE, so no caller can be handed an outline the size of the document; the caller cuts
#: again to what its own budget allows.
MAX_SECTIONS_PER_DOCUMENT = 50


def section_headings(rows: "Iterable[tuple[int, int, str]]", *,
                     limit: int = MAX_SECTIONS_PER_DOCUMENT) -> "tuple[str, ...]":
    """A document's SECTION headings from ``(depth, first_ordinal, heading)`` rows — PURE, the
    ONE rule both adapters apply.

    ``depth`` is the index into a chunk's ``heading_path`` and ``first_ordinal`` the lowest
    ordinal of a chunk carrying ``heading`` at that depth. The SECTION depth is the FIRST depth
    with at least two distinct headings: depth 0 is the document's own title (one value), a
    document whose top heading is unique (``# Manual``) has one value there too, and the next
    depth is where it splits into what it covers. Nothing is hard-coded to a level — a document
    with no unique top heading yields its level 1, one with a unique ``#`` its level 2, and one
    that never splits yields ``()``. Blank headings are dropped; each heading appears once, at
    its first appearance in the document; at most ``limit``."""
    first: "dict[int, dict[str, int]]" = {}
    for depth, ordinal, heading in rows:
        text = " ".join(str(heading or "").split())
        if not text:
            continue
        seen = first.setdefault(int(depth), {})
        if text not in seen or int(ordinal) < seen[text]:
            seen[text] = int(ordinal)
    for depth in sorted(first):
        if len(first[depth]) >= 2:
            ordered = sorted(first[depth].items(), key=lambda kv: (kv[1], kv[0]))
            return tuple(h for h, _ in ordered[:max(0, int(limit))])
    return ()


@dataclass(frozen=True)
class KbDocument:
    """A document and the two versions a caller needs to render its state.

    ``active`` is the version every search reads (``None`` until one is ready); ``latest`` is the
    most recent attempt — the same one when nothing is being rebuilt, a ``processing`` or
    ``error`` one beside a still-serving ``active`` when something is."""

    owner_key: str
    id: str
    title: str
    profiles: tuple[str, ...]
    media_type: str
    active: Optional[KbVersion] = None
    latest: Optional[KbVersion] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    #: The SECTION headings of the ACTIVE version, in document order (:func:`section_headings`)
    #: — filled by ``readable_documents`` only, because it is the reader's outline: what a
    #: caller can say the document is ABOUT when its title alone does not (a manual titled
    #: *Relatório 2026* whose sections name the rent, the payroll, the taxes). ``()`` from every
    #: other read, and for a document with no section depth.
    sections: tuple[str, ...] = ()

    @property
    def status(self) -> str:
        """``processing`` until a first attempt exists, then the LATEST attempt's state."""
        return self.latest.state if self.latest is not None else KB_PROCESSING


@dataclass(frozen=True)
class KbChunk:
    """One piece of a document as it is indexed.

    ``content`` is what is embedded, matched and returned: the heading path followed by the text
    (``"Manual › Horários › Sábado\\n\\n…"``), because the path is what gives a short passage its
    context in a search. ``heading_path`` keeps the same path structured, for provenance
    (``document › path › page``). ``page`` is 1-based, ``None`` for a format without pages."""

    ordinal: int
    content: str
    heading_path: tuple[str, ...] = ()
    page: Optional[int] = None
    embedding: Optional[list[float]] = None


def chunk_id(document_id: str, version: int, ordinal: int) -> str:
    """The id a search result carries — CONTENT-FREE by construction (a store uuid and two
    integers), so it can travel into a trace and be cited without carrying a byte of the text."""
    return f"kb:{document_id}.{int(version)}.{int(ordinal)}"


@dataclass(frozen=True)
class KbHit:
    """One chunk a search returned, with the three numbers that ranked it — RAW, each in [0, 1].

    * ``vector_score`` — ``1 − cosine distance`` to the query vector, cut to ``[0, 1]``; ``None``
      — not 0.0 — when the search was LEXICAL (see ``KbSearchResult``): "not measured" and
      "measured, unrelated" are different facts.
    * ``lexical_score`` — each adapter's own lexical measure, normalised into ``[0, 1]``: in
      Postgres ``ts_rank_cd`` with normalisation 32 (``rank / (rank + 1)``), in memory the share
      of the query's words the chunk carries. The RANGE is the contract; the two adapters'
      values for the same chunk are NOT equal and are not meant to be.
    * ``score`` — :func:`hybrid_score` of the two (``= lexical_score`` exactly when lexical).

    No floor is applied: which score is "relevant enough" is calibrated by the caller over the
    distribution these numbers actually have."""

    id: str
    document_id: str
    version: int
    ordinal: int
    title: str
    heading_path: tuple[str, ...]
    page: Optional[int]
    content: str
    embed_model: str
    vector_score: Optional[float]
    lexical_score: float
    score: float


@dataclass(frozen=True)
class KbSearchResult:
    """What a search returned and what it could NOT do.

    **All or none:** within one result either EVERY hit has a ``vector_score`` or NONE does, so
    the scores of one result are always on one scale. The vector is used only when every
    readable chunk was indexed by the model that made it; if ANY readable chunk was indexed by
    another model (a global swap not yet re-indexed), or there is no vector (the embedder is
    down), the whole search is lexical. ``models_unavailable`` names the models that forced it
    (every readable model when there was no vector); when non-empty, ``degradations`` carries
    :data:`KB_EMBED_SPACE_UNAVAILABLE`."""

    hits: tuple[KbHit, ...] = ()
    degradations: tuple[str, ...] = ()
    models_unavailable: tuple[str, ...] = ()


@dataclass(frozen=True)
class KbTombstone:
    """What a removal leaves behind: WHICH document, WHICH versions, WHEN, by WHOM, and WHY —
    never its title, its text or its bytes. ``actor`` is the caller's opaque label for who asked
    (blank when unknown)."""

    owner_key: str
    document_id: str
    versions: tuple[int, ...]
    kind: str
    actor: str = ""
    removed_at: Optional[datetime] = None


def clamp_unit(x: float) -> float:
    """``x`` cut to ``[0, 1]`` — both components of a score live on that scale, or the linear
    fusion below would be mixing units."""
    return 0.0 if x != x else max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class KbDraft:
    """A prepared version waiting for confirmation: its chunks WITHOUT vectors, the pages it was
    read from, the embedding cost ESTIMATED over those exact chunks, and when it expires.

    ``estimated_tokens`` and ``expires_at`` are the VERSION's (persisted there, so they outlive
    the draft); ``estimated_tokens`` is computed ONCE, at prepare, over the chunks stored here —
    the number shown to the uploader, the number the commit's gate is handed, and the chunks the
    commit embeds are therefore the same set by construction. ``chunks`` is empty in a listing
    (``DocumentStore.pending_drafts``), which carries ``chunk_count`` instead."""

    document_id: str
    version: int
    embed_model: str
    chunk_count: int
    pages: int
    estimated_tokens: int
    expires_at: datetime
    chunks: tuple[KbChunk, ...] = ()
    created_at: Optional[datetime] = None


def hybrid_score(vector_score: Optional[float], lexical_score: float, *,
                 vector_weight: float, lexical_weight: float) -> float:
    """The one fusion both adapters rank by, RENORMALISED over the components present.

    With a vector: ``(w_v·v + w_l·l) / (w_v + w_l)`` — ``0.6·v + 0.4·l`` at the defaults, the
    linear fusion ``load_memories`` uses (without its feedback term). **Without one:
    ``score = l`` exactly**, not ``w_l·l``: an absent component is renormalised away, never
    counted as zero (the convention of the anima's ``compute_cumulative``). Counting it as zero
    would put every degraded search under any floor a caller calibrated on full scores, and
    the degradation would read as "nothing relevant" — a silent false negative.

    Both components are in ``[0, 1]`` (the adapters normalise them), so the score is too."""
    lexical = clamp_unit(lexical_score)
    if vector_score is None:
        return lexical
    wv, wl = float(vector_weight), float(lexical_weight)
    if wv < 0 or wl < 0 or wv + wl <= 0:
        raise ValueError("hybrid weights must be non-negative with a positive sum")
    return clamp_unit((wv * clamp_unit(vector_score) + wl * lexical) / (wv + wl))


def hit_order_key(hit: KbHit) -> tuple:
    """Best first; ties broken by ``(document, version, ordinal)`` — deterministic, so two runs
    over the same rows return the same list and a replay can be compared byte for byte."""
    return (-hit.score, hit.document_id, hit.version, hit.ordinal)


# ── the extractor contract ───────────────────────────────────────────────────────────

#: The ceilings an extractor is handed, as plain keywords — see :class:`TextExtractor`.
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_PAGES = 300
DEFAULT_EXTRACT_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class ExtractionLimits:
    """The ceilings one ingestion applies. Defaults are a SAFE mechanism default, not a product
    decision: a host that sells a plan passes its own numbers."""

    max_bytes: int = DEFAULT_MAX_BYTES
    max_pages: int = DEFAULT_MAX_PAGES
    timeout_s: float = DEFAULT_EXTRACT_TIMEOUT_S


class DocumentLimitReached(Exception):
    """``create_document`` refused: the owner already holds ``max_documents``."""


class OriginalTooLarge(ValueError):
    """``begin_version`` refused an original past the store's byte ceiling, before writing it."""


class ExtractionError(Exception):
    """An extraction that ended for a reason the uploader must see. ``reason`` is sanitised to
    :data:`VALID_KB_REASONS`; ``detail`` is for the log and never reaches a stored row."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = sanitize_reason(reason)
        self.detail = str(detail or "")
        super().__init__(f"{self.reason}: {self.detail}" if self.detail else self.reason)


@runtime_checkable
class TextExtractor(Protocol):
    """Turns an UNTRUSTED file into text, page by page. Implemented OUTSIDE this library — the
    PDF one lives in ``cogno-vox`` (extra ``pdf``) — and this library depends on no
    implementation: the contract is STRUCTURAL, in both directions.

    **In:** the bytes and the ceilings, as plain keywords (``max_bytes``, ``max_pages``,
    ``timeout_s``). **Out:** pages of text, or an exception whose ``reason`` is one of
    :data:`EXTRACTOR_REASONS` — ``no_text``, ``over_limit``, ``encrypted``, ``invalid``,
    ``timeout`` — as a plain string. Anything without a readable ``reason`` is recorded as
    ``internal``.

    **What an implementation MUST do, because the input is hostile** (a file anybody with upload
    rights chose):

    1. refuse ``len(data) > max_bytes`` BEFORE parsing a byte, and a document of more than
       ``max_pages`` pages BEFORE extracting any page — reading everything and then checking is
       spending exactly the resource the ceiling exists to protect;
    2. parse in a SEPARATE PROCESS that is killed at ``timeout_s``, with NO network access — a
       parser is native code, a thread cannot be interrupted, and a file that spins or balloons
       must cost one process, not the caller's event loop;
    3. read the TEXT LAYER only — no JavaScript, no attachments, no links followed, no forms, no
       rendering to pixels.

    **What it returns** is read by ATTRIBUTE, so the implementation defines its own class and
    imports nothing from here: an object with ``pages`` — a sequence of objects with ``number``
    (1-based int) and ``text`` (str) — and, optionally, ``outline`` — a sequence of objects with
    ``level`` (1-based depth), ``title`` and ``page``, the document's bookmarks, used for the
    chunks' heading path. :func:`read_extracted` is the one place that reads it.
    """

    #: The media types this extractor handles, e.g. ``frozenset({"application/pdf"})``.
    media_types: frozenset

    async def extract(self, data: bytes, *, media_type: str, max_bytes: int, max_pages: int,
                      timeout_s: float) -> Any: ...


@dataclass(frozen=True)
class ExtractedPage:
    number: int
    text: str


@dataclass(frozen=True)
class OutlineEntry:
    level: int
    title: str
    page: int


@dataclass(frozen=True)
class ExtractedText:
    """This library's own reading of an extractor's answer — see :func:`read_extracted`."""

    pages: tuple[ExtractedPage, ...]
    outline: tuple[OutlineEntry, ...] = field(default=())


def read_extracted(raw: Any) -> ExtractedText:
    """An extractor's answer, read structurally and validated. Raises ``ExtractionError`` with
    ``invalid`` when the shape is wrong — an extractor that answers garbage has not read the
    file, and indexing its garbage would publish it."""
    try:
        pages = []
        for page in list(getattr(raw, "pages")):
            number = int(getattr(page, "number"))
            text = getattr(page, "text")
            if number < 1 or not isinstance(text, str):
                raise ValueError("bad page")
            pages.append(ExtractedPage(number=number, text=text))
        outline = []
        for entry in list(getattr(raw, "outline", ()) or ()):
            title = str(getattr(entry, "title", "") or "").strip()
            if not title:
                continue
            outline.append(OutlineEntry(level=max(1, int(getattr(entry, "level", 1))),
                                        title=title, page=max(1, int(getattr(entry, "page", 1)))))
    except Exception as exc:                            # noqa: BLE001 — a shape error is data
        raise ExtractionError(REASON_INVALID, f"extractor answered an unreadable shape: "
                                              f"{type(exc).__name__}") from exc
    return ExtractedText(pages=tuple(pages), outline=tuple(outline))
