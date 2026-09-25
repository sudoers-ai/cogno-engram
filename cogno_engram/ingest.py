"""
cogno_engram.ingest — turn an uploaded file into a served version, in the background.

Two steps compose the store's write path in the only order that is safe, and :func:`ingest`
runs them back to back:

    prepare:  size ceiling → record the attempt (idempotent) → extract → chunk → ESTIMATE
              → park as a draft (``awaiting_confirmation``) — nothing embedded, nothing spent
    commit:   claim the draft → [gate(estimate)] → embed → stage → atomic swap

and the commit returns what it spent. Splitting them is what lets an uploader SEE the cost —
the estimate over the exact chunks the commit will embed — before confirming it; a draft
nobody confirms expires (:func:`expire_drafts`, on the caller's clock). It never runs inside a conversation turn: indexing happens BEFORE a
document is readable, and the reader path only ever sees a version that finished.

**What each failure leaves behind.** Every failure after the attempt is recorded ends that
VERSION in ``error`` with a reason from ``documents.VALID_KB_REASONS``; the version that was
being served keeps being served. A document deleted (or its owner purged) while the job runs
writes nothing and the outcome says ``deleted`` — every write is conditional on the row. A
newer upload that lands first makes this one ``superseded``.

**Repeating is free.** The attempt is keyed by ``(document, sha256 of the bytes, embedding
model)``: running the same job twice returns ``unchanged`` once the first one finished, or
resumes the same version (chunks upserted by ordinal) when the first one died half-way.

**What it spends is RETURNED, not billed.** ``embedding_tokens``/``embedding_calls`` are the
embedder's own count (``embed_with_usage``), summed over the calls this job made. Who pays for
them, against which allowance, at which price, is the caller's business — this library knows no
tenant and no price. The ``gate`` hook is where a caller refuses BEFORE anything is embedded
(an allowance check, say): it is handed the estimate and a refusal costs zero embedding calls.
The ``pace`` hook is where a caller bounds the RATE (tokens per minute) so a 300-page upload
does not starve the model server that answers live conversations; :class:`TokensPerMinute` is
a ready one.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from cogno_engram.chunking import DEFAULT_CHUNKING, ChunkingConfig, chunk_markdown, chunk_pages
from cogno_engram.documents import (
    CLAIM_EXPIRED,
    CLAIM_OK,
    DRAFT_TTL,
    KB_AWAITING_CONFIRMATION,
    COMMIT_DELETED,
    COMMIT_READY,
    COMMIT_SUPERSEDED,
    KB_READY,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    REASON_EMBED_FAILED,
    REASON_EXTRACTOR_UNAVAILABLE,
    REASON_GATE_REFUSED,
    REASON_INTERNAL,
    REASON_INVALID,
    REASON_NO_TEXT,
    REASON_OVER_LIMIT,
    REASON_TIMEOUT,
    REASON_UNSUPPORTED_TYPE,
    ExtractionError,
    ExtractionLimits,
    KbChunk,
    OriginalTooLarge,
    read_extracted,
    require_model,
    require_owner,
    sanitize_reason,
)

logger = logging.getLogger("cogno_engram.ingest")

# ── outcomes ─────────────────────────────────────────────────────────────────────────
INGEST_READY = "ready"              # a new version is now served
INGEST_UNCHANGED = "unchanged"      # this content+model is already served — nothing done
INGEST_ERROR = "error"              # the version ended in error; ``reason`` says why
INGEST_DELETED = "deleted"          # the document is gone — nothing written
INGEST_SUPERSEDED = "superseded"    # a newer version got there first — this one discarded
INGEST_NO_ORIGINAL = "no_original"  # ``reindex`` found no stored original to rebuild from
INGEST_AWAITING = "awaiting_confirmation"   # ``prepare``: a draft waiting for confirmation
INGEST_NOT_PREPARED = "not_prepared"        # ``commit``: no draft for that version
INGEST_EXPIRED = "expired"                  # ``commit``: the draft is past ``expires_at``
VALID_INGEST_OUTCOMES = frozenset({INGEST_READY, INGEST_UNCHANGED, INGEST_ERROR, INGEST_DELETED,
                                   INGEST_SUPERSEDED, INGEST_NO_ORIGINAL, INGEST_AWAITING,
                                   INGEST_NOT_PREPARED, INGEST_EXPIRED})

#: Characters per token for the PRE-embedding estimate a gate is handed. An estimate, not a
#: bill: the bill is ``embedding_tokens``, the embedder's own count.
CHARS_PER_TOKEN = 4

#: Chunks staged per ``add_chunks`` call, so one statement never carries a whole book.
STAGE_BATCH = 200

#: How long past its own ``timeout_s`` an extractor is waited for before this side gives up.
#: The extractor owns the real deadline (it kills its own process); this is the belt for an
#: implementation that never returns.
EXTRACT_GRACE_S = 5.0


@dataclass(frozen=True)
class IngestOutcome:
    status: str
    document_id: str
    version: Optional[int] = None
    reason: str = ""
    chunks: int = 0
    pages: int = 0
    #: the pre-embedding estimate handed to ``gate`` (0 when extraction never finished)
    estimated_tokens: int = 0
    #: what the embedder REPORTED, summed — the number a caller bills from
    embedding_tokens: int = 0
    embedding_calls: int = 0
    #: ``False`` when the embedder has no ``embed_with_usage``: the calls happened and their
    #: tokens are unknown, which is not the same as zero
    usage_reported: bool = True
    #: ``prepare`` (and an ``expired`` commit): until when the draft can be confirmed
    expires_at: Optional[datetime] = None


def estimate_tokens(chunks: "list[KbChunk]") -> int:
    """A rough token count for ``chunks`` — ``ceil(chars / CHARS_PER_TOKEN)`` per chunk."""
    return sum(math.ceil(len(c.content) / CHARS_PER_TOKEN) for c in chunks)


class TokensPerMinute:
    """A ``pace`` hook: at most ``limit`` tokens per rolling minute, as a continuous bucket.

    ``await pace(n)`` returns at once while the bucket holds ``n``, and otherwise sleeps exactly
    until it does. A request larger than the whole bucket waits for a FULL bucket and then goes —
    refusing it would make one oversized chunk unindexable forever. The clock and the sleep are
    injected so the arithmetic is tested without waiting."""

    def __init__(self, limit: int, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        if int(limit) <= 0:
            raise ValueError("limit must be a positive number of tokens per minute")
        self.limit = int(limit)
        self._clock = clock
        self._sleep = sleep
        self._available = float(self.limit)
        self._at = clock()

    def _refill(self) -> None:
        now = self._clock()
        self._available = min(float(self.limit),
                              self._available + (now - self._at) * self.limit / 60.0)
        self._at = now

    async def __call__(self, tokens: int) -> None:
        need = float(min(max(0, int(tokens)), self.limit))
        self._refill()
        if self._available < need:
            await self._sleep((need - self._available) * 60.0 / self.limit)
            self._refill()
        self._available = max(0.0, self._available - need)


async def _embed(embedder: Any, text: str) -> "tuple[list[float], int, bool]":
    """``(vector, tokens, reported)`` — ``embed_with_usage`` when the embedder has it."""
    with_usage = getattr(embedder, "embed_with_usage", None)
    if with_usage is not None:
        vector, tokens = await with_usage(text)
        return list(vector), int(tokens or 0), True
    return list(await embedder.embed(text)), 0, False


def _clean(text: str) -> str:
    # NUL cannot be stored in a Postgres text column; a file carrying one must not fail the
    # insert with a driver error that reads like a database fault.
    return text.replace("\x00", "")


async def _extract(data: bytes, media_type: str, extractor: Any,
                   limits: ExtractionLimits) -> Any:
    """``str`` (Markdown) or the validated ``ExtractedText`` (pages), else ``ExtractionError``."""
    if media_type == MEDIA_MARKDOWN:
        try:
            return _clean(data.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise ExtractionError(REASON_INVALID, "not UTF-8") from exc
    if media_type != MEDIA_PDF:
        raise ExtractionError(REASON_UNSUPPORTED_TYPE, media_type)
    if extractor is None or media_type not in (getattr(extractor, "media_types", None) or ()):
        raise ExtractionError(REASON_EXTRACTOR_UNAVAILABLE, media_type)
    try:
        raw = await asyncio.wait_for(
            extractor.extract(data, media_type=media_type, max_bytes=limits.max_bytes,
                              max_pages=limits.max_pages, timeout_s=limits.timeout_s),
            timeout=limits.timeout_s + EXTRACT_GRACE_S)
    except asyncio.TimeoutError as exc:
        raise ExtractionError(REASON_TIMEOUT, "extractor did not return") from exc
    except ExtractionError:
        raise
    except Exception as exc:                            # noqa: BLE001 — the reason is the datum
        raise ExtractionError(sanitize_reason(getattr(exc, "reason", "")),
                              type(exc).__name__) from exc
    extracted = read_extracted(raw)
    if len(extracted.pages) > limits.max_pages:
        # An extractor that ignored the page ceiling has already spent what it protects; its
        # output is still refused, so the ceiling holds for whoever wrote the extractor.
        raise ExtractionError(REASON_OVER_LIMIT, f"{len(extracted.pages)} pages")
    return extracted


def utc_now() -> datetime:
    """The clock ``prepare``/``commit``/``expire_drafts`` use when the caller passes no ``now``."""
    return datetime.now(timezone.utc)


def _outcome_of_failure(status: str, document_id: str, version: int, reason: str,
                        **usage: Any) -> IngestOutcome:
    return IngestOutcome(status, document_id, version,
                         reason=sanitize_reason(reason) if status == INGEST_ERROR else "",
                         **usage)


async def _fail(store: Any, owner_key: str, document_id: str, version: int, reason: str,
                **usage: Any) -> IngestOutcome:
    marked = await store.fail_version(owner_key, document_id, version, reason=reason)
    if not marked and await store.get_document(owner_key, document_id) is None:
        return _outcome_of_failure(INGEST_DELETED, document_id, version, reason, **usage)
    return _outcome_of_failure(INGEST_ERROR, document_id, version, reason, **usage)


async def prepare(store: Any, owner_key: str, document_id: str, *, data: bytes, embed_model: str,
                  extractor: Any = None, limits: ExtractionLimits = ExtractionLimits(),
                  chunking: ChunkingConfig = DEFAULT_CHUNKING, keep_original: bool = True,
                  ttl: timedelta = DRAFT_TTL, now: Optional[datetime] = None) -> IngestOutcome:
    """STEP ONE: extract, chunk and ESTIMATE ``data`` as a new version, and park it as a draft
    (``awaiting_confirmation``) — without embedding anything.

    Calls neither the embedder nor any gate: nothing is spent here, so everything that can go
    wrong with the FILE (too big, no text, protected, corrupt) is known and recorded before the
    uploader is asked to confirm a cost. The outcome carries the ``estimated_tokens`` computed
    over the exact chunks the commit will embed, and ``expires_at`` = ``now`` + ``ttl`` — both
    also persisted on the version (``KbVersion.estimated_tokens``/``expires_at``).

    Outcomes: ``awaiting_confirmation``; ``unchanged`` (this content and model are already
    served); ``error`` (``reason`` from ``documents.VALID_KB_REASONS``); ``deleted``. Preparing the
    same content and model again re-prepares the SAME version, with a fresh expiry.
    """
    require_owner(owner_key)
    require_model(embed_model, store.embedding_dim)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    data = bytes(data)
    moment = now or utc_now()

    doc = await store.get_document(owner_key, document_id)
    if doc is None:
        return IngestOutcome(INGEST_DELETED, document_id)

    sha = hashlib.sha256(data).hexdigest()
    oversize = len(data) > limits.max_bytes
    original = data if (keep_original and not oversize) else None
    try:
        version = await store.begin_version(owner_key, document_id, sha256=sha,
                                            embed_model=embed_model, size_bytes=len(data),
                                            original=original, now=moment)
    except OriginalTooLarge:
        version = await store.begin_version(owner_key, document_id, sha256=sha,
                                            embed_model=embed_model, size_bytes=len(data),
                                            now=moment)
        oversize = True
    if version is None:
        return IngestOutcome(INGEST_DELETED, document_id)
    if version.state == KB_READY:
        return IngestOutcome(INGEST_UNCHANGED, document_id, version.version,
                             chunks=version.chunks, pages=version.pages)
    number = version.version
    if oversize:
        return await _fail(store, owner_key, document_id, number, REASON_OVER_LIMIT)

    pages = 0
    try:
        try:
            extracted = await _extract(data, doc.media_type, extractor, limits)
            if isinstance(extracted, str):
                chunks = chunk_markdown(doc.title, extracted, chunking)
            else:
                cleaned = [type(p)(number=p.number, text=_clean(p.text)) for p in extracted.pages]
                chunks = chunk_pages(doc.title, cleaned, extracted.outline, chunking)
                pages = len(cleaned)
        except ExtractionError as exc:
            logger.info("event=kb_extract_failed document=%s version=%d reason=%s detail=%s",
                        document_id, number, exc.reason, exc.detail)
            return await _fail(store, owner_key, document_id, number, exc.reason)
        if not chunks:
            return await _fail(store, owner_key, document_id, number, REASON_NO_TEXT, pages=pages)
        estimated = estimate_tokens(chunks)
        expires_at = moment + ttl
        parked = await store.save_draft(owner_key, document_id, number, chunks=chunks,
                                        pages=pages, estimated_tokens=estimated,
                                        expires_at=expires_at)
    except asyncio.CancelledError:
        raise
    except Exception:
        await store.fail_version(owner_key, document_id, number, reason=REASON_INTERNAL)
        raise
    if not parked:
        gone = await store.get_document(owner_key, document_id) is None
        return IngestOutcome(INGEST_DELETED if gone else INGEST_SUPERSEDED, document_id, number)
    return IngestOutcome(INGEST_AWAITING, document_id, number, chunks=len(chunks), pages=pages,
                         estimated_tokens=estimated, expires_at=expires_at)


async def commit(store: Any, owner_key: str, document_id: str, version: int, *, embedder: Any,
                 embed_model: str, gate: Optional[Callable[[int], Awaitable[bool]]] = None,
                 pace: Optional[Callable[[int], Awaitable[None]]] = None,
                 now: Optional[datetime] = None) -> IngestOutcome:
    """STEP TWO, on a prepared version: ``gate(estimated_tokens)`` → embed (``pace``) → stage →
    atomic swap. Returns the usage the embedder reported, exactly as :func:`ingest` always has.

    The draft is CLAIMED first (atomically: ``awaiting_confirmation`` → ``processing``, the draft
    removed), so two confirmations of the same upload cannot both embed it. The gate is handed
    the version's persisted ``estimated_tokens`` — the number the uploader was shown.

    Outcomes: ``ready``; ``unchanged`` (committed already — a repeat is free, zero calls);
    ``not_prepared`` (no draft for that version: never prepared, or another commit took it);
    ``expired`` (past ``expires_at`` on ``now``; nothing changes — the expiry sweep marks it);
    ``error``; ``deleted``; ``superseded``. ``embed_model`` must be the model the version was
    prepared for — anything else is a caller bug, ``ValueError``, and nothing changes.
    """
    require_owner(owner_key)
    require_model(embed_model, store.embedding_dim)
    number = int(version)
    moment = now or utc_now()

    doc = await store.get_document(owner_key, document_id)
    if doc is None:
        return IngestOutcome(INGEST_DELETED, document_id, number)
    found = await store.get_version(owner_key, document_id, number)
    if found is None:
        return IngestOutcome(INGEST_NOT_PREPARED, document_id, number)
    if found.embed_model != embed_model:
        raise ValueError(f"version {number} was prepared for {found.embed_model!r}, "
                         f"not {embed_model!r}")
    if found.state == KB_READY and doc.active is not None and doc.active.version == number:
        return IngestOutcome(INGEST_UNCHANGED, document_id, number, chunks=found.chunks,
                             pages=found.pages, estimated_tokens=found.estimated_tokens)
    if found.state != KB_AWAITING_CONFIRMATION:
        return IngestOutcome(INGEST_NOT_PREPARED, document_id, number)
    claim, draft = await store.claim_draft(owner_key, document_id, number, now=moment)
    if claim == CLAIM_EXPIRED:
        return IngestOutcome(INGEST_EXPIRED, document_id, number,
                             estimated_tokens=found.estimated_tokens, expires_at=found.expires_at)
    if claim != CLAIM_OK or draft is None:
        return IngestOutcome(INGEST_NOT_PREPARED, document_id, number)

    pages, estimated = draft.pages, draft.estimated_tokens
    tokens = calls = 0
    reported = True
    try:
        if gate is not None:
            try:
                approved = bool(await gate(estimated))
            except Exception:                            # noqa: BLE001 — a gate that cannot
                logger.warning("event=kb_gate_failed document=%s version=%d",  # answer did not
                               document_id, number, exc_info=True)              # approve
                approved = False
            if not approved:
                return await _fail(store, owner_key, document_id, number, REASON_GATE_REFUSED,
                                   pages=pages, estimated_tokens=estimated)

        embedded: list[KbChunk] = []
        for chunk in draft.chunks:
            if pace is not None:
                await pace(math.ceil(len(chunk.content) / CHARS_PER_TOKEN))
            try:
                vector, used, has_usage = await _embed(embedder, chunk.content)
            except Exception:                            # noqa: BLE001 — recorded as a reason
                logger.warning("event=kb_embed_failed document=%s version=%d",
                               document_id, number, exc_info=True)
                return await _fail(store, owner_key, document_id, number, REASON_EMBED_FAILED,
                                   pages=pages, estimated_tokens=estimated,
                                   embedding_tokens=tokens, embedding_calls=calls + 1,
                                   usage_reported=reported)
            calls += 1
            tokens += used
            reported = reported and has_usage
            if len(vector) != store.embedding_dim:
                return await _fail(store, owner_key, document_id, number, REASON_EMBED_FAILED,
                                   pages=pages, estimated_tokens=estimated,
                                   embedding_tokens=tokens, embedding_calls=calls,
                                   usage_reported=reported)
            embedded.append(KbChunk(ordinal=chunk.ordinal, content=chunk.content,
                                    heading_path=chunk.heading_path, page=chunk.page,
                                    embedding=vector))
    except asyncio.CancelledError:
        raise                                            # stays `processing` until a new prepare
    except Exception:                                    # of the bytes resumes it, or the
        # `interrupt_stale` sweep ends it — the claim consumed the draft, so nothing else can
        await store.fail_version(owner_key, document_id, number, reason=REASON_INTERNAL)
        raise

    def done(status: str, chunks: int = 0) -> IngestOutcome:
        return IngestOutcome(status, document_id, number, chunks=chunks, pages=pages,
                             estimated_tokens=estimated, embedding_tokens=tokens,
                             embedding_calls=calls, usage_reported=reported)

    for start in range(0, len(embedded), STAGE_BATCH):
        if not await store.add_chunks(owner_key, document_id, number,
                                      embedded[start:start + STAGE_BATCH]):
            gone = await store.get_document(owner_key, document_id) is None
            return done(INGEST_DELETED if gone else INGEST_SUPERSEDED)
    outcome = await store.commit_version(owner_key, document_id, number, pages=pages)
    status = {COMMIT_READY: INGEST_READY, COMMIT_DELETED: INGEST_DELETED,
              COMMIT_SUPERSEDED: INGEST_SUPERSEDED}.get(outcome, INGEST_ERROR)
    return done(status, len(embedded) if status == INGEST_READY else 0)


async def ingest(store: Any, owner_key: str, document_id: str, *, data: bytes, embedder: Any,
                 embed_model: str, extractor: Any = None,
                 limits: ExtractionLimits = ExtractionLimits(),
                 chunking: ChunkingConfig = DEFAULT_CHUNKING,
                 gate: Optional[Callable[[int], Awaitable[bool]]] = None,
                 pace: Optional[Callable[[int], Awaitable[None]]] = None,
                 keep_original: bool = True) -> IngestOutcome:
    """Index ``data`` as a new version of ``document_id`` and serve it, or say why not — in one
    call: :func:`prepare` then :func:`commit`, back to back, with no confirmation in between.

    ``embedder`` is duck-typed: ``embed_with_usage(text) -> (vector, tokens)`` when it has it
    (its tokens are what the outcome reports), else ``embed(text) -> vector``. ``embed_model``
    is the label of THAT embedder (``documents.embed_model_label``); it is recorded on the
    version and a search compares only with chunks of the same label. ``extractor`` is a
    ``documents.TextExtractor`` for PDF; Markdown needs none.

    Raises only for a STORE failure or a bug (after recording ``internal`` when it can);
    everything the uploader can act on is an outcome, not an exception.
    """
    prepared = await prepare(store, owner_key, document_id, data=data, embed_model=embed_model,
                             extractor=extractor, limits=limits, chunking=chunking,
                             keep_original=keep_original)
    if prepared.status != INGEST_AWAITING or prepared.version is None:
        return prepared
    return await commit(store, owner_key, document_id, prepared.version, embedder=embedder,
                        embed_model=embed_model, gate=gate, pace=pace)


async def discard_draft(store: Any, owner_key: str, document_id: str, version: int, *,
                        actor: str = "") -> str:
    """WITHDRAW a draft now — what :func:`expire_drafts` does at 24 h, on the uploader's request
    (an upload made by mistake must not keep its original stored for a day). The version becomes
    ``error``/``discarded``, its draft and original are removed, and a ``discarded`` tombstone
    records ``actor``. Returns ``documents.DISCARD_*``: ``discarded`` (a repeat too),
    ``not_a_draft`` (any version not awaiting confirmation — a served one included), ``missing``.
    """
    require_owner(owner_key)
    return await store.discard_draft(owner_key, document_id, int(version), actor=actor)


#: How long a ``processing`` version may go without progress before :func:`interrupt_stale`
#: calls it dead. A mechanism default; the host passes its own ``older_than``.
STALE_PROCESSING = timedelta(minutes=30)


async def interrupt_stale(store: Any, *, older_than: Optional[datetime] = None,
                          limit: int = 100) -> int:
    """The sweep that keeps "no version stays ``processing`` beyond its process" true — for a
    host's tick, cross-owner. Every ``processing`` version whose ``claimed_at`` is before
    ``older_than`` (``utc_now() - STALE_PROCESSING`` when omitted) becomes
    ``error``/``interrupted``, with a tombstone: a prepare that crashed half-way, and a commit
    that died after its claim — the claim consumed the draft, so nothing can resume it.
    ``claimed_at`` and not ``created_at``, because a draft confirmed a minute ago and still
    embedding may have been CREATED an hour ago."""
    return await store.interrupt_stale(older_than=older_than or utc_now() - STALE_PROCESSING,
                                       limit=limit)


async def expire_drafts(store: Any, *, now: Optional[datetime] = None, limit: int = 100) -> int:
    """The expiry sweep a host calls on its tick: every draft past ``expires_at`` on ``now``
    becomes ``error`` with reason ``expired``, its draft chunks and stored original are removed,
    and a tombstone is written (``DocumentStore.expire_drafts``). Returns how many expired."""
    return await store.expire_drafts(now=now or utc_now(), limit=limit)


async def reindex(store: Any, owner_key: str, document_id: str, *, embedder: Any,
                  embed_model: str, extractor: Any = None, **kwargs: Any) -> IngestOutcome:
    """Rebuild the SERVED version from its stored original with ``embedder`` — the job a global
    model swap needs, one document at a time (``DocumentStore.stale_documents`` lists them).
    Until it finishes, the old version keeps being served and its chunks are searched by words
    only (their model is not the query's)."""
    original = await store.get_original(owner_key, document_id)
    if original is None:
        gone = await store.get_document(owner_key, document_id) is None
        return IngestOutcome(INGEST_DELETED if gone else INGEST_NO_ORIGINAL, document_id)
    return await ingest(store, owner_key, document_id, data=original, embedder=embedder,
                        embed_model=embed_model, extractor=extractor, **kwargs)


#: An owner no host composes: the probe reads it and finds nothing.
PROBE_OWNER = "__documents_probe__/__schema__"
PROBE_PROFILE = "__documents_probe__"
_PROBE_DOCUMENT = "00000000-0000-0000-0000-000000000000"


async def documents_probe(store: Any, *, embed_model: str) -> None:
    """Run EVERY read the document store serves, against an owner that holds nothing — for a
    host's health check. Cheap (empty results, one ``LIMIT 1`` maintenance read), READ-ONLY, and
    it RAISES whatever the store raised.

    It exercises instead of enumerating, for the reason the host's own graph probe gives: a
    list of "the columns this library needs" would be a copy of a requirement that lives here
    and moves with the pin. A table or column the migration never created fails at PARSE time,
    whatever rows exist — so the reads themselves are the check. The test that holds this
    drops every column of every document table in turn and requires the probe to fail.
    """
    dim = int(store.embedding_dim)
    unit = [1.0] + [0.0] * (dim - 1)
    await store.list_documents(PROBE_OWNER)
    await store.get_document(PROBE_OWNER, _PROBE_DOCUMENT)
    await store.get_original(PROBE_OWNER, _PROBE_DOCUMENT, version=1)
    await store.version_text(PROBE_OWNER, _PROBE_DOCUMENT, version=1)
    await store.readable_documents(PROBE_OWNER, profile=PROBE_PROFILE)
    await store.search(PROBE_OWNER, profile=PROBE_PROFILE, text="probe", vector=unit,
                       embed_model=embed_model, limit=1)
    await store.tombstones(PROBE_OWNER, limit=1)
    await store.stored_original_bytes(PROBE_OWNER)
    await store.stale_documents(embed_model=embed_model, limit=1)
    await store.get_version(PROBE_OWNER, _PROBE_DOCUMENT, 1)
    await store.pending_drafts(PROBE_OWNER)
