"""
cogno_engram.ingest — turn an uploaded file into a served version, in the background.

One function, :func:`ingest`, composes the store's write path in the only order that is safe:

    size ceiling → record the attempt (idempotent) → extract → chunk → [gate] → embed → stage
    → atomic swap

and returns what it spent. It never runs inside a conversation turn: indexing happens BEFORE a
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
from typing import Any, Awaitable, Callable, Optional

from cogno_engram.chunking import DEFAULT_CHUNKING, ChunkingConfig, chunk_markdown, chunk_pages
from cogno_engram.documents import (
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
VALID_INGEST_OUTCOMES = frozenset({INGEST_READY, INGEST_UNCHANGED, INGEST_ERROR, INGEST_DELETED,
                                   INGEST_SUPERSEDED, INGEST_NO_ORIGINAL})

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


async def ingest(store: Any, owner_key: str, document_id: str, *, data: bytes, embedder: Any,
                 embed_model: str, extractor: Any = None,
                 limits: ExtractionLimits = ExtractionLimits(),
                 chunking: ChunkingConfig = DEFAULT_CHUNKING,
                 gate: Optional[Callable[[int], Awaitable[bool]]] = None,
                 pace: Optional[Callable[[int], Awaitable[None]]] = None,
                 keep_original: bool = True) -> IngestOutcome:
    """Index ``data`` as a new version of ``document_id`` and serve it, or say why not.

    ``embedder`` is duck-typed: ``embed_with_usage(text) -> (vector, tokens)`` when it has it
    (its tokens are what the outcome reports), else ``embed(text) -> vector``. ``embed_model``
    is the label of THAT embedder (``documents.embed_model_label``); it is recorded on the
    version and a search compares only with chunks of the same label. ``extractor`` is a
    ``documents.TextExtractor`` for PDF; Markdown needs none.

    Raises only for a STORE failure or a bug (after recording ``internal`` when it can);
    everything the uploader can act on is an outcome, not an exception.
    """
    require_owner(owner_key)
    require_model(embed_model, store.embedding_dim)
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes")
    data = bytes(data)

    doc = await store.get_document(owner_key, document_id)
    if doc is None:
        return IngestOutcome(INGEST_DELETED, document_id)

    sha = hashlib.sha256(data).hexdigest()
    oversize = len(data) > limits.max_bytes
    original = data if (keep_original and not oversize) else None
    try:
        version = await store.begin_version(owner_key, document_id, sha256=sha,
                                            embed_model=embed_model, size_bytes=len(data),
                                            original=original)
    except OriginalTooLarge:
        version = await store.begin_version(owner_key, document_id, sha256=sha,
                                            embed_model=embed_model, size_bytes=len(data))
        oversize = True
    if version is None:
        return IngestOutcome(INGEST_DELETED, document_id)
    if version.state == KB_READY:
        return IngestOutcome(INGEST_UNCHANGED, document_id, version.version,
                             chunks=version.chunks, pages=version.pages)
    number = version.version

    async def fail(reason: str, **usage: Any) -> IngestOutcome:
        marked = await store.fail_version(owner_key, document_id, number, reason=reason)
        if not marked and await store.get_document(owner_key, document_id) is None:
            return IngestOutcome(INGEST_DELETED, document_id, number, **usage)
        return IngestOutcome(INGEST_ERROR, document_id, number, reason=sanitize_reason(reason),
                             **usage)

    if oversize:
        return await fail(REASON_OVER_LIMIT)

    tokens = calls = 0
    reported = True
    estimated = pages = 0
    try:
        try:
            extracted = await _extract(data, doc.media_type, extractor, limits)
            if isinstance(extracted, str):
                chunks = chunk_markdown(doc.title, extracted, chunking)
                pages = 0
            else:
                cleaned = [type(p)(number=p.number, text=_clean(p.text)) for p in extracted.pages]
                chunks = chunk_pages(doc.title, cleaned, extracted.outline, chunking)
                pages = len(cleaned)
        except ExtractionError as exc:
            logger.info("event=kb_extract_failed document=%s version=%d reason=%s detail=%s",
                        document_id, number, exc.reason, exc.detail)
            return await fail(exc.reason)
        if not chunks:
            return await fail(REASON_NO_TEXT, pages=pages)

        estimated = estimate_tokens(chunks)
        if gate is not None:
            try:
                approved = bool(await gate(estimated))
            except Exception:                            # noqa: BLE001 — a gate that cannot
                logger.warning("event=kb_gate_failed document=%s version=%d",  # answer did not
                               document_id, number, exc_info=True)              # approve
                approved = False
            if not approved:
                return await fail(REASON_GATE_REFUSED, pages=pages, estimated_tokens=estimated)

        embedded: list[KbChunk] = []
        for chunk in chunks:
            if pace is not None:
                await pace(math.ceil(len(chunk.content) / CHARS_PER_TOKEN))
            try:
                vector, used, has_usage = await _embed(embedder, chunk.content)
            except Exception:                            # noqa: BLE001 — recorded as a reason
                logger.warning("event=kb_embed_failed document=%s version=%d",
                               document_id, number, exc_info=True)
                return await fail(REASON_EMBED_FAILED, pages=pages, estimated_tokens=estimated,
                                  embedding_tokens=tokens, embedding_calls=calls + 1,
                                  usage_reported=reported)
            calls += 1
            tokens += used
            reported = reported and has_usage
            if len(vector) != store.embedding_dim:
                return await fail(REASON_EMBED_FAILED, pages=pages, estimated_tokens=estimated,
                                  embedding_tokens=tokens, embedding_calls=calls,
                                  usage_reported=reported)
            embedded.append(KbChunk(ordinal=chunk.ordinal, content=chunk.content,
                                    heading_path=chunk.heading_path, page=chunk.page,
                                    embedding=vector))
    except asyncio.CancelledError:
        raise                                            # the version stays `processing`: a
    except Exception:                                    # retry resumes it (idempotent)
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
    await store.readable_documents(PROBE_OWNER, profile=PROBE_PROFILE)
    await store.search(PROBE_OWNER, profile=PROBE_PROFILE, text="probe", vector=unit,
                       embed_model=embed_model, limit=1)
    await store.tombstones(PROBE_OWNER, limit=1)
    await store.stored_original_bytes(PROBE_OWNER)
    await store.stale_documents(embed_model=embed_model, limit=1)
