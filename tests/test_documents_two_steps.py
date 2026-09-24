"""``prepare`` → confirm → ``commit`` — the ingestion in two steps, on BOTH adapters.

Every clock here is INJECTED (``T0``): the expiry is decided by the value handed in, never by the
day the suite runs. The Postgres leg runs when a test database answers (``tests/conftest.py``).
Invented content only.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogno_engram.documents import (
    DRAFT_TTL,
    KB_AWAITING_CONFIRMATION,
    KB_ERROR,
    KB_PROCESSING,
    KB_READY,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    TOMBSTONE_EXPIRED,
    ExtractionLimits,
    VALID_KB_REASONS,
    VALID_KB_STATES,
    VALID_TOMBSTONE_KINDS,
)
from cogno_engram.ingest import (
    INGEST_AWAITING,
    INGEST_DELETED,
    INGEST_ERROR,
    INGEST_EXPIRED,
    INGEST_NOT_PREPARED,
    INGEST_READY,
    INGEST_UNCHANGED,
    VALID_INGEST_OUTCOMES,
    commit,
    expire_drafts,
    ingest,
    prepare,
)

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import MODEL_A, MODEL_B, FakePdfExtractor, StubEmbedder, store_factory

DSN = resolve_test_dsn()      # module-level: the Postgres leg DROPs the kb_* tables

T0 = datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc)
MANUAL = ("# Guia\n\nBem-vindo.\n\n## Horários\n\nSábado das 8h às 12h.\n\n"
          "## Preços\n\nMensalidade de R$ 450,00.\n").encode()


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    return await store_factory(request.param, DSN)()


def owner() -> str:
    return f"escola{uuid4().hex[:6]}/secretaria"


async def new_doc(store, o, *, media_type=MEDIA_MARKDOWN):
    return (await store.create_document(o, title="Guia", profiles=["GUEST"],
                                        media_type=media_type)).id


class GateSpy:
    def __init__(self, answer: bool = True) -> None:
        self.answer = answer
        self.asked: list[int] = []

    async def __call__(self, estimated: int) -> bool:
        self.asked.append(estimated)
        return self.answer


# ── prepare: extract, chunk, estimate, park — and spend nothing ─────────────────────────

async def test_prepare_parks_a_costed_draft_and_spends_nothing(store):
    o = owner()
    doc_id = await new_doc(store, o)
    out = await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert out.status == INGEST_AWAITING and out.version == 1
    assert out.estimated_tokens > 0 and out.chunks >= 2
    assert out.expires_at == T0 + DRAFT_TTL == T0 + timedelta(hours=24)
    assert out.embedding_calls == 0 and out.embedding_tokens == 0

    doc = await store.get_document(o, doc_id)
    assert doc.status == KB_AWAITING_CONFIRMATION and doc.active is None
    # persisted ON THE VERSION — what the host's GET reads back
    assert (doc.latest.estimated_tokens, doc.latest.expires_at) == (out.estimated_tokens,
                                                                    out.expires_at)
    [draft] = await store.pending_drafts(o)
    assert (draft.version, draft.chunk_count, draft.estimated_tokens, draft.expires_at) == \
        (1, out.chunks, out.estimated_tokens, out.expires_at)
    assert draft.chunks == ()                              # a listing carries no text
    # not served: a draft is invisible to every reader
    assert (await store.search(o, profile="GUEST", text="sábado")).hits == ()
    assert await store.readable_documents(o, profile="GUEST") == []


def test_prepare_does_not_even_take_an_embedder_or_a_gate():
    """Structural: the step that runs before confirmation CANNOT spend — it is not handed the
    means to. (A keyword that is not there cannot be passed by mistake.)"""
    import inspect
    params = inspect.signature(prepare).parameters
    assert "embedder" not in params and "gate" not in params and "pace" not in params
    assert {"embedder", "gate", "pace", "now"} <= set(inspect.signature(commit).parameters)


async def test_an_error_in_prepare_costs_zero_embedding_calls(store):
    o = owner()
    for data, extractor, reason in ((b"%PDF-1.7 scanned", FakePdfExtractor(raises="no_text"),
                                     "no_text"),
                                    (b"%PDF" + b"0" * 64, FakePdfExtractor(), "over_limit")):
        doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
        embedder, gate = StubEmbedder(), GateSpy()
        out = await ingest(store, o, doc_id, data=data, embedder=embedder, embed_model=MODEL_A,
                           extractor=extractor, gate=gate,
                           limits=ExtractionLimits(max_bytes=32))
        assert out.status == INGEST_ERROR and out.reason == reason
        assert embedder.calls == 0 and gate.asked == [] and out.embedding_calls == 0
        assert await store.pending_drafts(o) == []


# ── commit: the refusals ────────────────────────────────────────────────────────────────

async def test_a_commit_without_a_prepare_is_refused(store):
    o = owner()
    doc_id = await new_doc(store, o)
    embedder = StubEmbedder()
    out = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert out.status == INGEST_NOT_PREPARED
    # a version that was BEGUN but never prepared is not a draft either
    v = await store.begin_version(o, doc_id, sha256="a" * 64, embed_model=MODEL_A, size_bytes=1)
    out = await commit(store, o, doc_id, v.version, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert out.status == INGEST_NOT_PREPARED and embedder.calls == 0
    assert (await store.get_version(o, doc_id, v.version)).state == KB_PROCESSING
    gone = await commit(store, o, str(uuid4()), 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert gone.status == INGEST_DELETED


async def test_a_commit_after_the_expiry_is_refused_and_changes_nothing(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    embedder = StubEmbedder()
    late = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A,
                        now=T0 + DRAFT_TTL)                              # the boundary itself
    assert late.status == INGEST_EXPIRED and late.expires_at == T0 + DRAFT_TTL
    assert embedder.calls == 0
    assert (await store.get_version(o, doc_id, 1)).state == KB_AWAITING_CONFIRMATION
    # CONTROL — one second earlier the same commit goes through
    ok = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A,
                      now=T0 + DRAFT_TTL - timedelta(seconds=1))
    assert ok.status == INGEST_READY and embedder.calls == ok.chunks


async def test_a_commit_for_another_model_is_a_caller_bug_and_changes_nothing(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    with pytest.raises(ValueError):
        await commit(store, o, doc_id, 1, embedder=StubEmbedder(), embed_model=MODEL_B, now=T0)
    assert (await store.get_version(o, doc_id, 1)).state == KB_AWAITING_CONFIRMATION


# ── commit: the happy path, the repeat, the race ────────────────────────────────────────

async def test_the_gate_is_handed_exactly_the_estimate_prepare_returned(store):
    o = owner()
    doc_id = await new_doc(store, o)
    prepared = await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    gate, embedder = GateSpy(), StubEmbedder()
    done = await commit(store, o, doc_id, prepared.version, embedder=embedder,
                        embed_model=MODEL_A, gate=gate, now=T0 + timedelta(minutes=5))
    assert gate.asked == [prepared.estimated_tokens]
    assert done.status == INGEST_READY and done.estimated_tokens == prepared.estimated_tokens
    assert done.embedding_calls == embedder.calls == prepared.chunks == done.chunks
    assert done.embedding_tokens == sum(embedder.reported)
    assert (await store.pending_drafts(o)) == []
    assert (await store.get_document(o, doc_id)).active.state == KB_READY
    assert len((await store.search(o, profile="GUEST", text="sábado")).hits) >= 1


async def test_a_refusing_gate_at_commit_embeds_nothing(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    embedder = StubEmbedder()
    out = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A,
                       gate=GateSpy(answer=False), now=T0)
    assert out.status == INGEST_ERROR and out.reason == "gate_refused"
    assert embedder.calls == 0 and await store.pending_drafts(o) == []


async def test_committing_twice_is_idempotent(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    embedder = StubEmbedder()
    first = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    calls = embedder.calls
    second = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert first.status == INGEST_READY and second.status == INGEST_UNCHANGED
    assert embedder.calls == calls and second.embedding_calls == 0
    assert second.embedding_tokens == 0                  # billed once, never twice
    assert (await store.get_document(o, doc_id)).active.chunks == first.chunks


async def test_two_confirmations_at_once_embed_the_draft_once(store):
    o = owner()
    doc_id = await new_doc(store, o)
    prepared = await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    embedder = StubEmbedder()
    outs = await asyncio.gather(*(commit(store, o, doc_id, 1, embedder=embedder,
                                         embed_model=MODEL_A, now=T0) for _ in range(3)))
    assert [o_.status for o_ in outs].count(INGEST_READY) == 1
    assert embedder.calls == prepared.chunks
    assert sum(o_.embedding_tokens for o_ in outs) == sum(embedder.reported)


async def test_ingest_is_prepare_then_commit(store):
    """The one-call path keeps its signature and gives the SAME result as the two steps."""
    one, two = owner(), owner()
    a, b = await new_doc(store, one), await new_doc(store, two)
    together = await ingest(store, one, a, data=MANUAL, embedder=StubEmbedder(),
                            embed_model=MODEL_A)
    prepared = await prepare(store, two, b, data=MANUAL, embed_model=MODEL_A)
    split = await commit(store, two, b, prepared.version, embedder=StubEmbedder(),
                         embed_model=MODEL_A)
    fields = ("status", "version", "reason", "chunks", "pages", "estimated_tokens",
              "embedding_tokens", "embedding_calls", "usage_reported")
    assert [getattr(together, f) for f in fields] == [getattr(split, f) for f in fields]
    assert together.status == INGEST_READY
    got = [[h.content for h in (await store.search(o, profile="GUEST", text="sábado preços",
                                                     limit=10)).hits] for o in (one, two)]
    assert got[0] == got[1] != []


async def test_preparing_the_same_bytes_again_refreshes_the_same_draft(store):
    o = owner()
    doc_id = await new_doc(store, o)
    first = await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    again = await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A,
                          now=T0 + timedelta(hours=20))
    assert again.version == first.version and again.expires_at == T0 + timedelta(hours=44)
    [draft] = await store.pending_drafts(o)
    assert draft.expires_at == again.expires_at


async def test_a_draft_beside_a_served_version_leaves_it_serving(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await ingest(store, o, doc_id, data=b"# Guia\n\nversao antiga sabado\n", embedder=StubEmbedder(),
                 embed_model=MODEL_A)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    doc = await store.get_document(o, doc_id)
    assert doc.active.version == 1 and doc.latest.version == 2
    assert doc.latest.state == KB_AWAITING_CONFIRMATION
    assert [h.version for h in (await store.search(o, profile="GUEST", text="sabado")).hits] == [1]


# ── expiry: the sweep, on the injected clock ────────────────────────────────────────────

async def test_an_unconfirmed_draft_expires_on_the_sweep_with_a_tombstone(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await store.stored_original_bytes(o) == len(MANUAL)            # CONTROL: bytes kept
    assert await expire_drafts(store, now=T0 + timedelta(hours=23)) == 0
    assert await expire_drafts(store, now=T0 + DRAFT_TTL) == 1

    v = await store.get_version(o, doc_id, 1)
    assert (v.state, v.reason) == (KB_ERROR, "expired")
    assert v.estimated_tokens > 0 and v.expires_at == T0 + DRAFT_TTL      # still answerable
    assert await store.pending_drafts(o) == []
    assert await store.stored_original_bytes(o) == 0                      # the draft's bytea
    [stone] = await store.tombstones(o)
    assert (stone.kind, stone.document_id, stone.versions) == (TOMBSTONE_EXPIRED, doc_id, (1,))
    assert stone.removed_at == T0 + DRAFT_TTL
    assert await expire_drafts(store, now=T0 + DRAFT_TTL) == 0             # idempotent
    after = await commit(store, o, doc_id, 1, embedder=StubEmbedder(), embed_model=MODEL_A,
                         now=T0 + DRAFT_TTL)
    assert after.status == INGEST_NOT_PREPARED


async def test_expiry_leaves_the_served_version_serving(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await ingest(store, o, doc_id, data=b"# Guia\n\nsabado aberto\n", embedder=StubEmbedder(),
                 embed_model=MODEL_A)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await expire_drafts(store, now=T0 + DRAFT_TTL) == 1
    doc = await store.get_document(o, doc_id)
    assert doc.active.version == 1 and doc.active.state == KB_READY
    assert doc.latest.reason == "expired"
    assert len((await store.search(o, profile="GUEST", text="sabado")).hits) == 1


async def test_a_draft_goes_with_its_owner_s_purge_and_with_a_delete(store):
    o = owner()
    a, b = await new_doc(store, o), await new_doc(store, o)
    for doc_id in (a, b):
        await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert len(await store.pending_drafts(o)) == 2
    assert await store.delete_document(o, a)
    assert [d.document_id for d in await store.pending_drafts(o)] == [b]
    assert await store.purge_owner_subtree(o.split("/")[0]) == 1
    assert await store.pending_drafts(o) == [] and await store.stored_original_bytes(o) == 0
    assert await expire_drafts(store, now=T0 + DRAFT_TTL) == 0            # nothing left to find


async def test_failing_a_draft_drops_it(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await store.fail_version(o, doc_id, 1, reason="internal")
    assert await store.pending_drafts(o) == []


async def test_another_owner_sees_no_draft_and_cannot_claim_one(store):
    o, other = owner(), owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await store.pending_drafts(other) == []
    out = await commit(store, other, doc_id, 1, embedder=StubEmbedder(), embed_model=MODEL_A, now=T0)
    assert out.status == INGEST_DELETED                   # not HIS document, so not there
    assert len(await store.pending_drafts(o)) == 1


# ── the alphabets ───────────────────────────────────────────────────────────────────────

def test_the_new_state_reason_and_tombstone_are_in_the_closed_alphabets():
    assert VALID_KB_STATES == {"processing", "awaiting_confirmation", "ready", "error"}
    assert "expired" in VALID_KB_REASONS and "expired" in VALID_TOMBSTONE_KINDS
    assert {INGEST_AWAITING, INGEST_NOT_PREPARED, INGEST_EXPIRED} <= VALID_INGEST_OUTCOMES


def test_the_adapters_never_spell_a_state_by_hand():
    """The SQL interpolates the constants; a literal state in an adapter is a second copy of the
    vocabulary that can drift without an error (a typo there matches nothing)."""
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent / "cogno_engram" / "adapters"
    for name in ("postgres.py", "in_memory.py"):
        text = (root / name).read_text(encoding="utf-8")
        for state in VALID_KB_STATES:
            assert f"'{state}'" not in text and f'"{state}"' not in text, (name, state)


# ── each guard of "no commit without a prepare", proved ALONE ───────────────────────────
#
# Two guards stand between a commit and a version nobody prepared, and each has its OWN
# question: the CLAIM asks "is there a draft?" (and nothing about the version's state), the
# COMMIT asks "is the version awaiting confirmation?". A guard no test proves alone can be
# removed without anyone noticing, so each test below builds the one shape where only ITS
# guard stands in the way.

async def _remove_draft_keep_state(store, doc_id: str, version: int) -> None:
    """A version still `awaiting_confirmation` whose draft is gone — the shape where only the
    claim's "is there a draft?" can refuse."""
    if hasattr(store, "_drafts"):
        store._drafts.pop((doc_id, version))
        return
    import psycopg
    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as conn:
        await conn.execute("DELETE FROM kb_drafts WHERE document_id = %s AND version = %s",
                           (doc_id, version))


async def test_the_claim_refuses_a_version_whose_draft_is_gone(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    await _remove_draft_keep_state(store, doc_id, 1)
    assert (await store.get_version(o, doc_id, 1)).state == KB_AWAITING_CONFIRMATION
    assert await store.claim_draft(o, doc_id, 1, now=T0) == ("missing", None)
    assert (await store.get_version(o, doc_id, 1)).state == KB_AWAITING_CONFIRMATION
    # CONTROL — with its draft, the same version IS claimed
    other = await new_doc(store, o)
    await prepare(store, o, other, data=MANUAL, embed_model=MODEL_A, now=T0)
    status, draft = await store.claim_draft(o, other, 1, now=T0)
    assert status == "claimed" and draft is not None and draft.chunk_count > 0


async def test_a_commit_refuses_a_version_out_of_awaiting_even_with_a_draft_present():
    """Built in the double: a draft that is THERE, beside a version that is no longer awaiting
    — the shape where only the commit's state check can refuse."""
    from cogno_engram.adapters.in_memory import InMemoryDocumentStore
    from documents_support import EMB_DIM
    store = InMemoryDocumentStore(embedding_dim=EMB_DIM)
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    store._versions[(doc_id, 1)].state = KB_PROCESSING          # a draft is still stored
    assert (doc_id, 1) in store._drafts
    embedder = StubEmbedder()
    out = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert out.status == INGEST_NOT_PREPARED and embedder.calls == 0
    # CONTROL — back in `awaiting_confirmation`, the same draft IS committed
    store._versions[(doc_id, 1)].state = KB_AWAITING_CONFIRMATION
    ok = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert ok.status == INGEST_READY and embedder.calls == ok.chunks > 0
