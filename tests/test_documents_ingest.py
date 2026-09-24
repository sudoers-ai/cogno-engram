"""``ingest()`` — the background job, end to end, with the risks it owns.

* the ceilings are applied BEFORE the extractor is called (risk 5, the half this library holds;
  the separate process with a deadline and no network is the extractor's — ``cogno-vox``);
* the gate runs BEFORE any embedding call, and a refusal costs ZERO calls;
* the usage returned is exactly what the embedder reported, summed;
* a document removed while the job runs is never put back;
* a failure keeps the served version serving; a repeat is free.

The end-to-end cases run on both adapters (``make_store``); the rest on the in-memory one.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from cogno_engram import ingest as ingest_module
from cogno_engram.adapters.in_memory import InMemoryDocumentStore
from cogno_engram.documents import (
    EXTRACTOR_REASONS,
    KB_ERROR,
    KB_PROCESSING,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    ExtractionLimits,
)
from cogno_engram.ingest import (
    INGEST_DELETED,
    INGEST_ERROR,
    INGEST_NO_ORIGINAL,
    INGEST_READY,
    INGEST_SUPERSEDED,
    INGEST_UNCHANGED,
    TokensPerMinute,
    documents_probe,
    estimate_tokens,
    ingest,
    reindex,
)

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import (
    EMB_DIM,
    MODEL_A,
    MODEL_B,
    Bookmark,
    Extracted,
    FakePdfExtractor,
    Page,
    PlainEmbedder,
    StubEmbedder,
    store_factory,
)

DSN = resolve_test_dsn()      # module-level: the Postgres leg DROPs the kb_* tables

MANUAL = """# Manual do aluno

Bem-vindo ao curso.

## Horários

### Sábado

Abrimos das 8h às 12h. Aulas práticas no laboratório.

### Domingo

Fechado.

## Mensalidade

O valor é R$ 450,00 por mês.
""".encode()


@pytest.fixture(params=["memory", "postgres"])
async def make_store(request):
    return store_factory(request.param, DSN)


def owner() -> str:
    return f"escola{uuid4().hex[:6]}/secretaria"


async def new_doc(store, o, *, media_type=MEDIA_MARKDOWN, title="Manual do aluno",
                  profiles=("GUEST",)):
    return (await store.create_document(o, title=title, profiles=list(profiles),
                                        media_type=media_type)).id


def memory_store(**kw) -> InMemoryDocumentStore:
    return InMemoryDocumentStore(embedding_dim=EMB_DIM, **kw)


# ── end to end, both adapters ────────────────────────────────────────────────────────────

async def test_a_markdown_upload_becomes_a_searchable_version_with_heading_paths(make_store):
    store = await make_store()
    o = owner()
    doc_id = await new_doc(store, o)
    embedder = StubEmbedder()
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A)
    assert out.status == INGEST_READY and out.version == 1 and out.chunks >= 3
    doc = await store.get_document(o, doc_id)
    assert doc.active.chunks == out.chunks and doc.active.has_original
    res = await store.search(o, profile="GUEST", text="sábado", limit=10)
    [hit] = [h for h in res.hits if "8h às 12h" in h.content]
    # the document's own `# Manual do aluno` is its title — said once, not twice
    assert hit.heading_path == ("Manual do aluno", "Horários", "Sábado")
    assert hit.content.startswith("Manual do aluno › Horários › Sábado\n\n")
    # the vector of the SAME text under the SAME model finds it first
    probe = embedder._vector(hit.content)
    top = await store.search(o, profile="GUEST", text="", vector=probe, embed_model=MODEL_A)
    assert top.hits[0].id == hit.id and top.hits[0].vector_score == pytest.approx(1.0)


async def test_repeating_the_job_is_free_and_duplicates_nothing(make_store):
    store = await make_store()
    o = owner()
    doc_id = await new_doc(store, o)
    first = await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A)
    again_embedder = StubEmbedder()
    again = await ingest(store, o, doc_id, data=MANUAL, embedder=again_embedder,
                         embed_model=MODEL_A)
    assert again.status == INGEST_UNCHANGED and again.version == first.version
    assert again_embedder.calls == 0 and again.embedding_tokens == 0
    assert (await store.get_document(o, doc_id)).active.chunks == first.chunks
    # with a vector every served chunk is a candidate, so this counts them all
    res = await store.search(o, profile="GUEST", text="", vector=[1.0] * EMB_DIM,
                             embed_model=MODEL_A, limit=50)
    assert len(res.hits) == first.chunks


async def test_a_document_deleted_while_the_job_runs_is_never_put_back(make_store):
    store = await make_store()
    o = owner()
    doc_id = await new_doc(store, o)

    class DeletingEmbedder(StubEmbedder):
        async def embed_with_usage(self, text):
            if self.calls == 1:
                assert await store.delete_document(o, doc_id, actor="admin")
            return await super().embed_with_usage(text)

    out = await ingest(store, o, doc_id, data=MANUAL, embedder=DeletingEmbedder(),
                       embed_model=MODEL_A)
    assert out.status == INGEST_DELETED
    assert (await store.search(o, profile="GUEST", text="sábado")).hits == ()
    assert await store.get_document(o, doc_id) is None
    assert await store.stored_original_bytes(o) == 0
    assert len(await store.tombstones(o)) == 1


# ── usage: returned exactly as reported ──────────────────────────────────────────────────

async def test_the_usage_returned_is_the_sum_of_what_the_embedder_reported():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    embedder = StubEmbedder(bonus=7)
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A)
    assert out.embedding_calls == embedder.calls == out.chunks == len(embedder.reported)
    assert out.embedding_tokens == sum(embedder.reported) > 7 * out.chunks
    assert out.usage_reported is True
    assert out.estimated_tokens > 0


async def test_an_embedder_without_usage_says_so_instead_of_reporting_zero():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    embedder = PlainEmbedder()
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A)
    assert out.status == INGEST_READY and out.embedding_calls == embedder.calls > 0
    assert out.embedding_tokens == 0 and out.usage_reported is False


# ── the gate: before any embedding, and a refusal costs zero calls ───────────────────────

async def test_a_refusing_gate_costs_zero_embedding_calls():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    asked: list[int] = []

    async def gate(estimated: int) -> bool:
        asked.append(estimated)
        return False

    embedder = StubEmbedder()
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A,
                       gate=gate)
    assert out.status == INGEST_ERROR and out.reason == "gate_refused"
    assert embedder.calls == 0 and out.embedding_calls == 0 and out.embedding_tokens == 0
    assert asked == [out.estimated_tokens] and asked[0] > 0
    assert (await store.get_document(o, doc_id)).latest.reason == "gate_refused"

    async def approve(estimated: int) -> bool:                       # CONTROL
        return True
    ok = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A,
                      gate=approve)
    assert ok.status == INGEST_READY and embedder.calls == ok.chunks


async def test_a_gate_that_cannot_answer_did_not_approve():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)

    async def broken(estimated: int) -> bool:
        raise ConnectionError("quota service down")

    embedder = StubEmbedder()
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A,
                       gate=broken)
    assert out.reason == "gate_refused" and embedder.calls == 0


def test_the_estimate_is_characters_over_four():
    from cogno_engram.documents import KbChunk
    assert estimate_tokens([KbChunk(0, "x" * 8), KbChunk(1, "y" * 9)]) == 2 + 3


# ── risk 5 (this library's half): the ceilings come BEFORE the extractor ────────────────

async def test_an_oversized_file_never_reaches_the_extractor_and_is_not_stored():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    extractor = FakePdfExtractor()
    out = await ingest(store, o, doc_id, data=b"%PDF" + b"0" * 2000, embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=extractor,
                       limits=ExtractionLimits(max_bytes=1000))
    assert out.status == INGEST_ERROR and out.reason == "over_limit"
    assert extractor.calls == []
    assert await store.stored_original_bytes(o) == 0
    # CONTROL — under the ceiling the extractor IS called, with the ceilings as keywords.
    ok = await ingest(store, o, doc_id, data=b"%PDF" + b"0" * 10, embedder=StubEmbedder(),
                      embed_model=MODEL_A, extractor=extractor,
                      limits=ExtractionLimits(max_bytes=1000, max_pages=7, timeout_s=3.0))
    assert ok.status == INGEST_READY
    assert extractor.calls == [dict(size=14, media_type=MEDIA_PDF, max_bytes=1000, max_pages=7,
                                    timeout_s=3.0)]


async def test_a_store_ceiling_below_the_limits_is_over_limit_too():
    store = memory_store(max_original_bytes=10)
    o = owner()
    doc_id = await new_doc(store, o)
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A)
    assert out.status == INGEST_ERROR and out.reason == "over_limit"
    assert await store.stored_original_bytes(o) == 0


async def test_an_extractor_that_ignores_the_page_ceiling_is_still_refused():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    extractor = FakePdfExtractor(pages=[Page(i, f"página {i}") for i in range(1, 6)])
    out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=extractor,
                       limits=ExtractionLimits(max_pages=4))
    assert out.reason == "over_limit"
    ok = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),     # CONTROL
                      embed_model=MODEL_A, extractor=extractor, limits=ExtractionLimits(max_pages=5))
    assert ok.status == INGEST_READY and ok.pages == 5


@pytest.mark.parametrize("reason", sorted(EXTRACTOR_REASONS))
async def test_every_extractor_reason_reaches_the_version_as_is(reason):
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=FakePdfExtractor(raises=reason))
    assert out.status == INGEST_ERROR and out.reason == reason
    assert (await store.get_document(o, doc_id)).latest.reason == reason


@pytest.mark.parametrize("raises", ["something_else", ""])
async def test_an_unknown_or_missing_extractor_reason_is_internal(raises):
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=FakePdfExtractor(raises=raises))
    assert out.reason == "internal"


async def test_an_extractor_answering_garbage_is_invalid_not_indexed():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    for answer in (object(), Extracted(pages=[Page(0, "x")]), Extracted(pages=[Page(1, 42)])):
        out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                           embed_model=MODEL_A, extractor=FakePdfExtractor(answer=answer))
        assert out.reason == "invalid", answer


async def test_an_extractor_that_never_returns_is_a_timeout(monkeypatch):
    monkeypatch.setattr(ingest_module, "EXTRACT_GRACE_S", 0.01)
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)

    class Hangs(FakePdfExtractor):
        async def extract(self, data, **kw):
            await asyncio.sleep(3600)

    out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=Hangs(),
                       limits=ExtractionLimits(timeout_s=0.01))
    assert out.reason == "timeout"


async def test_no_extractor_for_the_type_is_extractor_unavailable():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    for extractor in (None, type("X", (), {"media_types": frozenset({"image/png"})})()):
        out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                           embed_model=MODEL_A, extractor=extractor)
        assert out.reason == "extractor_unavailable"


async def test_markdown_that_is_not_utf8_is_invalid_and_blank_markdown_has_no_text():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    out = await ingest(store, o, doc_id, data=b"\xff\xfe\x00bad", embedder=StubEmbedder(),
                       embed_model=MODEL_A)
    assert out.reason == "invalid"
    out = await ingest(store, o, doc_id, data=b"  \n# \n\n", embedder=StubEmbedder(),
                       embed_model=MODEL_A)
    assert out.reason == "no_text"


async def test_a_pdf_whose_pages_have_no_text_is_no_text():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                       embed_model=MODEL_A,
                       extractor=FakePdfExtractor(pages=[Page(1, "  "), Page(2, "\n\n")]))
    assert out.status == INGEST_ERROR and out.reason == "no_text" and out.pages == 2


async def test_pdf_pages_keep_their_numbers_and_the_bookmark_trail():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF, title="Regulamento")
    extractor = FakePdfExtractor(
        pages=[Page(1, "Introdução."), Page(2, "Horário de sábado:\x00 8h às 12h."),
               Page(3, "Multas por atraso.")],
        outline=[Bookmark(1, "Funcionamento", 2), Bookmark(2, "Sábado", 2),
                 Bookmark(1, "Penalidades", 3)])
    out = await ingest(store, o, doc_id, data=b"%PDF", embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=extractor)
    assert out.status == INGEST_READY and out.pages == 3
    hits = {h.page: h for h in (await store.search(o, profile="GUEST", text="sábado multas "
                                                   "introdução", limit=10)).hits}
    assert hits[1].heading_path == ("Regulamento",)
    assert hits[2].heading_path == ("Regulamento", "Funcionamento", "Sábado")
    assert "\x00" not in hits[2].content
    assert hits[3].heading_path == ("Regulamento", "Penalidades")


# ── failures keep the served version; races resolve to the newest ────────────────────────

async def test_an_embedder_failing_mid_way_keeps_the_served_version_and_reports_what_it_spent():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    await ingest(store, o, doc_id, data=b"# V1\n\nsabado aberto\n", embedder=StubEmbedder(),
                 embed_model=MODEL_A)
    embedder = StubEmbedder(fail_after=2)
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=embedder, embed_model=MODEL_A)
    assert out.status == INGEST_ERROR and out.reason == "embed_failed"
    assert out.embedding_calls == 3 and out.embedding_tokens == sum(embedder.reported)
    doc = await store.get_document(o, doc_id)
    assert doc.status == KB_ERROR and doc.active.version == 1
    assert [h.content for h in (await store.search(o, profile="GUEST", text="sabado")).hits] \
        == ["Manual do aluno › V1\n\nsabado aberto"]


async def test_an_embedder_of_the_wrong_width_is_embed_failed():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(width=4),
                       embed_model=MODEL_A)
    assert out.reason == "embed_failed" and out.embedding_calls == 1


async def test_a_job_killed_half_way_is_resumed_as_the_same_version():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)

    class Killed(StubEmbedder):
        async def embed_with_usage(self, text):
            if self.calls == 1:
                raise asyncio.CancelledError()
            return await super().embed_with_usage(text)

    with pytest.raises(asyncio.CancelledError):
        await ingest(store, o, doc_id, data=MANUAL, embedder=Killed(), embed_model=MODEL_A)
    doc = await store.get_document(o, doc_id)
    assert doc.status == KB_PROCESSING and doc.latest.version == 1
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A)
    assert out.status == INGEST_READY and out.version == 1


async def test_a_bug_in_the_store_is_recorded_as_internal_and_raised():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)

    def broken(*a, **kw):
        raise RuntimeError("chunker exploded")

    original = ingest_module.chunk_markdown
    ingest_module.chunk_markdown = broken                                # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError):
            await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(),
                         embed_model=MODEL_A)
    finally:
        ingest_module.chunk_markdown = original
    assert (await store.get_document(o, doc_id)).latest.reason == "internal"


async def test_a_newer_upload_that_lands_first_supersedes_the_older_job():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)

    class RacesANewer(StubEmbedder):
        async def embed_with_usage(self, text):
            if self.calls == 0:
                newer = await ingest(store, o, doc_id, data=b"# Nova\n\nsabado novo\n",
                                     embedder=StubEmbedder(), embed_model=MODEL_A)
                assert newer.status == INGEST_READY
            return await super().embed_with_usage(text)

    out = await ingest(store, o, doc_id, data=MANUAL, embedder=RacesANewer(), embed_model=MODEL_A)
    assert out.status == INGEST_SUPERSEDED
    assert [h.content for h in (await store.search(o, profile="GUEST", text="sabado")).hits] \
        == ["Manual do aluno › Nova\n\nsabado novo"]


async def test_a_missing_document_is_deleted_before_anything_runs():
    store = memory_store()
    embedder = StubEmbedder()
    out = await ingest(store, owner(), str(uuid4()), data=MANUAL, embedder=embedder,
                       embed_model=MODEL_A)
    assert out.status == INGEST_DELETED and embedder.calls == 0
    with pytest.raises(TypeError):
        await ingest(store, owner(), "x", data="text", embedder=embedder, embed_model=MODEL_A)


# ── a global model swap: re-index from the stored original ──────────────────────────────

async def test_a_model_swap_is_reindexed_from_the_stored_original():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A)
    assert [d.id for d in await store.stale_documents(embed_model=MODEL_B)] == [doc_id]
    during = await store.search(o, profile="GUEST", text="sábado", vector=[1.0] * EMB_DIM,
                                embed_model=MODEL_B)
    assert during.models_unavailable == (MODEL_A,) and during.hits          # lexical, marked
    out = await reindex(store, o, doc_id, embedder=StubEmbedder(), embed_model=MODEL_B)
    assert out.status == INGEST_READY and out.version == 2
    assert await store.stale_documents(embed_model=MODEL_B) == []
    after = await store.search(o, profile="GUEST", text="sábado", vector=[1.0] * EMB_DIM,
                               embed_model=MODEL_B)
    assert after.degradations == () and after.hits[0].vector_score is not None


async def test_reindex_without_a_stored_original_says_so():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A,
                 keep_original=False)
    out = await reindex(store, o, doc_id, embedder=StubEmbedder(), embed_model=MODEL_B)
    assert out.status == INGEST_NO_ORIGINAL
    gone = await reindex(store, o, str(uuid4()), embedder=StubEmbedder(), embed_model=MODEL_B)
    assert gone.status == INGEST_DELETED


# ── pace ─────────────────────────────────────────────────────────────────────────────────

async def test_tokens_per_minute_waits_exactly_until_the_bucket_holds_the_request():
    now = [0.0]
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    pace = TokensPerMinute(600, clock=lambda: now[0], sleep=sleep)       # 10 tokens/second
    await pace(600)
    assert slept == []
    await pace(100)
    assert slept == [pytest.approx(10.0)]
    now[0] += 30.0                                                        # refills 300
    await pace(300)
    assert len(slept) == 1
    await pace(5000)                                                      # bigger than the bucket
    assert slept[-1] == pytest.approx(60.0)
    with pytest.raises(ValueError):
        TokensPerMinute(0)


async def test_ingest_paces_every_embedding_call():
    store = memory_store()
    o = owner()
    doc_id = await new_doc(store, o)
    paced: list[int] = []

    async def pace(tokens: int) -> None:
        paced.append(tokens)

    out = await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A,
                       pace=pace)
    assert len(paced) == out.embedding_calls and sum(paced) == out.estimated_tokens


# ── the health probe on the double ───────────────────────────────────────────────────────

async def test_the_probe_runs_on_the_double():
    await documents_probe(memory_store(), embed_model=MODEL_A)
