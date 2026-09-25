"""``DocumentStore.version_text`` — a version's extracted text read back, on BOTH adapters.

The management view of "what is in this document": the chunks the assistant reads, in order,
each with its page and heading path and the passage WITHOUT that path at its head. Only the
SERVED version and a DRAFT awaiting confirmation have text to show; everything else is ``None``.

Each "never" carries its CONTROL — the same setup with the one condition flipped, reading the
text that the "never" says is not there. The Postgres leg runs when a test database answers
(``tests/conftest.py``). Every document here is invented.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cogno_engram.chunking import chunk_markdown
from cogno_engram.documents import (
    COMMIT_READY,
    COMMIT_SUPERSEDED,
    DRAFT_TTL,
    KB_AWAITING_CONFIRMATION,
    KB_READY,
    MEDIA_MARKDOWN,
    MEDIA_PDF,
    VERSION_TEXT_LIMIT,
    VERSION_TEXT_MAX_LIMIT,
    KbChunk,
    KbTextChunk,
)
from cogno_engram.ingest import commit, discard_draft, expire_drafts, ingest, prepare

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import MODEL_A, FakePdfExtractor, Page, StubEmbedder, publish, store_factory, vec

DSN = resolve_test_dsn()      # module-level: the Postgres leg DROPs the kb_* tables

T0 = datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc)
MANUAL = ("# Guia\n\nBem-vindo à escola.\n\n## Horários\n\nSábado das 8h às 12h.\n\n"
          "Domingo fechado.\n\n## Preços\n\nMensalidade de R$ 450,00.\n").encode()


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    return await store_factory(request.param, DSN)()


def owner() -> str:
    return f"escola{uuid4().hex[:6]}/secretaria"


async def new_doc(store, o, *, media_type=MEDIA_MARKDOWN, title="Guia do aluno"):
    return (await store.create_document(o, title=title, profiles=["GUEST"],
                                        media_type=media_type)).id


def rows(text) -> "list[tuple]":
    return [(c.ordinal, c.page, c.heading_path, c.text) for c in text.chunks]


def expected(title: str, data: bytes) -> "list[tuple]":
    """What the chunker makes of ``data``, as ``(ordinal, page, heading_path, passage)`` — the
    passage taken off ``content`` HERE by the known head, independently of ``chunk_text``."""
    out = []
    for c in chunk_markdown(title, data.decode()):
        head = " › ".join(c.heading_path) + "\n\n"
        assert c.content.startswith(head)
        out.append((c.ordinal, c.page, c.heading_path, c.content[len(head):]))
    return out


# ── what is read: the served version, whole and in order ──────────────────────────────────

async def test_the_served_version_reads_back_in_order_with_the_path_off_the_text(store):
    o = owner()
    doc_id = await new_doc(store, o)
    out = await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A)
    assert out.status == "ready"

    text = await store.version_text(o, doc_id)
    assert (text.document_id, text.version, text.state) == (doc_id, 1, KB_READY)
    assert (text.pages, text.page, text.has_original) == (None, None, True)   # Markdown: no pages
    assert rows(text) == expected("Guia do aluno", MANUAL) and len(text.chunks) >= 3
    assert (text.has_more, text.next_after) == (False, None)
    assert all(isinstance(c, KbTextChunk) for c in text.chunks)
    # the SAME passages the search matches: path + blank line + text is the stored content
    res = await store.search(o, profile="GUEST", text="sábado", limit=10)
    [hit] = [h for h in res.hits if "8h às 12h" in h.content]
    [mine] = [c for c in text.chunks if c.ordinal == hit.ordinal]
    assert hit.content == " › ".join(mine.heading_path) + "\n\n" + mine.text
    assert await store.version_text(o, doc_id, version=1) == text          # explicit = active


async def test_a_draft_reads_back_BEFORE_its_cost_is_confirmed_and_is_what_gets_served(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await store.version_text(o, doc_id) is None       # CONTROL: nothing is served yet …
    draft = await store.version_text(o, doc_id, version=1)   # … but the draft has text to show
    assert (draft.version, draft.state, draft.has_original) == (1, KB_AWAITING_CONFIRMATION, True)
    assert rows(draft) == expected("Guia do aluno", MANUAL)

    embedder = StubEmbedder()
    done = await commit(store, o, doc_id, 1, embedder=embedder, embed_model=MODEL_A, now=T0)
    assert done.status == "ready" and embedder.calls == len(draft.chunks)
    served = await store.version_text(o, doc_id)
    assert served.state == KB_READY and rows(served) == rows(draft)   # what was checked is served


async def test_a_draft_beside_the_served_version_is_read_by_its_number(store):
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",), title="Guia",
                           chunks=(("Guia\n\nAbrimos das 8h às 12h.", vec(1.0)),))
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    served, draft = await store.version_text(o, doc_id), await store.version_text(o, doc_id,
                                                                                  version=2)
    assert (served.version, served.state) == (1, KB_READY)
    assert (draft.version, draft.state) == (2, KB_AWAITING_CONFIRMATION)
    assert [c.text for c in served.chunks] == ["Abrimos das 8h às 12h."]
    assert rows(draft) == expected("Guia", MANUAL)


# ── what is never read ──────────────────────────────────────────────────────────────────

async def test_nothing_is_read_from_a_version_being_built_failed_expired_or_discarded(store):
    o = owner()
    building = await new_doc(store, o)
    v = await store.begin_version(o, building, sha256="b" * 64, embed_model=MODEL_A, size_bytes=1)
    await store.add_chunks(o, building, v.version,
                           [KbChunk(ordinal=0, content="Guia\n\nparcial", heading_path=("Guia",),
                                    embedding=vec(1.0))])
    assert await store.version_text(o, building, version=v.version) is None       # processing
    assert await store.commit_version(o, building, v.version, pages=0) == COMMIT_READY
    assert [c.text for c in (await store.version_text(o, building)).chunks] == ["parcial"]  # CONTROL

    failed = await new_doc(store, o)
    await prepare(store, o, failed, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await store.version_text(o, failed, version=1) is not None              # CONTROL
    assert await store.fail_version(o, failed, 1, reason="internal")
    assert await store.version_text(o, failed, version=1) is None                  # error

    expired = await new_doc(store, o)
    await prepare(store, o, expired, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await expire_drafts(store, now=T0 + DRAFT_TTL) == 1
    assert await store.version_text(o, expired, version=1) is None

    discarded = await new_doc(store, o)
    await prepare(store, o, discarded, data=MANUAL, embed_model=MODEL_A, now=T0)
    assert await discard_draft(store, o, discarded, 1) == "discarded"
    assert await store.version_text(o, discarded, version=1) is None


async def test_a_claimed_draft_is_being_built_and_shows_nothing(store):
    """Between the claim and the swap the draft is gone and the chunks are partial: there is no
    WHOLE text to show, so there is none — never half of one."""
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    status, _ = await store.claim_draft(o, doc_id, 1, now=T0)
    assert status == "claimed"
    assert await store.version_text(o, doc_id, version=1) is None


async def test_a_version_the_swap_removed_and_a_deleted_document_read_nothing(store):
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",))
    v2 = await store.begin_version(o, doc_id, sha256="2" * 64, embed_model=MODEL_A, size_bytes=1)
    await store.add_chunks(o, doc_id, v2.version, [
        KbChunk(ordinal=0, content="Guia\n\nversão nova", heading_path=("Guia",), embedding=vec(1.0))])
    assert await store.version_text(o, doc_id, version=1) is not None               # CONTROL
    assert await store.commit_version(o, doc_id, v2.version, pages=0) == COMMIT_READY
    assert await store.version_text(o, doc_id, version=1) is None                   # superseded
    assert [c.text for c in (await store.version_text(o, doc_id)).chunks] == ["versão nova"]
    assert await store.delete_document(o, doc_id)
    assert await store.version_text(o, doc_id) is None
    assert await store.version_text(o, doc_id, version=2) is None


async def test_another_owner_reads_nothing_prefix_siblings_included(store):
    base = f"t{uuid4().hex[:6]}"
    mine, sibling, other = f"{base}1/p1", f"{base}10/p1", f"{base}1/p2"
    doc_id = await publish(store, mine, profiles=("GUEST",))
    assert await store.version_text(mine, doc_id) is not None                       # CONTROL
    for who in (sibling, other, f"{base}1"):
        assert await store.version_text(who, doc_id) is None, who
        assert await store.version_text(who, doc_id, version=1) is None, who
    with pytest.raises(ValueError):
        await store.version_text("  ", doc_id)
    assert await store.version_text(mine, "not-a-uuid") is None
    assert await store.version_text(mine, str(uuid4())) is None


# ── paging: an exclusive ordinal cursor, a ceiling per call ─────────────────────────────

async def _many(store, o, n: int) -> str:
    return await publish(store, o, profiles=("GUEST",), title="Guia",
                         chunks=tuple((f"Guia › Parte\n\ntrecho {i}", vec(1.0)) for i in range(n)))


async def test_the_text_pages_by_an_exclusive_ordinal_cursor(store):
    o = owner()
    doc_id = await _many(store, o, 7)
    got, after, calls = [], None, 0
    for _ in range(10):              # BOUNDED: a cursor that stops advancing must fail, not hang
        part = await store.version_text(o, doc_id, after=after, limit=3)
        calls += 1
        got += [c.ordinal for c in part.chunks]
        assert part.next_after == (part.chunks[-1].ordinal if part.has_more else None)
        if not part.has_more:
            break
        after = part.next_after
    assert got == list(range(7)) and calls == 3                  # 3 + 3 + 1, nothing twice
    end = await store.version_text(o, doc_id, after=6)
    assert (end.chunks, end.has_more, end.next_after) == ((), False, None)   # past the end: empty
    assert end.version == 1                                       # … and NOT None


async def test_a_draft_pages_by_the_same_cursor_one_chunk_at_a_time(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    whole = await store.version_text(o, doc_id, version=1)
    assert len(whole.chunks) >= 3 and not whole.has_more                     # the shape
    got, after = [], None
    for _ in range(len(whole.chunks) + 2):     # BOUNDED: a stuck cursor fails, never hangs
        part = await store.version_text(o, doc_id, version=1, after=after, limit=1)
        assert len(part.chunks) == 1
        got += part.chunks
        if not part.has_more:
            break
        after = part.next_after
    assert tuple(got) == whole.chunks


async def test_every_call_is_cut_to_the_ceiling_and_the_default_is_the_documented_one(store):
    o = owner()
    doc_id = await _many(store, o, VERSION_TEXT_MAX_LIMIT + 5)
    default = await store.version_text(o, doc_id)
    assert len(default.chunks) == VERSION_TEXT_LIMIT and default.has_more
    huge = await store.version_text(o, doc_id, limit=10 ** 6)
    assert len(huge.chunks) == VERSION_TEXT_MAX_LIMIT and huge.has_more
    rest = await store.version_text(o, doc_id, after=huge.next_after, limit=10 ** 6)
    assert [c.ordinal for c in rest.chunks] == list(range(VERSION_TEXT_MAX_LIMIT,
                                                          VERSION_TEXT_MAX_LIMIT + 5))
    assert not rest.has_more


@pytest.mark.parametrize("kwargs", [dict(page=0), dict(page=-1), dict(page="2"), dict(page=True),
                                    dict(after="3"), dict(after=1.5), dict(limit=0),
                                    dict(limit=-5), dict(limit="50"), dict(limit=True)])
async def test_a_window_that_is_not_integers_of_the_right_range_is_refused(store, kwargs):
    o = owner()
    doc_id = await publish(store, o)
    assert await store.version_text(o, doc_id) is not None                          # CONTROL
    with pytest.raises(ValueError):
        await store.version_text(o, doc_id, **kwargs)


# ── the page filter: one PDF page, and no filter where there are no pages ───────────────

PDF_PAGES = [Page(1, "Capa do guia."),
             Page(2, "\n\n".join(" ".join(f"regra{p}x{w:02d}" for w in range(40))
                                   for p in range(10))),                  # ~4400 chars: > 1 chunk
             Page(3, "Contactos da secretaria.")]


async def test_a_page_reads_one_pdf_page_and_continues_inside_it(store):
    o = owner()
    doc_id = await new_doc(store, o, media_type=MEDIA_PDF)
    out = await ingest(store, o, doc_id, data=b"%PDF-1.7 invented", embedder=StubEmbedder(),
                       embed_model=MODEL_A, extractor=FakePdfExtractor(pages=PDF_PAGES))
    assert out.status == "ready" and out.pages == 3
    whole = await store.version_text(o, doc_id)
    assert whole.pages == 3 and {c.page for c in whole.chunks} == {1, 2, 3}      # CONTROL
    on_two = [c for c in whole.chunks if c.page == 2]
    assert len(on_two) >= 2                          # the shape: a page of several chunks

    two = await store.version_text(o, doc_id, page=2)
    assert (two.page, two.pages) == (2, 3) and two.chunks == tuple(on_two)
    first = await store.version_text(o, doc_id, page=2, limit=1)
    assert first.chunks == (on_two[0],) and first.has_more
    nxt = await store.version_text(o, doc_id, page=2, after=first.next_after, limit=1)
    assert nxt.chunks == (on_two[1],)                # the cursor stays inside the page
    far = await store.version_text(o, doc_id, page=9)
    assert (far.chunks, far.has_more, far.page) == ((), False, 9)


async def test_a_page_filter_is_ignored_on_a_version_without_pages(store):
    o = owner()
    doc_id = await new_doc(store, o)
    await ingest(store, o, doc_id, data=MANUAL, embedder=StubEmbedder(), embed_model=MODEL_A)
    plain = await store.version_text(o, doc_id)
    filtered = await store.version_text(o, doc_id, page=2)
    assert plain.pages is None and plain.chunks                                   # the shape
    assert filtered.page is None and filtered.chunks == plain.chunks


# ── the numbering the host's two "not found" answers rest on ────────────────────────────

async def test_version_numbers_only_grow_and_the_latest_is_never_removed(store):
    """``None`` from ``version_text`` does not say WHY. A host tells "exists or existed, not
    served" from "never existed" by ``latest.version``: that only holds while numbers are never
    reused and the latest attempt is never the one a removal takes — pinned here, through every
    way this store removes a version."""
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",))                           # v1 served

    async def staged(tag: str) -> int:
        v = await store.begin_version(o, doc_id, sha256=tag * 64, embed_model=MODEL_A,
                                      size_bytes=1)
        await store.add_chunks(o, doc_id, v.version, [
            KbChunk(ordinal=0, content=f"Guia\n\n{tag}", heading_path=("Guia",),
                    embedding=vec(1.0))])
        return v.version

    late, newer = await staged("a"), await staged("b")                               # v2, v3
    assert await store.commit_version(o, doc_id, late, pages=0) == COMMIT_SUPERSEDED
    assert (await store.get_document(o, doc_id)).latest.version == newer == 3
    assert await store.commit_version(o, doc_id, newer, pages=0) == COMMIT_READY   # v1 removed
    assert (await store.get_document(o, doc_id)).latest.version == 3
    failed = await staged("c")
    assert failed == 4                                           # 2 is gone and is NOT reused
    assert await store.fail_version(o, doc_id, failed, reason="timeout")
    assert (await store.get_document(o, doc_id)).latest.version == 4
    for gone in (1, 2):
        assert await store.version_text(o, doc_id, version=gone) is None
    assert await store.version_text(o, doc_id, version=99) is None


# ── the original is never read to show the text ─────────────────────────────────────────

async def test_reading_the_text_never_reads_the_original():
    from cogno_engram.adapters.in_memory import InMemoryDocumentStore
    from documents_support import EMB_DIM
    store = InMemoryDocumentStore(embedding_dim=EMB_DIM)
    o = owner()
    doc_id = await publish(store, o, original=b"# Manual\nbytes nobody reads here\n")
    await prepare(store, o, doc_id, data=MANUAL, embed_model=MODEL_A, now=T0)
    before = store.original_reads
    for version in (None, 1, 2):
        assert (await store.version_text(o, doc_id, version=version)).has_original
    assert store.original_reads == before
    await store.get_original(o, doc_id)                                              # CONTROL
    assert store.original_reads == before + 1
