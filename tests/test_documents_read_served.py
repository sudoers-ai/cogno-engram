"""``DocumentStore.read_served`` — the text of ONE served document, for a READER, on BOTH adapters.

The reader path's whole-document read: what a turn calls when a document is needed WHOLE rather
than as its best passages. It applies THE filter of the reader path (this owner, the reader's
profile among the published ones, the served version, ``ready``) on every call, so an id a model
echoes back — or invents — reads nothing its reader could not have found.

The three "nevers" are the security twins of the whole-document read (P9): a document of
ANOTHER profile, a DRAFT, and a document of another owner never come back. Each carries its
CONTROL — the same setup with the one condition flipped, reading the text the "never" says is not
there — so a twin that passes cannot be a constant. The Postgres leg runs when a test database
answers (``tests/conftest.py``). Every document here is invented.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from cogno_engram.documents import (
    COMMIT_READY,
    VERSION_TEXT_LIMIT,
    VERSION_TEXT_MAX_LIMIT,
    KbChunk,
    KbServedText,
)
from cogno_engram.ingest import commit, ingest, prepare

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import MODEL_A, StubEmbedder, publish, store_factory, vec

DSN = resolve_test_dsn()      # module-level: the Postgres leg DROPs the kb_* tables

T0 = datetime(2030, 1, 7, 9, 0, tzinfo=timezone.utc)
GUIDE = ("# Guia\n\nBem-vindo ao instituto.\n\n## Horários\n\nSábado das 8h às 12h.\n\n"
         "Domingo fechado.\n\n## Preços\n\nMensalidade de R$ 450,00.\n").encode()
#: A word no other document here carries: when it shows up, the text it lives in was read.
MARKER = "zircônio"


@pytest.fixture(params=["memory", "postgres"])
async def store(request):
    return await store_factory(request.param, DSN)()


def owner() -> str:
    return f"inst{uuid4().hex[:6]}/secretaria"


def rows(text) -> "list[tuple]":
    return [(c.ordinal, c.page, c.heading_path, c.text) for c in text.chunks]


async def ingested(store, o, *, profiles=("GUEST",), title="Guia do aluno", data=GUIDE) -> str:
    doc = await store.create_document(o, title=title, profiles=list(profiles),
                                      media_type="text/markdown")
    out = await ingest(store, o, doc.id, data=data, embedder=StubEmbedder(), embed_model=MODEL_A)
    assert out.status == "ready"
    return doc.id


def whole_text(served) -> str:
    return "\n".join(c.text for c in served.chunks)


# ── what is read: the served version, in order, for a profile it is published to ────────

async def test_a_reader_reads_the_served_version_in_order_the_same_chunks_the_admin_sees(store):
    o = owner()
    doc_id = await ingested(store, o)

    served = await store.read_served(o, doc_id, profile="GUEST")
    assert isinstance(served, KbServedText)
    assert (served.document_id, served.version, served.title, served.pages) == \
        (doc_id, 1, "Guia do aluno", None)                          # Markdown: no pages
    assert len(served.chunks) >= 3 and (served.has_more, served.next_after) == (False, None)
    # the SAME chunks, in the same order, as the administrator's read of the served version —
    # the path OFF the text: one extraction, two doors
    assert rows(served) == rows(await store.version_text(o, doc_id))
    assert "Sábado das 8h às 12h." in whole_text(served)


async def test_it_pages_by_the_same_exclusive_cursor_and_is_cut_to_the_same_ceiling(store):
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",), title="Guia",
                           chunks=tuple((f"Guia › Parte\n\ntrecho {i}", vec(1.0))
                                        for i in range(VERSION_TEXT_MAX_LIMIT + 5)))
    got, after = [], None
    for _ in range(10):              # BOUNDED: a cursor that stops advancing must fail, not hang
        part = await store.read_served(o, doc_id, profile="GUEST", after=after, limit=90)
        got += [c.ordinal for c in part.chunks]
        assert part.next_after == (part.chunks[-1].ordinal if part.has_more else None)
        if not part.has_more:
            break
        after = part.next_after
    assert got == list(range(VERSION_TEXT_MAX_LIMIT + 5))            # nothing twice, nothing lost
    default = await store.read_served(o, doc_id, profile="GUEST")
    assert len(default.chunks) == VERSION_TEXT_LIMIT and default.has_more
    huge = await store.read_served(o, doc_id, profile="GUEST", limit=10 ** 6)
    assert len(huge.chunks) == VERSION_TEXT_MAX_LIMIT and huge.has_more
    end = await store.read_served(o, doc_id, profile="GUEST", after=VERSION_TEXT_MAX_LIMIT + 4)
    assert (end.chunks, end.has_more, end.next_after) == ((), False, None)   # past the end …
    assert end.version == 1                                                 # … and NOT None


@pytest.mark.parametrize("kwargs", [dict(after="3"), dict(after=1.5), dict(after=True),
                                    dict(limit=0), dict(limit=-5), dict(limit="50"),
                                    dict(limit=True)])
async def test_a_window_that_is_not_integers_of_the_right_range_is_refused(store, kwargs):
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",))
    assert await store.read_served(o, doc_id, profile="GUEST") is not None          # CONTROL
    with pytest.raises(ValueError):
        await store.read_served(o, doc_id, profile="GUEST", **kwargs)


async def test_the_profile_is_required_and_a_blank_one_is_no_wildcard(store):
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",))
    with pytest.raises(TypeError):
        await store.read_served(o, doc_id)                      # type: ignore[call-arg]
    for blank in ("", "   "):
        with pytest.raises(ValueError):
            await store.read_served(o, doc_id, profile=blank)
    with pytest.raises(ValueError):
        await store.read_served("  ", doc_id, profile="GUEST")


# ── TWIN 1: a document of ANOTHER profile never comes back — by any id ──────────────────

async def test_twin_a_document_of_another_profile_never_comes_back(store):
    o = owner()
    staff_only = await ingested(store, o, profiles=("ADMIN",), title="Notas internas",
                                data=f"# Notas\n\nO {MARKER} fica no cofre.\n".encode())
    for reader in ("EMPLOYEE", "GUEST", "admin", " ADMIN2"):
        assert await store.read_served(o, staff_only, profile=reader) is None, reader
    # CONTROL — the SAME id, the profile it IS published to: the text is there to be read
    control = await store.read_served(o, staff_only, profile="ADMIN")
    assert control is not None and MARKER in whole_text(control)
    # … and the search agrees: the profile that cannot read it cannot find it either
    assert (await store.search(o, profile="EMPLOYEE", text=MARKER)).hits == ()
    assert (await store.search(o, profile="ADMIN", text=MARKER)).hits != ()          # CONTROL


async def test_twin_a_change_of_profiles_takes_effect_on_the_next_read(store):
    o = owner()
    doc_id = await ingested(store, o, profiles=("EMPLOYEE",))
    assert await store.read_served(o, doc_id, profile="EMPLOYEE") is not None        # CONTROL
    assert await store.set_profiles(o, doc_id, ["ADMIN"])
    assert await store.read_served(o, doc_id, profile="EMPLOYEE") is None
    assert await store.read_served(o, doc_id, profile="ADMIN") is not None
    assert await store.set_profiles(o, doc_id, [])                  # nobody: the safe direction
    assert await store.read_served(o, doc_id, profile="ADMIN") is None


async def test_twin_another_owner_reads_nothing_prefix_siblings_included(store):
    base = f"t{uuid4().hex[:6]}"
    mine, sibling, other = f"{base}1/p1", f"{base}10/p1", f"{base}1/p2"
    doc_id = await publish(store, mine, profiles=("GUEST",))
    assert await store.read_served(mine, doc_id, profile="GUEST") is not None        # CONTROL
    for who in (sibling, other, f"{base}1"):
        assert await store.read_served(who, doc_id, profile="GUEST") is None, who
    assert await store.read_served(mine, "not-a-uuid", profile="GUEST") is None
    assert await store.read_served(mine, str(uuid4()), profile="GUEST") is None


# ── TWIN 2: a DRAFT never comes back — alone, or beside the served version ──────────────

async def test_twin_a_draft_never_comes_back_the_admin_read_does_show_it(store):
    o = owner()
    doc = await store.create_document(o, title="Rascunho", profiles=["GUEST"],
                                      media_type="text/markdown")
    await prepare(store, o, doc.id, data=f"# Rascunho\n\nO {MARKER} ainda não saiu.\n".encode(),
                  embed_model=MODEL_A, now=T0)
    assert await store.read_served(o, doc.id, profile="GUEST") is None
    # CONTROL — the draft HAS text, and the administrator's read shows it by its number
    draft = await store.version_text(o, doc.id, version=1)
    assert draft is not None and MARKER in "\n".join(c.text for c in draft.chunks)


async def test_twin_a_draft_beside_the_served_version_is_never_what_the_reader_gets(store):
    o = owner()
    doc_id = await ingested(store, o, title="Guia")
    await prepare(store, o, doc_id, data=f"# Guia\n\nNova regra: {MARKER}.\n".encode(),
                  embed_model=MODEL_A, now=T0)
    served = await store.read_served(o, doc_id, profile="GUEST")
    assert served.version == 1 and MARKER not in whole_text(served)
    assert "Sábado das 8h às 12h." in whole_text(served)            # the served text, whole
    # CONTROL — the draft is v2 and carries the marker; the reader sees it only once confirmed
    assert MARKER in "\n".join(c.text for c in (await store.version_text(o, doc_id,
                                                                         version=2)).chunks)
    done = await commit(store, o, doc_id, 2, embedder=StubEmbedder(), embed_model=MODEL_A, now=T0)
    assert done.status == "ready"
    after = await store.read_served(o, doc_id, profile="GUEST")
    assert after.version == 2 and MARKER in whole_text(after)


async def test_a_version_being_built_failed_or_deleted_reads_nothing(store):
    o = owner()
    building = (await store.create_document(o, title="Guia", profiles=["GUEST"],
                                            media_type="text/markdown")).id
    v = await store.begin_version(o, building, sha256="b" * 64, embed_model=MODEL_A, size_bytes=1)
    await store.add_chunks(o, building, v.version,
                           [KbChunk(ordinal=0, content=f"Guia\n\n{MARKER}", heading_path=("Guia",),
                                    embedding=vec(1.0))])
    assert await store.read_served(o, building, profile="GUEST") is None             # processing
    assert await store.commit_version(o, building, v.version, pages=0) == COMMIT_READY
    assert [c.text for c in (await store.read_served(o, building,
                                                     profile="GUEST")).chunks] == [MARKER]  # CONTROL
    # a NEWER version being built beside it: the reader keeps the served one, never the partial
    v2 = await store.begin_version(o, building, sha256="c" * 64, embed_model=MODEL_A, size_bytes=1)
    await store.add_chunks(o, building, v2.version,
                           [KbChunk(ordinal=0, content="Guia\n\nparcial", heading_path=("Guia",),
                                    embedding=vec(1.0))])
    assert [c.text for c in (await store.read_served(o, building, profile="GUEST")).chunks] == \
        [MARKER]
    assert await store.fail_version(o, building, v2.version, reason="internal")
    assert (await store.read_served(o, building, profile="GUEST")).version == 1   # still served
    assert await store.delete_document(o, building)
    assert await store.read_served(o, building, profile="GUEST") is None             # deleted


async def test_reading_the_served_text_never_reads_the_original():
    from cogno_engram.adapters.in_memory import InMemoryDocumentStore
    from documents_support import EMB_DIM
    store = InMemoryDocumentStore(embedding_dim=EMB_DIM)
    o = owner()
    doc_id = await publish(store, o, profiles=("GUEST",), original=b"# bytes nobody reads\n")
    before = store.original_reads
    assert await store.read_served(o, doc_id, profile="GUEST") is not None
    assert store.original_reads == before
    await store.get_original(o, doc_id)                                              # CONTROL
    assert store.original_reads == before + 1
