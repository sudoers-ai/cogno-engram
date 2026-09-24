"""The DocumentStore CONTRACT — every test here runs against BOTH adapters.

The in-memory leg always runs; the Postgres leg runs when a test database answers (see
``tests/conftest.py`` — the database name must contain "test", and the fixture DROPs the
document tables). One body, two stores: parity is not a second suite that can drift, it is the
same assertions.

Each risk test carries its CONTROL: the same setup with the one condition flipped, asserting
the thing the risk test says never happens DOES happen there. A "never" whose control does not
produce the "ever" proves nothing — the search could simply be returning nothing at all.

All documents are invented; no owner, profile or title names a real tenant or person.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cogno_engram import documents as kb
from cogno_engram.documents import (
    COMMIT_DELETED,
    COMMIT_READY,
    COMMIT_SUPERSEDED,
    KB_EMBED_SPACE_UNAVAILABLE,
    KB_ERROR,
    KB_PROCESSING,
    KB_READY,
    MEDIA_MARKDOWN,
    TOMBSTONE_DELETED,
    TOMBSTONE_PURGED,
    DocumentLimitReached,
    KbChunk,
    OriginalTooLarge,
    embed_model_label,
)
from cogno_engram.ports import DocumentStore

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import EMB_DIM, MODEL_A, MODEL_B, publish, store_factory, vec

# ENGRAM_TEST_DSN, else `engram_test` on the local server. Module-level ON PURPOSE: the Postgres
# leg DROPs the kb_* tables, and `tests/conftest.py` decides by this attribute whether a
# database may be touched at all (read at fixture time, so a blanked DSN skips the leg).
DSN = resolve_test_dsn()


@pytest.fixture(params=["memory", "postgres"])
async def make_store(request):
    return store_factory(request.param, DSN)


@pytest.fixture
async def docs(make_store):
    return await make_store()


def owner() -> str:
    return f"acme{uuid4().hex[:8]}/persona-a"


async def ids(store, owner_key, profile, text="horários sábado", **kw):
    res = await store.search(owner_key, profile=profile, text=text, **kw)
    return [h.document_id for h in res.hits]


# ── the port ─────────────────────────────────────────────────────────────────────────

async def test_both_adapters_satisfy_the_port(docs):
    assert isinstance(docs, DocumentStore)
    assert docs.embedding_dim == EMB_DIM


# ── risk 1: a wrong profile never sees a chunk, nor a title ─────────────────────────────

async def test_a_profile_reads_only_what_is_published_to_it(docs):
    o = owner()
    staff_only = await publish(docs, o, title="Manual interno", profiles=("EMPLOYEE",))
    shared = await publish(docs, o, title="Horários", profiles=("EMPLOYEE", "GUEST"))

    assert await ids(docs, o, "GUEST") == [shared]
    # CONTROL — the staff-only document IS findable by the query, for the profile it is for.
    assert set(await ids(docs, o, "EMPLOYEE")) == {staff_only, shared}


async def test_the_titles_that_describe_the_tool_follow_the_same_rule(docs):
    """The tool's description is generated from the readable titles, so a title is content: a
    GUEST must not learn an EMPLOYEE document exists by reading the tool list."""
    o = owner()
    await publish(docs, o, title="Manual interno", profiles=("EMPLOYEE",))
    await publish(docs, o, title="Horários", profiles=("GUEST",))
    guest = [d.title for d in await docs.readable_documents(o, profile="GUEST")]
    staff = [d.title for d in await docs.readable_documents(o, profile="EMPLOYEE")]
    assert guest == ["Horários"]
    assert staff == ["Manual interno"]                         # CONTROL: the title exists


async def test_profiles_are_exact_labels_no_case_folding_no_hierarchy(docs):
    o = owner()
    await publish(docs, o, profiles=("EMPLOYEE",))
    assert await ids(docs, o, "employee") == []
    assert await ids(docs, o, "ADMIN") == []
    assert len(await ids(docs, o, " EMPLOYEE ")) == 1          # stripped, like on write


async def test_a_blank_or_missing_profile_is_refused_not_widened(docs):
    o = owner()
    await publish(docs, o)
    with pytest.raises(ValueError):
        await docs.search(o, profile="  ", text="horários")
    with pytest.raises(ValueError):
        await docs.readable_documents(o, profile="")
    with pytest.raises(TypeError):
        await docs.search(o, text="horários")                  # type: ignore[call-arg]


async def test_a_document_published_to_nobody_is_read_by_nobody(docs):
    o = owner()
    await publish(docs, o, profiles=())
    for profile in ("EMPLOYEE", "GUEST", "DEFAULT"):
        assert await ids(docs, o, profile) == []


async def test_changing_who_may_read_takes_effect_on_the_next_search(docs):
    o = owner()
    doc_id = await publish(docs, o, profiles=("EMPLOYEE",))
    assert await ids(docs, o, "GUEST") == []
    assert await docs.set_profiles(o, doc_id, ["GUEST"])
    assert await ids(docs, o, "GUEST") == [doc_id]
    assert await ids(docs, o, "EMPLOYEE") == []


# ── risk 4: another owner never appears — by KEY, including a prefix sibling ─────────────

async def test_another_owner_never_appears_prefix_siblings_included(docs):
    base = f"t{uuid4().hex[:6]}"
    owners = [f"{base}1/p1", f"{base}1/p2", f"{base}10/p1", f"{base}1"]
    placed = {o: await publish(docs, o, profiles=("GUEST",)) for o in owners}
    for o in owners:
        # CONTROL built in: each owner DOES find its own document with the same query.
        assert await ids(docs, o, "GUEST") == [placed[o]], o
    assert await docs.get_document(owners[0], placed[owners[1]]) is None
    assert await docs.get_original(owners[0], placed[owners[1]]) is None
    assert not await docs.set_profiles(owners[0], placed[owners[1]], ["GUEST"])
    assert not await docs.delete_document(owners[0], placed[owners[1]])
    assert await ids(docs, owners[1], "GUEST") == [placed[owners[1]]]


async def test_a_blank_owner_is_refused(docs):
    with pytest.raises(ValueError):
        await docs.search("", profile="GUEST", text="x")
    with pytest.raises(ValueError):
        await docs.purge_owner_subtree(" ")


# ── risk 3: only the SERVED version in state ready is ever read ─────────────────────────

async def test_a_version_being_built_or_failed_is_never_read(docs):
    o = owner()
    doc = await docs.create_document(o, title="Manual", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    v = await docs.begin_version(o, doc.id, sha256="a" * 64, embed_model=MODEL_A, size_bytes=1)
    assert v.state == KB_PROCESSING
    await docs.add_chunks(o, doc.id, v.version,
                          [KbChunk(ordinal=0, content="Horários de sábado", embedding=vec(1.0))])
    assert await ids(docs, o, "GUEST", vector=vec(1.0), embed_model=MODEL_A) == []
    assert await docs.readable_documents(o, profile="GUEST") == []

    assert await docs.fail_version(o, doc.id, v.version, reason="no_text")
    got = await docs.get_document(o, doc.id)
    assert got.status == KB_ERROR and got.latest.reason == "no_text" and got.active is None
    assert await ids(docs, o, "GUEST", vector=vec(1.0), embed_model=MODEL_A) == []

    # CONTROL — the same content, committed, IS read.
    v2 = await docs.begin_version(o, doc.id, sha256="b" * 64, embed_model=MODEL_A, size_bytes=1)
    await docs.add_chunks(o, doc.id, v2.version,
                          [KbChunk(ordinal=0, content="Horários de sábado", embedding=vec(1.0))])
    assert await docs.commit_version(o, doc.id, v2.version, pages=0) == COMMIT_READY
    assert await ids(docs, o, "GUEST", vector=vec(1.0), embed_model=MODEL_A) == [doc.id]


async def test_the_old_version_answers_until_the_new_one_is_ready_then_the_swap_is_whole(docs):
    o = owner()
    doc_id = await publish(docs, o, profiles=("GUEST",),
                           chunks=(("versão antiga: sábado fechado", vec(1.0)),))
    v2 = await docs.begin_version(o, doc_id, sha256="c" * 64, embed_model=MODEL_A, size_bytes=1)
    await docs.add_chunks(o, doc_id, v2.version,
                          [KbChunk(ordinal=0, content="versão nova: sábado aberto", embedding=vec(1.0))])
    res = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_A)
    assert [h.content for h in res.hits] == ["versão antiga: sábado fechado"]
    assert await docs.commit_version(o, doc_id, v2.version, pages=0) == COMMIT_READY
    res = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_A)
    assert [h.content for h in res.hits] == ["versão nova: sábado aberto"]
    doc = await docs.get_document(o, doc_id)
    assert doc.active.version == v2.version == doc.latest.version


async def test_a_failed_rebuild_leaves_the_served_version_serving(docs):
    o = owner()
    doc_id = await publish(docs, o, profiles=("GUEST",))
    v2 = await docs.begin_version(o, doc_id, sha256="d" * 64, embed_model=MODEL_A, size_bytes=1)
    assert await docs.fail_version(o, doc_id, v2.version, reason="timeout")
    doc = await docs.get_document(o, doc_id)
    assert doc.status == KB_ERROR and doc.latest.reason == "timeout"
    assert doc.active is not None and doc.active.state == KB_READY
    assert await ids(docs, o, "GUEST") == [doc_id]


async def test_an_unknown_failure_reason_is_stored_as_internal(docs):
    o = owner()
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    v = await docs.begin_version(o, doc.id, sha256="e" * 64, embed_model=MODEL_A, size_bytes=1)
    await docs.fail_version(o, doc.id, v.version, reason="/home/x/secret.pdf exploded")
    assert (await docs.get_document(o, doc.id)).latest.reason == "internal"


async def test_an_older_version_committed_late_is_superseded_not_swapped_in(docs):
    o = owner()
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    v1 = await docs.begin_version(o, doc.id, sha256="1" * 64, embed_model=MODEL_A, size_bytes=1)
    v2 = await docs.begin_version(o, doc.id, sha256="2" * 64, embed_model=MODEL_A, size_bytes=1)
    for v, text in ((v1, "primeira sábado"), (v2, "segunda sábado")):
        await docs.add_chunks(o, doc.id, v.version,
                              [KbChunk(ordinal=0, content=text, embedding=vec(1.0))])
    assert await docs.commit_version(o, doc.id, v2.version, pages=0) == COMMIT_READY
    assert await docs.commit_version(o, doc.id, v1.version, pages=0) == COMMIT_SUPERSEDED
    res = await docs.search(o, profile="GUEST", text="sábado")
    assert [h.content for h in res.hits] == ["segunda sábado"]


async def test_beginning_the_same_content_and_model_twice_is_one_version(docs):
    o = owner()
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    a = await docs.begin_version(o, doc.id, sha256="f" * 64, embed_model=MODEL_A, size_bytes=1)
    b = await docs.begin_version(o, doc.id, sha256="f" * 64, embed_model=MODEL_A, size_bytes=1)
    assert a.version == b.version
    # a different MODEL over the same bytes is a different index — a new version
    c = await docs.begin_version(o, doc.id, sha256="f" * 64, embed_model=MODEL_B, size_bytes=1)
    assert c.version == a.version + 1


async def test_restaging_the_same_ordinals_does_not_duplicate_chunks(docs):
    o = owner()
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    v = await docs.begin_version(o, doc.id, sha256="9" * 64, embed_model=MODEL_A, size_bytes=1)
    batch = [KbChunk(ordinal=i, content=f"sábado parte {i}", embedding=vec(1.0)) for i in range(3)]
    await docs.add_chunks(o, doc.id, v.version, batch)
    await docs.add_chunks(o, doc.id, v.version, batch)
    await docs.commit_version(o, doc.id, v.version, pages=0)
    assert (await docs.get_document(o, doc.id)).active.chunks == 3
    res = await docs.search(o, profile="GUEST", text="sábado", limit=10)
    assert len(res.hits) == 3


# ── condition (e): delete leaves search AT ONCE, originals of every version, tombstone ────

async def test_a_deleted_document_leaves_every_search_at_once_and_leaves_a_tombstone(docs):
    o = owner()
    doc_id = await publish(docs, o, profiles=("GUEST",))
    v2 = await docs.begin_version(o, doc_id, sha256="7" * 64, embed_model=MODEL_A, size_bytes=4,
                                  original=b"next")
    assert await ids(docs, o, "GUEST") == [doc_id]              # CONTROL: it was searchable
    assert await docs.stored_original_bytes(o) > 0              # CONTROL: bytes were stored

    assert await docs.delete_document(o, doc_id, actor="admin-1")
    assert await ids(docs, o, "GUEST") == []
    assert await ids(docs, o, "GUEST", vector=vec(1.0), embed_model=MODEL_A) == []
    assert await docs.readable_documents(o, profile="GUEST") == []
    assert await docs.get_document(o, doc_id) is None
    assert await docs.stored_original_bytes(o) == 0             # EVERY version's bytes

    [stone] = await docs.tombstones(o)
    assert (stone.document_id, stone.kind, stone.actor) == (doc_id, TOMBSTONE_DELETED, "admin-1")
    assert set(stone.versions) == {1, v2.version}
    assert stone.removed_at is not None
    # no content in a tombstone: its fields are ids, numbers and labels only
    assert set(vars(stone)) == {"owner_key", "document_id", "versions", "kind", "actor",
                                "removed_at"}
    assert not await docs.delete_document(o, doc_id)            # twice is a no-op


async def test_a_purge_of_a_subtree_never_touches_a_sibling_prefix(docs):
    base = f"t{uuid4().hex[:6]}"
    inside = [f"{base}1", f"{base}1/p1", f"{base}1/p2"]
    outside = [f"{base}10/p1", f"{base}1x"]
    for o in inside + outside:
        await publish(docs, o, profiles=("GUEST",))
    assert await docs.stored_original_bytes(f"{base}1") > 0     # CONTROL

    assert await docs.purge_owner_subtree(f"{base}1", actor="offboarding") == 3
    for o in inside:
        assert await ids(docs, o, "GUEST") == [], o
        assert await docs.list_documents(o) == []
    assert await docs.stored_original_bytes(f"{base}1") == 0
    for o in outside:
        assert len(await ids(docs, o, "GUEST")) == 1, o
        assert await docs.stored_original_bytes(o) > 0
    stones = await docs.tombstones(f"{base}1")
    assert len(stones) == 3 and {s.kind for s in stones} == {TOMBSTONE_PURGED}
    assert await docs.tombstones(f"{base}10") == []


async def test_a_writer_that_finishes_after_the_delete_writes_nothing(docs):
    o = owner()
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    v = await docs.begin_version(o, doc.id, sha256="5" * 64, embed_model=MODEL_A, size_bytes=1,
                                 original=b"x")
    assert await docs.delete_document(o, doc.id)
    late = [KbChunk(ordinal=0, content="sábado", embedding=vec(1.0))]
    assert await docs.add_chunks(o, doc.id, v.version, late) is False
    assert await docs.commit_version(o, doc.id, v.version, pages=0) == COMMIT_DELETED
    assert await docs.begin_version(o, doc.id, sha256="5" * 64, embed_model=MODEL_A,
                                    size_bytes=1, original=b"x") is None
    assert await ids(docs, o, "GUEST") == []
    assert await docs.stored_original_bytes(o) == 0


async def test_old_tombstones_can_be_pruned(docs):
    from datetime import datetime, timedelta, timezone
    o = owner()
    doc_id = await publish(docs, o)
    await docs.delete_document(o, doc_id)
    assert await docs.prune_tombstones(before=datetime.now(timezone.utc) - timedelta(days=1)) == 0
    assert await docs.prune_tombstones(before=datetime.now(timezone.utc) + timedelta(days=1)) >= 1
    assert await docs.tombstones(o) == []


# ── the original: a ceiling before it is written, never read by a search ─────────────────

async def test_an_original_over_the_ceiling_is_refused_before_anything_is_written(make_store):
    store = await make_store(max_original_bytes=16)
    o = owner()
    doc = await store.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    with pytest.raises(OriginalTooLarge):
        await store.begin_version(o, doc.id, sha256="0" * 64, embed_model=MODEL_A,
                                  size_bytes=17, original=b"x" * 17)
    assert (await store.get_document(o, doc.id)).latest is None
    assert await store.stored_original_bytes(o) == 0
    ok = await store.begin_version(o, doc.id, sha256="0" * 64, embed_model=MODEL_A,
                                   size_bytes=16, original=b"x" * 16)        # CONTROL
    assert ok.has_original and await store.stored_original_bytes(o) == 16


async def test_the_original_round_trips_for_the_management_path(docs):
    o = owner()
    doc_id = await publish(docs, o, original=b"# Manual\nconteudo\n")
    assert await docs.get_original(o, doc_id) == b"# Manual\nconteudo\n"
    assert await docs.get_original(o, doc_id, version=99) is None


# ── models (condition b) — the full treatment is in test_documents_models.py ─────────────

async def test_a_vector_of_another_model_is_never_compared(docs):
    o = owner()
    await publish(docs, o, profiles=("GUEST",),
                  chunks=(("conteúdo sem a palavra da pergunta", vec(1.0)),), model=MODEL_A)
    same = await docs.search(o, profile="GUEST", text="zzz", vector=vec(1.0), embed_model=MODEL_A)
    # CONTROL: under ITS model the chunk is found by the vector alone, cosine 1.0 …
    assert len(same.hits) == 1 and same.hits[0].vector_score == pytest.approx(1.0)
    assert same.degradations == ()
    # … and the SAME numbers labelled with another model find nothing: no cosine was taken.
    other = await docs.search(o, profile="GUEST", text="zzz", vector=vec(1.0), embed_model=MODEL_B)
    assert other.hits == ()
    assert other.degradations == (KB_EMBED_SPACE_UNAVAILABLE,)
    assert other.models_unavailable == (MODEL_A,)


async def test_a_lexical_match_of_another_model_comes_back_words_only(docs):
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(("abrimos no sábado", vec(1.0)),))
    res = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_B)
    [hit] = res.hits
    assert hit.vector_score is None and hit.lexical_score > 0
    assert res.degradations == (KB_EMBED_SPACE_UNAVAILABLE,)


async def test_no_vector_at_all_is_lexical_only_and_says_so(docs):
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(("abrimos no sábado", vec(1.0)),))
    res = await docs.search(o, profile="GUEST", text="sábado", embed_model=MODEL_A)
    assert [h.vector_score for h in res.hits] == [None]
    assert res.degradations == (KB_EMBED_SPACE_UNAVAILABLE,)
    assert res.models_unavailable == (MODEL_A,)


async def test_the_query_vector_and_label_are_validated(docs):
    o = owner()
    with pytest.raises(ValueError):
        await docs.search(o, profile="GUEST", text="x", vector=vec(1.0))            # no label
    with pytest.raises(ValueError):
        await docs.search(o, profile="GUEST", text="x", vector=[1.0, 0.0], embed_model=MODEL_A)
    with pytest.raises(ValueError):
        await docs.search(o, profile="GUEST", text="x", vector=vec(1.0),
                          embed_model=embed_model_label("stub:alpha", 768))
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    with pytest.raises(ValueError):
        await docs.begin_version(o, doc.id, sha256="0" * 64, size_bytes=1,
                                 embed_model=embed_model_label("stub:alpha", 768))


async def test_the_documents_to_reindex_after_a_model_swap(docs):
    o = owner()
    old = await publish(docs, o, model=MODEL_A)
    await publish(docs, o, model=MODEL_B)
    stale = await docs.stale_documents(embed_model=MODEL_B, limit=1000)
    assert old in {d.id for d in stale if d.owner_key == o}
    assert {d.id for d in stale if d.owner_key == o} == {old}


# ── ranking: raw scores, deterministic order ─────────────────────────────────────────────

async def test_scores_are_raw_no_floor_is_applied(docs):
    """An unrelated chunk is still returned by the vector alone, with its low score — which
    score is "relevant" is calibrated by the caller, not decided here."""
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(("nada a ver", vec(0.1, 1.0)),))
    res = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_A)
    [hit] = res.hits
    assert 0 < hit.vector_score < 0.2 and hit.lexical_score == 0
    assert hit.score == pytest.approx(0.6 * hit.vector_score)


async def test_the_tie_break_decides_the_cut_whatever_the_physical_order(docs):
    """Rows written in the OPPOSITE of the tie order — the larger document id first, each one's
    ordinals backwards — so a store that relied on physical order would cut the wrong rows."""
    o = owner()

    async def create():
        return (await docs.create_document(o, title="M", profiles=["GUEST"],
                                           media_type=MEDIA_MARKDOWN)).id

    # The documents' rows must be WRITTEN larger-id first: the plan walks them in the order they
    # sit, so with the smaller one first a store that ignored the tie-break would still cut
    # right — a coin flip on two uuid4s, measured (the first version of this test passed under
    # the mutation). Keep the last one only if the next is smaller; otherwise drop it and retry.
    large = await create()
    while True:
        small = await create()
        if small < large:
            break
        await docs.delete_document(o, large)
        large = small
    for doc_id in (large, small):
        v = await docs.begin_version(o, doc_id, sha256="3" * 64, embed_model=MODEL_A, size_bytes=1)
        await docs.add_chunks(o, doc_id, v.version, [
            KbChunk(ordinal=i, content="sábado", embedding=vec(1.0)) for i in (2, 1, 0)])
        assert await docs.commit_version(o, doc_id, v.version, pages=0) == COMMIT_READY
    for kwargs in ({"vector": vec(1.0), "embed_model": MODEL_A}, {}):
        cut = await docs.search(o, profile="GUEST", text="sábado", limit=2, **kwargs)
        assert [(h.document_id, h.ordinal) for h in cut.hits] == [(small, 0), (small, 1)], kwargs


async def test_ties_break_by_document_version_ordinal(docs):
    o = owner()
    same = (("sábado", vec(1.0)), ("sábado", vec(1.0)), ("sábado", vec(1.0)))
    a = await publish(docs, o, profiles=("GUEST",), chunks=same)
    b = await publish(docs, o, profiles=("GUEST",), chunks=same)
    res = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                            embed_model=MODEL_A, limit=10)
    got = [(h.document_id, h.version, h.ordinal) for h in res.hits]
    assert got == sorted(got) and len(got) == 6
    assert {a, b} == {g[0] for g in got}
    # the tie-break decides WHICH rows survive a limit, not only their order
    cut = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                            embed_model=MODEL_A, limit=3)
    assert [(h.document_id, h.version, h.ordinal) for h in cut.hits] == got[:3]


async def test_a_hit_carries_its_provenance_and_a_content_free_id(docs):
    o = owner()
    doc_id = await publish(docs, o, title="Manual", profiles=("GUEST",))
    [hit] = (await docs.search(o, profile="GUEST", text="sábado")).hits
    assert hit.id == f"kb:{doc_id}.1.0" == kb.chunk_id(doc_id, 1, 0)
    assert hit.title == "Manual" and hit.heading_path == ("Manual",) and hit.page is None
    assert hit.embed_model == MODEL_A


# ── management ───────────────────────────────────────────────────────────────────────────

async def test_the_document_ceiling_is_enforced_on_insert(docs):
    o = owner()
    for _ in range(2):
        await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN,
                                   max_documents=2)
    with pytest.raises(DocumentLimitReached):
        await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN,
                                   max_documents=2)
    assert len(await docs.list_documents(o)) == 2


async def test_an_unsupported_media_type_is_refused(docs):
    with pytest.raises(ValueError):
        await docs.create_document(owner(), title="M", profiles=["GUEST"], media_type="text/html")


async def test_a_document_without_attempts_reads_processing(docs):
    o = owner()
    doc = await docs.create_document(o, title="  Manual   de  bolso ", profiles=["GUEST", "GUEST"],
                                     media_type=MEDIA_MARKDOWN)
    assert doc.status == KB_PROCESSING and doc.active is None and doc.latest is None
    assert doc.title == "Manual de bolso" and doc.profiles == ("GUEST",)
    [listed] = await docs.list_documents(o)
    assert listed.id == doc.id


# ── the score (consultor's notes on 3d5780c): renormalised, [0, 1], all or none ──────────

async def test_without_a_vector_the_score_IS_the_lexical_score(docs):
    """An absent component is renormalised away, never counted as zero: ``0.4·l`` would put
    every degraded search under any floor calibrated on full scores."""
    o = owner()
    await publish(docs, o, profiles=("GUEST",),
                  chunks=(("abrimos no sábado de manhã", vec(0.5, 1.0)),))
    lexical = await docs.search(o, profile="GUEST", text="sábado", embed_model=MODEL_A)
    hybrid = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                               embed_model=MODEL_A)
    [lx], [hy] = lexical.hits, hybrid.hits
    assert lx.lexical_score == hy.lexical_score > 0             # the SAME lexical component
    assert lx.vector_score is None and lx.score == lx.lexical_score
    # CONTROL: with the vector the fusion is the weighted one, and it is a different number.
    assert hy.score == pytest.approx(0.6 * hy.vector_score + 0.4 * hy.lexical_score)
    assert hy.score != lx.score


async def test_within_one_result_either_every_hit_has_a_vector_score_or_none_does(docs):
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(("sábado A", vec(1.0)),), model=MODEL_A)
    await publish(docs, o, profiles=("GUEST",), chunks=(("sábado B", vec(1.0)),), model=MODEL_B)
    mixed = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                              embed_model=MODEL_B, limit=10)
    assert len(mixed.hits) == 2
    assert {h.vector_score for h in mixed.hits} == {None}       # one scale: all lexical
    assert mixed.models_unavailable == (MODEL_A,)
    assert mixed.degradations == (KB_EMBED_SPACE_UNAVAILABLE,)
    # CONTROL — one model only: every hit carries a vector score.
    solo = owner()
    await publish(docs, solo, profiles=("GUEST",), chunks=(("sábado", vec(1.0)),) * 2,
                  model=MODEL_B)
    full = await docs.search(solo, profile="GUEST", text="sábado", vector=vec(1.0),
                             embed_model=MODEL_B)
    assert len(full.hits) == 2 and None not in {h.vector_score for h in full.hits}
    assert full.degradations == ()


async def test_a_chunk_without_its_embedding_can_never_be_stored(docs):
    """The other side of all-or-none: a ready version with an unembedded chunk would be a
    served version the vector cannot fully rank. It is refused on the way in, whole batch."""
    o = owner()
    doc = await docs.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN)
    v = await docs.begin_version(o, doc.id, sha256="8" * 64, embed_model=MODEL_A, size_bytes=1)
    batch = [KbChunk(ordinal=0, content="sábado com vector", embedding=vec(1.0)),
             KbChunk(ordinal=1, content="sábado sem vector", embedding=None)]
    with pytest.raises(ValueError):
        await docs.add_chunks(o, doc.id, v.version, batch)
    with pytest.raises(ValueError):
        await docs.add_chunks(o, doc.id, v.version,
                              [KbChunk(ordinal=0, content="largura errada", embedding=[1.0])])
    await docs.commit_version(o, doc.id, v.version, pages=0)
    assert (await docs.get_document(o, doc.id)).active.chunks == 0      # not even the good one


LONG = " ".join(["sábado"] * 6000 + ["horário"] * 3000)


@pytest.mark.parametrize("text", ["sábado", "sábado sábado sábado sábado", "sábado horário",
                                  "horário " * 50])
async def test_no_score_leaves_the_unit_interval(docs, text):
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(
        (LONG, vec(1.0)), ("sábado", vec(-1.0)), ("sábado curto", vec(0.3, 0.9)),
        ("nada", vec(0.0, 1.0))))
    for kwargs in ({}, {"vector": vec(1.0)}, {"vector": vec(-1.0)}):
        res = await docs.search(o, profile="GUEST", text=text, embed_model=MODEL_A, limit=10,
                                **kwargs)
        assert res.hits
        for h in res.hits:
            for x in (h.score, h.lexical_score) + ((h.vector_score,) if h.vector_score is not None
                                                   else ()):
                assert 0.0 <= x <= 1.0, (kwargs, h.content[:20], x)


async def test_an_anti_parallel_vector_is_cut_to_zero_not_negative(docs):
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(("nada", vec(-1.0)),))
    [hit] = (await docs.search(o, profile="GUEST", text="zzz", vector=vec(1.0),
                               embed_model=MODEL_A)).hits
    assert hit.vector_score == 0.0 and hit.score == 0.0


async def test_weights_are_renormalised_and_validated(docs):
    from cogno_engram.types import HybridWeights
    o = owner()
    await publish(docs, o, profiles=("GUEST",), chunks=(("sábado", vec(1.0)),))
    res = await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                            embed_model=MODEL_A, weights=HybridWeights(vector=3.0, lexical=1.0))
    [h] = res.hits
    assert h.score == pytest.approx((3.0 * h.vector_score + 1.0 * h.lexical_score) / 4.0)
    with pytest.raises(ValueError):
        await docs.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                          embed_model=MODEL_A, weights=HybridWeights(vector=0.0, lexical=0.0))
