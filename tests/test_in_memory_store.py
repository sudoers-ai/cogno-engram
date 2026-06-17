import pytest

from cogno_engram.types import HybridWeights, MemoryRecord, RetrievalQuery, TurnRecord


async def test_blank_scope_is_rejected(store):
    with pytest.raises(ValueError):
        await store.recent_sessions("")
    with pytest.raises(ValueError):
        await store.save_memory(MemoryRecord(scope="  ", category="fact", content="x"))


async def test_session_and_turn_roundtrip(store):
    session = await store.create_session("acme/phone1")
    await store.save_turn(TurnRecord(session.id, "acme/phone1", 2, "second"))
    await store.save_turn(TurnRecord(session.id, "acme/phone1", 1, "first"))
    turns = await store.load_turns(session.id)
    assert [t.turn_n for t in turns] == [1, 2]            # ordered by turn_n
    assert (await store.get_session(session.id)).scope == "acme/phone1"


async def test_close_session_records_summary(store):
    session = await store.create_session("s")
    await store.close_session(session.id, summary="all done")
    reloaded = await store.get_session(session.id)
    assert reloaded.ended_at is not None and reloaded.summary == "all done"


async def test_scope_isolation(store):
    await store.save_memory(MemoryRecord("tenantA/p", "fact", "A secret"))
    await store.save_memory(MemoryRecord("tenantB/p", "fact", "B secret"))
    a = await store.load_memories("tenantA/p")
    assert [m.content for m in a] == ["A secret"]          # never sees B


async def test_memory_upsert_dedup(store):
    await store.save_memory(MemoryRecord("s", "fact", "likes coffee", confidence=0.7))
    await store.save_memory(MemoryRecord("s", "fact", "likes coffee", confidence=1.0,
                                         embedding=[1.0, 0.0]))
    mems = await store.load_memories("s")
    assert len(mems) == 1 and mems[0].confidence == 1.0 and mems[0].embedding == [1.0, 0.0]


async def test_hybrid_lexical_ranks_match_first(store):
    await store.save_memory(MemoryRecord("s", "fact", "the user lives in Berlin"))
    await store.save_memory(MemoryRecord("s", "fact", "the user likes jazz music"))
    out = await store.load_memories("s", query=RetrievalQuery(text="where does the user live"))
    assert out[0].content == "the user lives in Berlin"


async def test_hybrid_vector_component(store):
    await store.save_memory(MemoryRecord("s", "fact", "alpha", embedding=[1.0, 0.0]))
    await store.save_memory(MemoryRecord("s", "fact", "beta", embedding=[0.0, 1.0]))
    out = await store.load_memories("s", query=RetrievalQuery(embedding=[0.9, 0.1]))
    assert out[0].content == "alpha"


async def test_feedback_score_breaks_tie(store):
    await store.save_memory(MemoryRecord("s", "fact", "same words here"))
    await store.save_memory(MemoryRecord("s", "fact", "same words here too"))
    # bump the second one's feedback so it wins on an otherwise-similar lexical score
    n = await store.adjust_feedback_score("s", "too", delta=5.0)
    assert n == 1
    out = await store.load_memories("s", query=RetrievalQuery(text="same words here too"),
                                    weights=HybridWeights(vector=0.0, lexical=0.4, feedback=0.05))
    assert out[0].content == "same words here too"


async def test_category_filter(store):
    await store.save_memory(MemoryRecord("s", "fact", "f"))
    await store.save_memory(MemoryRecord("s", "preference", "p"))
    out = await store.load_memories("s", query=RetrievalQuery(categories=["preference"]))
    assert [m.category for m in out] == ["preference"]


async def test_set_feedback_on_turn(store):
    session = await store.create_session("s")
    await store.save_turn(TurnRecord(session.id, "s", 1, "hi"))
    await store.set_feedback("s", session.id, 1, -1)
    assert (await store.load_turns(session.id))[0].feedback == -1


async def test_session_lock_serializes(store):
    order = []

    async def worker(tag):
        async with store.session_lock("s", "sess"):
            order.append(f"{tag}-in")
            order.append(f"{tag}-out")

    import asyncio
    await asyncio.gather(worker("a"), worker("b"))
    # each critical section is uninterrupted
    assert order in (["a-in", "a-out", "b-in", "b-out"], ["b-in", "b-out", "a-in", "a-out"])


def test_supports_vector_capability(store):
    from cogno_engram.ports import SupportsVectorSearch
    assert isinstance(store, SupportsVectorSearch) and store.supports_vector() is True
