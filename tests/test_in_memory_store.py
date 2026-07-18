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


async def test_save_turn_dedups_same_coordinate(store):
    # Match the Postgres ON CONFLICT (scope, session_id, turn_n) DO NOTHING: a re-saved turn
    # coordinate is a no-op, not a duplicate row (was a divergence — in-memory double-counted).
    session = await store.create_session("acme/phone1")
    await store.save_turn(TurnRecord(session.id, "acme/phone1", 1, "first"))
    await store.save_turn(TurnRecord(session.id, "acme/phone1", 1, "first-again"))   # dup coord
    turns = await store.load_turns(session.id)
    assert [t.turn_n for t in turns] == [1]              # one row, not two
    assert turns[0].user_input == "first"               # first write wins (DO NOTHING)


@pytest.mark.asyncio
async def test_turn_trace_roundtrip_and_upsert(store):
    from cogno_engram.types import TurnTrace
    sid = "sess-1"
    await store.save_turn_trace(TurnTrace(sid, "acme/u1", 2,
        {"ner": {"aristotelian": {"TIME": {"tag": "TOMORROW", "desc": "relative"}}}}))
    await store.save_turn_trace(TurnTrace(sid, "acme/u1", 1, {"ner": {"intent": "SOCIAL"}}))
    traces = await store.traces_for_session(sid)
    assert [t.turn_n for t in traces] == [1, 2]                       # ordered by turn_n
    assert traces[1].trace["ner"]["aristotelian"]["TIME"]["tag"] == "TOMORROW"
    # UPSERT: re-saving the same (scope, session, turn_n) replaces, not duplicates
    await store.save_turn_trace(TurnTrace(sid, "acme/u1", 1, {"ner": {"intent": "ACTION_REQUEST"}}))
    traces = await store.traces_for_session(sid)
    assert len(traces) == 2 and traces[0].trace["ner"]["intent"] == "ACTION_REQUEST"
    # blank scope rejected (engram invariant)
    with pytest.raises(ValueError):
        await store.save_turn_trace(TurnTrace(sid, "", 3, {}))


async def test_admin_turns_and_scopes_span_a_scope_subtree(store):
    # admin reads span a scope SUBTREE (a tenant's whole history: tenant_id/identity)
    await store.save_turn(TurnRecord("s1", "acme/u1", 0, "a"))
    await store.save_turn(TurnRecord("s2", "acme/u2", 0, "b"))
    await store.save_turn(TurnRecord("s3", "acme/u1", 1, "c"))
    await store.save_turn(TurnRecord("s9", "globex/u9", 0, "x"))   # other tenant — excluded

    turns, total = await store.admin_turns("acme")
    assert total == 3 and {t.scope for t in turns} == {"acme/u1", "acme/u2"}
    assert "globex/u9" not in {t.scope for t in turns}
    # distinct scopes (the identity sidebar)
    assert await store.admin_scopes("acme") == ["acme/u1", "acme/u2"]
    # an exact identity scope narrows to that subtree
    only_u1, total_u1 = await store.admin_turns("acme/u1")
    assert total_u1 == 2 and all(t.scope == "acme/u1" for t in only_u1)
    # pagination
    page, total = await store.admin_turns("acme", limit=1, offset=0)
    assert len(page) == 1 and total == 3
    # blank prefix is rejected (never silently span everything)
    with pytest.raises(ValueError):
        await store.admin_turns("")


async def test_close_session_records_summary(store):
    session = await store.create_session("s")
    await store.close_session(session.id, summary="all done")
    reloaded = await store.get_session(session.id)
    assert reloaded.ended_at is not None and reloaded.summary == "all done"


async def test_idle_sessions_finds_stale_turn_derived_sessions(store):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    # a turn-derived session (no create_session — the host only save_turn()s), last activity 45m ago
    await store.save_turn(TurnRecord("sess-idle", "t/u1", 0, "q0",
                                     created_at=now - timedelta(minutes=50)))
    await store.save_turn(TurnRecord("sess-idle", "t/u1", 1, "q1",
                                     created_at=now - timedelta(minutes=45)))
    # a fresh session — active 1m ago → NOT idle
    await store.save_turn(TurnRecord("sess-fresh", "t/u2", 0, "hi",
                                     created_at=now - timedelta(minutes=1)))

    idle = await store.idle_sessions(idle_seconds=1800)   # 30 min
    assert [s.id for s in idle] == ["sess-idle"]
    assert idle[0].scope == "t/u1" and idle[0].started_at == now - timedelta(minutes=50)


async def test_close_session_upserts_and_excludes_from_idle_scan(store):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    await store.save_turn(TurnRecord("sess-x", "t/u", 0, "q",
                                     created_at=now - timedelta(minutes=45)))
    assert [s.id for s in await store.idle_sessions(idle_seconds=1800)] == ["sess-x"]

    # close_session(scope=...) upserts a closed row for a session that never had create_session
    await store.close_session("sess-x", summary="consolidated", scope="t/u")
    closed = await store.get_session("sess-x")
    assert closed is not None and closed.ended_at is not None and closed.summary == "consolidated"
    assert await store.idle_sessions(idle_seconds=1800) == []      # not re-picked


async def test_idle_sessions_oldest_first_and_limit(store):
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    for i, mins in enumerate((90, 60, 120)):
        await store.save_turn(TurnRecord(f"s{i}", "t/u", 0, "q",
                                         created_at=now - timedelta(minutes=mins)))
    ordered = await store.idle_sessions(idle_seconds=1800)
    assert [s.id for s in ordered] == ["s2", "s0", "s1"]           # 120m, 90m, 60m — oldest first
    assert len(await store.idle_sessions(idle_seconds=1800, limit=2)) == 2


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


async def test_purge_scope_removes_all_scope_data_and_isolates(store):
    from cogno_engram.types import TurnTrace
    keep_scope, drop_scope = "acme/keep", "acme/drop"
    # populate BOTH scopes across sessions/turns/traces/memories
    for sc in (keep_scope, drop_scope):
        sess = await store.create_session(sc)
        await store.save_turn(TurnRecord(sess.id, sc, 1, "hi"))
        await store.save_turn_trace(TurnTrace(sess.id, sc, 1, {"ner": {"intent": "SOCIAL"}}))
        await store.save_memory(MemoryRecord(scope=sc, category="fact", content="x"))

    removed = await store.purge_scope(drop_scope)
    assert removed == 4                                    # session + turn + trace + memory

    # the purged scope is empty on every surface …
    assert await store.memory_count(drop_scope) == 0
    assert await store.recent_turns(drop_scope) == []
    assert await store.recent_sessions(drop_scope) == []
    # … and the neighbour scope is untouched
    assert await store.memory_count(keep_scope) == 1
    assert len(await store.recent_sessions(keep_scope)) == 1


async def test_purge_scope_rejects_blank_scope(store):
    with pytest.raises(ValueError):
        await store.purge_scope("")
