"""
Integration tests for the Postgres + pgvector adapter.

Gated: they need a real Postgres with the ``vector`` extension. Set
``ENGRAM_TEST_DSN`` to a scratch database; the suite auto-skips if it is unset
or unreachable. The fixture DROPs and recreates the engram tables (with a tiny
embedding dim) so runs are deterministic — point it at a throwaway DB.

**The database name must contain "test"** or ``tests/conftest.py`` aborts the run: these
tests do not just delete rows, they leave the schema with a ``vector(8)`` embedding column
that a real 768-dimension embedder cannot write to. The example below used to say
``/postgres`` — a database that exists on every server, production ones included.

    docker run -d --rm --name engram-pg -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=engram_test -p 55432:5432 pgvector/pgvector:pg16
    ENGRAM_TEST_DSN=postgresql://postgres:postgres@localhost:55432/engram_test \
        python3 -m pytest tests/test_postgres_integration.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.rows import dict_row  # noqa: E402

from cogno_engram.adapters.postgres import (  # noqa: E402
    PostgresKnowledgeGraph,
    PostgresStore,
    ensure_schema,
)
from cogno_engram.types import (  # noqa: E402
    GraphEdge,
    GraphNode,
    HybridWeights,
    MemoryRecord,
    RetrievalQuery,
    TurnRecord,
)

DSN = os.getenv("ENGRAM_TEST_DSN", "")
EMB_DIM = 8

# (asyncio_mode=auto in pyproject handles the async marker — no module pytestmark)


async def _connect():
    return await psycopg.AsyncConnection.connect(DSN, autocommit=True, connect_timeout=3)


async def _pg_available() -> bool:
    if not DSN:
        return False
    try:
        conn = await _connect()
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pg():
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to a reachable Postgres+pgvector to run")
    conn = await _connect()
    for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns", "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM)
    await conn.close()
    yield DSN


@pytest.fixture
def store(pg):
    return PostgresStore(dsn=pg, mask_pii=True)


@pytest.fixture
def graph(pg):
    return PostgresKnowledgeGraph(dsn=pg)


def _emb(*head):
    """Build an EMB_DIM-wide vector from a few leading values (rest zeros)."""
    v = list(head) + [0.0] * (EMB_DIM - len(head))
    return v[:EMB_DIM]


# ── sessions / turns ──────────────────────────────────────────────────────

async def test_session_and_turn_roundtrip(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.save_turn(TurnRecord(s.id, scope, 2, "second"))
    await store.save_turn(TurnRecord(s.id, scope, 1, "first"))
    turns = await store.load_turns(s.id)
    assert [t.turn_n for t in turns] == [1, 2]
    assert (await store.get_session(s.id)).scope == scope


@pytest.mark.asyncio
async def test_turn_trace_jsonb_roundtrip_and_upsert(store):
    from cogno_engram.types import TurnTrace
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    aristo = {"ner": {"aristotelian": {"QUANTITY": {"tag": "45 REAIS", "desc": "amount"}}}}
    await store.save_turn_trace(TurnTrace(s.id, scope, 1, aristo))
    got = await store.traces_for_session(s.id)
    assert len(got) == 1
    assert got[0].trace["ner"]["aristotelian"]["QUANTITY"]["tag"] == "45 REAIS"   # jsonb → dict
    # UPSERT replaces, not duplicates
    await store.save_turn_trace(TurnTrace(s.id, scope, 1, {"ner": {"intent": "SOCIAL"}}))
    got = await store.traces_for_session(s.id)
    assert len(got) == 1 and got[0].trace["ner"]["intent"] == "SOCIAL"


async def test_admin_traces_subtree_since_and_pagination(store):
    """Mirror of the in-memory contract on the real adapter: subtree (not prefix) match,
    newest-first, inclusive ``since``, total independent of the page."""
    from datetime import datetime, timedelta, timezone

    from cogno_engram.types import TurnTrace

    tenant = f"t{uuid4().hex[:8]}"
    t0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    sids = [str(uuid4()) for _ in range(4)]
    for i, (scope, sid) in enumerate([(f"{tenant}/u1", sids[0]), (f"{tenant}/u1", sids[0]),
                                      (f"{tenant}/u2", sids[1]), (f"{tenant}x/u9", sids[2]),
                                      (f"other/{tenant}", sids[3])]):
        await store.save_turn_trace(TurnTrace(sid, scope, i, {"i": i},
                                              created_at=t0 + timedelta(hours=i)))
    rows, total = await store.admin_traces(tenant)
    assert total == 3 and [r.trace["i"] for r in rows] == [2, 1, 0]
    rows, total = await store.admin_traces(tenant, since=t0 + timedelta(hours=1))
    assert total == 2 and [r.trace["i"] for r in rows] == [2, 1]
    rows, total = await store.admin_traces(tenant, limit=1, offset=2)
    assert total == 3 and [r.trace["i"] for r in rows] == [0]


async def test_admin_turns_subtree_pagination_and_scopes(store):
    # a unique tenant prefix so the assertion is isolated from other tests' rows
    tenant = f"adm{uuid4().hex[:8]}"
    s1 = await store.create_session(f"{tenant}/u1")
    s2 = await store.create_session(f"{tenant}/u2")
    await store.save_turn(TurnRecord(s1.id, f"{tenant}/u1", 0, "a"))
    await store.save_turn(TurnRecord(s2.id, f"{tenant}/u2", 0, "b"))
    await store.save_turn(TurnRecord(s1.id, f"{tenant}/u1", 1, "c"))
    # a different tenant must NOT leak in
    other = await store.create_session(f"oth{uuid4().hex[:8]}/u9")
    await store.save_turn(TurnRecord(other.id, other.scope, 0, "x"))

    turns, total = await store.admin_turns(tenant)
    assert total == 3 and {t.scope for t in turns} == {f"{tenant}/u1", f"{tenant}/u2"}
    assert await store.admin_scopes(tenant) == [f"{tenant}/u1", f"{tenant}/u2"]
    # exact identity subtree
    _t, total_u1 = await store.admin_turns(f"{tenant}/u1")
    assert total_u1 == 2
    # pagination keeps the running total
    page, total = await store.admin_turns(tenant, limit=2, offset=0)
    assert len(page) == 2 and total == 3


async def test_close_session_and_recent(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.close_session(s.id, summary="done")
    assert (await store.get_session(s.id)).summary == "done"
    recent = await store.recent_sessions(scope)
    assert recent and recent[0].id == s.id


async def test_idle_sessions_and_close_upsert(store):
    # turn-derived sessions (the host save_turn()s without create_session): idle scan groups the
    # turns table, keeps those past the cutoff and not-yet-closed, and close_session(scope=) upserts
    # a closed row so the next scan skips it. Backdate created_at via a raw conn (save_turn defaults
    # created_at to now()).
    scope = f"jan{uuid4().hex[:8]}/u"
    idle_id, fresh_id = str(uuid4()), str(uuid4())
    await store.save_turn(TurnRecord(idle_id, scope, 0, "q0"))
    await store.save_turn(TurnRecord(idle_id, scope, 1, "q1"))
    await store.save_turn(TurnRecord(fresh_id, scope, 0, "hi"))
    conn = await _connect()
    await conn.execute("UPDATE turns SET created_at = now() - interval '45 min' WHERE session_id = %s",
                       (idle_id,))

    idle = await store.idle_sessions(idle_seconds=1800)
    ids = [s.id for s in idle]
    assert idle_id in ids and fresh_id not in ids

    await store.close_session(idle_id, summary="consolidated", scope=scope)
    closed = await store.get_session(idle_id)
    assert closed is not None and closed.ended_at is not None and closed.summary == "consolidated"
    assert idle_id not in [s.id for s in await store.idle_sessions(idle_seconds=1800)]
    await conn.close()


async def test_a_session_that_GREW_after_closing_comes_back_for_consolidation(store):
    """The SQL half of the frozen-memory fix — the in-memory adapter cannot prove this one,
    because the `t.created_at > s.ended_at` comparison is the query.

    A host that derives `session_id` from (tenant, channel, sender), as a messaging gateway
    must so an out-of-band message lands in the contact's own thread, never mints a second
    session for a contact. Excluding every closed session therefore froze Tier 3 at the first
    quiet spell, permanently. Measured live (2026-08): 20 of 22, 10 of 13 and 8 of 10 turns
    arrived after their session had been declared over.

    Mutation: drop `OR t.created_at > s.ended_at` from the WHERE and this dies."""
    scope = f"grow{uuid4().hex[:8]}/u"
    sid = str(uuid4())
    conn = await _connect()
    try:
        await store.save_turn(TurnRecord(sid, scope, 0, "oi"))
        await conn.execute("UPDATE turns SET created_at = now() - interval '3 h' "
                           "WHERE session_id = %s", (sid,))
        await store.close_session(sid, summary="turno 1", scope=scope)
        await conn.execute("UPDATE sessions SET ended_at = now() - interval '3 h' WHERE id = %s",
                           (sid,))
        assert sid not in [s.id for s in await store.idle_sessions(idle_seconds=1800)]

        # the contact comes back later — SAME session id, because it never rotates
        await store.save_turn(TurnRecord(sid, scope, 1, "voltei"))
        await conn.execute("UPDATE turns SET created_at = now() - interval '45 min' "
                           "WHERE session_id = %s AND turn_n = 1", (sid,))
        assert sid in [s.id for s in await store.idle_sessions(idle_seconds=1800)], (
            "turns after the close must bring the session back")

        # …and re-closing quiets it again: once per new burst, not once per tick
        await store.close_session(sid, summary="turnos 1-2", scope=scope)
        assert sid not in [s.id for s in await store.idle_sessions(idle_seconds=1800)]
    finally:
        await conn.close()


async def test_pii_masking_on_write(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.save_turn(TurnRecord(s.id, scope, 1, "meu email é joao@x.com", pii_types=["EMAIL"]))
    [t] = await store.load_turns(s.id)
    assert "joao@x.com" not in t.user_input and "[EMAIL MASKED]" in t.user_input


async def test_set_feedback(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.save_turn(TurnRecord(s.id, scope, 1, "hi"))
    await store.set_feedback(scope, s.id, 1, -1)
    assert (await store.load_turns(s.id))[0].feedback == -1


# ── memories / hybrid retrieval ───────────────────────────────────────────

async def test_memory_upsert_and_scope_isolation(store):
    a, b = f"t/{uuid4()}", f"t/{uuid4()}"
    await store.save_memory(MemoryRecord(a, "fact", "A secret"))
    await store.save_memory(MemoryRecord(b, "fact", "B secret"))
    await store.save_memory(MemoryRecord(a, "fact", "A secret", confidence=0.5))  # upsert
    got = await store.load_memories(a)
    assert [m.content for m in got] == ["A secret"]
    assert got[0].confidence == 0.5


async def test_bm25_retrieval(store):
    scope = f"t/{uuid4()}"
    await store.save_memory(MemoryRecord(scope, "fact", "o usuário mora em Berlim"))
    await store.save_memory(MemoryRecord(scope, "fact", "o usuário gosta de jazz"))
    # plainto_tsquery ANDs all terms, so every term must be present in the match.
    out = await store.load_memories(scope, query=RetrievalQuery(text="usuário mora"))
    assert [m.content for m in out] == ["o usuário mora em Berlim"]


async def test_vector_retrieval(store):
    scope = f"t/{uuid4()}"
    await store.save_memory(MemoryRecord(scope, "fact", "alpha", embedding=_emb(1.0, 0.0)))
    await store.save_memory(MemoryRecord(scope, "fact", "beta", embedding=_emb(0.0, 1.0)))
    out = await store.load_memories(scope, query=RetrievalQuery(embedding=_emb(0.9, 0.1)))
    assert out and out[0].content == "alpha"


async def test_hybrid_and_feedback_boost(store):
    scope = f"t/{uuid4()}"
    await store.save_memory(MemoryRecord(scope, "fact", "pagamento via pix", embedding=_emb(1.0)))
    await store.save_memory(MemoryRecord(scope, "fact", "pagamento via boleto", embedding=_emb(0.9, 0.4)))
    n = await store.adjust_feedback_score(scope, "boleto", delta=8.0)
    assert n == 1
    out = await store.load_memories(
        scope, query=RetrievalQuery(text="pagamento boleto", embedding=_emb(1.0)),
        weights=HybridWeights(vector=0.1, lexical=0.4, feedback=0.05))
    assert out[0].content == "pagamento via boleto"


async def test_category_filter(store):
    scope = f"t/{uuid4()}"
    await store.save_memory(MemoryRecord(scope, "fact", "f"))
    await store.save_memory(MemoryRecord(scope, "preference", "p"))
    out = await store.load_memories(scope, query=RetrievalQuery(categories=["preference"]))
    assert [m.category for m in out] == ["preference"]


async def test_session_lock(store):
    scope = f"t/{uuid4()}"
    async with store.session_lock(scope, "sess"):
        pass  # acquires + releases the pg advisory lock without error


# ── knowledge graph ───────────────────────────────────────────────────────

async def _build_jose_rex(graph, scope):
    await graph.upsert_node(GraphNode(scope, "José", "PERSON"))
    await graph.upsert_node(GraphNode(scope, "Rex", "ANIMAL"))
    await graph.upsert_node(GraphNode(scope, "Pastor Alemão", "CONCEPT"))
    await graph.upsert_edge(GraphEdge(scope, "José", "Rex", "OWNS", source_session="sess1"))
    await graph.upsert_edge(GraphEdge(scope, "Rex", "Pastor Alemão", "BREED", source_session="sess1"))


async def test_node_upsert_and_find(graph):
    scope = f"t/{uuid4()}"
    a = await graph.upsert_node(GraphNode(scope, "José", "PERSON"))
    b = await graph.upsert_node(GraphNode(scope, "josé", "PERSON", attributes={"age": 40}))
    assert a == b
    node = await graph.find_node(scope, "JOSÉ")
    assert node is not None and node.attributes == {"age": 40}


async def test_node_label_matched_literally_not_as_like_pattern(graph):
    # Node labels with SQL wildcard chars (_ and %) must match LITERALLY (exact, case-insensitive),
    # like the in-memory adapter — not as ILIKE patterns, or find/delete would hit the wrong nodes.
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "conta_corrente", "CONCEPT"))
    await graph.upsert_node(GraphNode(scope, "contaXcorrente", "CONCEPT"))   # matches _ as wildcard
    await graph.upsert_node(GraphNode(scope, "100%", "CONCEPT"))
    await graph.upsert_node(GraphNode(scope, "100pct", "CONCEPT"))

    found = await graph.find_node(scope, "conta_corrente")
    assert found is not None and found.label == "conta_corrente"            # not contaXcorrente
    # delete by a %-bearing label must remove ONLY that node, never everything matching "100%"
    assert await graph.delete_node(scope, "100%") is True
    assert await graph.find_node(scope, "100pct") is not None               # survived


async def test_walk_terminates_on_a_cycle(graph):
    # A cyclic subgraph must not re-expand nodes (path guard) — bounded, terminating result.
    scope = f"t/{uuid4()}"
    for a, b in (("A", "B"), ("B", "C"), ("C", "A")):
        await graph.upsert_edge(GraphEdge(scope, a, b, "LINK"))
    edges = await graph.walk(scope, "A", max_depth=5)
    assert {e.relation for e in edges} == {"LINK"} and len(edges) == 3      # each edge once


async def test_multi_hop_walk(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    edges = await graph.walk(scope, "José", max_depth=2)
    assert {e.relation for e in edges} == {"OWNS", "BREED"}


async def test_walk_depth_limit(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    edges = await graph.walk(scope, "José", max_depth=1)
    assert {e.relation for e in edges} == {"OWNS"}


async def test_neighbors(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    assert {n.label for n in await graph.neighbors(scope, "Rex")} == {"José", "Pastor Alemão"}


async def test_node_embedding_search(graph):
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "alpha", "CONCEPT", embedding=_emb(1.0, 0.0)))
    await graph.upsert_node(GraphNode(scope, "beta", "CONCEPT", embedding=_emb(0.0, 1.0)))
    out = await graph.find_nodes_by_embedding(scope, _emb(0.95, 0.05), limit=1)
    assert out and out[0].label == "alpha"


async def test_delete_edges_by_session_prunes(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    await graph.upsert_edge(GraphEdge(scope, "José", "Maria", "KNOWS", source_session="sess2"))
    deleted = await graph.delete_edges_by_session(scope, "sess1")
    assert deleted == 2
    assert {e.relation for e in await graph.walk(scope, "José", max_depth=3)} == {"KNOWS"}


# ── parent-parity surface (added for completeness) ─────────────────────────

async def test_get_active_session_and_ttl(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    assert (await store.get_active_session(scope)).id == s.id
    assert await store.get_active_session(scope, within_seconds=0) is None  # outside TTL
    await store.close_session(s.id)
    assert await store.get_active_session(scope) is None                    # closed


async def test_update_turn_response_and_counts(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.save_turn(TurnRecord(s.id, scope, 1, "oi"))
    await store.save_turn(TurnRecord(s.id, scope, 2, "tudo bem?"))
    await store.update_turn_response(scope, s.id, 1, "olá!")
    turns = await store.load_turns(s.id)
    assert turns[0].response == "olá!"
    assert await store.turn_count(s.id) == 2
    await store.save_memory(MemoryRecord(scope, "fact", "x"))
    assert await store.memory_count(scope) == 1


async def test_graph_node_context_list_delete(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    ctx = await graph.get_node_context(scope, "José")
    assert ctx is not None and {e.relation for e in ctx.edges} == {"OWNS"}
    assert {n.label for n in ctx.neighbors} == {"Rex"}
    assert {n.label for n in await graph.list_nodes(scope)} == {"José", "Rex", "Pastor Alemão"}
    assert [n.label for n in await graph.list_nodes(scope, node_type="ANIMAL")] == ["Rex"]
    assert await graph.delete_node(scope, "Rex") is True
    assert await graph.find_node(scope, "Rex") is None
    assert await graph.walk(scope, "José", max_depth=2) == []   # edges cascaded


# ── maintenance + partitioning ─────────────────────────────────────────────

class _Embedder:
    async def embed(self, text):
        return [float(len(text) % 7)] + [0.0] * (EMB_DIM - 1)


async def test_maintenance_prune_and_reembed(store):
    from datetime import timedelta
    from cogno_engram import maintenance
    scope = f"t/{uuid4()}"
    # an old low-confidence memory (force created_at into the past) + a durable fact
    await store.save_memory(MemoryRecord(scope, "fact", "stale", confidence=0.6))
    await store.save_memory(MemoryRecord(scope, "fact", "durable", confidence=1.0))
    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as c:
        await c.execute("UPDATE memories SET created_at = now() - interval '400 days' WHERE scope = %s",
                        (scope,))
    n = await maintenance.prune_memories(store, scope, older_than=timedelta(days=180),
                                         max_confidence=0.75)
    assert n == 1
    remaining = await store.load_memories(scope)
    assert [m.content for m in remaining] == ["durable"]
    # reembed the survivor
    assert await maintenance.reembed_memories(store, _Embedder(), scope) == 1


async def test_maintenance_prune_orphan_nodes(graph):
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "A", "CONCEPT"))
    await graph.upsert_node(GraphNode(scope, "B", "CONCEPT"))
    await graph.upsert_edge(GraphEdge(scope, "A", "B", "REL"))
    await graph.upsert_node(GraphNode(scope, "Lonely", "CONCEPT"))
    from cogno_engram import maintenance
    assert await maintenance.prune_orphan_nodes(graph, scope) == 1
    assert await graph.find_node(scope, "Lonely") is None
    assert await graph.find_node(scope, "A") is not None


async def test_hash_partitioning_roundtrip():
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns", "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)
    # partitions were created
    cur = await conn.execute(
        "SELECT count(*) FROM pg_inherits i "
        "JOIN pg_class p ON p.oid = i.inhparent WHERE p.relname = 'turns'")
    assert (await cur.fetchone())[0] == 4   # 4 hash partitions of turns
    await conn.close()

    store = PostgresStore(dsn=DSN)
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.save_turn(TurnRecord(s.id, scope, 1, "particionado"))
    await store.save_turn(TurnRecord(s.id, scope, 1, "dup ignored"))   # ON CONFLICT (scope,sess,turn)
    assert [t.user_input for t in await store.load_turns(s.id)] == ["particionado"]
    await store.save_memory(MemoryRecord(scope, "fact", "x", embedding=[1.0] + [0.0] * (EMB_DIM - 1)))
    out = await store.load_memories(scope, query=RetrievalQuery(text="x"))
    assert out and out[0].content == "x"


# ── edge curation: the two adapters must agree ────────────────────────────
#
# Every behaviour below is pinned against the in-memory adapter in `test_edge_curation.py`.
# Repeated here because the invariant lives in two implementations — an in-Python `continue`
# and a SQL predicate inside a recursive CTE — and "the prompt never carries an unreviewed
# claim about a person" is not a guarantee one of them may hold alone.

async def test_pg_walk_excludes_a_proposal_and_the_hop_behind_it(graph):
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Rex", "OWNS_PET", status="proposed"))
    await graph.upsert_edge(GraphEdge(scope, "Rex", "Pastor Alemão", "BREED"))

    assert await graph.walk(scope, "José", max_depth=3) == []

    assert await graph.set_edge_status(scope, "José", "Rex", "OWNS_PET", "accepted")
    reached = {e.target for e in await graph.walk(scope, "José", max_depth=3)}
    assert reached == {"Rex", "Pastor Alemão"}


async def test_pg_queue_holds_only_proposals_and_carries_the_detail(graph):
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"age": 8}))
    await graph.upsert_edge(GraphEdge(scope, "José", "Rex", "OWNS_PET",
                                      attributes={"note": "vira-lata caramelo"},
                                      status="proposed"))

    queued = await graph.pending_edges(scope)
    assert [e.target for e in queued] == ["Rex"]
    assert queued[0].attributes == {"note": "vira-lata caramelo"}   # the jsonb round-trips


async def test_pg_reassertion_merges_and_promotes_but_never_demotes(graph):
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"note": "joga futebol"}, status="proposed"))
    # a human asserts the same edge: merge the detail, promote the status
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"age": 8}))
    edge = (await graph.walk(scope, "José"))[0]
    assert edge.attributes == {"note": "joga futebol", "age": 8} and edge.status == "accepted"

    # the next extraction proposes it again — a verdict does not expire on its own
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", status="proposed"))
    assert (await graph.walk(scope, "José"))[0].status == "accepted"
    assert await graph.pending_edges(scope) == []


async def test_pg_verdict_on_an_absent_edge_is_reported(graph):
    scope = "tenant:acme|identity:jose"
    assert await graph.set_edge_status(scope, "José", "Ninguém", "OWNS_PET", "accepted") is False


async def test_an_existing_database_GETS_the_new_columns_and_keeps_its_edges(pg):
    """The half that only breaks in production, and only once.

    Takes ``pg`` — which it does not otherwise need — because that fixture is what carries the
    module's skip. Without it this test never reaches the skip path: with ``ENGRAM_TEST_DSN``
    unset, ``psycopg.connect("")`` falls back to libpq's own defaults (``PGHOST``/``PGDATABASE``,
    then the OS user's database) and the FIRST statement below is ``DROP SCHEMA public CASCADE``.
    A review caught it; on a box with a local Postgres it would have connected. `conftest`'s
    2026-08-04 guard does not cover this path — it returns early precisely when the DSN is empty,
    on the reasoning that "those modules skip on their own", which this test did not.

    `CREATE TABLE IF NOT EXISTS` is a NO-OP against a live table, so without the `ALTER TABLE`
    statements the new code would ship to a deployment that already has a graph and find the
    columns missing — a 500 on the first walk. And the backfill direction is load-bearing: the
    DEFAULT is `accepted`, so an edge the host asserted before curation existed keeps being
    spoken. Backfilling to `proposed` would mute every existing graph until someone reviewed it
    one by one, which is the same outage wearing a policy's clothes.

    Manages its own connection and schema: the shared fixture creates the CURRENT tables, which
    is precisely the state this test must not start from.
    """
    # Belt AND braces: the fixture already skips, and this refuses to aim a DROP at whatever
    # libpq would pick. A destructive test may not depend on one guard.
    assert DSN, "refusing to run a schema-dropping test without an explicit ENGRAM_TEST_DSN"

    old_schema = """
        DROP SCHEMA public CASCADE; CREATE SCHEMA public;
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE TABLE knowledge_nodes (
            id bigserial PRIMARY KEY, scope text NOT NULL, label text NOT NULL,
            node_type text NOT NULL DEFAULT 'CONCEPT', attributes jsonb NOT NULL DEFAULT '{}',
            embedding vector(8), created_at timestamptz DEFAULT now(),
            updated_at timestamptz DEFAULT now());
        CREATE UNIQUE INDEX uq_nodes_scope_label_type
            ON knowledge_nodes (scope, lower(label), node_type);
        CREATE TABLE knowledge_edges (
            id bigserial PRIMARY KEY, scope text NOT NULL,
            source_id bigint NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            target_id bigint NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            relation text NOT NULL, confidence real NOT NULL DEFAULT 1.0,
            source_session text NOT NULL DEFAULT '', created_at timestamptz DEFAULT now(),
            UNIQUE (source_id, target_id, relation));
        INSERT INTO knowledge_nodes (scope, label) VALUES ('t:mig', 'Jose'), ('t:mig', 'Pedro');
        INSERT INTO knowledge_edges (scope, source_id, target_id, relation)
            VALUES ('t:mig', 1, 2, 'PARENT_OF');
    """

    async def columns(conn) -> set:
        cur = await conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'knowledge_edges'")
        return {r["column_name"] for r in await cur.fetchall()}

    async with await psycopg.AsyncConnection.connect(
            pg, row_factory=dict_row, autocommit=True) as conn:
        await conn.execute(old_schema)
        assert not {"attributes", "status"} & await columns(conn)     # the pre-curation world

        await ensure_schema(conn, embedding_dim=EMB_DIM)
        assert {"attributes", "status"} <= await columns(conn)

        cur = await conn.execute("SELECT status, attributes FROM knowledge_edges")
        row = await cur.fetchone()
        assert row["status"] == "accepted" and row["attributes"] == {}

    # ...and it is still spoken
    kg = PostgresKnowledgeGraph(dsn=pg)
    assert [(e.relation, e.status) for e in await kg.walk("t:mig", "Jose")] == \
        [("PARENT_OF", "accepted")]


async def test_pg_a_proposal_leaks_through_neither_neighbors_nor_node_context(graph):
    """Both were unfiltered — `neighbors` in both adapters, `get_node_context` in the in-memory
    one only. `NodeContext` hands edges and neighbors to the same caller, so filtering one and
    not the other leaks the association through the other field."""
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", status="proposed"))

    assert await graph.neighbors(scope, "José") == []
    ctx = await graph.get_node_context(scope, "José")
    assert ctx is not None and ctx.edges == [] and ctx.neighbors == []


async def test_pg_queue_drains_oldest_first(graph):
    """The two adapters disagreed on ordering and the disagreement was invisible: newest-first
    plus a `limit` makes the oldest proposals permanently unreachable."""
    scope = "tenant:acme|identity:jose"
    for i in range(5):
        await graph.upsert_edge(GraphEdge(scope, "José", f"Contato {i}", "FRIEND_OF",
                                          status="proposed"))
    assert [e.target for e in await graph.pending_edges(scope, limit=2)] == \
        ["Contato 0", "Contato 1"]


async def test_pg_a_value_json_cannot_serialise_does_not_take_the_turn_down(graph):
    """A host stamping a `datetime`/`UUID` wrote fine in memory and raised `TypeError` here —
    a divergence that only surfaces in production."""
    from datetime import datetime
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"since": datetime(2020, 1, 1)}))
    edge = (await graph.walk(scope, "José"))[0]
    assert "2020-01-01" in str(edge.attributes["since"])


async def test_pg_an_explicit_None_attributes_does_not_poison_the_next_merge(graph):
    """`json.dumps(None)` → `'null'::jsonb` passes NOT NULL, and then the NEXT upsert's `||`
    fails on concatenating an object with a scalar."""
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", attributes=None))
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", attributes={"age": 8}))
    assert (await graph.walk(scope, "José"))[0].attributes == {"age": 8}


@pytest.mark.asyncio
async def test_admin_traces_a_scope_with_LIKE_metacharacters_does_not_leak(store):
    """O comportamento que os testes puros só conseguem provar pela FORMA.

    O scope é opaco: pode conter `%`, `_` e `\\`, e cada um vira curinga dentro de um `LIKE`. Sem
    o escaping, prefixo `_` puxa todo scope de um caractere e prefixo `%` puxa TUDO — e o que
    volta aqui é o traço COMPLETO do turno de outro tenant.

    Este caso corre contra Postgres a sério; `tests/test_subtree_escaping.py` cobre o mesmo pelo
    padrão, sem banco, para a rede existir também na máquina de quem desenvolve."""
    from cogno_engram.types import TurnTrace

    marca = uuid4().hex[:8]
    prefixo = f"t{marca}_a"                       # `_` é curinga de UM caractere no LIKE
    intrusos = [f"t{marca}Xa/u1", f"t{marca}za/u1", f"t{marca}-a/u1"]

    # `turn_traces.session_id` é UUID no schema — string livre rebenta com
    # InvalidTextRepresentation. O in-memory aceita qualquer string, e foi por isso que a
    # primeira versão passou 196 verdes sem nunca tocar o banco.
    await store.save_turn_trace(TurnTrace(str(uuid4()), prefixo, 0, {"quem": "dono"}))
    await store.save_turn_trace(TurnTrace(str(uuid4()), f"{prefixo}/filho", 1,
                                          {"quem": "filho"}))
    for i, alheio in enumerate(intrusos):
        await store.save_turn_trace(TurnTrace(str(uuid4()), alheio, 2 + i, {"quem": alheio}))

    rows, total = await store.admin_traces(prefixo)
    voltaram = {r.scope for r in rows}

    assert voltaram == {prefixo, f"{prefixo}/filho"}, (
        f"vazou para fora da subárvore: {sorted(voltaram - {prefixo, f'{prefixo}/filho'})}"
    )
    assert total == 2, "o total tem de respeitar a mesma fronteira que a página"


@pytest.mark.asyncio
async def test_admin_traces_the_since_window_applies_to_the_PREFIX_row_too(store):
    """O `since` tem de valer para a linha que está NO prefixo, não só para as descendentes.

    O teste vizinho semeia apenas descendentes, então o ramo `scope = %s` nunca chega a ter linha
    — e a mutação que tira os parênteses do `_SUBTREE` (fazendo o `AND created_at >= %s` ligar-se
    só ao ramo do LIKE) passava despercebida. É preciso semear o PRÓPRIO prefixo."""
    from datetime import datetime, timedelta, timezone

    from cogno_engram.types import TurnTrace

    tenant = f"t{uuid4().hex[:8]}"
    t0 = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    await store.save_turn_trace(TurnTrace(str(uuid4()), tenant, 0, {"quando": "velho"},
                                          created_at=t0))
    await store.save_turn_trace(TurnTrace(str(uuid4()), f"{tenant}/u1", 1, {"quando": "novo"},
                                          created_at=t0 + timedelta(hours=2)))

    rows, total = await store.admin_traces(tenant, since=t0 + timedelta(hours=1))
    assert [r.trace["quando"] for r in rows] == ["novo"], (
        "a linha NO prefixo atravessou a janela do `since`"
    )
    assert total == 1


# ── count_nodes: the query the uniqueness question needed ─────────────────

async def test_pg_count_nodes_answers_beyond_the_page(graph):
    """`list_nodes` is `ORDER BY id LIMIT n` with no label filter — the host had to refuse to
    answer whenever that page came back full, because a homonym past the cut is invisible."""
    scope = f"t/{uuid4()}"
    for i in range(120):
        await graph.upsert_node(GraphNode(scope, f"Contato {i}", "PERSON"))
    await graph.upsert_node(GraphNode(scope, "Maria", "PERSON"))

    assert await graph.count_nodes(scope) == 121
    assert await graph.count_nodes(scope, label="maria") == 1        # case-insensitive
    assert len(await graph.list_nodes(scope, limit=50)) == 50        # the page, for contrast


async def test_pg_count_nodes_SEES_the_homonym_the_double_collapses(graph):
    """The case the whole feature exists for, and the one the in-memory double gets wrong.

    The unique index is `(scope, lower(label), node_type)`, so `José/PERSON` and `José/CONCEPT`
    are two rows — and `walk` seeds on the LABEL, so it expands from both. A Tier-2 extraction
    creating "José" as a CONCEPT while the contact is a PERSON is exactly how a tenant gets
    there. See `test_count_nodes.py::test_the_DOUBLE_collapses_a_homonym_the_real_store_keeps`."""
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "José", "PERSON"))
    await graph.upsert_node(GraphNode(scope, "José", "CONCEPT"))

    assert await graph.count_nodes(scope, label="José") == 2


async def test_pg_count_nodes_scopes_and_zeroes(graph):
    mine, theirs = f"t/{uuid4()}", f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(theirs, "José", "PERSON"))
    assert await graph.count_nodes(mine) == 0
    assert await graph.count_nodes(mine, label="José") == 0
