"""
Integration tests for the Postgres + pgvector adapter.

Gated: they need a real Postgres with the ``vector`` extension. Set
``ENGRAM_TEST_DSN`` to a scratch database; the suite auto-skips if it is unset
or unreachable. The fixture DROPs and recreates the engram tables (with a tiny
embedding dim) so runs are deterministic — point it at a throwaway DB.

    docker run -d --rm --name engram-pg -e POSTGRES_PASSWORD=postgres \
        -p 55432:5432 pgvector/pgvector:pg16
    ENGRAM_TEST_DSN=postgresql://postgres:postgres@localhost:55432/postgres \
        python3 -m pytest tests/test_postgres_integration.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")

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
