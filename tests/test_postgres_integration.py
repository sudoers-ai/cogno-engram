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
    for tbl in ("knowledge_edges", "knowledge_nodes", "memories", "turns", "sessions"):
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


async def test_close_session_and_recent(store):
    scope = f"t/{uuid4()}"
    s = await store.create_session(scope)
    await store.close_session(s.id, summary="done")
    assert (await store.get_session(s.id)).summary == "done"
    recent = await store.recent_sessions(scope)
    assert recent and recent[0].id == s.id


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
