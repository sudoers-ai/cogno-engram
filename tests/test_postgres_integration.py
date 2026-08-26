"""
Integration tests for the Postgres + pgvector adapter.

Gated: they need a real Postgres with the ``vector`` extension. They aim at ``engram_test``
by themselves — on the server ``COGNO_PG_DSN`` already names, else on libpq's defaults — and
auto-skip when nothing is listening there. The fixture DROPs and recreates the engram tables
(with a tiny embedding dim) so runs are deterministic, which is why the destination is chosen
for you rather than typed.

**The database name must contain "test"** or ``tests/conftest.py`` aborts the run: these
tests do not just delete rows, they leave the schema with a ``vector(8)`` embedding column
that a real 768-dimension embedder cannot write to. The example below used to say
``/postgres`` — a database that exists on every server, production ones included.

    docker run -d --rm --name engram-pg -e POSTGRES_PASSWORD=postgres \
        -e POSTGRES_DB=engram_test -p 5432:5432 pgvector/pgvector:pg16
    python3 -m pytest tests/test_postgres_integration.py     # ← no DSN to type

``ENGRAM_TEST_DSN`` still overrides, for a server that is neither of those.
"""

from __future__ import annotations

import logging
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.rows import dict_row  # noqa: E402

from cogno_engram.adapters.postgres import (  # noqa: E402
    _partition_existing_table,
    PostgresKnowledgeGraph,
    PostgresStore,
    ensure_schema,
)
from cogno_engram.types import (  # noqa: E402
    AUDIENCE_STAFF,
    GraphEdge,
    GraphNode,
    HybridWeights,
    MemoryRecord,
    RetrievalQuery,
    TurnRecord,
)

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path

DSN = resolve_test_dsn()      # ENGRAM_TEST_DSN, else `engram_test` on the local server
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
    node = await graph.find_node(scope, "JOSÉ", audience=AUDIENCE_STAFF)
    assert node is not None and node.attributes == {"age": 40}


async def test_node_label_matched_literally_not_as_like_pattern(graph):
    # Node labels with SQL wildcard chars (_ and %) must match LITERALLY (exact, case-insensitive),
    # like the in-memory adapter — not as ILIKE patterns, or find/delete would hit the wrong nodes.
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "conta_corrente", "CONCEPT"))
    await graph.upsert_node(GraphNode(scope, "contaXcorrente", "CONCEPT"))   # matches _ as wildcard
    await graph.upsert_node(GraphNode(scope, "100%", "CONCEPT"))
    await graph.upsert_node(GraphNode(scope, "100pct", "CONCEPT"))

    found = await graph.find_node(scope, "conta_corrente", audience=AUDIENCE_STAFF)
    assert found is not None and found.label == "conta_corrente"            # not contaXcorrente
    # delete by a %-bearing label must remove ONLY that node, never everything matching "100%"
    assert await graph.delete_node(scope, "100%") is True
    assert await graph.find_node(scope, "100pct", audience=AUDIENCE_STAFF) is not None               # survived


async def test_walk_terminates_on_a_cycle(graph):
    # A cyclic subgraph must not re-expand nodes (path guard) — bounded, terminating result.
    scope = f"t/{uuid4()}"
    for a, b in (("A", "B"), ("B", "C"), ("C", "A")):
        await graph.upsert_edge(GraphEdge(scope, a, b, "LINK"))
    edges = await graph.walk(scope, "A", max_depth=5, audience=AUDIENCE_STAFF)
    assert {e.relation for e in edges} == {"LINK"} and len(edges) == 3      # each edge once


async def test_multi_hop_walk(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    edges = await graph.walk(scope, "José", max_depth=2, audience=AUDIENCE_STAFF)
    assert {e.relation for e in edges} == {"OWNS", "BREED"}


async def test_walk_depth_limit(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    edges = await graph.walk(scope, "José", max_depth=1, audience=AUDIENCE_STAFF)
    assert {e.relation for e in edges} == {"OWNS"}


async def test_neighbors(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    assert {n.label for n in await graph.neighbors(scope, "Rex", audience=AUDIENCE_STAFF)} == {"José", "Pastor Alemão"}


async def test_node_embedding_search(graph):
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "alpha", "CONCEPT", embedding=_emb(1.0, 0.0)))
    await graph.upsert_node(GraphNode(scope, "beta", "CONCEPT", embedding=_emb(0.0, 1.0)))
    out = await graph.find_nodes_by_embedding(scope, _emb(0.95, 0.05), limit=1, audience=AUDIENCE_STAFF)
    assert out and out[0].label == "alpha"


async def test_delete_edges_by_session_prunes(graph):
    scope = f"t/{uuid4()}"
    await _build_jose_rex(graph, scope)
    await graph.upsert_edge(GraphEdge(scope, "José", "Maria", "KNOWS", source_session="sess2"))
    deleted = await graph.delete_edges_by_session(scope, "sess1")
    assert deleted == 2
    assert {e.relation for e in await graph.walk(scope, "José", max_depth=3, audience=AUDIENCE_STAFF)} == {"KNOWS"}


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
    ctx = await graph.get_node_context(scope, "José", audience=AUDIENCE_STAFF)
    assert ctx is not None and {e.relation for e in ctx.edges} == {"OWNS"}
    assert {n.label for n in ctx.neighbors} == {"Rex"}
    assert {n.label for n in await graph.list_nodes(scope, audience=AUDIENCE_STAFF)} == {"José", "Rex", "Pastor Alemão"}
    assert [n.label for n in await graph.list_nodes(scope, node_type="ANIMAL", audience=AUDIENCE_STAFF)] == ["Rex"]
    assert await graph.delete_node(scope, "Rex") is True
    assert await graph.find_node(scope, "Rex", audience=AUDIENCE_STAFF) is None
    assert await graph.walk(scope, "José", max_depth=2, audience=AUDIENCE_STAFF) == []   # edges cascaded


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
    assert await graph.find_node(scope, "Lonely", audience=AUDIENCE_STAFF) is None
    assert await graph.find_node(scope, "A", audience=AUDIENCE_STAFF) is not None


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


async def test_the_subtree_index_reaches_every_PARTITION():
    """O modo que a produção corre — e o único em que o teste de plano acima não vale.

    `cogno_host/migrate.py::init_db` chama `ensure_schema(..., partition_by_scope=True)`. Nesse
    modo o índice declarado no pai é propagado a cada filho sob nome AUTO-GERADO
    (`turn_traces_p0_scope_created_at_idx`, …), portanto o nome do pai nunca aparece num plano —
    e uma asserção sobre plano exigiria linhas suficientes para CADA partição passar o limiar de
    Seq Scan (medido: 2000 linhas por 8 partições ficam todas abaixo), o que é um teste de
    minutos para verificar uma coisa que o Postgres garante.

    Por isso esta é, deliberadamente, uma asserção de DDL — a forma que critiquei no teste acima
    e que aqui é a garantia honesta disponível. O que ela apanha é real e já aconteceu noutras
    tabelas: um índice declarado no pai que NÃO chega aos filhos deixa a produção inteira sem
    ele, com a suíte verde porque o modo não particionado é o que os testes exercitam.

    TECTO, dito em voz alta: isto prova que o índice EXISTE em cada partição, não que o
    planeador o escolhe lá. Essa continua por medir."""
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns",
                "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)

    cur = await conn.execute(
        "SELECT c.relname FROM pg_inherits i "
        "  JOIN pg_class p ON p.oid = i.inhparent "
        "  JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE p.relname = 'turn_traces' ORDER BY c.relname")
    filhas = [r[0] for r in await cur.fetchall()]
    assert len(filhas) == 4, f"esperava 4 partições de turn_traces, achei {filhas}"

    # `scope` sob uma opclass de padrão em cada filha — é o que serve o ramo do LIKE, e é a
    # metade que um btree comum não faz (ver o comentário do índice em `postgres.py`)
    sem_indice = []
    for filha in filhas:
        c = await conn.execute(
            "SELECT count(*) FROM pg_index x "
            "  JOIN pg_class t ON t.oid = x.indrelid "
            "  JOIN pg_opclass o ON o.oid = x.indclass[0] "
            "WHERE t.relname = %s AND o.opcname IN ('text_pattern_ops', 'varchar_pattern_ops')",
            (filha,))
        if (await c.fetchone())[0] == 0:
            sem_indice.append(filha)
    await conn.close()

    assert not sem_indice, (
        f"o índice da subárvore não chegou a {sem_indice} — em produção, que corre "
        f"particionada, essas partições ficam sem ele e a suíte não notaria")


@pytest.mark.asyncio
@pytest.mark.parametrize("leitor", ["admin_turns", "admin_scopes"])
async def test_the_turns_subtree_readers_do_not_seq_scan(store, leitor):
    """Os dois irmãos que o índice dos traços deixou para trás.

    `idx_turns_scope_time` era btree COMUM, e este ficheiro citava-o como o irmão que "já tinha"
    o índice — ao contrário: pelo mesmo argumento, num collation que não seja `C` um btree comum
    não serve `LIKE 'prefixo/%'`. Medido a 200k COM esse índice presente, ambos os leitores davam
    `Parallel Seq Scan on turns`.

    Como no teste gémeo do `turn_traces`: o SQL é CAPTURADO do próprio método, não copiado à mão
    (uma garantia sobre uma consulta que ninguém emite não é garantia), e a asserção é a ausência
    de `Seq Scan` e não o nome do índice (o nome falha sob collation `C`, onde a UNIQUE já serve
    ambos os ramos e não há varrimento nenhum)."""
    import contextlib

    marca = uuid4().hex[:8]
    # muitos escopos, um alvo raro: é a forma da produção, e a única em que um índice ganha
    for i in range(2000):
        escopo = f"{marca}{i % 400:03d}/u{i % 7}"
        sess = await store.create_session(escopo)
        await store.save_turn(TurnRecord(sess.id, escopo, i % 30, f"in{i}", f"out{i}"))

    emitido: "list[tuple]" = []

    class _Espiao:
        """Regista o SQL e delega o resto — `__getattr__`, não uma lista de métodos à mão."""

        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, nome):
            return getattr(self._conn, nome)

        async def execute(self, sql, params=None, *a, **kw):
            emitido.append((sql, params))
            return await self._conn.execute(sql, params, *a, **kw)

    real = store._conn                                             # type: ignore[attr-defined]

    @contextlib.asynccontextmanager
    async def _espiado():
        async with real() as c:
            yield _Espiao(c)

    store._conn = _espiado                                         # type: ignore[attr-defined]
    try:
        if leitor == "admin_scopes":
            await store.admin_scopes(f"{marca}007")
        else:
            await store.admin_turns(f"{marca}007")
    finally:
        store._conn = real                                         # type: ignore[attr-defined]

    sql, params = emitido[0]
    assert "FROM turns" in sql, f"não capturei a consulta de {leitor}: {sql!r}"

    async with real() as conn:
        await conn.execute("ANALYZE turns")
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + sql, params)
            plano = "\n".join(
                (r["QUERY PLAN"] if isinstance(r, dict) else r[0])
                for r in await cur.fetchall())

    # Não-vacuidade primeiro: uma asserção NEGATIVA sobre uma string passa trivialmente se a
    # string vier vazia. Igual ao gémeo do `turn_traces`.
    assert "on turns" in plano, f"o EXPLAIN não falou de turns:\n{plano}"
    assert "Seq Scan on turns" not in plano, (
        f"{leitor} varreu a tabela inteira — o plano foi:\n{plano}")


@pytest.mark.asyncio
async def test_the_OLD_turns_index_is_gone_and_the_new_one_is_there():
    """A migração, e é a metade que quase subiu inerte.

    `CREATE INDEX IF NOT EXISTS idx_turns_scope_time` com definição NOVA e nome ANTIGO é um
    no-op silencioso — o nome existe, nada acontece, e o conserto ficaria a não fazer nada em
    toda a instalação já criada, com a suíte verde porque um banco de teste nasce do zero. Por
    isso o nome mudou e o antigo é derrubado.

    Este teste corre `ensure_schema` DUAS vezes sobre um banco que já tinha o índice antigo — que
    é o estado real de qualquer instalação existente — e afirma os dois lados: o novo está lá com
    a opclass de padrão, e o antigo saiu. A segunda passagem prova a idempotência."""
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns",
                "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM)

    # o estado de uma instalação ANTIGA: recria à mão o índice que este commit aposenta
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_turns_scope_time "
        "ON turns (scope, created_at DESC, id DESC)")

    for _ in range(2):                                     # idempotência: correr duas vezes
        await ensure_schema(conn, embedding_dim=EMB_DIM)

    cur = await conn.execute(
        "SELECT x.indexrelid::regclass::text, o.opcname FROM pg_index x "
        "  JOIN pg_class t ON t.oid = x.indrelid "
        "  JOIN pg_opclass o ON o.oid = x.indclass[0] "
        "WHERE t.relname = 'turns'")
    indices = {nome: opc for nome, opc in await cur.fetchall()}
    await conn.close()

    assert "idx_turns_scope_time" not in indices, (
        "o índice antigo (btree comum) sobreviveu — a instalação fica a pagar escrita por um "
        "índice que o novo supersede em todas as formas medidas")
    assert indices.get("idx_turns_scope_pattern") in ("text_pattern_ops", "varchar_pattern_ops"), (
        f"o índice novo não está lá, ou não tem a opclass de padrão: {indices}")


async def test_the_turns_subtree_index_reaches_every_PARTITION():
    """O mesmo para `turns` — e aqui pesa mais do que nos traços.

    `turns` é uma das DUAS tabelas que o `partition_by_scope` activa (a outra é `memories`); os
    traços não são particionados de todo. Portanto em produção o índice desta correcção vive
    inteiramente nas filhas, e um índice declarado no pai que não chegue lá deixa TODA a
    instalação sem ele, com a suíte verde porque o modo não particionado é o que os testes
    exercitam por omissão.

    A segunda metade é o `DROP` do índice antigo, que nesta correcção é o que faz o novo valer a
    pena: um `DROP INDEX` do pai tem de levar as filhas consigo, senão cada partição fica a pagar
    escrita por um índice que o novo supersede.

    Asserção de DDL, pela mesma razão do teste gémeo acima: uma asserção de plano exigiria linhas
    suficientes para CADA partição passar o limiar de Seq Scan."""
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns",
                "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)

    # O estado de uma instalação PARTICIONADA já existente — e sem isto a asserção do índice
    # antigo era VÁCUA: numa base nova ele nunca chega a existir, portanto tirar o `DROP` do
    # `ensure_schema` não mudava nada aqui e a mutação sobrevivia (medido). Criá-lo à mão é o que
    # torna a segunda metade deste teste capaz de falhar.
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_turns_scope_time "
                       "ON turns (scope, created_at DESC, id DESC)")
    for _ in range(2):                                     # e a idempotência, como no gémeo
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)

    cur = await conn.execute(
        "SELECT c.relname FROM pg_inherits i "
        "  JOIN pg_class p ON p.oid = i.inhparent "
        "  JOIN pg_class c ON c.oid = i.inhrelid "
        "WHERE p.relname = 'turns' ORDER BY c.relname")
    filhas = [r[0] for r in await cur.fetchall()]
    assert len(filhas) == 4, f"esperava 4 partições de turns, achei {filhas}"

    sem_padrao = []
    for filha in filhas:
        c = await conn.execute(
            "SELECT count(*) FROM pg_index x "
            "  JOIN pg_class t ON t.oid = x.indrelid "
            "  JOIN pg_opclass o ON o.oid = x.indclass[0] "
            "WHERE t.relname = %s AND o.opcname IN ('text_pattern_ops', 'varchar_pattern_ops')",
            (filha,))
        if (await c.fetchone())[0] == 0:
            sem_padrao.append(filha)
    # e o antigo não pode ter sobrevivido em partição nenhuma — é o `DROP` que faz o índice
    # novo valer a pena, e em modo particionado ele tem de levar as 4 filhas consigo
    c = await conn.execute(
        "SELECT x.indexrelid::regclass::text FROM pg_index x "
        "  JOIN pg_class t ON t.oid = x.indrelid "
        "WHERE t.relname LIKE %s AND x.indexrelid::regclass::text LIKE %s",
        ("turns%", "%scope_time%"))
    com_antigo = [r[0] for r in await c.fetchall()]
    await conn.close()

    assert not sem_padrao, (
        f"o índice novo não chegou a {sem_padrao} — em produção, que corre particionada, essas "
        f"partições varrem a tabela inteira e a suíte não notaria")
    assert not com_antigo, (
        f"o índice antigo sobreviveu em {com_antigo} — cada partição paga escrita por um índice "
        f"que o novo supersede em todas as formas medidas")


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

    assert await graph.walk(scope, "José", max_depth=3, audience=AUDIENCE_STAFF) == []

    assert await graph.set_edge_status(scope, "José", "Rex", "OWNS_PET", "accepted")
    reached = {e.target for e in await graph.walk(scope, "José", max_depth=3, audience=AUDIENCE_STAFF)}
    assert reached == {"Rex", "Pastor Alemão"}


async def test_pg_queue_holds_only_proposals_and_carries_the_detail(graph):
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"age": 8}))
    await graph.upsert_edge(GraphEdge(scope, "José", "Rex", "OWNS_PET",
                                      attributes={"note": "vira-lata caramelo"},
                                      status="proposed"))

    queued = await graph.pending_edges(scope, audience=AUDIENCE_STAFF)
    assert [e.target for e in queued] == ["Rex"]
    assert queued[0].attributes == {"note": "vira-lata caramelo"}   # the jsonb round-trips


async def test_pg_reassertion_merges_and_promotes_but_never_demotes(graph):
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"note": "joga futebol"}, status="proposed"))
    # a human asserts the same edge: merge the detail, promote the status
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"age": 8}))
    edge = (await graph.walk(scope, "José", audience=AUDIENCE_STAFF))[0]
    assert edge.attributes == {"note": "joga futebol", "age": 8} and edge.status == "accepted"

    # the next extraction proposes it again — a verdict does not expire on its own
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", status="proposed"))
    assert (await graph.walk(scope, "José", audience=AUDIENCE_STAFF))[0].status == "accepted"
    assert await graph.pending_edges(scope, audience=AUDIENCE_STAFF) == []


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
    assert DSN, "refusing to run a schema-dropping test without a resolved test DSN"

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
    assert [(e.relation, e.status) for e in await kg.walk("t:mig", "Jose", audience=AUDIENCE_STAFF)] == \
        [("PARENT_OF", "accepted")]


async def test_pg_a_proposal_leaks_through_neither_neighbors_nor_node_context(graph):
    """Both were unfiltered — `neighbors` in both adapters, `get_node_context` in the in-memory
    one only. `NodeContext` hands edges and neighbors to the same caller, so filtering one and
    not the other leaks the association through the other field."""
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", status="proposed"))

    assert await graph.neighbors(scope, "José", audience=AUDIENCE_STAFF) == []
    ctx = await graph.get_node_context(scope, "José", audience=AUDIENCE_STAFF)
    assert ctx is not None and ctx.edges == [] and ctx.neighbors == []


async def test_pg_queue_drains_oldest_first(graph):
    """The two adapters disagreed on ordering and the disagreement was invisible: newest-first
    plus a `limit` makes the oldest proposals permanently unreachable."""
    scope = "tenant:acme|identity:jose"
    for i in range(5):
        await graph.upsert_edge(GraphEdge(scope, "José", f"Contato {i}", "FRIEND_OF",
                                          status="proposed"))
    assert [e.target for e in await graph.pending_edges(scope, limit=2, audience=AUDIENCE_STAFF)] == \
        ["Contato 0", "Contato 1"]


async def test_pg_a_value_json_cannot_serialise_does_not_take_the_turn_down(graph):
    """A host stamping a `datetime`/`UUID` wrote fine in memory and raised `TypeError` here —
    a divergence that only surfaces in production."""
    from datetime import datetime
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF",
                                      attributes={"since": datetime(2020, 1, 1)}))
    edge = (await graph.walk(scope, "José", audience=AUDIENCE_STAFF))[0]
    assert "2020-01-01" in str(edge.attributes["since"])


async def test_pg_an_explicit_None_attributes_does_not_poison_the_next_merge(graph):
    """`json.dumps(None)` → `'null'::jsonb` passes NOT NULL, and then the NEXT upsert's `||`
    fails on concatenating an object with a scalar."""
    scope = "tenant:acme|identity:jose"
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", attributes=None))
    await graph.upsert_edge(GraphEdge(scope, "José", "Pedro", "PARENT_OF", attributes={"age": 8}))
    assert (await graph.walk(scope, "José", audience=AUDIENCE_STAFF))[0].attributes == {"age": 8}


@pytest.mark.asyncio
@pytest.mark.parametrize("meta", ["_", "%", "\\"])
async def test_admin_traces_a_scope_with_LIKE_metacharacters_does_not_leak(store, meta):
    """O comportamento que os testes puros só conseguem provar pela FORMA.

    O scope é opaco: pode conter `%`, `_` e `\\`, e cada um vira curinga dentro de um `LIKE`. Sem
    o escaping, prefixo `_` puxa todo scope de um caractere e prefixo `%` puxa TUDO — e o que
    volta aqui é o traço COMPLETO do turno de outro tenant.

    Este caso corre contra Postgres a sério; `tests/test_subtree_escaping.py` cobre o mesmo pelo
    padrão, sem banco, para a rede existir também na máquina de quem desenvolve."""
    from cogno_engram.types import TurnTrace

    marca = uuid4().hex[:8]
    prefixo = f"t{marca}{meta}a"
    # Os três metacaracteres do LIKE, e são exactamente três — um censo contra o banco sobre os
    # 31 caracteres de pontuação/espaço não encontra mais nenhum. Antes só o `_` chegava ao
    # banco: medido, uma mutação que escapasse `\\` e `_` mas NÃO `%` sobrevivia a este nível
    # inteiro (40 verdes), porque o único caso que aqui corria não continha `%`.
    #   `_`  puxa um caractere qualquer   →  `tXa` entra
    #   `%`  puxa TUDO                    →  toda a tabela entra
    #   `\\`  escapa o seguinte             →  desarma o escaping do vizinho
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
@pytest.mark.parametrize("leitor", ["admin_turns", "admin_scopes"])
async def test_the_OTHER_subtree_readers_do_not_leak_either(store, leitor):
    """Os irmãos do `admin_traces`, e a razão pela qual eles precisam de caso PRÓPRIO.

    `_subtree_like` e `_SUBTREE` são partilhados pelos três leitores, portanto qualquer mutação
    DENTRO do helper morre pelos testes do `admin_traces` acima — o que dava a impressão de que
    o padrão inteiro estava coberto. Não estava: nada fixava que os irmãos continuassem a CHAMAR
    o helper. Medido, trocar `like = self._subtree_like(scope_prefix)` por
    `like = scope_prefix + "/%"` dentro do `admin_turns` (ou do `admin_scopes`) sobrevivia à
    suíte inteira — 236 verdes — e devolvia isto:

        admin_turns('t_a')  ANTES  ['t_a', 't_a/u1']
        admin_turns('t_a')  DEPOIS ['tXa/u1', 'tZa/u9', 't_a', 't_a/u1']
                                   com o `user_input` de outros tenants dentro

    É a forma "conserto nascendo inerte" na sua versão mais cara: a regra vive num sítio, está
    correcta, e nada prova que os sítios de consumo a usam. Uma regra que cada chamador pode
    deixar de aplicar sozinho é uma regra que cada chamador esquece sozinho."""
    marca = uuid4().hex[:8]
    prefixo = f"t{marca}_a"                       # `_` é curinga de UM caractere no LIKE
    intrusos = [f"t{marca}Xa/u1", f"t{marca}za/u1"]
    meus = {prefixo, f"{prefixo}/filho"}

    for i, escopo in enumerate([*meus, *intrusos]):
        sess = await store.create_session(escopo)
        await store.save_turn(TurnRecord(sess.id, escopo, i, f"segredo de {escopo}"))

    if leitor == "admin_scopes":
        voltaram = set(await store.admin_scopes(prefixo))
    else:
        rows, total = await store.admin_turns(prefixo)
        voltaram = {r.scope for r in rows}
        assert total == len(meus), "o total tem de respeitar a mesma fronteira que a página"

    assert voltaram == meus, (
        f"{leitor} vazou para fora da subárvore: {sorted(voltaram - meus)} — o escaping do LIKE "
        f"vive no helper e este chamador deixou de o usar")


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

    assert await graph.count_nodes(scope, audience=AUDIENCE_STAFF) == 121
    assert await graph.count_nodes(scope, label="maria", audience=AUDIENCE_STAFF) == 1        # case-insensitive
    assert len(await graph.list_nodes(scope, limit=50, audience=AUDIENCE_STAFF)) == 50        # the page, for contrast


async def test_pg_count_nodes_SEES_the_homonym_the_double_collapses(graph):
    """The case the whole feature exists for, and the one the in-memory double gets wrong.

    The unique index is `(scope, lower(label), node_type)`, so `José/PERSON` and `José/CONCEPT`
    are two rows — and `walk` seeds on the LABEL, so it expands from both. A Tier-2 extraction
    creating "José" as a CONCEPT while the contact is a PERSON is exactly how a tenant gets
    there. See `test_count_nodes.py::test_the_DOUBLE_collapses_a_homonym_the_real_store_keeps`."""
    scope = f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(scope, "José", "PERSON"))
    await graph.upsert_node(GraphNode(scope, "José", "CONCEPT"))

    assert await graph.count_nodes(scope, label="José", audience=AUDIENCE_STAFF) == 2


async def test_pg_count_nodes_scopes_and_zeroes(graph):
    mine, theirs = f"t/{uuid4()}", f"t/{uuid4()}"
    await graph.upsert_node(GraphNode(theirs, "José", "PERSON"))
    assert await graph.count_nodes(mine, audience=AUDIENCE_STAFF) == 0
    assert await graph.count_nodes(mine, label="José", audience=AUDIENCE_STAFF) == 0


@pytest.mark.asyncio
async def test_admin_traces_does_not_seq_scan_the_whole_table(store):
    """O índice da subárvore é USADO — medido pelo plano do SQL que o `admin_traces` emite.

    Duas coisas que a primeira versão deste teste fazia mal, ambas medidas em review:

    **Afirmava o NOME do índice, não a propriedade do próprio título.** Numa base com collation
    `C` — `initdb --locale=C`, `postgres:alpine`, qualquer cluster criado `LC_COLLATE 'C'`, que é
    uma escolha comum por desempenho — a UNIQUE pré-existente já serve os DOIS ramos do OR, não
    há Seq Scan nenhum, e o teste ficava VERMELHO em código correcto com uma mensagem a dizer o
    contrário do que o plano mostrava. A asserção passou a ser a ausência de Seq Scan: verdadeira
    sob `C`, sob `en_US.utf8`, sob ICU, e sobrevive a alguém renomear o índice.

    **Copiava à mão uma aproximação do SQL** — outra ordem de colunas, e `ORDER BY created_at
    DESC` onde o real é `created_at DESC, session_id DESC, turn_n DESC`. Uma garantia de
    desempenho sobre uma consulta que ninguém emite não é garantia: medido, trocar o predicado do
    `admin_traces` por um `LIKE '%' || %s` (curinga à cabeça, que nenhum índice pode servir)
    deixava este teste VERDE. Agora o SQL é capturado do próprio método e é esse que vai ao
    EXPLAIN — o que este teste passa a apanhar é deriva do PREDICADO (medido: a mutação do
    curinga, que sobrevivia, agora mata).

    **O que ele continua a NÃO apanhar, medido e não presumido:** deriva do ORDER BY. Trocar
    `created_at DESC, session_id DESC, turn_n DESC` por `turn_n ASC` sobrevive — e sobrevive com
    razão, porque um `BitmapOr` nunca preserva ordem de índice e o plano acaba num `Sort`
    explícito seja qual for o ORDER BY. A propriedade sob teste é "não varreu a tabela", e essa é
    insensível à ordenação. Dizer o contrário na docstring seria a mesma sobre-afirmação que este
    conserto veio remover.

    A asserção continua a ser sobre o plano e não sobre a DDL, e essa parte estava certa: um
    índice que o planeador nunca escolhe passa em qualquer teste de DDL — foi exactamente o que
    a correcção "óbvia" (btree comum) produziu."""
    import contextlib
    from datetime import datetime, timedelta, timezone

    from cogno_engram.types import TurnTrace

    marca = uuid4().hex[:8]
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # muitos escopos, um alvo raro: é a forma da produção, e a única em que um índice ganha
    for i in range(2000):
        await store.save_turn_trace(TurnTrace(str(uuid4()), f"{marca}{i % 400:03d}/u{i % 7}",
                                              i % 30, {"i": i},
                                              created_at=t0 + timedelta(minutes=i)))

    emitido: "list[tuple]" = []

    class _Espiao:
        """Regista o SQL que passa e delega TUDO o resto.

        `__getattr__` e não uma lista de métodos à mão: um wrapper que enumera o que reencaminha
        cala-se sobre o que esqueceu e mente ao `isinstance` — defeito já pago noutro sítio deste
        ecossistema."""

        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, nome):
            return getattr(self._conn, nome)

        async def execute(self, sql, params=None, *a, **kw):
            emitido.append((sql, params))
            return await self._conn.execute(sql, params, *a, **kw)

    real = store._conn                                             # type: ignore[attr-defined]

    @contextlib.asynccontextmanager
    async def _espiado():
        async with real() as c:
            yield _Espiao(c)

    store._conn = _espiado                                         # type: ignore[attr-defined]
    try:
        await store.admin_traces(f"{marca}007", since=t0, limit=30)
    finally:
        store._conn = real                                         # type: ignore[attr-defined]

    sql, params = emitido[0]                                       # o SELECT; o [1] é o count
    assert "FROM turn_traces" in sql and "ORDER BY" in sql, f"não capturei o SELECT: {sql!r}"

    async with real() as conn:
        await conn.execute("ANALYZE turn_traces")
        async with conn.cursor() as cur:
            await cur.execute("EXPLAIN " + sql, params)
            # a conexão usa row_factory de dict — a coluna do EXPLAIN chama-se "QUERY PLAN"
            plano = "\n".join(
                (r["QUERY PLAN"] if isinstance(r, dict) else r[0])
                for r in await cur.fetchall())

    # A não-vacuidade primeiro: uma asserção NEGATIVA sobre uma string passa trivialmente se a
    # string vier vazia — inalcançável aqui na prática, mas é barato tornar isso explícito em vez
    # de confiar que continua inalcançável.
    assert "on turn_traces" in plano, f"o EXPLAIN não falou de turn_traces:\n{plano}"
    assert "Seq Scan on turn_traces" not in plano, (
        f"a leitura de subárvore varreu a tabela inteira — o plano foi:\n{plano}")

# ── audience: the leak boundary, on the store that actually runs in production ──

async def test_the_POSTGRES_reads_do_not_leak_one_contact_to_another(graph):
    """The "0/9" of this feature was first measured on the IN-MEMORY adapter — the one that
    does not run in production. The SQL is a separate implementation of the same rule, and the
    first cut of it bound the caller's audience raw: `audience=""` (what `audience_for(None)`
    returns before a contact registers) read as `e.audience = ''` and matched **every
    unclassified row** — the whole legacy graph. In memory the same input said no.

    A/B/staff, through the reads the turn and the dashboard actually use.
    """
    from cogno_engram.types import (AUDIENCE_STAFF, AUDIENCE_TENANT, AUDIENCE_UNCLASSIFIED,
                                    audience_for)

    scope = f"aud-{uuid4().hex[:8]}"
    A, B = audience_for("aaa"), audience_for("bbb")
    for label in ("Ana", "Maria", "Bruno", "Carlos", "Clinica", "Unimed"):
        await graph.upsert_node(GraphNode(scope, label, "PERSON"))
    await graph.upsert_edge(GraphEdge(scope, "Ana", "Maria", "SPOUSE_OF", audience=A))
    await graph.upsert_edge(GraphEdge(scope, "Bruno", "Carlos", "PARENT_OF", audience=B))
    await graph.upsert_edge(GraphEdge(scope, "Clinica", "Unimed", "ACCEPTS",
                                      audience=AUDIENCE_TENANT))
    await graph.upsert_edge(GraphEdge(scope, "Ana", "Carlos", "FRIEND_OF"))   # unclassified

    async def _reads(who):
        return {
            "walk": await graph.walk(scope, "Bruno", audience=who, max_depth=3),
            "neighbors": await graph.neighbors(scope, "Bruno", audience=who),
            "node_context": await graph.get_node_context(scope, "Bruno", audience=who),
            "list_nodes": await graph.list_nodes(scope, audience=who, limit=100),
            "find_node": await graph.find_node(scope, "Carlos", audience=who),
            "count_nodes": await graph.count_nodes(scope, audience=who),
            "scan_nodes": await graph.scan_nodes(scope, audience=who, limit=100),
            "pending": await graph.pending_edges(scope, audience=who, limit=100),
        }

    for who, name in ((A, "A"), (AUDIENCE_UNCLASSIFIED, "no-identity")):
        got = await _reads(who)
        for read, value in got.items():
            assert "PARENT_OF" not in repr(value), f"{read} leaked B's relation to {name}"
            assert "Carlos" not in repr(value), f"{read} leaked B's node to {name}"
        # ...and the unclassified edge is staff-only, for BOTH of them
        assert "FRIEND_OF" not in repr(got["walk"])

    staff = await _reads(AUDIENCE_STAFF)
    assert "PARENT_OF" in repr(staff["walk"]), "the filter blinded staff — that is deletion"
    assert "Carlos" in repr(staff["list_nodes"])

    # the owner still hears its own life, and everyone hears a tenant fact
    assert "Maria" in repr(await graph.walk(scope, "Ana", audience=A, max_depth=2))
    for who in (A, B):
        assert "ACCEPTS" in repr(await graph.walk(scope, "Clinica", audience=who, max_depth=2))


async def test_a_FLAT_legacy_table_does_not_abort_the_rest_of_the_schema():
    """One legacy shape must not cost every statement written after it.

    `CREATE TABLE IF NOT EXISTS turns (...) PARTITION BY HASH (scope)` is a NO-OP when `turns`
    already exists — Postgres does not check the definition matches — so a database created flat
    (an older host, or a run with `partition_by_scope=False`) reaches the partition loop with a
    plain table, and `PARTITION OF` raises `InvalidObjectDefinition: "turns" is not partitioned`.

    That used to abort the WHOLE call, and the knowledge graph is created ELEVEN statements
    later. Measured on a real box 2026-08-25: `sessions`/`turns`/`memories`/`turn_traces` existed
    and `knowledge_edges` did NOT, so the host ran with no graph, `/health` said `stale`, and the
    only clue in the log was a Postgres error about partitioning. The documented remedy
    (`python -m cogno_host.migrate`, advertised as idempotent) could never fix it, because it is
    the very call that died.

    Partitioning is THROUGHPUT; the tables and columns after it are CORRECTNESS.
    """
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await _connect()
    try:
        for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories",
                    "turns", "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        # The box's history: born FLAT.
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=False)
        kind = await (await conn.execute(
            "SELECT relkind FROM pg_class WHERE relname = 'turns' "
            "AND relnamespace = 'public'::regnamespace")).fetchone()
        assert kind[0] == "r", "fixture is wrong: `turns` must start UNPARTITIONED"
        await conn.execute("DROP TABLE knowledge_edges CASCADE")   # the statement after the loop

        # Now the run a deployer makes: partitioned. It must not die on the legacy shape.
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)

        got = await (await conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'knowledge_edges'")).fetchone()
        assert got[0] == 1, "the schema after the partition loop was never created"
        # …and `turns` is untouched: skipping is not converting. Moving the data is the
        # operator's call, never a side effect of asking for a schema.
        kind = await (await conn.execute(
            "SELECT relkind FROM pg_class WHERE relname = 'turns' "
            "AND relnamespace = 'public'::regnamespace")).fetchone()
        assert kind[0] == "r"
    finally:
        await conn.close()


async def test_a_PARTITIONED_table_still_gets_its_partitions():
    """The other half: the skip must be about the legacy shape, not about giving up on
    partitioning. Without this, deleting the whole loop passes the test above."""
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await _connect()
    try:
        for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories",
                    "turns", "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)
        got = await (await conn.execute(
            # `relkind='r'` matters: without it this counts the partitions' INDEXES too and
            # answers 21 for four partitions — a number that looks like a bug in the code
            # instead of one in the question.
            "SELECT count(*) FROM pg_class WHERE relname LIKE 'turns\\_p%' "
            "AND relkind = 'r' AND relnamespace = 'public'::regnamespace")).fetchone()
        assert got[0] == 4, f"expected 4 partitions of `turns`, found {got[0]}"
    finally:
        await conn.close()


async def test_a_MIXED_database_only_skips_the_flat_table(caplog):
    """Skipping is per TABLE, and the loop must not stop at the first flat one.

    `continue` vs `break` is one word and both leave the two tests above green: there every
    table is flat, or none is. The loop runs `turns, memories, turn_traces`, so the fixture puts
    the flat tables FIRST and the one that still needs partitions LAST — with `break`,
    `turn_traces` would come out with zero partitions and nothing would say so.

    Also pins the LEVEL. `logger.error -> logger.debug` is another one-word change no behavioural
    assertion can see, and the whole point of the line is that an operator finds it: a skipped
    partition is a real divergence between what was asked for and what exists.
    """
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await _connect()
    try:
        for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories",
                    "turns", "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        # Both shapes come from engram's OWN DDL — never hand-written columns. The first cut of
        # these tests invented a flat `turns` and died on a column that does not exist, which
        # measured the fixture instead of the code.
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=False)
        await conn.execute("DROP TABLE turn_traces CASCADE")     # this one will be created fresh

        with caplog.at_level("DEBUG", logger="cogno_engram.postgres"):
            await ensure_schema(conn, embedding_dim=EMB_DIM,
                                partition_by_scope=True, partitions=4)

        got = await (await conn.execute(
            "SELECT count(*) FROM pg_class WHERE relname LIKE 'turn\\_traces\\_p%' "
            "AND relkind = 'r' AND relnamespace = 'public'::regnamespace")).fetchone()
        assert got[0] == 4, (
            f"`turn_traces` came out with {got[0]} partitions — the loop stopped at a flat "
            "sibling instead of skipping only it")
        for tbl in ("turns", "memories"):
            kind = await (await conn.execute(
                "SELECT relkind FROM pg_class WHERE relname = %s "
                "AND relnamespace = 'public'::regnamespace", (tbl,))).fetchone()
            assert kind[0] == "r", f"`{tbl}` was converted — skipping must not move data"

        skipped = [r for r in caplog.records if "partitioning_skipped" in r.getMessage()]
        assert {r.levelname for r in skipped} == {"ERROR"}, (
            "the skip must be findable by an operator grepping for errors")
        assert len(skipped) == 2, f"expected turns + memories, got {len(skipped)}"
    finally:
        await conn.close()


async def test_the_partition_probe_works_on_a_DICT_ROW_connection():
    """`ensure_schema` runs on both row factories and must not read a value out of either.

    `migrate.py` hands it tuples; `PostgresStore._conn` hands it `dict_row`. The first cut of the
    probe read `kind[0]` — a `KeyError: 0` under dict_row — and measured on a FRESH database it
    produced zero partitions and no knowledge graph. That is the guard breaking the healthy path
    worse than the bug it came to fix, and no existing test saw it: the dict_row test in this
    file runs with `partition_by_scope=False`, so the probe never executed over a dict.
    """
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True, connect_timeout=3,
                                                 row_factory=dict_row)
    try:
        for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories",
                    "turns", "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)
        got = await (await conn.execute(
            "SELECT count(*) AS n FROM pg_class WHERE relname LIKE 'turns\\_p%' "
            "AND relkind = 'r' AND relnamespace = 'public'::regnamespace")).fetchone()
        assert got["n"] == 4
        got = await (await conn.execute(
            "SELECT count(*) AS n FROM information_schema.tables "
            "WHERE table_name = 'knowledge_edges'")).fetchone()
        assert got["n"] == 1, "the schema after the partition loop was never created"
    finally:
        await conn.close()


async def test_a_DIFFERENT_modulus_does_not_abort_the_rest_of_the_schema(caplog):
    """`relkind` says PARTITIONED; it does not say WITH WHAT.

    Measured on the demo box 2026-08-26, immediately after the flat-table fix shipped: the box's
    `turn_traces` had four children and the host asks for eight, so `turn_traces_p0..p3` were
    no-ops and `turn_traces_p4` raised `would overlap partition "turn_traces_p0"`. Same class as
    the flat table, same consequence — the knowledge graph is created after this loop — and the
    first fix could not see it, because it only asked whether the table was partitioned AT ALL.

    The counts belong in the log: "has 4, asked for 8" is actionable; "would overlap" is not.
    """
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await _connect()
    try:
        for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories",
                    "turns", "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=4)
        await conn.execute("DROP TABLE knowledge_edges CASCADE")   # after the loop

        with caplog.at_level("DEBUG", logger="cogno_engram.postgres"):
            await ensure_schema(conn, embedding_dim=EMB_DIM,
                                partition_by_scope=True, partitions=8)

        got = await (await conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'knowledge_edges'")).fetchone()
        assert got[0] == 1, "the schema after the partition loop was never created"
        got = await (await conn.execute(
            "SELECT count(*) FROM pg_inherits WHERE inhparent = to_regclass('turns')")).fetchone()
        assert got[0] == 4, "the existing partitioning was changed — skipping must not convert"
        # `caplog.at_level("DEBUG")` captura TUDO, portanto filtrar pela mensagem não diz nada
        # sobre o nível: `logger.error -> logger.debug` sobrevivia. O nível É o contrato aqui —
        # uma partição saltada é divergência real, e o operador encontra-a a grepar erros.
        records = [r for r in caplog.records if "partitioning_skipped" in r.getMessage()]
        assert records and all(r.levelno == logging.ERROR for r in records), (
            f"skips must be ERROR; got {[(r.levelname, r.getMessage()[:40]) for r in records]}")
        skipped = [r.getMessage() for r in records]
        # ONE line per table, carrying the PRECISE reason. Dropping the early return still
        # completes the schema — the defensive DDL below catches the overlap — so without this
        # the return reads as dead code. It is not: falling through runs eight statements that
        # are all going to be refused and then logs `incompatible_shape` on top of the accurate
        # `exists_with_4_partitions`, which is the reason an operator would act on buried under
        # one they cannot.
        for tbl in ("turns", "memories", "turn_traces"):
            lines = [m for m in skipped if f"table={tbl} " in m]
            assert len(lines) == 1, f"expected one skip line for `{tbl}`, got {lines}"
            assert "reason=exists_with_4_partitions requested=8" in lines[0], (
                f"the log must carry both counts, and the precise reason; got {lines[0]}")


    finally:
        await conn.close()


async def test_an_UNASKABLE_shape_is_caught_by_the_DDL_itself(caplog):
    """The probes cannot name every shape, so the DDL runs defensively too.

    A table partitioned by LIST with exactly as many children as we ask for passes BOTH probes —
    it is partitioned, and the count matches — and only the `PARTITION OF ... MODULUS` statement
    can discover that the strategy is wrong. Without the nested transaction and the downgrade,
    that raises and the knowledge graph is lost again, for a third distinct shape.

    The fixture's columns come from engram's own `turn_traces` (`LIKE`), never hand-written: an
    invented table dies on a column that does not exist and measures the fixture.
    """
    if not await _pg_available():
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await _connect()
    try:
        for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories",
                    "turns", "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        await ensure_schema(conn, embedding_dim=EMB_DIM, partition_by_scope=True, partitions=2)
        await conn.execute(
            "CREATE TABLE tt_list (LIKE turn_traces INCLUDING DEFAULTS) PARTITION BY LIST (scope)")
        await conn.execute("DROP TABLE turn_traces CASCADE")
        await conn.execute("ALTER TABLE tt_list RENAME TO turn_traces")
        for name in ("a", "b"):                       # exactly `partitions` children: both probes pass
            await conn.execute(
                f"CREATE TABLE turn_traces_{name} PARTITION OF turn_traces FOR VALUES IN ('{name}')")
        await conn.execute("DROP TABLE knowledge_edges CASCADE")

        with caplog.at_level("DEBUG", logger="cogno_engram.postgres"):
            await ensure_schema(conn, embedding_dim=EMB_DIM,
                                partition_by_scope=True, partitions=2)

        got = await (await conn.execute(
            "SELECT count(*) FROM information_schema.tables "
            "WHERE table_name = 'knowledge_edges'")).fetchone()
        assert got[0] == 1, "a LIST-partitioned legacy table still aborted the schema"
        recs = [r for r in caplog.records
                if "partitioning_skipped" in r.getMessage() and "turn_traces" in r.getMessage()]
        # UMA linha, e a ERROR. Sem o `len == 1`, trocar o `return` do `except` por `pass`
        # sobrevive: as oito instruções seguintes falham todas e logam oito vezes a mesma coisa,
        # e um `any(...)` fica contente com isso.
        assert len(recs) == 1, f"expected ONE skip line for `turn_traces`, got {len(recs)}"
        assert recs[0].levelno == logging.ERROR, f"got {recs[0].levelname}"
        assert "reason=incompatible_shape" in recs[0].getMessage(), (
            f"the DDL's own refusal must be downgraded to the same event; got {recs[0].getMessage()}")
    finally:
        await conn.close()

class _NetProbe:
    """A connection just real enough to drive `_partition_existing_table`'s two probes and then
    fail the `PARTITION OF` with whatever error the test names."""

    def __init__(self, boom: BaseException, *, children: int = 2) -> None:
        self._boom, self._children, self.attempts = boom, children, 0

    async def execute(self, sql, params=None):          # noqa: ANN001 — a stand-in, not a port
        outer = self

        class _Cur:
            async def fetchone(self):
                return None if "relkind <> 'p'" in sql else (1,)

            async def fetchall(self):
                return [(1,)] * outer._children

        if "PARTITION OF" in sql:
            outer.attempts += 1
            raise outer._boom
        return _Cur()

    def transaction(self):
        class _Tx:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False
        return _Tx()


@pytest.mark.parametrize("exc, swallowed", [
    (psycopg.errors.InvalidObjectDefinition("would overlap"), True),
    (psycopg.errors.InvalidTableDefinition("wrong strategy"), True),
    (psycopg.errors.InsufficientPrivilege("permission denied"), False),
    (psycopg.errors.DiskFull("no space left"), False),
    (RuntimeError("a real bug"), False),
])
async def test_the_net_catches_legacy_shapes_and_NOTHING_else(exc, swallowed, caplog):
    """The net must stay narrow, and only a test of the net itself can say so.

    `InvalidObjectDefinition` and `InvalidTableDefinition` mean "this table has a history". A
    permission error, a full disk or a genuine bug are not that, and swallowing them turns a
    schema call into a best-effort no-op that reports success over a database it never fixed.

    Driven through a stand-in rather than a restricted Postgres role, because that recipe cannot
    reach the loop: measured, `CREATE TABLE IF NOT EXISTS sessions` — well before the partitions
    — already needs CREATE on the schema, so a restricted role dies there. The first version of
    this test was written that way and passed while proving nothing: `pytest.raises` was
    satisfied by a LATER statement, and widening the `except` to `Exception` left it green.
    """
    caplog.set_level("DEBUG", logger="cogno_engram.postgres")
    conn = _NetProbe(exc)
    if swallowed:
        await _partition_existing_table(conn, "turns", 2)
        assert conn.attempts == 1, "the loop must stop at the refused statement, not retry it"
        msgs = [r.getMessage() for r in caplog.records if "partitioning_skipped" in r.getMessage()]
        assert len(msgs) == 1 and "reason=incompatible_shape" in msgs[0]
        assert all(r.levelno == logging.ERROR for r in caplog.records
                   if "partitioning_skipped" in r.getMessage())
    else:
        with pytest.raises(type(exc)):
            await _partition_existing_table(conn, "turns", 2)
        assert not [r for r in caplog.records if "partitioning_skipped" in r.getMessage()], (
            "a non-legacy failure was reported as a legacy shape — the net is too wide")
