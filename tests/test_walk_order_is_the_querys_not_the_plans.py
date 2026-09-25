"""The walk's ORDER is the query's, not the plan's — and a ranking over it keys on the STORE's id.

**The defect, measured on a live deployment (read-only):** ``PostgresKnowledgeGraph.walk`` from
one node returned the SAME set of edges in a DIFFERENT order under the custom and the generic
plan of the same statement. Its final ``SELECT DISTINCT`` had no ``ORDER BY``, so the row order
was whatever the plan's de-duplication produced — and a pooled connection that has PREPARED the
statement may be answered with a generic plan. ``lexical.graph_candidates`` then numbered each
edge by its POSITION, and ``rank`` breaks ties by that position: with dozens of edges tied on
score (every one of them names the person asked about) the top-k was a function of the plan.

**Here, with invented data** (one teacher, 80 classes, rooms at depth 2, eight small scopes
beside it — the shape that makes the generic estimate of ``scope = $1`` differ from the custom
one): the statement is prepared at its FIRST execution (``prepare_threshold=0``) and the plan is
forced per connection (``plan_cache_mode``), which is what a pooled connection reaches after its
fifth execution without waiting for it. The twin: the same ORDER and the same top-5 under both
plans. Measured before the fix on this shape: 99 and 100 of 100 positions differed (two runs), and
the top-5 and the rendered block did too.

The unit half needs no database: ``graph_candidates`` over the same edges in two orders gives the
same ids and the same top-k when the edges carry store ids, and the positional fallback (no store
id) is unchanged. Postgres DSN: ``ENGRAM_TEST_DSN`` (the conftest guard), skips without one.
"""

from __future__ import annotations

import random
from contextlib import asynccontextmanager

import pytest

from cogno_engram import AUDIENCE_STAFF, EDGE_ACCEPTED, GraphEdge
from cogno_engram.lexical import chosen, graph_candidates, rank, render

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path

DSN = resolve_test_dsn()
EMB_DIM = 8
SCOPE = "t-walk-order"
TEACHER = "Zulmira Quaresma"
QUESTION = ["Zulmira aulas"]            # every edge below names her → they ALL tie on score


# ── the unit half: ids and priors from the store's id, never from the position ────────────────

def _edges(n: int = 30) -> "list[GraphEdge]":
    return [GraphEdge(scope=SCOPE, source=TEACHER, target=f"turma {i:03d}", relation="TEACHES",
                      id=1000 + i) for i in range(n)]


def test_the_same_edges_in_another_order_give_the_same_ids_and_the_same_top_k():
    edges = _edges()
    shuffled = list(edges)
    random.Random(7).shuffle(shuffled)
    assert shuffled != edges or [e.id for e in shuffled] != [e.id for e in edges]
    a = rank(graph_candidates([(0, 0, 1, edges)]), QUESTION)
    b = rank(graph_candidates([(0, 0, 1, shuffled)]), QUESTION)
    assert {s for s, _ in a} == {0.5}, "the control: every candidate ties on score"
    assert [c.id for _, c in chosen(a, k=5)] == [c.id for _, c in chosen(b, k=5)]
    assert sorted(c.id for _, c in a) == sorted(f"edge:{1000 + i}" for i in range(30))
    assert [c.prior for _, c in a] == [1000 + i for i in range(30)]


def test_an_edge_with_no_store_id_keeps_the_positional_id_and_prior():
    edges = [GraphEdge(scope=SCOPE, source=TEACHER, target=f"turma {i}", relation="TEACHES")
             for i in range(3)]
    cands = graph_candidates([(0, 0, 42, edges)])
    assert [c.id for c in cands] == ["edge:42.0", "edge:42.1", "edge:42.2"]
    assert [c.prior for c in cands] == [0, 1, 2]


def test_the_store_id_is_not_part_of_an_edges_equality():
    a = GraphEdge(scope=SCOPE, source="a", target="b", relation="R", id=1)
    assert a == GraphEdge(scope=SCOPE, source="a", target="b", relation="R", id=2)
    assert a == GraphEdge(scope=SCOPE, source="a", target="b", relation="R")


# ── the Postgres half: the same order, and the same top-k, under both plans ──────────────────

class _OneConnection:
    """The ``pool`` the adapter accepts, lending ONE connection that prepares every statement at
    its first execution and answers it with the plan this test forces."""

    def __init__(self, conn) -> None:
        self.conn = conn

    @asynccontextmanager
    async def connection(self):
        yield self.conn


@pytest.fixture
async def world():
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to a reachable Postgres+pgvector to run")
    psycopg = pytest.importorskip("psycopg")
    from cogno_engram.adapters.postgres import PostgresKnowledgeGraph, ensure_schema
    try:
        conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True, connect_timeout=3)
    except Exception:  # noqa: BLE001
        pytest.skip("test Postgres unreachable")
    try:
        for t in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns",
                  "sessions"):
            await conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
        await ensure_schema(conn, embedding_dim=EMB_DIM)
    finally:
        await conn.close()
    g = PostgresKnowledgeGraph(dsn=DSN)
    for i in range(80):
        await g.upsert_edge(GraphEdge(scope=SCOPE, source=TEACHER, target=f"turma {i:03d}",
                                      relation="TEACHES", status=EDGE_ACCEPTED, audience="tenant"))
        if i % 4 == 0:
            await g.upsert_edge(GraphEdge(scope=SCOPE, source=f"turma {i:03d}",
                                          target=f"sala {i:03d} Zulmira", relation="MEETS_IN",
                                          status=EDGE_ACCEPTED, audience="tenant"))
    for o in range(8):
        for i in range(5):
            await g.upsert_edge(GraphEdge(scope=f"t-other-{o}", source=f"pessoa {o}-{i}",
                                          target=f"coisa {o}-{i}", relation="HAS",
                                          status=EDGE_ACCEPTED, audience="tenant"))
    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as c:
        await c.execute("ANALYZE knowledge_edges")
        await c.execute("ANALYZE knowledge_nodes")
    yield


async def _walk_under(plan_mode: str) -> "list[GraphEdge]":
    import psycopg
    from cogno_engram.adapters.postgres import PostgresKnowledgeGraph
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True, prepare_threshold=0)
    try:
        await conn.execute(f"SET plan_cache_mode = {plan_mode}")
        g = PostgresKnowledgeGraph(pool=_OneConnection(conn))
        edges: "list[GraphEdge]" = []
        for _ in range(3):              # the same statement, again, on the same connection
            edges = await g.walk(SCOPE, TEACHER, audience=AUDIENCE_STAFF, max_depth=2)
        cur = await conn.execute("SELECT generic_plans, custom_plans FROM pg_prepared_statements")
        plans = [tuple(r) for r in await cur.fetchall()]
        # the CONDITION: the walk statement really ran, prepared, under the forced kind of plan
        want = 0 if plan_mode == "force_generic_plan" else 1
        assert any(p[want] >= 3 for p in plans), (plan_mode, plans)
        return edges
    finally:
        await conn.close()


def _keys(edges: "list[GraphEdge]") -> "list[tuple]":
    return [(e.source, e.relation, e.target) for e in edges]


async def test_the_walk_returns_the_same_ORDER_under_the_custom_and_the_generic_plan(world):
    custom = await _walk_under("force_custom_plan")
    generic = await _walk_under("force_generic_plan")
    assert len(custom) == 100
    assert sorted(_keys(custom)) == sorted(_keys(generic)), "the control: the SAME set"
    assert _keys(custom) == _keys(generic), \
        f"{sum(1 for a, b in zip(_keys(custom), _keys(generic)) if a != b)} positions differ"
    # the documented order: the shallowest depth first, then the store's edge id
    assert all(isinstance(e.id, int) for e in custom)
    depth1 = [e for e in custom if e.relation == "TEACHES"]
    assert custom[:len(depth1)] == depth1, "depth 1 before depth 2"
    assert [e.id for e in depth1] == sorted(e.id for e in depth1)


async def test_the_TOP_K_and_the_BLOCK_over_edges_that_tie_are_the_same_under_both_plans(world):
    """What a caller hands on — the top-k and the rendered block (``render``) — is identical under
    the two plans. Before the fix, on this fixture, both differed."""
    tops, blocks = {}, {}
    for mode in ("force_custom_plan", "force_generic_plan"):
        edges = await _walk_under(mode)
        ranked = rank(graph_candidates([(0, 0, 1, edges)]), QUESTION)
        assert sum(1 for s, _ in ranked if s == ranked[0][0]) >= 25, "the condition: a tie"
        picked = chosen(ranked, k=5)
        tops[mode] = [c.id for _, c in picked]
        blocks[mode] = render(QUESTION[0], picked)
    assert tops["force_custom_plan"] == tops["force_generic_plan"], tops
    assert blocks["force_custom_plan"] == blocks["force_generic_plan"]
