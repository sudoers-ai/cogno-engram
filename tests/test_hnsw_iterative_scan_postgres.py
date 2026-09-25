"""The HNSW index under a scope filter: ``find_nodes_by_embedding`` must not come back EMPTY.

**The defect, PRODUCED here.** One HNSW index serves every scope of ``knowledge_nodes``, and the
``scope`` condition is applied to what the index returns. With ``hnsw.ef_search`` at its default
of 40, the index hands over the 40 nearest nodes of the WHOLE table; a scope whose nodes are all
farther than 40 nodes of OTHER scopes then gets ZERO rows although it holds hundreds. The shape:
600 nodes in the asked scope, 5 000 in another scope, all of those nearer the query.

**The cure:** pgvector 0.8's iterative scan, set for the query's own transaction
(``HNSW_ITERATIVE_SCAN``, ``HNSW_MAX_SCAN_TUPLES``). This file shows, against a real
Postgres + pgvector:

* the premise — the planner DOES answer the query from the HNSW index (``EXPLAIN``), otherwise
  an exact scan would hide the defect and the test would prove nothing;
* the world before — the same query with the iterative scan OFF returns 0 rows;
* the world after — ``find_nodes_by_embedding`` returns ``limit`` rows, and they are EXACTLY the
  nearest ones in order (compared with an exact scan with the index disabled);
* the control — a small scope that nothing crowds out answers exactly, before and after;
* the setting never outlives the query (a later statement on the same session sees the server
  default again).

Gated like the rest of the Postgres suite: it aims at ``engram_test`` by itself and skips when
nothing is listening. Every scope, label and vector here is invented.
"""

from __future__ import annotations

import math
import random

import pytest

psycopg = pytest.importorskip("psycopg")
from psycopg.rows import dict_row  # noqa: E402

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path

from cogno_engram.adapters.postgres import (  # noqa: E402
    HNSW_ITERATIVE_SCAN,
    HNSW_MAX_SCAN_TUPLES,
    PostgresKnowledgeGraph,
    ensure_schema,
)
from cogno_engram.types import AUDIENCE_STAFF  # noqa: E402

DSN = resolve_test_dsn()
EMB_DIM = 8
IN_SCOPE, CROWD, SMALL = "t-escola/f22e", "t-outro/f22e", "t-pequeno/f22e"
N_IN, N_CROWD, LIMIT = 600, 5_000, 5
QUERY = [1.0] + [0.0] * (EMB_DIM - 1)


def _unit(v: "list[float]") -> "list[float]":
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _near(rng: random.Random) -> "list[float]":
    """A vector close to the query (axis 0) — the crowd of the OTHER scope."""
    return _unit([1.0] + [rng.uniform(-0.05, 0.05) for _ in range(EMB_DIM - 1)])


def _nearest(rng: random.Random) -> "list[float]":
    """Closer to the query than any of the crowd — the small scope nothing crowds out."""
    return _unit([1.0] + [rng.uniform(-0.001, 0.001) for _ in range(EMB_DIM - 1)])


def _far(rng: random.Random) -> "list[float]":
    """A vector far from the query (around axis 1) — every node of the asked scope."""
    return _unit([rng.uniform(0.0, 0.05), 1.0] + [rng.uniform(-0.05, 0.05)
                                                   for _ in range(EMB_DIM - 2)])


def _lit(v: "list[float]") -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


async def _connect():
    return await psycopg.AsyncConnection.connect(DSN, autocommit=True, connect_timeout=3,
                                                 row_factory=dict_row)


@pytest.fixture
async def world():
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to a reachable Postgres+pgvector to run")
    try:
        conn = await _connect()
    except Exception as exc:
        pytest.skip(f"set ENGRAM_TEST_DSN to a reachable Postgres+pgvector to run "
                    f"({type(exc).__name__})")
    for tbl in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns",
                "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM)
    ver = await (await conn.execute(
        "SELECT extversion FROM pg_extension WHERE extname = 'vector'")).fetchone()
    major_minor = tuple(int(x) for x in ver["extversion"].split("-")[0].split(".")[:2])
    if major_minor < (0, 8):
        await conn.close()
        pytest.skip(f"pgvector {ver['extversion']} has no iterative scan (needs 0.8+)")
    rng = random.Random(20260925)
    rows = ([(CROWD, f"crowd-{i}", _lit(_near(rng))) for i in range(N_CROWD)]
            + [(IN_SCOPE, f"asked-{i}", _lit(_far(rng))) for i in range(N_IN)]
            + [(SMALL, f"small-{i}", _lit(_nearest(rng))) for i in range(3)])
    async with conn.cursor() as cur:
        await cur.executemany(
            "INSERT INTO knowledge_nodes (scope, label, embedding) VALUES (%s, %s, %s::vector)",
            rows)
    await conn.execute("ANALYZE knowledge_nodes")
    yield conn
    await conn.close()


_SQL = ("SELECT n.id FROM knowledge_nodes n WHERE n.scope = %s AND n.embedding IS NOT NULL "
        "ORDER BY n.embedding <=> %s::vector LIMIT %s")


async def _plain(conn, scope: str) -> "list[int]":
    """The query WITHOUT the iterative scan — the world before the fix."""
    cur = await conn.execute(_SQL, (scope, _lit(QUERY), LIMIT))
    return [r["id"] for r in await cur.fetchall()]


async def _exact(conn, scope: str) -> "list[int]":
    """The TRUE nearest rows: the same query with index scans disabled (an exact sort)."""
    async with conn.transaction():
        await conn.execute("SET LOCAL enable_indexscan = off")
        cur = await conn.execute(_SQL, (scope, _lit(QUERY), LIMIT))
        return [r["id"] for r in await cur.fetchall()]


async def test_PREMISE_the_planner_answers_from_the_HNSW_index(world):
    cur = await world.execute("EXPLAIN " + _SQL, (IN_SCOPE, _lit(QUERY), LIMIT))
    plan = "\n".join(r["QUERY PLAN"] for r in await cur.fetchall())
    assert "idx_nodes_embedding" in plan, plan


async def test_the_DEFECT_without_the_iterative_scan_a_crowded_scope_comes_back_EMPTY(world):
    assert await _plain(world, IN_SCOPE) == [], "the defect: 600 nodes in scope, 0 returned"
    assert len(await _exact(world, IN_SCOPE)) == LIMIT, "the pair: the rows ARE there"


async def _distances(conn, ids: "list[int]") -> "list[float]":
    cur = await conn.execute(
        "SELECT id, embedding <=> %s::vector AS d FROM knowledge_nodes WHERE id = ANY(%s)",
        (_lit(QUERY), ids))
    by = {r["id"]: float(r["d"]) for r in await cur.fetchall()}
    return [by[i] for i in ids]


async def test_the_FIX_find_nodes_returns_limit_rows_in_STRICT_distance_order(world):
    """HNSW is approximate, so "the nearest" is measured against the exact scan rather than
    asserted equal to it: the rows are the asked scope's, in non-decreasing distance (the
    `strict_order` promise), and they are the TRUE nearest ones for this fixed seed."""
    graph = PostgresKnowledgeGraph(dsn=DSN)
    nodes = await graph.find_nodes_by_embedding(IN_SCOPE, QUERY, audience=AUDIENCE_STAFF,
                                                limit=LIMIT)
    ids = [n.id for n in nodes]
    assert len(ids) == LIMIT, "600 nodes in scope: LIMIT of them come back, not 0"
    assert all(n.scope == IN_SCOPE for n in nodes)
    d = await _distances(world, ids)
    assert d == sorted(d), "strict_order: the rows arrive nearest first"
    exact = await _exact(world, IN_SCOPE)
    assert len(set(ids) & set(exact)) >= LIMIT - 1, (ids, exact)


async def test_CONTROL_a_small_scope_nothing_crowds_out_is_right_before_and_after(world):
    """The fix changes nothing where the index already answered: a small scope NEARER the query
    than the crowd gets its 3 rows with the iterative scan off and on alike, and the crowd's own
    scope still gets its LIMIT nearest."""
    graph = PostgresKnowledgeGraph(dsn=DSN)
    after = [n.id for n in await graph.find_nodes_by_embedding(SMALL, QUERY,
                                                               audience=AUDIENCE_STAFF,
                                                               limit=LIMIT)]
    before = await _plain(world, SMALL)
    exact = await _exact(world, SMALL)
    assert len(exact) == 3 and set(after) == set(before) == set(exact)
    crowd = await graph.find_nodes_by_embedding(CROWD, QUERY, audience=AUDIENCE_STAFF,
                                                limit=LIMIT)
    assert len(crowd) == LIMIT and all(n.scope == CROWD for n in crowd)


async def test_the_MEMORY_search_returns_its_rows_in_the_SAME_crowded_shape(world):
    """The memory search is the other vector read, and it is NOT given the iterative scan — on a
    measurement, not by omission: its ORDER BY is an EXPRESSION (the hybrid score; the distance
    minus a feedback term), which an HNSW index cannot serve, so it scans the scope exactly. This
    pins the BEHAVIOUR that makes that safe, in the defect's own shape (600 of the asked scope, far;
    5 000 of another, near): the search returns its LIMIT, vector-only and hybrid alike. If a
    rewrite ever lets the index answer it, this goes red and says what to add."""
    from uuid import uuid4

    from cogno_engram.adapters.postgres import PostgresStore
    from cogno_engram.types import RetrievalQuery

    rng = random.Random(9)
    rows = ([(str(uuid4()), "m-outro/f22e", "fact", f"crowd note {i}", _lit(_near(rng)))
             for i in range(N_CROWD)]
            + [(str(uuid4()), "m-escola/f22e", "fact", f"asked note {i}", _lit(_far(rng)))
               for i in range(N_IN)])
    async with world.cursor() as cur:
        await cur.executemany("INSERT INTO memories (id, scope, category, content, embedding) "
                              "VALUES (%s, %s, %s, %s, %s::vector)", rows)
    await world.execute("ANALYZE memories")
    store = PostgresStore(dsn=DSN)
    vector_only = await store.load_memories("m-escola/f22e",
                                            query=RetrievalQuery(embedding=QUERY), limit=LIMIT)
    hybrid = await store.load_memories("m-escola/f22e",
                                       query=RetrievalQuery(text="asked note", embedding=QUERY),
                                       limit=LIMIT)
    assert len(vector_only) == LIMIT and all(m.scope == "m-escola/f22e" for m in vector_only)
    assert len(hybrid) == LIMIT and all(m.scope == "m-escola/f22e" for m in hybrid)


async def test_the_setting_lives_ONLY_for_its_transaction(world):
    async with world.transaction():
        await world.execute(HNSW_ITERATIVE_SCAN, (str(HNSW_MAX_SCAN_TUPLES),))
        inside = await (await world.execute("SHOW hnsw.iterative_scan")).fetchone()
        tuples = await (await world.execute("SHOW hnsw.max_scan_tuples")).fetchone()
    after = await (await world.execute("SHOW hnsw.iterative_scan")).fetchone()
    assert inside["hnsw.iterative_scan"] == "strict_order"
    assert tuples["hnsw.max_scan_tuples"] == str(HNSW_MAX_SCAN_TUPLES)
    assert after["hnsw.iterative_scan"] == "off", "a pooled connection must return clean"
