"""``graph_stats`` — the dashboard summary in two aggregated reads instead of ``1 + 3N``.

The host's ``knowledge_stats`` route needed four numbers — total nodes, total edges, a histogram
by type, and the five most connected nodes — and **nothing in the port could answer "how connected
is each node" in bulk**. So it listed every node and called ``get_node_context`` for each; that
helper is itself ``find_node`` + ``walk`` + ``neighbors``, which makes the real cost ``1 + 3N``:

    388 nodes on the live box (27/08/2026)  ->  1165 queries per page open, growing with the graph

**This is a COST change and not a MEANING change**, and that is the hard part to prove: a
performance PR that quietly moves a number is worse than the slow version. So the first test here
does not assert the numbers — it recomputes them THE OLD WAY, node by node, and demands the two
agree. Rewrite the aggregate wrong and that test fails, whatever the new numbers look like.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import (AUDIENCE_STAFF, AUDIENCE_TENANT, EDGE_PROPOSED,
                                GraphEdge, GraphNode)

pytestmark = pytest.mark.asyncio


async def _the_old_way(kg, scope: str, audience: str, top: int = 5):
    """The host's previous algorithm, verbatim in shape: list, then one context per node.

    It is the ORACLE. Keeping it here — rather than asserting hand-written numbers — is what
    makes this a proof of equivalence instead of a restatement of the new code's opinion.
    """
    nodes = await kg.list_nodes(scope, audience=audience, limit=100000)
    by_type: dict = {}
    for n in nodes:
        by_type[n.node_type] = by_type.get(n.node_type, 0) + 1
    edge_keys: set = set()
    degree: dict = {}
    for n in nodes:
        ctx = await kg.get_node_context(scope, n.label, audience=audience)
        es = ctx.edges if ctx else []
        degree[n.label] = len(es)
        for e in es:
            edge_keys.add((e.source, e.target, e.relation))
    ranked = sorted(degree.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return len(nodes), len(edge_keys), by_type, ranked


async def _a_graph():
    """A shape with every case that decides a number: a hub, leaves, an orphan, an unreviewed
    edge, and an edge only staff may read."""
    kg = InMemoryGraph()
    scope = "t/stats"
    for label, ntype in [("Acme", "ORG"), ("Ana", "PERSON"), ("Bruno", "PERSON"),
                         ("Carla", "PERSON"), ("Solo", "CONCEPT")]:
        await kg.upsert_node(GraphNode(scope, label, ntype))
    await kg.upsert_edge(GraphEdge(scope, "Ana", "Acme", "WORKS_AT", audience=AUDIENCE_TENANT))
    await kg.upsert_edge(GraphEdge(scope, "Bruno", "Acme", "WORKS_AT", audience=AUDIENCE_TENANT))
    await kg.upsert_edge(GraphEdge(scope, "Carla", "Acme", "OWNS", audience=AUDIENCE_TENANT))
    # PROPOSED: `walk` skips it, so the old degree never counted it — and neither may we.
    await kg.upsert_edge(GraphEdge(scope, "Ana", "Bruno", "KNOWS",
                                   status=EDGE_PROPOSED, audience=AUDIENCE_TENANT))
    return kg, scope


async def test_the_aggregate_says_exactly_what_the_NODE_BY_NODE_version_said():
    """The whole point of the change: same answer, fewer queries.

    Asserted against the recomputation, not against constants — a constant would have to be
    updated by whoever breaks the semantics, which is precisely the person who should be stopped.
    """
    kg, scope = await _a_graph()
    for audience in (AUDIENCE_STAFF, AUDIENCE_TENANT):
        want_n, want_e, want_by_type, want_top = await _the_old_way(kg, scope, audience)
        got = await kg.graph_stats(scope, audience=audience)
        assert got.total_nodes == want_n, audience
        assert got.total_edges == want_e, audience
        assert got.by_type == want_by_type, audience
        assert [(n.label, d) for n, d in got.top_connected] == want_top, audience


async def test_an_UNREVIEWED_edge_changes_no_number():
    """An edge nobody accepted is not a relation yet. ``walk`` refuses to traverse it, so the
    old degree never saw it; counting it here would inflate the dashboard with proposals."""
    kg, scope = await _a_graph()
    before = await kg.graph_stats(scope, audience=AUDIENCE_STAFF)
    await kg.upsert_edge(GraphEdge(scope, "Carla", "Solo", "MENTIONS",
                                   status=EDGE_PROPOSED, audience=AUDIENCE_TENANT))
    after = await kg.graph_stats(scope, audience=AUDIENCE_STAFF)
    assert after.total_edges == before.total_edges
    assert dict((n.label, d) for n, d in after.top_connected) == \
           dict((n.label, d) for n, d in before.top_connected)


async def test_the_orphan_is_staff_only_and_still_RANKED():
    """Two rules at once, and the second is the one an aggregate loses by accident.

    An orphan is invisible to a non-staff reader (node visibility is DERIVED from visible edges)
    — but to STAFF it is a real node with degree zero, and it must still appear in the count and
    in the ranking's tail. An inner join would have dropped it and shortened the list in silence.
    """
    kg, scope = await _a_graph()
    staff = await kg.graph_stats(scope, audience=AUDIENCE_STAFF, top=10)
    tenant = await kg.graph_stats(scope, audience=AUDIENCE_TENANT, top=10)
    assert ("Solo", 0) in [(n.label, d) for n, d in staff.top_connected]
    assert "Solo" not in [n.label for n, _ in tenant.top_connected]
    assert staff.total_nodes == tenant.total_nodes + 1
    assert staff.by_type.get("CONCEPT") == 1 and "CONCEPT" not in tenant.by_type


async def test_degree_counts_BOTH_ends():
    """Half the relations point AT the node (``Ana WORKS_AT Acme``), so counting only the source
    would halve the hub and make the ranking meaningless."""
    kg, scope = await _a_graph()
    st = await kg.graph_stats(scope, audience=AUDIENCE_STAFF)
    top = dict((n.label, d) for n, d in st.top_connected)
    assert top["Acme"] == 3, "o hub é alvo das três arestas, não origem"
    assert top["Ana"] == 1


async def test_top_is_a_CUT_not_the_whole_list():
    """``top`` bounds the ranking and nothing else — the totals stay whole."""
    kg, scope = await _a_graph()
    two = await kg.graph_stats(scope, audience=AUDIENCE_STAFF, top=2)
    none = await kg.graph_stats(scope, audience=AUDIENCE_STAFF, top=0)
    assert len(two.top_connected) == 2
    assert none.top_connected == []
    assert none.total_nodes == two.total_nodes == 5


async def test_TIES_keep_the_store_order_not_alphabetical():
    """A tie-break is behaviour, and this method was supposed to change only COST.

    The caller's previous version ranked over whatever ``list_nodes`` gave it — ``ORDER BY id``,
    i.e. insertion order. The first cut of this method sorted ties by LABEL, which is equally
    deterministic and quietly different: the host's own `test_knowledge_walk_and_stats_shape`
    failed because two nodes of degree 1 had swapped places. Caught by a test nobody wrote for
    this change, which is the only reason it was not shipped as "a rewrite, tests still pass".
    """
    kg = InMemoryGraph()
    scope = "t/ties"
    # Inserted in an order the alphabet disagrees with, so the two rules cannot both pass.
    for label in ("Zeta", "Alfa"):
        await kg.upsert_node(GraphNode(scope, label, "PERSON"))
    await kg.upsert_node(GraphNode(scope, "Hub", "ORG"))
    await kg.upsert_edge(GraphEdge(scope, "Zeta", "Hub", "AT", audience=AUDIENCE_TENANT))
    await kg.upsert_edge(GraphEdge(scope, "Alfa", "Hub", "AT", audience=AUDIENCE_TENANT))
    st = await kg.graph_stats(scope, audience=AUDIENCE_STAFF, top=10)
    tied = [n.label for n, d in st.top_connected if d == 1]
    assert tied == ["Zeta", "Alfa"], "empate segue a ordem do store (id), não o alfabeto"
