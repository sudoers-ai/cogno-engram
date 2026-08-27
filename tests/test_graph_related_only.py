"""``related_only`` — the search filter that keeps isolated nodes out of the caller's slots.

A node with no edge is a legitimate row: the NER write channel creates one per entity, the
dashboard lists it, staff search it. It is only wrong as a WALK START — it spends one of the
caller's few slots and the walk from it returns nothing.

Measured on the live box before this existed (host's ``_graph_context``: ``walk(max_depth=2)``
then ``_durable_edges``, over 40 real ``user_input`` queries): **6/40 turns got any relation
at all — 15%. With the filter, 40/40 — 100%**, and 169 → 2268 durable edges delivered.

The two adapters must agree, so every case here runs against both (see
``test_audience_parity.py`` for the same discipline applied to the audience wall).
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import (AUDIENCE_STAFF, EDGE_ACCEPTED, EDGE_PROPOSED,
                                GraphEdge, GraphNode)

pytestmark = pytest.mark.asyncio


async def _graph_with_one_of_each():
    """``lonely`` has no edge; ``joined`` has one. Both are embedded and both are visible."""
    kg = InMemoryGraph()
    scope = "t/related"
    await kg.upsert_node(GraphNode(scope, "lonely", "CONCEPT", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode(scope, "joined", "CONCEPT", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode(scope, "other", "CONCEPT", embedding=[0.0, 1.0]))
    await kg.upsert_edge(GraphEdge(scope, "joined", "other", "KNOWS"))
    return kg, scope


async def test_default_still_returns_isolated_nodes():
    """THE DEFAULT IS TODAY'S BEHAVIOUR — this test dies the day that stops being true.

    Not a style preference: the two callers want opposite things. ``_graph_context`` walks
    from these nodes and wants ones it can walk from; the boot schema probe exists to confirm
    the query EXECUTES and must run its simplest form, not a caller-specific variant that
    silently joins ``knowledge_edges``. Other readers — a staff search, the dashboard node
    list — legitimately want the isolated nodes. A default decides for all of them.
    """
    kg, scope = await _graph_with_one_of_each()
    labels = {n.label for n in
              await kg.find_nodes_by_embedding(scope, [1.0, 0.0], audience=AUDIENCE_STAFF, limit=5)}
    assert "lonely" in labels, "the default must NOT filter — the schema probe depends on it"


async def test_related_only_drops_the_isolated_node():
    kg, scope = await _graph_with_one_of_each()
    labels = {n.label for n in
              await kg.find_nodes_by_embedding(scope, [1.0, 0.0], audience=AUDIENCE_STAFF,
                                               limit=5, related_only=True)}
    assert "lonely" not in labels
    assert "joined" in labels


async def test_related_only_counts_an_edge_arriving_at_the_node():
    """Either END of an edge makes a node walkable — the walk is undirected in both adapters.

    Separated from the case above deliberately: a filter written as ``source_id = n.id`` alone
    passes that one and silently halves the recall of a graph whose relations mostly point AT
    the person (``Rex OWNED_BY José``).
    """
    kg = InMemoryGraph()
    scope = "t/inbound"
    await kg.upsert_node(GraphNode(scope, "target-only", "CONCEPT", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode(scope, "src", "CONCEPT", embedding=[0.0, 1.0]))
    await kg.upsert_edge(GraphEdge(scope, "src", "target-only", "KNOWS"))
    labels = {n.label for n in
              await kg.find_nodes_by_embedding(scope, [1.0, 0.0], audience=AUDIENCE_STAFF,
                                               limit=5, related_only=True)}
    assert "target-only" in labels


async def test_related_only_is_scoped():
    """An edge in ANOTHER tenant's scope must not rescue a node from this one."""
    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode("t/a", "shared-name", "CONCEPT", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode("t/b", "shared-name", "CONCEPT", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode("t/b", "friend", "CONCEPT", embedding=[0.0, 1.0]))
    await kg.upsert_edge(GraphEdge("t/b", "shared-name", "friend", "KNOWS"))
    labels = {n.label for n in
              await kg.find_nodes_by_embedding("t/a", [1.0, 0.0], audience=AUDIENCE_STAFF,
                                               limit=5, related_only=True)}
    assert labels == set(), "t/b's edge leaked across the scope boundary"


# ── the two blind spots the first cut shipped with ──────────────────────────────────────


async def test_related_only_ignores_an_UNREVIEWED_edge():
    """`walk` traverses ACCEPTED edges only, so a `proposed` edge does not make a node walkable.

    Reachable by design, not a corner case: the host writes proximity relations as PROPOSED
    (`propose_relations`), which is exactly the class of edge a turn wants. Counting one here
    is worse than missing it — see the pessimisation test below.
    """
    kg = InMemoryGraph()
    scope = "t/unreviewed"
    await kg.upsert_node(GraphNode(scope, "jose", "PERSON", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode(scope, "pedro", "PERSON", embedding=[0.9, 0.1]))
    await kg.upsert_edge(GraphEdge(scope, "jose", "pedro", "PARENT_OF", status=EDGE_PROPOSED))
    out = await kg.find_nodes_by_embedding(scope, [1.0, 0.0], audience=AUDIENCE_STAFF,
                                           limit=5, related_only=True)
    assert out == [], "an unreviewed edge must not make a node look walkable"


async def test_counting_an_unreviewed_edge_would_be_a_PESSIMISATION():
    """Not merely a miss: it EVICTS a nearer candidate for a farther one that also goes nowhere.

    Unfiltered the search returns `jose` + `lonely`; counting the proposed edge would return
    `jose` + `pedro` — `pedro` is FARTHER from the probe than `lonely`, and neither walks. The
    filter would have made the answer strictly worse than no filter at all.
    """
    kg = InMemoryGraph()
    scope = "t/pess"
    await kg.upsert_node(GraphNode(scope, "jose", "PERSON", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode(scope, "lonely", "CONCEPT", embedding=[0.99, 0.01]))
    await kg.upsert_node(GraphNode(scope, "pedro", "PERSON", embedding=[0.5, 0.5]))
    await kg.upsert_node(GraphNode(scope, "ana", "PERSON", embedding=[0.4, 0.6]))
    await kg.upsert_edge(GraphEdge(scope, "jose", "pedro", "PARENT_OF", status=EDGE_PROPOSED))
    await kg.upsert_edge(GraphEdge(scope, "pedro", "ana", "KNOWS", status=EDGE_ACCEPTED))
    out = await kg.find_nodes_by_embedding(scope, [1.0, 0.0], audience=AUDIENCE_STAFF,
                                           limit=2, related_only=True)
    labels = [n.label for n in out]
    assert "jose" not in labels, (
        "`jose` has only a PROPOSED edge — picking it spends a slot on a node that cannot walk")
    assert labels == ["pedro", "ana"]


async def test_the_filter_runs_BEFORE_the_slice():
    """THE SLOT COMPETITION IS THE WHOLE FEATURE — and the first cut never tested it.

    Every test in the original PR used `limit=5` against <= 4 nodes, so the limit never bit and
    a filter applied AFTER the slice passed all of them (measured: 396 green). The real caller
    asks for `limit=2`. Here three isolated nodes sit NEARER the probe than the one connected
    node, so a post-slice filter returns [] — the feature 100% dead with the board 100% green.
    """
    kg = InMemoryGraph()
    scope = "t/slots"
    for i in range(3):
        await kg.upsert_node(GraphNode(scope, f"iso-{i}", "CONCEPT", embedding=[1.0, 0.0]))
    await kg.upsert_node(GraphNode(scope, "joined", "CONCEPT", embedding=[0.7, 0.3]))
    await kg.upsert_node(GraphNode(scope, "friend", "CONCEPT", embedding=[0.0, 1.0]))
    await kg.upsert_edge(GraphEdge(scope, "joined", "friend", "KNOWS"))
    out = await kg.find_nodes_by_embedding(scope, [1.0, 0.0], audience=AUDIENCE_STAFF,
                                           limit=2, related_only=True)
    labels = {n.label for n in out}
    assert labels == {"joined", "friend"}, (
        "the filter must narrow the CANDIDATES, not the already-chosen slots")
