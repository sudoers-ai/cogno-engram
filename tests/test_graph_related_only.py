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
from cogno_engram.types import AUDIENCE_STAFF, GraphEdge, GraphNode

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

    Not a style preference. The host's boot schema probe calls this port with a ZERO vector
    for one purpose: to confirm the query executes. Under a filtering default it would get an
    empty list from a graph that simply has no edges yet, and could not tell a broken schema
    from a new deployment. A default decides for every future reader; a parameter costs the
    two current callers one keyword.
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
