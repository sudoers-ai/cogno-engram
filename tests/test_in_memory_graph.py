import pytest

from cogno_engram.types import GraphEdge, GraphNode


async def _build_jose_rex(graph):
    # José --OWNS--> Rex --BREED--> Pastor Alemão   (the doc's example)
    await graph.upsert_node(GraphNode("s", "José", "PERSON"))
    await graph.upsert_node(GraphNode("s", "Rex", "ANIMAL"))
    await graph.upsert_node(GraphNode("s", "Pastor Alemão", "CONCEPT"))
    await graph.upsert_edge(GraphEdge("s", "José", "Rex", "OWNS", source_session="sess1"))
    await graph.upsert_edge(GraphEdge("s", "Rex", "Pastor Alemão", "BREED", source_session="sess1"))


async def test_upsert_node_dedups_case_insensitive(graph):
    a = await graph.upsert_node(GraphNode("s", "José", "PERSON"))
    b = await graph.upsert_node(GraphNode("s", "josé", "PERSON", attributes={"age": 40}))
    assert a == b
    node = await graph.find_node("s", "JOSÉ")
    assert node is not None and node.attributes == {"age": 40}


async def test_multi_hop_walk(graph):
    await _build_jose_rex(graph)
    edges = await graph.walk("s", "José", max_depth=2)
    relations = {e.relation for e in edges}
    assert relations == {"OWNS", "BREED"}                 # reached Pastor Alemão (2 hops)


async def test_walk_depth_limit(graph):
    await _build_jose_rex(graph)
    edges = await graph.walk("s", "José", max_depth=1)
    assert {e.relation for e in edges} == {"OWNS"}        # 1 hop only


async def test_walk_loop_prevention(graph):
    await graph.upsert_edge(GraphEdge("s", "A", "B", "R"))
    await graph.upsert_edge(GraphEdge("s", "B", "A", "R2"))   # cycle
    edges = await graph.walk("s", "A", max_depth=5)
    assert len(edges) == 2                                 # terminates, no infinite loop


async def test_neighbors(graph):
    await _build_jose_rex(graph)
    names = {n.label for n in await graph.neighbors("s", "Rex")}
    assert names == {"José", "Pastor Alemão"}


async def test_delete_edges_by_session_prunes(graph):
    await _build_jose_rex(graph)
    await graph.upsert_edge(GraphEdge("s", "José", "Maria", "KNOWS", source_session="sess2"))
    deleted = await graph.delete_edges_by_session("s", "sess1")
    assert deleted == 2
    remaining = await graph.walk("s", "José", max_depth=3)
    assert {e.relation for e in remaining} == {"KNOWS"}   # only sess2 survives


async def test_scope_isolation(graph):
    await graph.upsert_node(GraphNode("a", "X", "CONCEPT"))
    assert await graph.find_node("b", "X") is None


async def test_blank_scope_rejected(graph):
    with pytest.raises(ValueError):
        await graph.find_node("", "X")


async def test_upsert_edge_auto_creates_missing_endpoints():
    # Parity with the Postgres adapter (_resolve_node_id INSERTs on miss): an edge whose
    # endpoints were never declared must not dangle — they materialise as CONCEPT nodes.
    from cogno_engram.adapters.in_memory import InMemoryGraph
    g = InMemoryGraph()
    await g.upsert_edge(GraphEdge("s", "Maria", "Mimi", "OWNS", source_session="sess"))
    src = await g.find_node("s", "Maria")
    tgt = await g.find_node("s", "Mimi")
    assert src is not None and tgt is not None
    assert src.node_type == "CONCEPT" and tgt.node_type == "CONCEPT"
    assert [e.relation for e in await g.walk("s", "Maria", max_depth=1)] == ["OWNS"]


async def test_upsert_node_sets_and_bumps_timestamps():
    from cogno_engram.adapters.in_memory import InMemoryGraph
    g = InMemoryGraph()
    await g.upsert_node(GraphNode("s", "Rex", "ANIMAL"))
    first = await g.find_node("s", "Rex")
    assert first.created_at is not None and first.updated_at is not None
    await g.upsert_node(GraphNode("s", "Rex", "ANIMAL", attributes={"k": "v"}))
    again = await g.find_node("s", "Rex")
    assert again.created_at == first.created_at
    assert again.updated_at >= first.updated_at


async def test_ingest_entities_stamps_attributes():
    from cogno_engram.adapters.in_memory import InMemoryGraph
    from cogno_engram.graph_context import ingest_entities
    g = InMemoryGraph()
    n = await ingest_entities(g, "acme", [("Maria", "PERSON"), "troca"],
                              attributes={"identity_id": "id-7"})
    assert n == 2
    maria = await g.find_node("acme", "Maria")
    assert maria.attributes["identity_id"] == "id-7"
    assert (await g.find_node("acme", "troca")).attributes["identity_id"] == "id-7"
