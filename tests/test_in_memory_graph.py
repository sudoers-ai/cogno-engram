import pytest

from cogno_engram.types import AUDIENCE_STAFF, GraphEdge, GraphNode


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
    node = await graph.find_node("s", "JOSÉ", audience=AUDIENCE_STAFF)
    assert node is not None and node.attributes == {"age": 40}


async def test_multi_hop_walk(graph):
    await _build_jose_rex(graph)
    edges = await graph.walk("s", "José", max_depth=2, audience=AUDIENCE_STAFF)
    relations = {e.relation for e in edges}
    assert relations == {"OWNS", "BREED"}                 # reached Pastor Alemão (2 hops)


async def test_walk_depth_limit(graph):
    await _build_jose_rex(graph)
    edges = await graph.walk("s", "José", max_depth=1, audience=AUDIENCE_STAFF)
    assert {e.relation for e in edges} == {"OWNS"}        # 1 hop only


async def test_walk_loop_prevention(graph):
    await graph.upsert_edge(GraphEdge("s", "A", "B", "R"))
    await graph.upsert_edge(GraphEdge("s", "B", "A", "R2"))   # cycle
    edges = await graph.walk("s", "A", max_depth=5, audience=AUDIENCE_STAFF)
    assert len(edges) == 2                                 # terminates, no infinite loop


async def test_neighbors(graph):
    await _build_jose_rex(graph)
    names = {n.label for n in await graph.neighbors("s", "Rex", audience=AUDIENCE_STAFF)}
    assert names == {"José", "Pastor Alemão"}


async def test_delete_edges_by_session_prunes(graph):
    await _build_jose_rex(graph)
    await graph.upsert_edge(GraphEdge("s", "José", "Maria", "KNOWS", source_session="sess2"))
    deleted = await graph.delete_edges_by_session("s", "sess1")
    assert deleted == 2
    remaining = await graph.walk("s", "José", max_depth=3, audience=AUDIENCE_STAFF)
    assert {e.relation for e in remaining} == {"KNOWS"}   # only sess2 survives


async def test_scope_isolation(graph):
    await graph.upsert_node(GraphNode("a", "X", "CONCEPT"))
    assert await graph.find_node("b", "X", audience=AUDIENCE_STAFF) is None


async def test_blank_scope_rejected(graph):
    with pytest.raises(ValueError):
        await graph.find_node("", "X", audience=AUDIENCE_STAFF)


async def test_upsert_edge_auto_creates_missing_endpoints():
    # Parity with the Postgres adapter (_resolve_node_id INSERTs on miss): an edge whose
    # endpoints were never declared must not dangle — they materialise as CONCEPT nodes.
    from cogno_engram.adapters.in_memory import InMemoryGraph
    g = InMemoryGraph()
    await g.upsert_edge(GraphEdge("s", "Maria", "Mimi", "OWNS", source_session="sess"))
    src = await g.find_node("s", "Maria", audience=AUDIENCE_STAFF)
    tgt = await g.find_node("s", "Mimi", audience=AUDIENCE_STAFF)
    assert src is not None and tgt is not None
    assert src.node_type == "CONCEPT" and tgt.node_type == "CONCEPT"
    assert [e.relation for e in await g.walk("s", "Maria", max_depth=1, audience=AUDIENCE_STAFF)] == ["OWNS"]


async def test_upsert_node_sets_and_bumps_timestamps():
    from cogno_engram.adapters.in_memory import InMemoryGraph
    g = InMemoryGraph()
    await g.upsert_node(GraphNode("s", "Rex", "ANIMAL"))
    first = await g.find_node("s", "Rex", audience=AUDIENCE_STAFF)
    assert first.created_at is not None and first.updated_at is not None
    await g.upsert_node(GraphNode("s", "Rex", "ANIMAL", attributes={"k": "v"}))
    again = await g.find_node("s", "Rex", audience=AUDIENCE_STAFF)
    assert again.created_at == first.created_at
    assert again.updated_at >= first.updated_at


async def test_ingest_entities_stamps_attributes():
    from cogno_engram.adapters.in_memory import InMemoryGraph
    from cogno_engram.graph_context import ingest_entities
    g = InMemoryGraph()
    n = await ingest_entities(g, "acme", [("Maria", "PERSON"), "troca"],
                              attributes={"identity_id": "id-7"})
    assert n == 2
    maria = await g.find_node("acme", "Maria", audience=AUDIENCE_STAFF)
    assert maria.attributes["identity_id"] == "id-7"
    assert (await g.find_node("acme", "troca", audience=AUDIENCE_STAFF)).attributes["identity_id"] == "id-7"


async def test_purge_scope_drops_nodes_and_edges_isolated(graph):
    await _build_jose_rex(graph)                            # scope "s": 3 nodes, 2 edges
    await graph.upsert_node(GraphNode("other", "Maria", "PERSON"))
    await graph.upsert_edge(GraphEdge("other", "Maria", "Rex", "OWNS"))

    removed = await graph.purge_scope("s")
    assert removed == 5                                     # 3 nodes + 2 edges

    assert await graph.find_node("s", "José", audience=AUDIENCE_STAFF) is None
    assert await graph.list_nodes("s", audience=AUDIENCE_STAFF) == []
    # neighbour scope intact
    assert await graph.find_node("other", "Maria", audience=AUDIENCE_STAFF) is not None


async def test_purge_scope_graph_rejects_blank_scope(graph):
    with pytest.raises(ValueError):
        await graph.purge_scope("")


async def test_an_edge_remembers_WHEN_and_both_adapters_answer_the_same():
    """`created_at` was written on every edge since the table existed and thrown away on the way
    out: the column is `NOT NULL DEFAULT now()`, and the dataclass had nowhere to put it.

    A contact-graph view needs it for a reason that is not decoration: **a wrong fact with no
    date is not correctable** — the operator cannot tell whether it is from yesterday or from
    March, so they cannot judge whether it still holds.

    Parity is asserted here and not left to the Postgres suite because the two adapters have
    drifted before (label normalisation): the in-memory one must stamp on insert the way the
    column's `DEFAULT now()` does.
    """
    from datetime import datetime, timezone

    from cogno_engram import AUDIENCE_STAFF, GraphEdge, InMemoryGraph

    g = InMemoryGraph()
    antes = datetime.now(timezone.utc)
    await g.upsert_edge(GraphEdge("s", "José", "Rex", "OWNS_PET"))
    lida = (await g.walk("s", "José", audience=AUDIENCE_STAFF))[0]

    assert lida.created_at is not None, "a aresta veio sem data — a coluna existe desde sempre"
    assert lida.created_at >= antes, "a data não é do momento da inserção"

    # …e uma data que o CHAMADOR trouxe é preservada: um replay ou um teste que fixa a data não
    # pode vê-la substituída pelo relógio, ou deixa de poder reproduzir o passado.
    fixa = datetime(2026, 3, 14, 9, 0, tzinfo=timezone.utc)
    await g.upsert_edge(GraphEdge("s", "José", "Ana", "KNOWS", created_at=fixa))
    ana = [e for e in await g.walk("s", "José", audience=AUDIENCE_STAFF) if e.target == "Ana"][0]
    assert ana.created_at == fixa, "a data trazida pelo chamador foi sobrescrita"
