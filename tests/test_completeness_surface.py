"""Unit tests for the parent-parity surface added for completeness."""

from cogno_engram import format_graph_context, ingest_entities
from cogno_engram.adapters.in_memory import InMemoryStore
from cogno_engram.types import AUDIENCE_STAFF, GraphEdge, GraphNode, MemoryRecord, TurnRecord


# ── MemoryStore additions ─────────────────────────────────────────────────

async def test_get_active_session_within_and_expired(store):
    s = await store.create_session("scope")
    assert (await store.get_active_session("scope")).id == s.id
    # a closed session is not active
    await store.close_session(s.id)
    assert await store.get_active_session("scope") is None


async def test_get_active_session_respects_ttl():
    store = InMemoryStore()
    await store.create_session("scope")
    assert await store.get_active_session("scope", within_seconds=3600) is not None
    assert await store.get_active_session("scope", within_seconds=0) is None


async def test_update_turn_response(store):
    s = await store.create_session("scope")
    await store.save_turn(TurnRecord(s.id, "scope", 1, "oi"))
    await store.update_turn_response("scope", s.id, 1, "olá!")
    assert (await store.load_turns(s.id))[0].response == "olá!"


async def test_turn_count_and_memory_count(store):
    s = await store.create_session("scope")
    await store.save_turn(TurnRecord(s.id, "scope", 1, "a"))
    await store.save_turn(TurnRecord(s.id, "scope", 2, "b"))
    await store.save_memory(MemoryRecord("scope", "fact", "x"))
    assert await store.turn_count(s.id) == 2
    assert await store.memory_count("scope") == 1
    assert await store.memory_count("other") == 0


# ── KnowledgeGraph additions ──────────────────────────────────────────────

async def _jose_rex(graph):
    await graph.upsert_node(GraphNode("s", "José", "PERSON"))
    await graph.upsert_node(GraphNode("s", "Rex", "ANIMAL"))
    await graph.upsert_edge(GraphEdge("s", "José", "Rex", "OWNS"))


async def test_get_node_context(graph):
    await _jose_rex(graph)
    ctx = await graph.get_node_context("s", "José", audience=AUDIENCE_STAFF)
    assert ctx is not None
    assert ctx.node.label == "José"
    assert [e.relation for e in ctx.edges] == ["OWNS"]
    assert {n.label for n in ctx.neighbors} == {"Rex"}
    assert await graph.get_node_context("s", "missing", audience=AUDIENCE_STAFF) is None


async def test_list_nodes_with_type_filter(graph):
    await _jose_rex(graph)
    assert {n.label for n in await graph.list_nodes("s", audience=AUDIENCE_STAFF)} == {"José", "Rex"}
    assert [n.label for n in await graph.list_nodes("s", node_type="ANIMAL", audience=AUDIENCE_STAFF)] == ["Rex"]


async def test_delete_node_cascades_edges(graph):
    await _jose_rex(graph)
    assert await graph.delete_node("s", "Rex") is True
    assert await graph.find_node("s", "Rex", audience=AUDIENCE_STAFF) is None
    assert await graph.walk("s", "José", max_depth=2, audience=AUDIENCE_STAFF) == []     # edge cascaded
    assert await graph.delete_node("s", "Rex") is False         # already gone


# ── graph_context helpers ─────────────────────────────────────────────────

async def test_ingest_entities_creates_nodes(graph):
    n = await ingest_entities(graph, "s", ["Maria", ("Acme", "ORG"), ("", "PERSON"), "bad-type-skipped"])
    assert n == 3      # the empty label is skipped
    assert (await graph.find_node("s", "Acme", audience=AUDIENCE_STAFF)).node_type == "ORG"
    assert (await graph.find_node("s", "Maria", audience=AUDIENCE_STAFF)).node_type == "CONCEPT"   # default


async def test_ingest_entities_normalizes_unknown_type(graph):
    await ingest_entities(graph, "s", [("Thing", "WIDGET")])
    assert (await graph.find_node("s", "Thing", audience=AUDIENCE_STAFF)).node_type == "CONCEPT"


def test_format_graph_context_block():
    edges = [GraphEdge("s", "José", "Rex", "OWNS"), GraphEdge("s", "Rex", "Pastor", "BREED")]
    out = format_graph_context(edges)
    assert out.startswith("[Knowledge Graph]")
    assert "- José --[OWNS]--> Rex" in out and "- Rex --[BREED]--> Pastor" in out


def test_format_graph_context_budget_and_empty():
    assert format_graph_context([]) == ""
    edges = [GraphEdge("s", f"A{i}", f"B{i}", "R") for i in range(100)]
    out = format_graph_context(edges, max_chars=60)
    assert len(out) <= 60 + 40   # roughly bounded; not all 100 lines included
    assert out.count("\n") < 100
