"""Unit tests for sleep-time maintenance ops (in-memory adapters, no model)."""

from datetime import datetime, timedelta, timezone

from cogno_engram import maintenance
from cogno_engram.types import GraphEdge, GraphNode, MemoryRecord

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def _old(content, *, age_days, conf=0.7):
    return MemoryRecord("s", "fact", content, confidence=conf,
                        created_at=NOW - timedelta(days=age_days))


# ── prune_memories ────────────────────────────────────────────────────────

async def test_prune_memories_by_age(store):
    await store.save_memory(_old("ancient", age_days=400))
    await store.save_memory(_old("recent", age_days=1))
    n = await maintenance.prune_memories(store, "s", older_than=timedelta(days=180), now=NOW)
    assert n == 1
    assert [m.content for m in await store.load_memories("s")] == ["recent"]


async def test_prune_memories_keeps_high_confidence(store):
    await store.save_memory(_old("low-old", age_days=400, conf=0.6))
    await store.save_memory(_old("fact-old", age_days=400, conf=1.0))   # durable fact
    n = await maintenance.prune_memories(store, "s", older_than=timedelta(days=180),
                                         max_confidence=0.75, now=NOW)
    assert n == 1
    assert [m.content for m in await store.load_memories("s")] == ["fact-old"]


async def test_prune_memories_scope_isolation(store):
    await store.save_memory(_old("a", age_days=400))
    await store.save_memory(MemoryRecord("other", "fact", "b", created_at=NOW - timedelta(days=400)))
    await maintenance.prune_memories(store, "s", older_than=timedelta(days=180), now=NOW)
    assert await store.memory_count("other") == 1   # untouched


# ── reembed_memories ──────────────────────────────────────────────────────

class _Embedder:
    async def embed(self, text):
        return [float(len(text)), 1.0]


async def test_reembed_memories(store):
    await store.save_memory(MemoryRecord("s", "fact", "no embedding yet"))
    n = await maintenance.reembed_memories(store, _Embedder(), "s")
    assert n == 1
    # the upsert preserved the single memory and set an embedding
    [m] = [m for m in store._memories if m.scope == "s"]   # in-memory introspection
    assert m.embedding is not None


# ── prune_orphan_nodes ────────────────────────────────────────────────────

async def test_prune_orphan_nodes(graph):
    await graph.upsert_node(GraphNode("s", "Connected", "CONCEPT"))
    await graph.upsert_node(GraphNode("s", "Other", "CONCEPT"))
    await graph.upsert_edge(GraphEdge("s", "Connected", "Other", "REL"))
    await graph.upsert_node(GraphNode("s", "Orphan", "CONCEPT"))     # no edges
    n = await maintenance.prune_orphan_nodes(graph, "s")
    assert n == 1
    assert await graph.find_node("s", "Orphan") is None
    assert await graph.find_node("s", "Connected") is not None
