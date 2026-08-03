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


async def test_reembed_memories_walks_past_one_page(store):
    """``batch`` is a PAGE SIZE, not a cap.

    It used to be a cap: one `limit=batch` read and done, so a scope with more rows than the
    batch was left half-migrated in the old vector space while the caller printed the truncated
    count as a success — the exact silent failure this module exists to prevent. Every fixture
    here had three rows, so nothing caught it. This one is deliberately larger than its page."""
    for i in range(25):
        await store.save_memory(MemoryRecord("s", "fact", f"memory number {i}"))
    n = await maintenance.reembed_memories(store, _Embedder(), "s", batch=10)
    assert n == 25
    assert all(m.embedding is not None for m in store._memories if m.scope == "s")


async def test_reembed_memories_keeps_the_old_vector_when_the_embedder_returns_nothing(store):
    """An empty vector is an embedder failure, not an instruction to blank the column.

    ``save_memory`` reads ``embedding=None`` as "leave/clear", so writing an empty result
    through turned a transient provider hiccup into permanent loss of the stored vector."""
    class _Broken:
        async def embed(self, text):
            return []

    await store.save_memory(MemoryRecord("s", "fact", "keep me", embedding=[9.0, 9.0]))
    n = await maintenance.reembed_memories(store, _Broken(), "s")
    assert n == 0
    [m] = [m for m in store._memories if m.scope == "s"]
    assert m.embedding == [9.0, 9.0]


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


# ── batch preference + knowledge-node re-embedding ────────────────────────

class _BatchEmbedder(_Embedder):
    """Records which path the caller took. Re-embedding is bulk by definition, so a
    batch-capable embedder must not be driven one item at a time."""

    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    async def embed(self, text):
        self.single_calls += 1
        return await super().embed(text)

    async def embed_batch(self, texts):
        self.batch_calls += 1
        return [[float(len(t)), 1.0] for t in texts]


async def test_reembed_memories_uses_the_batch_path(store):
    for c in ("alpha", "beta", "gamma"):
        await store.save_memory(MemoryRecord("s", "fact", c))
    emb = _BatchEmbedder()
    assert await maintenance.reembed_memories(store, emb, "s") == 3
    assert emb.batch_calls == 1 and emb.single_calls == 0     # one request, not three


async def test_reembed_memories_falls_back_without_batch(store):
    """The Embedder protocol does not require embed_batch — degrade, never crash."""
    await store.save_memory(MemoryRecord("s", "fact", "solo"))
    assert await maintenance.reembed_memories(store, _Embedder(), "s") == 1


async def test_reembed_knowledge_nodes(graph):
    await graph.upsert_node(GraphNode("s", "Ada Lovelace", "PERSON"))
    await graph.upsert_node(GraphNode("s", "analytical engine", "CONCEPT"))
    emb = _BatchEmbedder()
    assert await maintenance.reembed_knowledge_nodes(graph, emb, "s") == 2
    assert emb.batch_calls == 1
    for node in await graph.list_nodes("s"):
        assert node.embedding is not None


async def test_reembed_knowledge_nodes_walks_past_one_page(graph):
    """The graph half of the page-size-not-a-cap fix — see the memory counterpart."""
    for i in range(25):
        await graph.upsert_node(GraphNode("s", f"concept {i}", "CONCEPT"))
    assert await maintenance.reembed_knowledge_nodes(graph, _BatchEmbedder(), "s", batch=10) == 25
    assert all(n.embedding is not None for n in await graph.list_nodes("s", limit=100))


async def test_reembed_knowledge_nodes_is_idempotent(graph):
    """Keyed on (scope, label), so a second pass updates in place rather than duplicating."""
    await graph.upsert_node(GraphNode("s", "Ada Lovelace", "PERSON"))
    await maintenance.reembed_knowledge_nodes(graph, _Embedder(), "s")
    await maintenance.reembed_knowledge_nodes(graph, _Embedder(), "s")
    assert len(await graph.list_nodes("s")) == 1


async def test_reembed_knowledge_nodes_scope_isolation(graph):
    await graph.upsert_node(GraphNode("s", "mine", "CONCEPT"))
    await graph.upsert_node(GraphNode("other", "theirs", "CONCEPT"))
    assert await maintenance.reembed_knowledge_nodes(graph, _Embedder(), "s") == 1
    [untouched] = await graph.list_nodes("other")
    assert untouched.embedding is None                        # a switch never crosses scopes


async def test_reembed_empty_scope_is_a_noop(store, graph):
    assert await maintenance.reembed_memories(store, _Embedder(), "empty") == 0
    assert await maintenance.reembed_knowledge_nodes(graph, _Embedder(), "empty") == 0
