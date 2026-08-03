"""
cogno_engram.maintenance — sleep-time upkeep over the substrate (host schedules).

The same contract as ``hypnos``: engram provides the *operations*, the host runs
the loop (the "janitor"). These keep the substrate bounded and consistent over
time — pruning stale memories, re-embedding after a model change, and dropping
orphaned graph nodes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from cogno_engram.ports import KnowledgeGraph, MemoryStore
from cogno_engram.types import MemoryRecord

logger = logging.getLogger("cogno_engram.maintenance")


async def prune_memories(
    store: MemoryStore,
    scope: str,
    *,
    older_than: timedelta,
    max_confidence: Optional[float] = None,
    category: Optional[str] = None,
    now: Optional[datetime] = None,
) -> int:
    """Delete memories older than ``older_than`` (optionally only low-confidence /
    a single category). Returns the number deleted.

    A sensible default for bounding growth: ``prune_memories(store, scope,
    older_than=timedelta(days=180), max_confidence=0.75)`` keeps durable facts
    (confidence ~1.0) while clearing stale, low-value inferences.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - older_than
    deleted = await store.delete_memories(scope, older_than=cutoff, category=category,
                                          max_confidence=max_confidence)
    logger.info("stage=maintenance event=prune_memories scope=%s removed=%d category=%s",
                scope, deleted, category or "*")
    return deleted


async def reembed_memories(
    store: MemoryStore,
    embedder: Any,
    scope: str,
    *,
    batch: int = 1000,
) -> int:
    """Recompute + upsert embeddings for EVERY memory in a scope. Returns the number re-embedded.

    ``embedder`` is a duck-typed cogno-anima ``Embedder``. Idempotent: re-saving a
    memory upserts on ``(scope, category, content)`` and updates the embedding.

    ``batch`` is the page size, **not a cap**: this walks until the scope is exhausted. It used
    to be a cap — one `limit=batch` read and done — so a scope with more than 1000 memories was
    silently left half-migrated in the OLD vector space while the caller printed the truncated
    count as a success. That is the exact failure this module exists to prevent, and no test
    caught it because every fixture had three rows.

    Residual, stated rather than hidden: the walk is keyset-ordered on ``id``, which is stable
    for rows that already exist but cannot see a row INSERTED behind the cursor mid-walk (the
    ids are random uuids, not monotonic). Re-embedding is a maintenance step run against a
    quiescent store; a row written during the walk belongs to the writer's model generation.
    """
    count, cursor = 0, None
    while True:
        mems = await store.scan_memories(scope, after_id=cursor, limit=batch)
        if not mems:
            break
        vectors = await _embed_all(embedder, [m.content for m in mems])
        for m, vec in zip(mems, vectors):
            # An empty vector means the embedder failed for this text. Writing it through would
            # blank the stored embedding (``save_memory`` treats None as "clear"), turning a
            # transient provider hiccup into permanent data loss — leave the old vector alone.
            if not vec:
                logger.warning("stage=maintenance event=reembed_skip_empty scope=%s id=%s",
                               scope, m.id)
                continue
            await store.save_memory(MemoryRecord(scope, m.category, m.content,
                                                 confidence=m.confidence, embedding=vec))
            count += 1
        cursor = mems[-1].id
        if cursor is None or len(mems) < batch:
            break
    logger.info("stage=maintenance event=reembed scope=%s reprocessed=%d", scope, count)
    return count


async def reembed_knowledge_nodes(
    kg: KnowledgeGraph,
    embedder: Any,
    scope: str,
    *,
    batch: int = 1000,
) -> int:
    """Recompute + upsert embeddings for a scope's graph nodes. Returns the number re-embedded.

    The counterpart to :func:`reembed_memories`. Both must run after an embedding-model
    switch: ``knowledge_nodes.embedding`` feeds ``find_nodes_by_embedding``, so a node left
    in the OLD vector space is silently unreachable by semantic lookup — no error, just a
    graph that stops answering. Re-embedding one store and not the other leaves the system
    half-migrated in a way nothing reports.

    Nodes are keyed by ``(scope, label)`` and the embedding is derived from the label, so
    ``upsert_node`` updates in place — the operation is idempotent.

    ``batch`` is the page size, **not a cap** — see :func:`reembed_memories` for why that
    distinction cost a silent half-migration. Node ids are monotonic, so this walk has none of
    the concurrent-insert residual the memory walk carries.
    """
    count, cursor = 0, None
    while True:
        nodes = await kg.scan_nodes(scope, after_id=cursor, limit=batch)
        if not nodes:
            break
        vectors = await _embed_all(embedder, [n.label for n in nodes])
        for node, vec in zip(nodes, vectors):
            if not vec:      # never blank a stored vector over a transient embedder failure
                logger.warning("stage=maintenance event=reembed_nodes_skip_empty scope=%s id=%s",
                               scope, node.id)
                continue
            node.embedding = vec
            await kg.upsert_node(node)
            count += 1
        cursor = nodes[-1].id
        if cursor is None or len(nodes) < batch:
            break
    logger.info("stage=maintenance event=reembed_nodes scope=%s reprocessed=%d", scope, count)
    return count


async def _embed_all(embedder: Any, texts: "list[str]") -> "list[list[float]]":
    """Embed a list, preferring the embedder's batch path.

    Re-embedding is the bulk operation by definition — one request beats N round trips, and
    against a metered cloud provider the difference is latency AND rate-limit headroom. Falls
    back to sequential ``embed`` for an embedder that offers no batch (duck-typed: the
    ``Embedder`` protocol does not require one).
    """
    batch_fn = getattr(embedder, "embed_batch", None)
    if batch_fn is not None:
        return list(await batch_fn(texts))
    return [await embedder.embed(t) for t in texts]


async def prune_orphan_nodes(kg: KnowledgeGraph, scope: str, *, limit: int = 1000) -> int:
    """Delete knowledge nodes with no incident edges. Returns the number deleted.

    Composed from the port (``list_nodes`` + ``neighbors`` + ``delete_node``), so
    it works on any adapter.
    """
    deleted = 0
    for node in await kg.list_nodes(scope, limit=limit):
        if not await kg.neighbors(scope, node.label):
            if await kg.delete_node(scope, node.label):
                deleted += 1
    logger.info("stage=maintenance event=prune_orphan_nodes scope=%s removed=%d", scope, deleted)
    return deleted
