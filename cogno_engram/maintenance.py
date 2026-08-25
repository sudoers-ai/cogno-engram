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

from cogno_engram.types import (
    AUDIENCE_STAFF,
    AUDIENCE_TENANT,
    audience_for,
    sanitize_audience,
)
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
        # STAFF: a maintenance walk re-embeds the whole scope, and a contact-scoped read
        # would silently skip every node whose edges belong to someone else.
        nodes = await kg.scan_nodes(scope, audience=AUDIENCE_STAFF,
                                    after_id=cursor, limit=batch)
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
    # `has_edges`, not `neighbors`, and the difference is the whole reason it exists: this
    # branch DELETES. A filtered read here would not narrow what is seen, it would widen what
    # is destroyed — a node whose edges all belong to another contact would look unattached.
    # The orphan question has no audience, so the predicate takes none, and STAFF never appears
    # in a write decision.
    for node in await kg.list_nodes(scope, audience=AUDIENCE_STAFF, limit=limit):
        if not await kg.has_edges(scope, node.label):
            if await kg.delete_node(scope, node.label):
                deleted += 1
    logger.info("stage=maintenance event=prune_orphan_nodes scope=%s removed=%d", scope, deleted)
    return deleted

async def classify_edge_audience(
    kg: KnowledgeGraph,
    scope: str,
    *,
    identity_of_session: "Optional[Any]" = None,
    dry_run: bool = True,
) -> "dict[str, int]":
    """Give every unclassified edge an audience. Returns the counts, by outcome.

    Edges written before the column existed carry ``''`` — staff sees them, no contact does. So
    a deployment that upgrades and stops there loses the contact's relation block until this
    runs. It is a deliberate act on live data, and it is safe to run twice: only ``''`` rows are
    touched, so a second pass finds nothing.

    The rule is the one the writers now follow, applied backwards:

    * ``source_session`` **empty** → nothing automated wrote it (``ingest_entities`` only makes
      NODES), so it came from staff, an admin API or a KB import → ``AUDIENCE_TENANT``;
    * ``source_session`` **set** → Hypnos extracted it from one contact's conversation → that
      contact's own life.

    Mapping a session to an identity is the HOST's knowledge, not engram's, so it arrives as
    ``identity_of_session(session_id) -> str`` (return ``""`` when it cannot be resolved). With
    no resolver, stamped edges are left ``''`` rather than guessed at — staff keeps seeing them
    and no contact does, which is the safe direction and an honest "unknown" instead of a wrong
    owner.
    """
    seen: set[tuple[str, str, str]] = set()
    counts = {"tenant": 0, "identity": 0, "unresolved": 0}
    # `walk` returns ACCEPTED only, so a proposal written before the column would stay `''` and,
    # once a human accepted it, be invisible to the very contact it is about. `pending_edges`
    # closes that. REJECTED edges are deliberately left alone: nothing ever speaks them, and
    # classifying a verdict nobody reads is work with no reader.
    edges = list(await kg.pending_edges(scope, audience=AUDIENCE_STAFF, limit=100_000))
    for node in await kg.list_nodes(scope, audience=AUDIENCE_STAFF, limit=100_000):
        edges.extend(await kg.walk(scope, node.label, audience=AUDIENCE_STAFF, max_depth=1))
    for e in edges:
        key = (e.source, e.target, e.relation)
        if key in seen or sanitize_audience(e.audience):
            continue                     # a walk reaches an edge from both endpoints
        seen.add(key)
        stamp = (e.source_session or "").strip()
        if not stamp:
            want = AUDIENCE_TENANT
            counts["tenant"] += 1
        else:
            who = str(identity_of_session(stamp) or "") if identity_of_session else ""
            want = audience_for(who) if who else ""
            counts["identity" if want else "unresolved"] += 1
        if want and not dry_run:
            # `set_edge_audience`, not `upsert_edge`: the latter is narrow-never-widen, so
            # it cannot write `tenant` over `''` — and that promotion is the widest step
            # this function takes. The explicit setter is also what makes the migration
            # REVERSIBLE, which is the property the deploy note rests on.
            await kg.set_edge_audience(scope, e.source, e.target, e.relation, want)
    logger.info("stage=maintenance event=classify_audience scope=%s dry_run=%s %s",
                scope, dry_run, counts)
    return counts
