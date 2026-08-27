"""
cogno_engram.ports — the capability-scoped Protocols (hexagonal ports).

Three independent ports, each matching the data model that fits it:

  * ``MemoryStore``       — sessions / turns / memories + hybrid retrieval
                            (relational/document with optional vector search)
  * ``ConversationBuffer``— the sliding short-term window (KV-shaped)
  * ``KnowledgeGraph``    — nodes / edges / multi-hop walk (graph-shaped)

There is deliberately NO universal "any DB" store — forcing one interface over
relational + KV + graph collapses to a lowest-common-denominator that throws
away vector search and graph traversal. Each port is backed by the storage
engine that suits it; a host mixes adapters.

Every method takes an opaque ``scope`` and isolates strictly by it. ``scope``
must be non-empty — adapters reject blank scopes (a forgotten scope must fail
loud, never silently span tenants).
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import Optional, Protocol, runtime_checkable

from cogno_engram.types import (
    GraphEdge,
    GraphNode,
    GraphStats,
    HybridWeights,
    MemoryRecord,
    NodeContext,
    RetrievalQuery,
    Session,
    TurnRecord,
    TurnTrace,
)


@runtime_checkable
class MemoryStore(Protocol):
    """System-of-record for sessions, turns, and long-term memories."""

    # ── sessions ─────────────────────────────────────────────────────────
    async def create_session(self, scope: str) -> Session: ...
    # ``scope`` (optional) isolates the read to that scope — pass it whenever the caller knows the
    # scope. A session id is host-derived (a deterministic uuid5 of a caller string), so it can
    # COLLIDE across scopes; reading by id alone (no scope) let a tenant-A consolidation pick up
    # tenant-B's session/turns. Omitted → back-compat id-only read.
    async def get_session(self, session_id: str, *, scope: str = "") -> Optional[Session]: ...
    # Mark a session ended. With ``scope`` given, UPSERTS a closed row when none exists — a host
    # that persists turns WITHOUT a matching ``create_session`` (the common case: it only calls
    # ``save_turn``) closes a turn-derived session this way, so the janitor's idle scan won't
    # re-pick it. Without ``scope`` it only updates an existing row (back-compat).
    async def close_session(self, session_id: str, *, summary: str = "", scope: str = "") -> None: ...
    async def recent_sessions(self, scope: str, *, limit: int = 5) -> list[Session]: ...
    # Most recent still-open session within a TTL window (None if none/expired).
    async def get_active_session(self, scope: str, *,
                                 within_seconds: int = 12 * 3600) -> Optional[Session]: ...
    # Cross-scope idle scan for the host's janitor (Tier-3 consolidation): turn-derived sessions
    # (grouped from the ``turns`` table, since the host may never call ``create_session``) whose
    # LAST turn is older than ``idle_seconds`` and that are NOT already closed — oldest-idle first,
    # capped at ``limit``. Each carries its ``scope`` + ``id`` so the caller can consolidate + close.
    async def idle_sessions(self, *, idle_seconds: int = 1800,
                            limit: int = 100) -> list[Session]: ...

    # ── turns ────────────────────────────────────────────────────────────
    async def save_turn(self, turn: TurnRecord) -> None: ...
    async def update_turn_response(self, scope: str, session_id: str, turn_n: int,
                                   response: str) -> None: ...
    # ``scope`` (optional) isolates the read to that scope — see get_session on why id alone is
    # unsafe. hypnos consolidation MUST pass it (it reads turns then writes the extract back into
    # the caller's scope). Omitted → back-compat id-only read.
    async def load_turns(self, session_id: str, *, scope: str = "") -> list[TurnRecord]: ...
    async def recent_turns(self, scope: str, *, limit: int = 5,
                           exclude_session: str = "") -> list[TurnRecord]: ...
    async def set_feedback(self, scope: str, session_id: str, turn_n: int,
                           feedback: int) -> None: ...
    async def turn_count(self, session_id: str, *, scope: str = "") -> int: ...

    # ── turn traces (own table) ──────────────────────────────────────────
    # The per-turn pipeline trace (audit/inspector). Kept OUT of the flat ``turns`` row:
    # variable-shape, debug-oriented, opt-in. UPSERT by (scope, session_id, turn_n).
    async def save_turn_trace(self, trace: TurnTrace) -> None: ...
    async def traces_for_session(self, session_id: str, *, scope: str = "") -> list[TurnTrace]: ...
    # Admin / cross-scope reads: all turns at or under a scope SUBTREE (``scope_prefix`` itself or
    # any ``scope_prefix + "/" + …`` descendant), newest-first + a total for pagination. This is
    # the one place a query spans scopes — e.g. a tenant's whole chat history, where the host's
    # scope is ``tenant_id/identity`` (``make_scope``). NOT partition-pruned (the hash is over the
    # full scope), so it's an admin/maintenance read, not a hot path.
    async def admin_turns(self, scope_prefix: str, *, limit: int = 30,
                          offset: int = 0) -> "tuple[list[TurnRecord], int]": ...
    async def admin_scopes(self, scope_prefix: str) -> list[str]: ...   # distinct scopes w/ turns
    # The traces of a scope SUBTREE, newest-first + a total — the admin/offline read that
    # ``traces_for_session`` (one session) cannot serve: an audit over a tenant's whole
    # history, or a janitor pass over "everything since T". ``since`` is inclusive on
    # ``created_at``. Same subtree semantics and pagination shape as ``admin_turns``; same
    # caveat (not partition-pruned — a maintenance read, not a hot path).
    async def admin_traces(self, scope_prefix: str, *, since: Optional[datetime] = None,
                           limit: int = 1000, offset: int = 0) -> "tuple[list[TurnTrace], int]": ...

    # ── memories ─────────────────────────────────────────────────────────
    async def save_memory(self, memory: MemoryRecord) -> None: ...   # upsert
    async def load_memories(self, scope: str, *, query: Optional[RetrievalQuery] = None,
                            limit: int = 50,
                            weights: Optional[HybridWeights] = None) -> list[MemoryRecord]: ...
    async def adjust_feedback_score(self, scope: str, query_text: str, delta: float,
                                    *, limit: int = 10) -> int: ...
    # Bulk walk for maintenance (re-embedding), NOT retrieval: ``load_memories`` is a ranked
    # hybrid search whose order depends on the query, so paging it would revisit and skip rows.
    # Keyset on the id — stable while the table is being written, which OFFSET is not.
    async def scan_memories(self, scope: str, *, after_id: Optional[str] = None,
                            limit: int = 1000) -> list[MemoryRecord]: ...
    async def memory_count(self, scope: str) -> int: ...
    async def delete_memories(self, scope: str, *, older_than: Optional[datetime] = None,
                              category: Optional[str] = None,
                              max_confidence: Optional[float] = None) -> int: ...
    # Right-to-be-forgotten: hard-delete EVERY row this store owns for a scope — sessions, turns,
    # turn_traces, and memories. For when the host offboards an identity/scope entirely (a
    # deleted contact must not leave stale recall behind that re-pollutes a later re-onboarding).
    # Returns the total rows removed. Pair with ``KnowledgeGraph.purge_scope`` for the graph half.
    async def purge_scope(self, scope: str) -> int: ...

    # ── concurrency primitive (host decides when to hold it) ─────────────
    def session_lock(self, scope: str, session_id: str) -> AbstractAsyncContextManager[None]: ...


@runtime_checkable
class SupportsVectorSearch(Protocol):
    """Optional capability — a MemoryStore that can score by embedding.

    Separate from ``MemoryStore`` (like ``ToolCallingBackend`` is separate from
    ``LLMBackend`` in cogno-anima) so a store without vectors (SQLite, Mongo
    without Atlas) degrades retrieval to lexical/chronological instead of
    breaking.
    """
    def supports_vector(self) -> bool: ...


@runtime_checkable
class ConversationBuffer(Protocol):
    """The short-term sliding window (KV-shaped; TTL is an adapter concern)."""

    async def push(self, scope: str, session_id: str, turn: TurnRecord) -> None: ...
    async def window(self, scope: str, session_id: str, *, size: int = 10) -> list[TurnRecord]: ...
    async def clear(self, scope: str, session_id: str) -> None: ...


@runtime_checkable
class KnowledgeGraph(Protocol):
    """Associative memory: typed nodes + directed, confidence-weighted edges.

    **Every read that can return contact data takes ``audience`` as a REQUIRED keyword.** It is
    required, not defaulted, and that is the whole design: "tenant sees everything; an identity
    sees only its own life; they do not mix". With an optional argument, forgetting it returns
    EVERYTHING — the failure would be silent and would be a leak. Required, forgetting it is a
    `TypeError` at the call, which is a test that writes itself.

    Pass ``AUDIENCE_STAFF`` for a tenant/staff read and ``audience_for(identity_id)`` for a
    read inside one contact's turn. Nobody hand-writes the string.

    The filter lives on the EDGE (``knowledge_nodes`` is unique on
    ``(scope, lower(label), node_type)``, so the node "Maria" is one row for the whole tenant —
    there is no "José's Maria" to mark). Node visibility is DERIVED: a node is visible when some
    edge this reader may see touches it; an orphan node is staff-only.
    """

    async def upsert_node(self, node: GraphNode) -> int: ...
    async def upsert_edge(self, edge: GraphEdge) -> None: ...
    async def find_node(self, scope: str, label: str, *,
                        audience: str) -> Optional[GraphNode]: ...
    # ``related_only`` restricts the candidates to nodes that participate in at least one edge.
    # A PARAMETER, and never a default, because THE TWO CALLERS WANT OPPOSITE THINGS. One walks
    # from these nodes and wants candidates it can walk from; the other is a boot schema probe
    # that exists to check the query still EXECUTES, and must therefore run the SIMPLEST form
    # of it — a filtering default would silently make it exercise a join against
    # ``knowledge_edges`` that it never asked for. Add the readers that legitimately want
    # isolated nodes (a staff search, the dashboard's node list) and a default is a silent
    # behaviour change for every deployment that upgrades. A default decides for all of them;
    # a parameter costs the one caller that wants it a single keyword.
    async def find_nodes_by_embedding(self, scope: str, embedding: list[float],
                                      *, audience: str, limit: int = 5,
                                      related_only: bool = False) -> list[GraphNode]: ...
    # Returns ACCEPTED edges only, and deliberately has no flag to say otherwise: a walk feeds
    # the prompt, and "show me the unreviewed ones too" is a curation question, not a retrieval
    # one. A keyword that could turn the filter off is a keyword someone eventually passes.
    async def walk(self, scope: str, start_label: str, *, audience: str,
                   max_depth: int = 2) -> list[GraphEdge]: ...
    # Curation (see ``types.VALID_EDGE_STATUS``): what is waiting for a human, and the verdict.
    async def pending_edges(self, scope: str, *, audience: str,
                            limit: int = 100) -> list[GraphEdge]: ...
    async def set_edge_status(self, scope: str, source: str, target: str, relation: str,
                              status: str) -> bool: ...
    # The way BACK. `upsert_edge` is narrow-never-widen, so re-writing cannot undo a
    # classification — and `classify_edge_audience` promotes `''` to `tenant`, the widest step
    # there is. Without an explicit setter that migration would be irreversible.
    async def set_edge_audience(self, scope: str, source: str, target: str, relation: str,
                                audience: str) -> bool: ...

    async def neighbors(self, scope: str, label: str, *,
                        audience: str) -> list[GraphNode]: ...
    async def get_node_context(self, scope: str, label: str, *,
                               audience: str) -> Optional[NodeContext]: ...
    async def list_nodes(self, scope: str, *, audience: str,
                         node_type: Optional[str] = None,
                         limit: int = 100) -> list[GraphNode]: ...
    # How many nodes carry a label, as a QUERY. `list_nodes` is a page (`ORDER BY id LIMIT n`,
    # no label filter, no offset), so a caller asking "is this label unique?" over it gets the
    # right answer only while the tenant stays smaller than the page — and a homonym created
    # past the cut is simply invisible. That is a live defect the host had to work around by
    # refusing to answer whenever the page came back full.
    async def count_nodes(self, scope: str, *, audience: str,
                          label: Optional[str] = None) -> int: ...
    # The whole dashboard summary in ONE aggregated read per shape, because there was no way
    # to ask "how connected is each node" in bulk. The caller (the host's `knowledge_stats`)
    # was rebuilding it a node at a time — `list_nodes` then `get_node_context` per node, and
    # that helper is itself `find_node` + `walk` + `neighbors`, so the real cost was `1 + 3N`:
    # 1165 queries for the 388 nodes of the live box, on every page open, growing with the graph.
    # Visibility follows the one-at-a-time version EXACTLY: nodes by the audience rule (DERIVED
    # for a non-staff reader), degree over DISTINCT accepted edges the audience may see — an
    # unreviewed edge is not walkable, and the old code counted what `walk` returned.
    async def graph_stats(self, scope: str, *, audience: str,
                          top: int = 5) -> GraphStats: ...
    # The graph half of the maintenance walk — see ``MemoryStore.scan_memories``.
    async def scan_nodes(self, scope: str, *, audience: str,
                         after_id: Optional[int] = None,
                         limit: int = 1000) -> list[GraphNode]: ...
    # ORPHAN PREDICATE — every edge, ANY audience, any status, deliberately. It takes no
    # `audience` and that is the point: the caller is `prune_orphan_nodes`, which DELETES, and
    # a filtered read there does not narrow what is seen, it widens what is destroyed — a node
    # whose edges all belong to another contact would look unattached and be removed. The
    # question "does anything point at this node" has no audience.
    async def has_edges(self, scope: str, label: str) -> bool: ...

    async def delete_node(self, scope: str, label: str) -> bool: ...
    # Feedback-driven pruning: drop every edge a (disliked) session asserted. Returns the rows
    # removed, and RAISES ``ValueError`` on a blank ``session_id``: a blank is not a wildcard —
    # it matches every edge whose ``source_session`` is empty, which is exactly the class
    # nothing automated writes (the notes a human or an admin API put there). An adapter that
    # skips this check keeps that hazard verbatim.
    async def delete_edges_by_session(self, scope: str, session_id: str) -> int: ...
    # Right-to-be-forgotten: hard-delete EVERY node + edge for a scope (the graph half of the
    # host's scope offboarding; pair with ``MemoryStore.purge_scope``). Returns rows removed.
    async def purge_scope(self, scope: str) -> int: ...
