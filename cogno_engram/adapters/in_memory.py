"""
cogno_engram.adapters.in_memory — zero-dependency reference adapters.

Pure-Python implementations of all three ports, for tests, local dev, and as the
executable proof that the Protocols are honest. They implement the same hybrid
retrieval fusion and multi-hop graph walk as the Postgres adapter, just over
in-process structures (no persistence, single-process).
"""

from __future__ import annotations

import asyncio
import math
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional
from uuid import uuid4

from dataclasses import replace


from cogno_engram.types import (
    AUDIENCE_STAFF,
    audience_can_read,
    sanitize_audience,
    EDGE_ACCEPTED,
    EDGE_PROPOSED,
    require_edge_status,
    GraphEdge,
    GraphNode,
    HybridWeights,
    MemoryRecord,
    NodeContext,
    RetrievalQuery,
    Session,
    TurnRecord,
    TurnTrace,
)


def _detached(edge: "GraphEdge") -> "GraphEdge":
    """A caller-safe copy of a stored edge.

    Every read used to hand back the STORED object, so a caller that touched what it was given
    changed what the prompt says — and Postgres, which builds fresh rows, did not. A review
    measured `walk(...)[0].attributes["note"] = "LEAKED"` rendering into the in-memory block and
    not into the Postgres one: same code, two prompts, on the one invariant this module is.

    `dataclasses.replace` alone is not enough: it is SHALLOW, so the copy shares its
    `attributes` dict with the store and a mutation of that dict still lands. The dict is
    rebuilt here for the same reason the object is.
    """
    return replace(edge, attributes=dict(edge.attributes or {}))



def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require_scope(scope: str) -> str:
    if not scope or not scope.strip():
        raise ValueError("scope must be a non-empty string (engram isolates every row by scope)")
    return scope

def _require_session(session_id: str) -> str:
    """A session id that is blank is NOT a wildcard, and the prune must not treat it as one.

    ``delete_edges_by_session(scope, "")`` matches every edge whose ``source_session`` is empty
    — which is precisely the class nothing automated writes: the notes a HUMAN or an admin API
    put there. One disliked turn arriving with a blank id would erase them all, silently, and a
    `DELETE ... WHERE source_session = ''` looks entirely ordinary in a log.

    Refusing is right rather than returning 0: an empty id here is a caller bug (a missing
    session on the feedback path), and swallowing it hides the bug while pretending the prune
    ran.
    """
    if not (session_id or "").strip():
        raise ValueError("session_id must be non-empty: a blank id is not a wildcard")
    return session_id


def _cosine(a: Optional[list[float]], b: Optional[list[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _lexical(query: str, content: str) -> float:
    """A tiny BM25 stand-in: fraction of query terms present in the content."""
    q = set(query.lower().split())
    if not q:
        return 0.0
    c = set(content.lower().split())
    return len(q & c) / len(q)


class InMemoryStore:
    """Reference ``MemoryStore`` + ``SupportsVectorSearch``."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._turns: list[TurnRecord] = []
        self._traces: list[TurnTrace] = []
        self._memories: list[MemoryRecord] = []
        self._locks: dict[str, asyncio.Lock] = {}

    # ── sessions ─────────────────────────────────────────────────────────
    async def create_session(self, scope: str) -> Session:
        _require_scope(scope)
        session = Session(id=str(uuid4()), scope=scope, started_at=_now())
        self._sessions[session.id] = session
        return session

    async def get_session(self, session_id: str, *, scope: str = "") -> Optional[Session]:
        s = self._sessions.get(session_id)
        if s is not None and scope and s.scope != scope:
            return None      # id collides across scopes → isolate to the requested scope
        return s

    async def close_session(self, session_id: str, *, summary: str = "", scope: str = "") -> None:
        session = self._sessions.get(session_id)
        if session is not None and scope and session.scope != scope:
            return          # a colliding id owned by ANOTHER scope — never write its summary
        if session is not None:
            session.ended_at = _now()
            session.summary = summary
        elif scope:
            # UPSERT a closed row for a turn-derived session that never had a create_session
            # (host-persisted turns) — so the janitor's idle scan won't re-pick it.
            self._sessions[session_id] = Session(
                id=session_id, scope=scope, started_at=_now(), ended_at=_now(), summary=summary)

    async def idle_sessions(self, *, idle_seconds: int = 1800,
                            limit: int = 100) -> list[Session]:
        cutoff = _now() - timedelta(seconds=idle_seconds)
        # last activity per session_id, derived from turns (the host may never create_session)
        last: dict[str, tuple[str, "datetime", "datetime"]] = {}   # sid → (scope, first, last)
        for t in self._turns:
            ts = t.created_at or _now()
            cur = last.get(t.session_id)
            if cur is None:
                last[t.session_id] = (t.scope, ts, ts)
            else:
                sc, first, lst = cur
                last[t.session_id] = (sc, min(first, ts), max(lst, ts))
        out = []
        for sid, (scope, first, lst) in last.items():
            if lst >= cutoff:
                continue                                   # still recently active
            sess = self._sessions.get(sid)
            # Closed is not the same as FINISHED: a host whose session_id is derived from
            # (tenant, channel, sender) reuses one session per contact forever, so turns keep
            # arriving after a close. Skip only when nothing has happened since — see the
            # Postgres adapter for the measured effect of getting this wrong.
            if sess is not None and sess.ended_at is not None and lst <= sess.ended_at:
                continue
            out.append(Session(id=sid, scope=scope, started_at=first))
        out.sort(key=lambda s: s.started_at)               # oldest-idle first
        return out[:limit]

    async def recent_sessions(self, scope: str, *, limit: int = 5) -> list[Session]:
        _require_scope(scope)
        sessions = [s for s in self._sessions.values() if s.scope == scope]
        sessions.sort(key=lambda s: s.started_at, reverse=True)
        return sessions[:limit]

    async def get_active_session(self, scope: str, *,
                                 within_seconds: int = 12 * 3600) -> Optional[Session]:
        _require_scope(scope)
        cutoff = _now() - timedelta(seconds=within_seconds)
        candidates = [s for s in self._sessions.values()
                      if s.scope == scope and s.ended_at is None and s.started_at >= cutoff]
        candidates.sort(key=lambda s: s.started_at, reverse=True)
        return candidates[0] if candidates else None

    # ── turns ────────────────────────────────────────────────────────────
    async def save_turn(self, turn: TurnRecord) -> None:
        _require_scope(turn.scope)
        # Match the Postgres adapter's ON CONFLICT (scope, session_id, turn_n) DO NOTHING: a
        # retried/idempotent re-save of the same turn coordinate is a no-op, not a duplicate row
        # (the in-memory adapter otherwise diverged, double-counting turns in local/test runs).
        if any(t.scope == turn.scope and t.session_id == turn.session_id
               and t.turn_n == turn.turn_n for t in self._turns):
            return
        if turn.created_at is None:
            turn.created_at = _now()
        self._turns.append(turn)

    async def update_turn_response(self, scope: str, session_id: str, turn_n: int,
                                   response: str) -> None:
        _require_scope(scope)
        for turn in self._turns:
            if turn.scope == scope and turn.session_id == session_id and turn.turn_n == turn_n:
                turn.response = response

    async def load_turns(self, session_id: str, *, scope: str = "") -> list[TurnRecord]:
        turns = [t for t in self._turns if t.session_id == session_id
                 and (not scope or t.scope == scope)]
        turns.sort(key=lambda t: t.turn_n)
        return turns

    async def turn_count(self, session_id: str, *, scope: str = "") -> int:
        return sum(1 for t in self._turns if t.session_id == session_id
                   and (not scope or t.scope == scope))

    # ── turn traces (own table) ──────────────────────────────────────────
    async def save_turn_trace(self, trace: TurnTrace) -> None:
        _require_scope(trace.scope)
        if trace.created_at is None:
            trace.created_at = _now()
        # UPSERT by (scope, session_id, turn_n).
        self._traces = [t for t in self._traces
                        if not (t.scope == trace.scope and t.session_id == trace.session_id
                                and t.turn_n == trace.turn_n)]
        self._traces.append(trace)

    async def traces_for_session(self, session_id: str, *, scope: str = "") -> list[TurnTrace]:
        traces = [t for t in self._traces if t.session_id == session_id
                  and (not scope or t.scope == scope)]
        traces.sort(key=lambda t: t.turn_n)
        return traces

    async def recent_turns(self, scope: str, *, limit: int = 5,
                           exclude_session: str = "") -> list[TurnRecord]:
        _require_scope(scope)
        turns = [t for t in self._turns if t.scope == scope and t.session_id != exclude_session]
        turns.sort(key=lambda t: (t.created_at or _now()), reverse=True)
        return turns[:limit]

    async def set_feedback(self, scope: str, session_id: str, turn_n: int, feedback: int) -> None:
        _require_scope(scope)
        for turn in self._turns:
            if turn.scope == scope and turn.session_id == session_id and turn.turn_n == turn_n:
                turn.feedback = feedback

    @staticmethod
    def _under(scope: str, prefix: str) -> bool:
        # the scope IS the prefix, or a descendant ``prefix/…`` (subtree match)
        return scope == prefix or scope.startswith(prefix + "/")

    async def admin_turns(self, scope_prefix: str, *, limit: int = 30,
                          offset: int = 0) -> "tuple[list[TurnRecord], int]":
        _require_scope(scope_prefix)
        turns = [t for t in self._turns if self._under(t.scope, scope_prefix)]
        turns.sort(key=lambda t: (t.created_at or _now()), reverse=True)
        return turns[offset:offset + limit], len(turns)

    async def admin_scopes(self, scope_prefix: str) -> list[str]:
        _require_scope(scope_prefix)
        return sorted({t.scope for t in self._turns if self._under(t.scope, scope_prefix)})

    async def admin_traces(self, scope_prefix: str, *, since: Optional[datetime] = None,
                           limit: int = 1000, offset: int = 0) -> "tuple[list[TurnTrace], int]":
        _require_scope(scope_prefix)
        rows = [t for t in self._traces if self._under(t.scope, scope_prefix)
                and (since is None or (t.created_at or _now()) >= since)]
        rows.sort(key=lambda t: (t.created_at or _now(), t.session_id, t.turn_n), reverse=True)
        return rows[offset:offset + limit], len(rows)

    # ── memories ─────────────────────────────────────────────────────────
    async def save_memory(self, memory: MemoryRecord) -> None:
        _require_scope(memory.scope)
        for existing in self._memories:  # upsert by (scope, category, content)
            if (existing.scope == memory.scope and existing.category == memory.category
                    and existing.content == memory.content):
                if memory.embedding is not None:
                    existing.embedding = memory.embedding
                existing.confidence = memory.confidence
                return
        if memory.created_at is None:
            memory.created_at = _now()
        if memory.id is None:
            memory.id = str(uuid4())
        self._memories.append(memory)

    async def load_memories(self, scope: str, *, query: Optional[RetrievalQuery] = None,
                            limit: int = 50,
                            weights: Optional[HybridWeights] = None) -> list[MemoryRecord]:
        _require_scope(scope)
        weights = weights or HybridWeights()
        mems = [m for m in self._memories if m.scope == scope]
        if query is not None and query.categories:
            cats = set(query.categories)
            mems = [m for m in mems if m.category in cats]

        # No query signal → chronological (most recent first).
        if query is None or (not query.text and query.embedding is None):
            mems.sort(key=lambda m: (m.created_at or _now()), reverse=True)
            return mems[:limit]

        scored: list[tuple[float, MemoryRecord]] = []
        for m in mems:
            vec = _cosine(query.embedding, m.embedding) if (query.embedding and m.embedding) else 0.0
            lex = _lexical(query.text, m.content) if query.text else 0.0
            score = weights.vector * vec + weights.lexical * lex + weights.feedback * m.feedback_score
            scored.append((score, m))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [m for _, m in scored[:limit]]

    async def scan_memories(self, scope: str, *, after_id: Optional[str] = None,
                            limit: int = 1000) -> list[MemoryRecord]:
        _require_scope(scope)
        mems = sorted((m for m in self._memories if m.scope == scope), key=lambda m: m.id or "")
        if after_id is not None:
            mems = [m for m in mems if (m.id or "") > after_id]
        return mems[:limit]

    async def adjust_feedback_score(self, scope: str, query_text: str, delta: float,
                                    *, limit: int = 10) -> int:
        _require_scope(scope)
        touched = 0
        for m in self._memories:
            if m.scope == scope and _lexical(query_text, m.content) > 0:
                m.feedback_score = max(-10.0, min(10.0, m.feedback_score + delta))
                touched += 1
                if touched >= limit:
                    break
        return touched

    async def memory_count(self, scope: str) -> int:
        _require_scope(scope)
        return sum(1 for m in self._memories if m.scope == scope)

    async def delete_memories(self, scope: str, *, older_than: Optional[datetime] = None,
                              category: Optional[str] = None,
                              max_confidence: Optional[float] = None) -> int:
        _require_scope(scope)

        def keep(m: MemoryRecord) -> bool:
            if m.scope != scope:
                return True
            if older_than is not None and (m.created_at is None or m.created_at >= older_than):
                return True
            if category is not None and m.category != category:
                return True
            if max_confidence is not None and m.confidence > max_confidence:
                return True
            return False   # matches all active filters → delete

        before = len(self._memories)
        self._memories = [m for m in self._memories if keep(m)]
        return before - len(self._memories)

    async def purge_scope(self, scope: str) -> int:
        _require_scope(scope)
        total = 0
        sids = [sid for sid, s in self._sessions.items() if s.scope == scope]
        total += len(sids)
        for sid in sids:
            del self._sessions[sid]
        for coll_attr in ("_turns", "_traces", "_memories"):
            coll = getattr(self, coll_attr)
            before = len(coll)
            setattr(self, coll_attr, [r for r in coll if r.scope != scope])
            total += before - len(getattr(self, coll_attr))
        return total

    # ── concurrency ──────────────────────────────────────────────────────
    def session_lock(self, scope: str, session_id: str):
        _require_scope(scope)
        lock = self._locks.setdefault(f"{scope}:{session_id}", asyncio.Lock())

        @asynccontextmanager
        async def _cm() -> AsyncIterator[None]:
            async with lock:
                yield

        return _cm()

    # ── capability flag ──────────────────────────────────────────────────
    def supports_vector(self) -> bool:
        return True


class InMemoryBuffer:
    """Reference ``ConversationBuffer`` — a per-(scope,session) sliding window."""

    def __init__(self) -> None:
        self._buf: dict[str, list[TurnRecord]] = {}

    async def push(self, scope: str, session_id: str, turn: TurnRecord) -> None:
        _require_scope(scope)
        self._buf.setdefault(f"{scope}:{session_id}", []).append(turn)

    async def window(self, scope: str, session_id: str, *, size: int = 10) -> list[TurnRecord]:
        _require_scope(scope)
        return self._buf.get(f"{scope}:{session_id}", [])[-size:]

    async def clear(self, scope: str, session_id: str) -> None:
        _require_scope(scope)
        self._buf.pop(f"{scope}:{session_id}", None)


class InMemoryGraph:
    """Reference ``KnowledgeGraph`` — typed nodes + directed edges + BFS walk."""

    def __init__(self) -> None:
        self._nodes: dict[tuple[str, str], GraphNode] = {}   # (scope, label.lower()) -> node
        self._edges: list[GraphEdge] = []
        self._next_id = 1

    # ── audience ────────────────────────────────────────────────────────────
    #
    # "Tenant sees everything; an identity sees only its own life. They do not mix."
    # The filter is on the EDGE (see `types.audience_can_read` for why the node cannot carry
    # it), and node visibility is DERIVED: a node is visible to a reader if some edge that
    # reader may see touches it. An orphan node — one no visible edge reaches — is staff-only,
    # which is right: a bare label is the weakest thing the graph holds and belongs to nobody.

    def _readable(self, scope: str, audience: str) -> "list[GraphEdge]":
        return [e for e in self._edges
                if e.scope == scope and audience_can_read(audience, e.audience)]

    def _visible_labels(self, scope: str, audience: str) -> "set[str]":
        if audience == AUDIENCE_STAFF:
            return {lbl for (sc, lbl) in self._nodes if sc == scope}
        out: set[str] = set()
        for e in self._readable(scope, audience):
            out.add(e.source.lower())
            out.add(e.target.lower())
        return out

    async def upsert_node(self, node: GraphNode) -> int:
        _require_scope(node.scope)
        key = (node.scope, node.label.lower())
        existing = self._nodes.get(key)
        now = datetime.now(timezone.utc)
        if existing is not None:
            existing.attributes.update(node.attributes)
            if node.embedding is not None:
                existing.embedding = node.embedding
            existing.updated_at = now                        # parity with Pg's updated_at = now()
            assert existing.id is not None
            return existing.id
        node.id = self._next_id
        self._next_id += 1
        node.created_at = node.created_at or now
        node.updated_at = node.updated_at or now
        self._nodes[key] = node
        return node.id

    async def upsert_edge(self, edge: GraphEdge) -> None:
        _require_scope(edge.scope)
        # Parity with the Postgres adapter: an edge's endpoints are auto-created when missing
        # (Pg's _resolve_node_id INSERTs with the column default node_type='CONCEPT'), so an
        # LLM extraction that lists an edge without declaring both nodes never dangles.
        for label in (edge.source, edge.target):
            if (edge.scope, label.lower()) not in self._nodes:
                await self.upsert_node(GraphNode(edge.scope, label, "CONCEPT"))
        for existing in self._edges:
            if (existing.scope == edge.scope and existing.source.lower() == edge.source.lower()
                    and existing.target.lower() == edge.target.lower()
                    and existing.relation == edge.relation):
                if existing.status == EDGE_ACCEPTED and edge.status == EDGE_PROPOSED:
                    # A PROPOSAL cannot modify a VERDICT — in any field, not just `status`.
                    # The gate used to cover `status` alone while `attributes` merged straight
                    # through, and `_detail` puts attributes in the prompt: a caller that marked
                    # the whole edge unreviewed had its relation held and its free text SPOKEN
                    # ("Pedro (note: expelled from school for cheating)"). Reviewed means
                    # reviewed as it stood; a proposal with something to add needs its own turn
                    # through the queue.
                    return
                existing.confidence = edge.confidence
                existing.source_session = edge.source_session
                # Audience may be NARROWED but never widened — the same shape as `status`, and
                # the direction that cannot leak: an edge already private to someone stays
                # private even when a later writer forgets to declare one. The Postgres twin
                # does this in its `ON CONFLICT`, and the two adapters diverging on a write is
                # exactly how an edge ends up visible in one store and not the other.
                if not existing.audience:
                    existing.audience = sanitize_audience(edge.audience)
                # Merge, never replace: the LLM that re-proposes an edge must not wipe the
                # detail a human typed. (Key-level: the LLM CAN overwrite a value for a key it
                # also emits — what is protected is the keys it omits.)
                #
                # A re-assertion promotes a PROPOSAL and never demotes a verdict. `rejected` is
                # deliberately sticky, and a review was right that the code did not say so: a
                # rejection is a person stating the claim is WRONG about this contact, and the
                # next extraction pass must not be able to undo it — `upsert_edge` cannot tell a
                # deliberate correction from the LLM re-emitting the same edge, and it defaults
                # to `accepted`, so promoting from `rejected` here would resurrect every
                # rejected edge on the next Tier-2 run. `set_edge_status` is the way back, and
                # it is the way back on purpose: undoing a human verdict takes a human.
                existing.attributes = {**existing.attributes, **edge.attributes}
                if existing.status == EDGE_PROPOSED:
                    existing.status = edge.status
                return
        self._edges.append(_detached(edge))   # never alias the caller's object

    async def find_node(self, scope: str, label: str, *,
                        audience: str) -> Optional[GraphNode]:
        _require_scope(scope)
        if label.lower() not in self._visible_labels(scope, audience):
            return None
        return self._nodes.get((scope, label.lower()))

    async def find_nodes_by_embedding(self, scope: str, embedding: list[float],
                                      *, audience: str, limit: int = 5) -> list[GraphNode]:
        _require_scope(scope)
        visible = self._visible_labels(scope, audience)
        scored = [(_cosine(embedding, n.embedding), n)
                  for n in self._nodes.values()
                  if n.scope == scope and n.embedding and n.label.lower() in visible]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [n for _, n in scored[:limit]]

    async def pending_edges(self, scope: str, *, audience: str,
                            limit: int = 100) -> list[GraphEdge]:
        """Oldest first — see the Postgres adapter for why the two must agree on this.

        COPIES, not the stored objects. Postgres builds fresh rows and this returned live
        references, so a curation UI that edited a queue item published the edge here and did
        nothing there — the two stores disagreeing on exactly the invariant this feature is.
        """
        _require_scope(scope)
        if limit <= 0:                  # Postgres raises on a negative LIMIT; agree on empty
            return []
        return [_detached(e) for e in self._readable(scope, audience)
                if e.status == EDGE_PROPOSED][:limit]

    async def set_edge_status(self, scope: str, source: str, target: str, relation: str,
                              status: str) -> bool:
        _require_scope(scope)
        want = require_edge_status(status)
        for e in self._edges:
            if (e.scope == scope and e.source.lower() == source.lower()
                    and e.target.lower() == target.lower() and e.relation == relation):
                e.status = want
                return True
        return False

    async def set_edge_audience(self, scope: str, source: str, target: str, relation: str,
                                audience: str) -> bool:
        """Explicit re-classification — the only way back from a migration."""
        _require_scope(scope)
        want = sanitize_audience(audience)
        for e in self._edges:
            if (e.scope == scope and e.source.lower() == source.lower()
                    and e.target.lower() == target.lower() and e.relation == relation):
                e.audience = want
                return True
        return False

    async def walk(self, scope: str, start_label: str, *, audience: str,
                   max_depth: int = 2) -> list[GraphEdge]:
        _require_scope(scope)
        readable = self._readable(scope, audience)
        result: list[GraphEdge] = []
        seen_edges: set[int] = set()
        visited = {start_label.lower()}
        frontier: list[tuple[str, int]] = [(start_label.lower(), 0)]
        while frontier:
            label, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for edge in readable:
                if edge.status != EDGE_ACCEPTED:
                    # Skipped, not merely unreturned: an unreviewed edge must not decide what
                    # the walk can REACH either. Returning it later while letting it route the
                    # traversal now would leak the same unverified claim one hop further away.
                    continue
                if edge.source.lower() == label:
                    nxt = edge.target
                elif edge.target.lower() == label:
                    nxt = edge.source
                else:
                    continue
                if id(edge) not in seen_edges:
                    seen_edges.add(id(edge))
                    result.append(_detached(edge))
                if nxt.lower() not in visited:
                    visited.add(nxt.lower())
                    frontier.append((nxt.lower(), depth + 1))
        return result

    async def neighbors(self, scope: str, label: str, *, audience: str) -> list[GraphNode]:
        _require_scope(scope)
        labels: set[str] = set()
        for edge in self._readable(scope, audience):
            if edge.status != EDGE_ACCEPTED:
                # An unreviewed edge still DISCLOSES its endpoint. The relation label is gone,
                # but "this person is connected to José" is exactly the unverified claim the
                # feature holds back — and `NodeContext` hands edges and neighbors to the same
                # caller, so filtering one and not the other leaks it through the other field.
                continue
            if edge.source.lower() == label.lower():
                labels.add(edge.target.lower())
            elif edge.target.lower() == label.lower():
                labels.add(edge.source.lower())
        return [n for (s, lbl), n in self._nodes.items() if s == scope and lbl in labels]

    async def get_node_context(self, scope: str, label: str, *,
                               audience: str) -> Optional[NodeContext]:
        _require_scope(scope)
        node = self._nodes.get((scope, label.lower()))
        if node is None or label.lower() not in self._visible_labels(scope, audience):
            return None
        edges = [_detached(e) for e in self._readable(scope, audience)
                 if e.status == EDGE_ACCEPTED
                 and label.lower() in (e.source.lower(), e.target.lower())]
        return NodeContext(node=node, edges=edges,
                           neighbors=await self.neighbors(scope, label, audience=audience))

    async def list_nodes(self, scope: str, *, audience: str,
                         node_type: Optional[str] = None,
                         limit: int = 100) -> list[GraphNode]:
        _require_scope(scope)
        visible = self._visible_labels(scope, audience)
        nodes = [n for (s, lbl), n in self._nodes.items()
                 if s == scope and lbl in visible
                 and (node_type is None or n.node_type == node_type)]
        return nodes[:limit]

    async def count_nodes(self, scope: str, *, audience: str,
                          label: Optional[str] = None) -> int:
        """How many nodes this scope holds, or how many carry ``label`` (case-insensitively).

        Case-insensitive because that is how every other node read matches: `find_node` and
        `walk` both compare `lower(label)`, so a count that were case-SENSITIVE would answer a
        different question from the one the caller is about to act on.
        """
        _require_scope(scope)
        want = label.strip().casefold() if label is not None else None
        visible = self._visible_labels(scope, audience)
        return sum(1 for (s, lbl), n in self._nodes.items()
                   if s == scope and lbl in visible
                   and (want is None
                        or (n.label or "").strip().casefold() == want))

    async def scan_nodes(self, scope: str, *, audience: str,
                         after_id: Optional[int] = None,
                         limit: int = 1000) -> list[GraphNode]:
        _require_scope(scope)
        visible = self._visible_labels(scope, audience)
        nodes = sorted((n for (s, lbl), n in self._nodes.items()
                        if s == scope and lbl in visible),
                       key=lambda n: n.id or 0)
        if after_id is not None:
            nodes = [n for n in nodes if (n.id or 0) > after_id]
        return nodes[:limit]

    async def has_edges(self, scope: str, label: str) -> bool:
        """Every edge, any audience, any status — see the port for why it takes no audience."""
        _require_scope(scope)
        want = label.lower()
        return any(e.scope == scope and want in (e.source.lower(), e.target.lower())
                   for e in self._edges)

    async def delete_node(self, scope: str, label: str) -> bool:
        _require_scope(scope)
        key = (scope, label.lower())
        if key not in self._nodes:
            return False
        del self._nodes[key]
        # cascade: drop edges touching the node
        self._edges = [e for e in self._edges
                       if not (e.scope == scope and label.lower() in (e.source.lower(), e.target.lower()))]
        return True

    async def delete_edges_by_session(self, scope: str, session_id: str) -> int:
        _require_scope(scope)
        _require_session(session_id)
        before = len(self._edges)
        self._edges = [e for e in self._edges
                       if not (e.scope == scope and e.source_session == session_id)]
        return before - len(self._edges)

    async def purge_scope(self, scope: str) -> int:
        _require_scope(scope)
        before_edges = len(self._edges)
        self._edges = [e for e in self._edges if e.scope != scope]
        before_nodes = len(self._nodes)
        self._nodes = {k: n for k, n in self._nodes.items() if k[0] != scope}
        return (before_edges - len(self._edges)) + (before_nodes - len(self._nodes))
