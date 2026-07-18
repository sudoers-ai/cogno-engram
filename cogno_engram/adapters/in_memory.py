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

from cogno_engram.types import (
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


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require_scope(scope: str) -> str:
    if not scope or not scope.strip():
        raise ValueError("scope must be a non-empty string (engram isolates every row by scope)")
    return scope


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

    async def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    async def close_session(self, session_id: str, *, summary: str = "", scope: str = "") -> None:
        session = self._sessions.get(session_id)
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
            if sess is not None and sess.ended_at is not None:
                continue                                   # already closed/consolidated
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

    async def load_turns(self, session_id: str) -> list[TurnRecord]:
        turns = [t for t in self._turns if t.session_id == session_id]
        turns.sort(key=lambda t: t.turn_n)
        return turns

    async def turn_count(self, session_id: str) -> int:
        return sum(1 for t in self._turns if t.session_id == session_id)

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

    async def traces_for_session(self, session_id: str) -> list[TurnTrace]:
        traces = [t for t in self._traces if t.session_id == session_id]
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
                existing.confidence = edge.confidence
                existing.source_session = edge.source_session
                return
        self._edges.append(edge)

    async def find_node(self, scope: str, label: str) -> Optional[GraphNode]:
        _require_scope(scope)
        return self._nodes.get((scope, label.lower()))

    async def find_nodes_by_embedding(self, scope: str, embedding: list[float],
                                      *, limit: int = 5) -> list[GraphNode]:
        _require_scope(scope)
        scored = [(_cosine(embedding, n.embedding), n)
                  for n in self._nodes.values() if n.scope == scope and n.embedding]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [n for _, n in scored[:limit]]

    async def walk(self, scope: str, start_label: str, *, max_depth: int = 2) -> list[GraphEdge]:
        _require_scope(scope)
        result: list[GraphEdge] = []
        seen_edges: set[int] = set()
        visited = {start_label.lower()}
        frontier: list[tuple[str, int]] = [(start_label.lower(), 0)]
        while frontier:
            label, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            for edge in self._edges:
                if edge.scope != scope:
                    continue
                if edge.source.lower() == label:
                    nxt = edge.target
                elif edge.target.lower() == label:
                    nxt = edge.source
                else:
                    continue
                if id(edge) not in seen_edges:
                    seen_edges.add(id(edge))
                    result.append(edge)
                if nxt.lower() not in visited:
                    visited.add(nxt.lower())
                    frontier.append((nxt.lower(), depth + 1))
        return result

    async def neighbors(self, scope: str, label: str) -> list[GraphNode]:
        _require_scope(scope)
        labels: set[str] = set()
        for edge in self._edges:
            if edge.scope != scope:
                continue
            if edge.source.lower() == label.lower():
                labels.add(edge.target.lower())
            elif edge.target.lower() == label.lower():
                labels.add(edge.source.lower())
        return [n for (s, lbl), n in self._nodes.items() if s == scope and lbl in labels]

    async def get_node_context(self, scope: str, label: str) -> Optional[NodeContext]:
        _require_scope(scope)
        node = self._nodes.get((scope, label.lower()))
        if node is None:
            return None
        edges = [e for e in self._edges
                 if e.scope == scope and label.lower() in (e.source.lower(), e.target.lower())]
        return NodeContext(node=node, edges=edges, neighbors=await self.neighbors(scope, label))

    async def list_nodes(self, scope: str, *, node_type: Optional[str] = None,
                         limit: int = 100) -> list[GraphNode]:
        _require_scope(scope)
        nodes = [n for (s, _), n in self._nodes.items()
                 if s == scope and (node_type is None or n.node_type == node_type)]
        return nodes[:limit]

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
