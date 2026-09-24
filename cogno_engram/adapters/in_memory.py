"""
cogno_engram.adapters.in_memory — zero-dependency reference adapters.

Pure-Python implementations of all four ports, for tests, local dev, and as the
executable proof that the Protocols are honest. They implement the same hybrid
retrieval fusion and multi-hop graph walk as the Postgres adapter, just over
in-process structures (no persistence, single-process).
"""

from __future__ import annotations

import asyncio
import math
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator, Optional
from uuid import uuid4


from cogno_engram import textfold
from cogno_engram.folding import fold_label, has_diacritics
from cogno_engram.documents import (
    COMMIT_DELETED,
    COMMIT_READY,
    COMMIT_SUPERSEDED,
    DEFAULT_MAX_BYTES,
    KB_EMBED_SPACE_UNAVAILABLE,
    KB_ERROR,
    KB_PROCESSING,
    KB_READY,
    TOMBSTONE_DELETED,
    TOMBSTONE_PURGED,
    VALID_MEDIA_TYPES,
    DocumentLimitReached,
    KbChunk,
    KbDocument,
    KbHit,
    KbSearchResult,
    KbTombstone,
    KbVersion,
    OriginalTooLarge,
    chunk_id,
    clamp_unit,
    hit_order_key,
    hybrid_score,
    owner_in_subtree,
    profile_can_read,
    require_model,
    require_owner,
    require_profile,
    require_vector,
    sanitize_profiles,
    sanitize_reason,
)
from cogno_engram import write_loss
from cogno_engram.trace_policy import TRACE_REVISION_WINDOW_S
from cogno_engram.types import (
    AUDIENCE_STAFF,
    audience_can_read,
    sanitize_audience,
    EDGE_ACCEPTED,
    GraphStats,
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

    #: This store ALLOCATES ``turn_n`` when handed ``ALLOCATE_TURN_N``.
    #:
    #: A capability flag, not decoration. A caller cannot detect the ability from
    #: the signature — the old ``save_turn`` accepted the same argument and simply
    #: wrote the sentinel into the column — so a host that probed by trying would
    #: corrupt a row against an older engram. The host pins the two libraries by
    #: git SHA and they land in separate deploys, so the mixed combination is a
    #: state the ecosystem WILL be in, not one it might be. Absent → the caller
    #: keeps numbering turns itself, exactly as before.
    allocates_turn_n = True

    def __init__(self, *, trace_revision_window_s: float = TRACE_REVISION_WINDOW_S) -> None:
        # Same knob, same default, same meaning as the Postgres adapter's — the two
        # are hand-written copies of one rule and a divergence here is invisible
        # until production.
        self._trace_revision_window_s = float(trace_revision_window_s)
        self._sessions: dict[str, Session] = {}
        self._turns: list[TurnRecord] = []
        # The Postgres `turns.consolidated_at` column, mirrored. Keyed by the same
        # triple that table declares UNIQUE, so "the mark rides on the turn" is true
        # here too: it survives a `sessions` deletion and dies with `purge_scope`.
        # NOT a field on `TurnRecord` — that dataclass is what the HOST wrote; this is
        # what the janitor did to it, and a host has no business setting it.
        self._consolidated: set[tuple[str, str, int]] = set()
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
        """Mirror of ``PostgresStore.close_session``: close the row AND stamp the turns.

        The stamp is the marker a `sessions` deletion cannot reach — see that method and
        ``idle_sessions`` for the loop it closes. Same watermark rule (`created_at <=
        ended_at`), same refusal to stamp when nothing was actually closed.
        """
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
        else:
            return          # no row and no scope to make one → nothing closed, nothing to stamp
        closed = self._sessions[session_id]
        ended = closed.ended_at
        assert ended is not None                       # just written, both branches above
        # The scope comes from the row that was closed, never from the argument — the Postgres
        # adapter reads it back with RETURNING for the same reason: ids collide across scopes.
        for turn in self._turns:
            if (turn.session_id == session_id and turn.scope == closed.scope
                    and turn.created_at is not None and turn.created_at <= ended):
                self._consolidated.add((turn.scope, turn.session_id, turn.turn_n))

    async def idle_sessions(self, *, idle_seconds: int = 1800,
                            limit: int = 100) -> list[Session]:
        cutoff = _now() - timedelta(seconds=idle_seconds)
        # last activity per session_id, derived from turns (the host may never create_session)
        last: dict[str, tuple[str, "datetime", "datetime"]] = {}   # sid → (scope, first, last)
        for t in self._turns:
            if (t.scope, t.session_id, t.turn_n) in self._consolidated:
                continue        # already read into long-term memory — the mark a purge can't erase
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
            # Postgres adapter for the measured effect of getting this wrong. The `sessions`
            # row is only HALF the answer: it is deletable, and the loop that fact opened is
            # closed by the `_consolidated` filter above, not by this test.
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
    async def save_turn(self, turn: TurnRecord) -> int:
        """Mirror of ``PostgresStore.save_turn`` — see there for the design.

        ``turn_n = ALLOCATE_TURN_N`` (any negative) allocates ``max + 1`` for the
        ``(scope, session_id)``; a pinned number — ``0`` included, it is a real
        coordinate — is honoured, and a collision on it is a counted loss rather
        than a duplicate row. The two adapters are hand-written copies of one rule,
        so a divergence here is a bug that only shows up in production.
        """
        _require_scope(turn.scope)
        mine = [t for t in self._turns
                if t.scope == turn.scope and t.session_id == turn.session_id]
        if turn.turn_n is not None and turn.turn_n >= 0:
            # Match the Postgres adapter's ON CONFLICT (scope, session_id, turn_n) DO NOTHING:
            # a re-save of the same turn coordinate is a no-op, not a duplicate row (the
            # in-memory adapter otherwise diverged, double-counting turns in local/test runs).
            if any(t.turn_n == turn.turn_n for t in mine):
                write_loss.record(write_loss.TURN_DISCARDED, scope=turn.scope,
                                  session=turn.session_id, turn=turn.turn_n,
                                  reason="coordinate_taken", pinned="true")
                return 0
        else:
            turn.turn_n = max((t.turn_n for t in mine), default=0) + 1
        if turn.created_at is None:
            turn.created_at = _now()
        self._turns.append(turn)
        return turn.turn_n

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
    async def save_turn_trace(self, trace: TurnTrace) -> bool:
        """Mirror of ``PostgresStore.save_turn_trace`` — a stored trace stops being
        revisable once it is older than ``TRACE_REVISION_WINDOW_S``, and its
        ``created_at`` never moves."""
        _require_scope(trace.scope)
        if trace.created_at is None:
            trace.created_at = _now()
        # UPSERT by (scope, session_id, turn_n).
        prior = [t for t in self._traces
                 if t.scope == trace.scope and t.session_id == trace.session_id
                 and t.turn_n == trace.turn_n]
        window = timedelta(seconds=self._trace_revision_window_s)
        if prior and prior[0].created_at is not None \
                and trace.created_at > prior[0].created_at + window:
            write_loss.record(write_loss.TRACE_OVERWRITE_REFUSED, scope=trace.scope,
                              session=trace.session_id, turn=trace.turn_n,
                              reason="stored_trace_is_older")
            return False
        if prior:
            # The stored row keeps its own ``created_at``: an upsert revises CONTENT,
            # it does not restate when the turn happened.
            trace.created_at = prior[0].created_at
        self._traces = [t for t in self._traces
                        if not (t.scope == trace.scope and t.session_id == trace.session_id
                                and t.turn_n == trace.turn_n)]
        self._traces.append(trace)
        return True

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

    async def memory_scopes(self) -> "list[str]":
        """Twin of the Postgres one — scopes with MEMORIES, tenant-owned or not."""
        return sorted({m.scope for m in self._memories})

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
                # `first_heard_by` is NOT copied over, mirroring the Postgres `DO UPDATE` list
                # that omits it: the column answers "who did the contact tell this to FIRST", so
                # the second persona to mention the same fact must not take the credit. Adding it
                # here to "match the other fields" would make this double disagree with the real
                # store, and the tests that pass against it would stop saying anything about
                # production.
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
                              max_confidence: Optional[float] = None,
                              dry_run: bool = False) -> int:
        """Twin of the Postgres one: ONE predicate, two verbs — see it for why."""
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

        survivors = [m for m in self._memories if keep(m)]
        going = len(self._memories) - len(survivors)
        if dry_run:
            return going
        self._memories = survivors
        return going

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
        # The marks go with the turns they were stamped on — in Postgres they are a COLUMN of
        # the deleted row and cannot do otherwise. Not counted: a mark is not a row.
        self._consolidated = {k for k in self._consolidated if k[0] != scope}
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
        self._nodes: dict[tuple[str, str], GraphNode] = {}   # (scope, fold_label(label)) -> node
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
            out.add(fold_label(e.source))
            out.add(fold_label(e.target))
        return out

    async def upsert_node(self, node: GraphNode) -> int:
        _require_scope(node.scope)
        key = (node.scope, fold_label(node.label))
        existing = self._nodes.get(key)
        now = datetime.now(timezone.utc)
        if existing is not None:
            # A grafia com DIACRÍTICOS sobe, nunca desce — paridade com o `ON CONFLICT` do
            # Postgres. Sem isto o primeiro a chegar ficava para sempre, e como o contacto
            # escreve o nome sem acento metade das vezes, `Jose` chegava primeiro e a linha
            # nunca mais voltava a ser `José`: o contrário do que o `folding` promete.
            if has_diacritics(node.label) and not has_diacritics(existing.label):
                existing.label = node.label
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
            if (edge.scope, fold_label(label)) not in self._nodes:
                await self.upsert_node(GraphNode(edge.scope, label, "CONCEPT"))
        for existing in self._edges:
            if (existing.scope == edge.scope and fold_label(existing.source) == fold_label(edge.source)
                    and fold_label(existing.target) == fold_label(edge.target)
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
        # Stamp `created_at` on FIRST insert, the way the Postgres column's `DEFAULT now()`
        # does — parity, and the direction that matters: a caller who already set one (a test
        # pinning a date, a replay) keeps it. Without this the two adapters answer the same
        # read differently, which is exactly how the label normalisation drifted before.
        novo = _detached(edge)                # never alias the caller's object
        if novo.created_at is None:
            novo.created_at = datetime.now(timezone.utc)
        self._edges.append(novo)

    async def find_node(self, scope: str, label: str, *,
                        audience: str) -> Optional[GraphNode]:
        _require_scope(scope)
        if fold_label(label) not in self._visible_labels(scope, audience):
            return None
        return self._nodes.get((scope, fold_label(label)))

    async def find_nodes_by_embedding(self, scope: str, embedding: list[float],
                                      *, audience: str, limit: int = 5,
                                      related_only: bool = False) -> list[GraphNode]:
        _require_scope(scope)
        visible = self._visible_labels(scope, audience)
        related: set = set()
        if related_only:
            # ACCEPTED only — `walk` skips every other status, so an unreviewed edge does not
            # make a node walkable. See the Postgres twin for why this is a pessimisation and
            # not merely a miss.
            related = {fold_label(lbl)
                       for e in self._edges
                       if e.scope == scope and e.status == EDGE_ACCEPTED
                       for lbl in (e.source, e.target)}
        scored = [(_cosine(embedding, n.embedding), n)
                  for n in self._nodes.values()
                  if n.scope == scope and n.embedding and fold_label(n.label) in visible
                  and (not related_only or fold_label(n.label) in related)]
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
            if (e.scope == scope and fold_label(e.source) == fold_label(source)
                    and fold_label(e.target) == fold_label(target) and e.relation == relation):
                e.status = want
                return True
        return False

    async def set_edge_audience(self, scope: str, source: str, target: str, relation: str,
                                audience: str) -> bool:
        """Explicit re-classification — the only way back from a migration."""
        _require_scope(scope)
        want = sanitize_audience(audience)
        for e in self._edges:
            if (e.scope == scope and fold_label(e.source) == fold_label(source)
                    and fold_label(e.target) == fold_label(target) and e.relation == relation):
                e.audience = want
                return True
        return False

    async def walk(self, scope: str, start_label: str, *, audience: str,
                   max_depth: int = 2) -> list[GraphEdge]:
        _require_scope(scope)
        readable = self._readable(scope, audience)
        result: list[GraphEdge] = []
        seen_edges: set[int] = set()
        visited = {fold_label(start_label)}
        frontier: list[tuple[str, int]] = [(fold_label(start_label), 0)]
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
                if fold_label(edge.source) == label:
                    nxt = edge.target
                elif fold_label(edge.target) == label:
                    nxt = edge.source
                else:
                    continue
                if id(edge) not in seen_edges:
                    seen_edges.add(id(edge))
                    result.append(_detached(edge))
                if fold_label(nxt) not in visited:
                    visited.add(fold_label(nxt))
                    frontier.append((fold_label(nxt), depth + 1))
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
            if fold_label(edge.source) == fold_label(label):
                labels.add(fold_label(edge.target))
            elif fold_label(edge.target) == fold_label(label):
                labels.add(fold_label(edge.source))
        return [n for (s, lbl), n in self._nodes.items() if s == scope and lbl in labels]

    async def get_node_context(self, scope: str, label: str, *,
                               audience: str) -> Optional[NodeContext]:
        _require_scope(scope)
        node = self._nodes.get((scope, fold_label(label)))
        if node is None or fold_label(label) not in self._visible_labels(scope, audience):
            return None
        edges = [_detached(e) for e in self._readable(scope, audience)
                 if e.status == EDGE_ACCEPTED
                 and fold_label(label) in (fold_label(e.source), fold_label(e.target))]
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
                          label: Optional[str] = None,
                          node_type: Optional[str] = None) -> int:
        """How many nodes this scope holds, or how many carry ``label`` (case-insensitively).

        Dobrado por `folding.fold_label`, como TODA leitura de nó — `find_node` e `walk`
        incluídas. Usava `casefold()`, que dobra caixa mas não acento nem transliteração: o host
        usa este método como GUARDA (`memory.py:536`, "existe exactamente 1?") e ele respondia a
        uma pergunta diferente da que o `find_node` ao lado responde. Uma definição de identidade
        que vive num sítio e é re-derivada noutro é a forma de defeito que este módulo veio
        fechar; escapou à varredura porque a varredura procurava `.lower()`.
        """
        _require_scope(scope)
        want = fold_label(label.strip()) if label is not None else None
        visible = self._visible_labels(scope, audience)
        return sum(1 for (s, lbl), n in self._nodes.items()
                   if s == scope and lbl in visible
                   and (want is None
                        or fold_label((n.label or "").strip()) == want)
                   # EXACT, like the Postgres twin and like `list_nodes`: node types are a
                   # closed vocabulary the writer already normalised, and folding here would
                   # make this count answer about a wider set than the list it accompanies.
                   and (node_type is None or n.node_type == node_type))

    async def graph_stats(self, scope: str, *, audience: str, top: int = 5) -> GraphStats:
        """The dashboard summary in one pass — the twin of the Postgres aggregate.

        The point of the method is the CALLER's cost, not this one's: in memory a loop is a loop.
        It exists here so the two stores answer the same question the same way, which is what
        ``tests/test_graph_stats_parity.py`` pins.

        Edges are counted DISTINCT by ``(source, target, relation)`` and only when ACCEPTED,
        because the caller's previous version summed what ``walk`` returned and ``walk`` skips
        every other status.

        **Deduplication is on the RAW labels — a decision taken on 2026-08-27, not an
        inheritance.** Folding would give the same number today (the edges come from one store),
        so the key was left raw to keep this change to COST alone; folding it is a semantic
        change and gets its own PR. Saying "the same as the caller did" would be provenance
        dressed as a reason — and that is the exact shape `José`/`Jose` had before it was a defect.
        """
        _require_scope(scope)
        visible = self._visible_labels(scope, audience)
        nodes = [n for (sc, lbl), n in self._nodes.items() if sc == scope and lbl in visible]
        by_type: dict = {}
        for n in nodes:
            by_type[n.node_type] = by_type.get(n.node_type, 0) + 1
        seen: set = set()
        degree: dict = {}
        for e in self._readable(scope, audience):
            if e.status != EDGE_ACCEPTED:
                continue
            key = (e.source, e.target, e.relation)
            if key in seen:
                continue
            seen.add(key)
            for lbl in (fold_label(e.source), fold_label(e.target)):
                degree[lbl] = degree.get(lbl, 0) + 1
        # Ties keep the store's own order (`id`), which is what the caller's previous version
        # inherited from `list_nodes` (`ORDER BY id`). Sorting ties by LABEL is equally
        # deterministic and quietly different — see the Postgres twin.
        ranked = sorted(nodes, key=lambda n: (-degree.get(fold_label(n.label), 0), n.id or 0))
        return GraphStats(
            total_nodes=len(nodes), total_edges=len(seen), by_type=by_type,
            top_connected=[(n, degree.get(fold_label(n.label), 0)) for n in ranked[:max(0, top)]])

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
        want = fold_label(label)
        return any(e.scope == scope and want in (fold_label(e.source), fold_label(e.target))
                   for e in self._edges)

    async def delete_node(self, scope: str, label: str) -> bool:
        _require_scope(scope)
        key = (scope, fold_label(label))
        if key not in self._nodes:
            return False
        del self._nodes[key]
        # cascade: drop edges touching the node
        self._edges = [e for e in self._edges
                       if not (e.scope == scope and fold_label(label) in (fold_label(e.source), fold_label(e.target)))]
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


# ── documents ────────────────────────────────────────────────────────────────────────

@dataclass
class _DocRow:
    owner_key: str
    id: str
    title: str
    profiles: tuple
    media_type: str
    active_version: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class _VersionRow:
    document_id: str
    version: int
    state: str
    sha256: str
    embed_model: str
    reason: str = ""
    size_bytes: int = 0
    pages: int = 0
    chunks: int = 0
    created_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


def _doc_cosine(a: list[float], b: list[float]) -> float:
    """The ONLY place the document store compares two vectors — a seam the space tests watch:
    a search that reaches it with vectors of two different models is the defect."""
    return _cosine(a, b)


def _doc_terms(text: str) -> "set[str]":
    """The words of ``text`` for the in-memory lexical stand-in: the GENERAL fold
    (:func:`cogno_engram.textfold.fold`, no keyword step), then ``\\w+``, and NO stopwords — a
    neutral stand-in, like Postgres ``simple`` plus accent folding. Not the graph's LABEL fold
    (``cogno_engram.folding``): that is an IDENTITY rule, and borrowing it would make a change to
    how labels are folded silently move how documents are matched — this section may not even
    name it (``tests/test_documents_use_the_general_fold.py``).
    Not ``lexical.tokens`` either: that one drops stopwords and cuts prefixes, because it is the
    relevance engine's ruler, and this is the store's candidate test."""
    return set(re.findall(r"\w+", textfold.fold(text)))


def _doc_lexical(query_terms: "set[str]", content: str) -> float:
    """Share of the query's DISTINCT words the chunk carries — a DECLARED stand-in for the
    Postgres side's ``ts_rank_cd(tsv, query, 32)``, and NOT the same number.

    What the two agree on, and what they do not, said in full because a double is only useful
    when its limits are written down:

    * **the same candidates**: ANY shared word makes a chunk a lexical candidate here, and the
      Postgres side ORs the query's lexemes, so a chunk that carries no query word scores 0 on
      both sides — and, on a lexical search, is absent from both;
    * **the same range**: both are in ``[0, 1]``;
    * **the same ORDER on a corpus where each query word occurs at most once per chunk**, under a
      configuration that only lowercases (Postgres ``simple``, no stemming, no unaccent):
      ``tests/test_documents_postgres.py::test_the_lexical_order_is_the_same_in_both_adapters``
      pins it, and the ``0`` in the same place;
    * **NOT the same value, and not the same order in general**: ``ts_rank_cd`` counts
      OCCURRENCES and weighs their PROXIMITY (a word said three times outranks a word said once;
      two words side by side outrank the same two far apart), while this counts distinct words
      and nothing else. And the Postgres configuration may stem and fold accents (``portuguese``,
      ``unaccent``), which this does not model beyond its accent fold. A test that needs the
      real lexical VALUE belongs on the Postgres leg."""
    if not query_terms:
        return 0.0
    return len(query_terms & _doc_terms(content)) / len(query_terms)


class InMemoryDocumentStore:
    """Reference ``DocumentStore`` — the executable statement of the rules in
    ``cogno_engram.documents``, and the double a host codes against.

    No ``await`` happens inside a method, so every method is atomic with respect to every other
    coroutine: the swap in :meth:`commit_version` cannot be observed half-done, which is the
    property the Postgres adapter buys with a transaction.

    The originals live in their OWN mapping, apart from the rows a search walks — the same
    separation the Postgres adapter makes with a table of their own — so no search can touch
    them. :attr:`original_reads` counts every read of that mapping, for the test that proves it.
    """

    def __init__(self, *, embedding_dim: int = 768,
                 max_original_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self.embedding_dim = int(embedding_dim)
        self.max_original_bytes = int(max_original_bytes)
        self._docs: dict[str, _DocRow] = {}
        self._versions: dict[tuple[str, int], _VersionRow] = {}
        self._chunks: dict[tuple[str, int], dict[int, KbChunk]] = {}
        # (document, version) → (owner_key, bytes). The OWNER rides with the bytes, so a count
        # by owner sees an original whose document row is gone — the leftover a purge must not
        # leave, which a count THROUGH the documents could never see.
        self._originals: dict[tuple[str, int], tuple[str, bytes]] = {}
        self._tombstones: list[KbTombstone] = []
        self.original_reads = 0

    # ── rows → records ───────────────────────────────────────────────────
    def _version(self, row: Optional[_VersionRow]) -> Optional[KbVersion]:
        if row is None:
            return None
        return KbVersion(document_id=row.document_id, version=row.version, state=row.state,
                         sha256=row.sha256, embed_model=row.embed_model, reason=row.reason,
                         size_bytes=row.size_bytes, pages=row.pages, chunks=row.chunks,
                         has_original=(row.document_id, row.version) in self._originals,
                         created_at=row.created_at, finished_at=row.finished_at)

    def _versions_of(self, document_id: str) -> "list[_VersionRow]":
        return sorted((v for (d, _), v in self._versions.items() if d == document_id),
                      key=lambda v: v.version)

    def _document(self, row: _DocRow) -> KbDocument:
        versions = self._versions_of(row.id)
        active = self._versions.get((row.id, row.active_version)) \
            if row.active_version is not None else None
        return KbDocument(owner_key=row.owner_key, id=row.id, title=row.title,
                          profiles=tuple(row.profiles), media_type=row.media_type,
                          active=self._version(active),
                          latest=self._version(versions[-1] if versions else None),
                          created_at=row.created_at, updated_at=row.updated_at)

    def _owned(self, owner_key: str, document_id: str) -> Optional[_DocRow]:
        row = self._docs.get(str(document_id or ""))
        return row if row is not None and row.owner_key == owner_key else None

    def _drop_version(self, document_id: str, version: int) -> None:
        self._versions.pop((document_id, version), None)
        self._chunks.pop((document_id, version), None)
        self._originals.pop((document_id, version), None)

    # ── management ───────────────────────────────────────────────────────
    async def create_document(self, owner_key: str, *, title: str, profiles, media_type: str,
                              max_documents: Optional[int] = None) -> KbDocument:
        require_owner(owner_key)
        if media_type not in VALID_MEDIA_TYPES:
            raise ValueError(f"unsupported media type {media_type!r}")
        if max_documents is not None and \
                sum(1 for d in self._docs.values() if d.owner_key == owner_key) >= max_documents:
            raise DocumentLimitReached(f"owner already holds {max_documents} documents")
        now = _now()
        row = _DocRow(owner_key=owner_key, id=str(uuid4()), title=" ".join(str(title or "").split()),
                      profiles=sanitize_profiles(profiles), media_type=media_type,
                      created_at=now, updated_at=now)
        self._docs[row.id] = row
        return self._document(row)

    async def get_document(self, owner_key: str, document_id: str) -> Optional[KbDocument]:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        return self._document(row) if row is not None else None

    async def list_documents(self, owner_key: str) -> "list[KbDocument]":
        require_owner(owner_key)
        rows = sorted((d for d in self._docs.values() if d.owner_key == owner_key),
                      key=lambda d: (d.created_at or _now(), d.id))
        return [self._document(r) for r in rows]

    async def set_profiles(self, owner_key: str, document_id: str, profiles) -> bool:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        if row is None:
            return False
        row.profiles = sanitize_profiles(profiles)
        row.updated_at = _now()
        return True

    async def get_original(self, owner_key: str, document_id: str, *,
                           version: Optional[int] = None) -> Optional[bytes]:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        if row is None:
            return None
        v = row.active_version if version is None else int(version)
        if v is None:
            return None
        self.original_reads += 1
        kept = self._originals.get((row.id, v))
        return kept[1] if kept is not None else None

    # ── ingestion ────────────────────────────────────────────────────────
    async def begin_version(self, owner_key: str, document_id: str, *, sha256: str,
                            embed_model: str, size_bytes: int,
                            original: Optional[bytes] = None) -> Optional[KbVersion]:
        require_owner(owner_key)
        require_model(embed_model, self.embedding_dim)
        if original is not None and len(original) > self.max_original_bytes:
            raise OriginalTooLarge(f"original of {len(original)} bytes is over the "
                                   f"{self.max_original_bytes}-byte ceiling")
        row = self._owned(owner_key, document_id)
        if row is None:
            return None
        now = _now()
        for v in self._versions_of(row.id):
            if v.sha256 == sha256 and v.embed_model == embed_model:
                if v.state != KB_READY:
                    v.state, v.reason, v.finished_at = KB_PROCESSING, "", None
                if original is not None and (row.id, v.version) not in self._originals:
                    self._originals[(row.id, v.version)] = (row.owner_key, bytes(original))
                return self._version(v)
        existing = self._versions_of(row.id)
        number = (existing[-1].version + 1) if existing else 1
        v = _VersionRow(document_id=row.id, version=number, state=KB_PROCESSING, sha256=sha256,
                        embed_model=embed_model, size_bytes=int(size_bytes), created_at=now)
        self._versions[(row.id, number)] = v
        if original is not None:
            self._originals[(row.id, number)] = (row.owner_key, bytes(original))
        return self._version(v)

    async def add_chunks(self, owner_key: str, document_id: str, version: int,
                         chunks) -> bool:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        v = self._versions.get((row.id, int(version))) if row is not None else None
        if v is None or v.state != KB_PROCESSING:
            return False
        rows = [(c, require_vector(c.embedding, self.embedding_dim, what="chunk embedding"))
                for c in chunks]                          # validate ALL before staging any
        staged = self._chunks.setdefault((v.document_id, v.version), {})
        for c, emb in rows:
            staged[int(c.ordinal)] = KbChunk(ordinal=int(c.ordinal), content=c.content,
                                              heading_path=tuple(c.heading_path), page=c.page,
                                              embedding=emb)
        return True

    async def commit_version(self, owner_key: str, document_id: str, version: int, *,
                             pages: int) -> str:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        if row is None:
            return COMMIT_DELETED
        v = self._versions.get((row.id, int(version)))
        if v is None:
            return COMMIT_SUPERSEDED              # discarded by a newer version's swap
        newest = self._versions_of(row.id)[-1].version
        if v.version != newest:
            self._drop_version(row.id, v.version)
            return COMMIT_SUPERSEDED
        if v.state == KB_READY and row.active_version == v.version:
            return COMMIT_READY
        if v.state != KB_PROCESSING:
            return COMMIT_SUPERSEDED
        staged = self._chunks.get((row.id, v.version), {})
        v.state, v.reason, v.finished_at = KB_READY, "", _now()
        v.pages, v.chunks = int(pages), len(staged)
        row.active_version = v.version
        row.updated_at = v.finished_at
        for old in self._versions_of(row.id):
            if old.version < v.version:
                self._drop_version(row.id, old.version)
        return COMMIT_READY

    async def fail_version(self, owner_key: str, document_id: str, version: int, *,
                           reason: str) -> bool:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        v = self._versions.get((row.id, int(version))) if row is not None else None
        if v is None or v.state == KB_READY:
            return False
        v.state, v.reason, v.finished_at = KB_ERROR, sanitize_reason(reason), _now()
        self._chunks.pop((v.document_id, v.version), None)
        self._originals.pop((v.document_id, v.version), None)
        return True

    # ── removal ──────────────────────────────────────────────────────────
    def _remove(self, row: _DocRow, kind: str, actor: str) -> None:
        versions = tuple(v.version for v in self._versions_of(row.id))
        for number in versions:
            self._drop_version(row.id, number)
        del self._docs[row.id]
        self._tombstones.append(KbTombstone(owner_key=row.owner_key, document_id=row.id,
                                            versions=versions, kind=kind,
                                            actor=str(actor or ""), removed_at=_now()))

    async def delete_document(self, owner_key: str, document_id: str, *, actor: str = "") -> bool:
        require_owner(owner_key)
        row = self._owned(owner_key, document_id)
        if row is None:
            return False
        self._remove(row, TOMBSTONE_DELETED, actor)
        return True

    async def purge_owner_subtree(self, owner_prefix: str, *, actor: str = "") -> int:
        require_owner(owner_prefix)
        rows = [d for d in self._docs.values() if owner_in_subtree(d.owner_key, owner_prefix)]
        for row in rows:
            self._remove(row, TOMBSTONE_PURGED, actor)
        return len(rows)

    async def tombstones(self, owner_prefix: str, *, limit: int = 100) -> "list[KbTombstone]":
        require_owner(owner_prefix)
        rows = [(i, t) for i, t in enumerate(self._tombstones)
                if owner_in_subtree(t.owner_key, owner_prefix)]
        # newest first, then most recently written — the Postgres `removed_at DESC, id DESC`
        rows.sort(key=lambda it: (it[1].removed_at or _now(), it[0]), reverse=True)
        return [t for _, t in rows[:limit]]

    async def prune_tombstones(self, *, before: datetime) -> int:
        keep = [t for t in self._tombstones if (t.removed_at or _now()) >= before]
        gone = len(self._tombstones) - len(keep)
        self._tombstones = keep
        return gone

    async def stored_original_bytes(self, owner_prefix: str) -> int:
        require_owner(owner_prefix)
        return sum(len(data) for owner, data in self._originals.values()
                   if owner_in_subtree(owner, owner_prefix))

    # ── the reader path ──────────────────────────────────────────────────
    def _served(self, owner_key: str, profile: str) -> "list[tuple[_DocRow, _VersionRow]]":
        """Every (document, served version) ``profile`` may read — THE filter of the reader path:
        this owner, a served version, that version ``ready``, the profile published to."""
        out = []
        for row in self._docs.values():
            if row.owner_key != owner_key or row.active_version is None:
                continue
            if not profile_can_read(profile, row.profiles):
                continue
            v = self._versions.get((row.id, row.active_version))
            if v is None or v.state != KB_READY:
                continue
            out.append((row, v))
        return out

    async def readable_documents(self, owner_key: str, *, profile: str) -> "list[KbDocument]":
        require_owner(owner_key)
        require_profile(profile)
        rows = sorted(self._served(owner_key, profile),
                      key=lambda rv: (rv[0].created_at or _now(), rv[0].id))
        return [self._document(r) for r, _ in rows]

    async def search(self, owner_key: str, *, profile: str, text: str,
                     vector: Optional[list[float]] = None, embed_model: Optional[str] = None,
                     limit: int = 5,
                     weights: Optional[HybridWeights] = None) -> KbSearchResult:
        require_owner(owner_key)
        require_profile(profile)
        if vector is not None and embed_model is None:
            raise ValueError("a query vector needs the label of the model that made it")
        if embed_model is not None:
            require_model(embed_model, self.embedding_dim)
        query = require_vector(vector, self.embedding_dim, what="query vector") \
            if vector is not None else None
        w = weights or HybridWeights()
        terms = _doc_terms(text)
        served = self._served(owner_key, profile)
        models = {v.embed_model for _, v in served}
        unavailable = sorted(models if query is None else models - {embed_model})
        use_vector = query is not None and not unavailable        # all or none
        hits = []
        for row, v in served:
            for chunk in self._chunks.get((row.id, v.version), {}).values():
                vs = clamp_unit(_doc_cosine(query, chunk.embedding)) \
                    if use_vector and query is not None and chunk.embedding is not None else None
                ls = clamp_unit(_doc_lexical(terms, chunk.content))
                if vs is None and ls <= 0:
                    continue
                hits.append(KbHit(
                    id=chunk_id(row.id, v.version, chunk.ordinal), document_id=row.id,
                    version=v.version, ordinal=chunk.ordinal, title=row.title,
                    heading_path=tuple(chunk.heading_path), page=chunk.page,
                    content=chunk.content, embed_model=v.embed_model, vector_score=vs,
                    lexical_score=ls,
                    score=hybrid_score(vs, ls, vector_weight=w.vector, lexical_weight=w.lexical)))
        hits.sort(key=hit_order_key)
        return KbSearchResult(
            hits=tuple(hits[:max(0, int(limit))]),
            degradations=(KB_EMBED_SPACE_UNAVAILABLE,) if unavailable else (),
            models_unavailable=tuple(unavailable))

    # ── maintenance ──────────────────────────────────────────────────────
    async def stale_documents(self, *, embed_model: str, limit: int = 100) -> "list[KbDocument]":
        require_model(embed_model, self.embedding_dim)
        rows = []
        for row in self._docs.values():
            v = self._versions.get((row.id, row.active_version)) \
                if row.active_version is not None else None
            if v is not None and v.embed_model != embed_model:
                rows.append(row)
        rows.sort(key=lambda d: (d.created_at or _now(), d.id))
        return [self._document(r) for r in rows[:limit]]
