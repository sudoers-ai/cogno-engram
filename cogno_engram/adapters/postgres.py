"""
cogno_engram.adapters.postgres — the reference Postgres + pgvector adapter.

Ported clean-room from the parent's ``memory/postgres_store.py`` +
``core/db_knowledge.py``, with the business identity (``tenant_id``/
``identity_id``) collapsed into a single opaque ``scope`` column. Implements
``MemoryStore`` (+ ``SupportsVectorSearch``) and ``KnowledgeGraph`` over one
Postgres database:

  * hybrid memory retrieval — ``0.60·vector + 0.40·BM25 + 0.05·feedback``
    (pgvector ``<=>`` + ``ts_rank_cd`` over a generated ``tsvector``);
  * recursive-CTE multi-hop graph walk (loop-bounded by depth);
  * a session advisory lock (``pg_advisory_lock`` keyed on a sha256 of the id);
  * optional PII masking on write.

``psycopg`` (v3) + ``pgvector`` are required (``pip install
"cogno-engram[postgres]"``). Call :func:`ensure_schema` once to create the
idempotent DDL (tables + indexes; HASH-by-scope partitioning is left opt-in).
"""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from cogno_engram.types import (
    EDGE_ACCEPTED,
    EDGE_PROPOSED,
    require_edge_status,
    sanitize_edge_status,
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

# Default embedding width — nomic-embed-text (the parent's default embedder).
DEFAULT_EMBEDDING_DIM = 768
# Text-search config for BM25; Portuguese is the parent's primary language.
DEFAULT_TS_CONFIG = "portuguese"

# ``ts_config`` is interpolated into SQL by name (a regconfig identifier cannot be
# a bound parameter), so it MUST be a bare SQL identifier — never tenant-supplied
# free text. Accept an optionally schema-qualified lowercase identifier only; this
# allows a host's custom dictionary (e.g. ``my_schema.unaccent_pt``) while making
# injection via the config impossible.
_TS_CONFIG_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _validate_ts_config(ts_config: str) -> str:
    if not _TS_CONFIG_RE.match(ts_config or ""):
        raise ValueError(
            f"invalid ts_config {ts_config!r}: must be a bare SQL identifier "
            "(letters/digits/underscore, optionally schema-qualified) — it is "
            "interpolated into SQL and must not be tenant-supplied free text")
    return ts_config


def _require_scope(scope: str) -> str:
    if not scope or not scope.strip():
        raise ValueError("scope must be a non-empty string (engram isolates every row by scope)")
    return scope


def _vec(v: Optional[list[float]]) -> Optional[str]:
    """Render an embedding as a pgvector literal, e.g. ``[0.1,0.2]``."""
    return None if v is None else "[" + ",".join(repr(float(x)) for x in v) + "]"


def _mask_pii(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b", "[CPF MASKED]", text)
    text = re.sub(r"\b\d{11}\b", "[CPF MASKED]", text)
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL MASKED]", text)
    text = re.sub(r"\b(?:\+55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b", "[PHONE MASKED]", text)
    return text


async def ensure_schema(conn, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                        ts_config: str = DEFAULT_TS_CONFIG,
                        partition_by_scope: bool = False, partitions: int = 8) -> None:
    """Create the engram schema idempotently (extension + tables + indexes).

    No alembic required — this is the zero-friction path. Migrations can be
    layered on top by a host that wants versioned schema.

    ``partition_by_scope`` opts the high-volume ``turns`` and ``memories`` tables
    into HASH(scope) partitioning over a fixed number of buckets (``partitions``),
    the generic equivalent of the parent's LIST(tenant) partitioning — zero DDL
    per new scope. Every query carries ``scope`` so partition pruning applies.
    """
    ts_config = _validate_ts_config(ts_config)  # interpolated into the tsvector DDL
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # sessions / knowledge_* stay unpartitioned (low volume); turns/memories opt in.
    pk = "PRIMARY KEY (id, scope)" if partition_by_scope else "PRIMARY KEY (id)"
    part = "PARTITION BY HASH (scope)" if partition_by_scope else ""

    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id          uuid PRIMARY KEY,
            scope       text NOT NULL,
            started_at  timestamptz NOT NULL DEFAULT now(),
            ended_at    timestamptz,
            summary     text NOT NULL DEFAULT ''
        )
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS turns (
            id           bigserial,
            scope        text NOT NULL,
            session_id   uuid NOT NULL,
            turn_n       integer NOT NULL,
            user_input   text NOT NULL,
            response     text NOT NULL DEFAULT '',
            feedback     smallint NOT NULL DEFAULT 0,
            goal         text NOT NULL DEFAULT '',
            goal_status  text NOT NULL DEFAULT '',
            sentiment    text NOT NULL DEFAULT '',
            domains      text[] NOT NULL DEFAULT '{{}}',
            pii_types    text[] NOT NULL DEFAULT '{{}}',
            created_at   timestamptz NOT NULL DEFAULT now(),
            {pk},
            UNIQUE (scope, session_id, turn_n)
        ) {part}
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS memories (
            id             uuid,
            scope          text NOT NULL,
            category       text NOT NULL,
            content        text NOT NULL,
            confidence     real NOT NULL DEFAULT 1.0,
            feedback_score real NOT NULL DEFAULT 0.0,
            embedding      vector({embedding_dim}),
            tsv            tsvector GENERATED ALWAYS AS (to_tsvector('{ts_config}', content)) STORED,
            created_at     timestamptz NOT NULL DEFAULT now(),
            {pk},
            UNIQUE (scope, category, content)
        ) {part}
        """
    )
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS turn_traces (
            id           bigserial,
            scope        text NOT NULL,
            session_id   uuid NOT NULL,
            turn_n       integer NOT NULL,
            trace        jsonb NOT NULL DEFAULT '{{}}'::jsonb,
            created_at   timestamptz NOT NULL DEFAULT now(),
            {pk},
            UNIQUE (scope, session_id, turn_n)
        ) {part}
        """
    )
    if partition_by_scope:
        for tbl in ("turns", "memories", "turn_traces"):
            for k in range(partitions):
                await conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {tbl}_p{k} PARTITION OF {tbl} "
                    f"FOR VALUES WITH (MODULUS {partitions}, REMAINDER {k})")
    await conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS knowledge_nodes (
            id          bigserial PRIMARY KEY,
            scope       text NOT NULL,
            label       text NOT NULL,
            node_type   text NOT NULL DEFAULT 'CONCEPT',
            attributes  jsonb NOT NULL DEFAULT '{{}}',
            embedding   vector({embedding_dim}),
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    await conn.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            id             bigserial PRIMARY KEY,
            scope          text NOT NULL,
            source_id      bigint NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            target_id      bigint NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            relation       text NOT NULL,
            confidence     real NOT NULL DEFAULT 1.0,
            source_session text NOT NULL DEFAULT '',
            attributes     jsonb NOT NULL DEFAULT '{}',
            status         text NOT NULL DEFAULT 'accepted',
            created_at     timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source_id, target_id, relation)
        )
        """
    )
    # ── migration for databases created before edge curation ──────────────
    # `CREATE TABLE IF NOT EXISTS` above is a NO-OP against a live table, so a deployment that
    # already has a graph would get the new code and none of the columns. Additive and
    # idempotent: the DEFAULT backfills every existing edge as `accepted`, which is what it
    # was — nothing a host already asserted becomes unreviewed overnight.
    # Asked BEFORE altering: `ADD COLUMN IF NOT EXISTS` is a no-op when the column is there, but
    # it still takes an ACCESS EXCLUSIVE lock on `knowledge_edges` — so several workers booting
    # against a live graph queue every reader behind a statement that changes nothing. The
    # catalogue read is cheap and takes no lock.
    # Existence asked per column with a bare `SELECT 1`, whose ROW is truthy whatever row
    # factory the caller configured — `ensure_schema` runs on a plain connection here and on a
    # `dict_row` one elsewhere, and reading a column by NAME crashed on the tuple shape.
    for column, ddl in (
        ("attributes",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS attributes jsonb NOT NULL DEFAULT '{}'"),
        ("status",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'accepted'"),
    ):
        cur = await conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'knowledge_edges' AND column_name = %s", (column,))
        if await cur.fetchone() is None:
            await conn.execute(ddl)

    # ── indexes (see engram-blueprint indexing strategy) ──
    stmts = [
        # Case-insensitive node identity (matches the in-memory adapter's dedup).
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_label_type "
        "ON knowledge_nodes (scope, lower(label), node_type)",
        # The curation queue reads one status within one scope; the prompt walk reads the other.
        "CREATE INDEX IF NOT EXISTS idx_edges_scope_status ON knowledge_edges (scope, status)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_scope_time ON sessions (scope, started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_turns_scope_time ON turns (scope, created_at DESC, id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, turn_n)",
        "CREATE INDEX IF NOT EXISTS idx_turn_traces_session ON turn_traces (session_id, turn_n)",
        "CREATE INDEX IF NOT EXISTS idx_memories_scope_time ON memories (scope, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING gin (tsv)",
        "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories "
        "USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_embedding ON knowledge_nodes "
        "USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS idx_edges_session ON knowledge_edges (scope, source_session)",
        "CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges (source_id)",
        "CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges (target_id)",
    ]
    for stmt in stmts:
        await conn.execute(stmt)


class _PgBase:
    """Shared connection plumbing for the Postgres adapters."""

    def __init__(self, *, dsn: Optional[str] = None, pool=None,
                 ts_config: str = DEFAULT_TS_CONFIG, mask_pii: bool = False) -> None:
        if not dsn and pool is None:
            raise ValueError("provide either dsn= or pool=")
        self._dsn = dsn
        self._pool = pool
        self._ts = _validate_ts_config(ts_config)
        self._mask_pii = mask_pii

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[Any]:
        # Typed Any: the dict_row factory makes rows dicts at runtime, which the
        # static psycopg row type (tuple) doesn't reflect.
        if self._pool is not None:
            async with self._pool.connection() as conn:
                prev = conn.row_factory
                conn.row_factory = dict_row
                try:
                    yield conn
                finally:
                    # restore the pooled connection's factory so a later borrower that expects
                    # tuple rows isn't handed dicts (a shared-pool consumer must not inherit this).
                    conn.row_factory = prev
        else:
            assert self._dsn is not None  # __init__ guarantees dsn when pool is None
            conn = await psycopg.AsyncConnection.connect(
                self._dsn, autocommit=True, row_factory=dict_row)
            try:
                yield conn
            finally:
                await conn.close()


class PostgresStore(_PgBase):
    """Reference ``MemoryStore`` + ``SupportsVectorSearch``."""

    def supports_vector(self) -> bool:
        return True

    # ── sessions ─────────────────────────────────────────────────────────
    async def create_session(self, scope: str) -> Session:
        _require_scope(scope)
        sid = str(uuid4())
        async with self._conn() as conn:
            cur = await conn.execute(
                "INSERT INTO sessions (id, scope) VALUES (%s, %s) RETURNING started_at",
                (sid, scope))
            row = await cur.fetchone()
        return Session(id=sid, scope=scope, started_at=row["started_at"])

    async def get_session(self, session_id: str, *, scope: str = "") -> Optional[Session]:
        async with self._conn() as conn:
            if scope:
                cur = await conn.execute(
                    "SELECT id, scope, started_at, ended_at, summary FROM sessions "
                    "WHERE id = %s AND scope = %s", (session_id, scope))
            else:
                cur = await conn.execute(
                    "SELECT id, scope, started_at, ended_at, summary FROM sessions WHERE id = %s",
                    (session_id,))
            row = await cur.fetchone()
        if not row:
            return None
        return Session(id=str(row["id"]), scope=row["scope"], started_at=row["started_at"],
                       ended_at=row["ended_at"], summary=row["summary"])

    async def close_session(self, session_id: str, *, summary: str = "", scope: str = "") -> None:
        async with self._conn() as conn:
            if scope:
                # UPSERT: a host that only save_turn()s has no sessions row to update — insert a
                # closed one (keyed by the same session id) so the janitor's idle scan skips it.
                # `sessions` is keyed by id alone, so a colliding id from ANOTHER scope would take
                # this scope's summary (the cross-scope write the read-side fix was written to
                # stop). Guard the conflict update with the scope.
                await conn.execute(
                    "INSERT INTO sessions (id, scope, ended_at, summary) "
                    "VALUES (%s, %s, now(), %s) "
                    "ON CONFLICT (id) DO UPDATE SET ended_at = now(), summary = EXCLUDED.summary "
                    "WHERE sessions.scope = EXCLUDED.scope",
                    (session_id, scope, summary))
            else:
                await conn.execute(
                    "UPDATE sessions SET ended_at = now(), summary = %s WHERE id = %s",
                    (summary, session_id))

    async def idle_sessions(self, *, idle_seconds: int = 1800,
                            limit: int = 100) -> list[Session]:
        # Turn-derived (the host persists turns without a sessions row): group the turns table by
        # session, take the last activity, and keep those idle past the cutoff and not already
        # consolidated UP TO THEIR CURRENT END.
        #
        # That last clause is the whole point, and its absence froze long-term memory. A closed
        # session used to be excluded forever — but a host that derives `session_id` from
        # (tenant, channel, sender), as a messaging gateway must so an out-of-band message lands
        # in the contact's own thread, NEVER mints a second session for that contact. So the
        # first idle period consolidated the conversation as it stood and every later turn was
        # invisible to Tier 3 for good. Measured on a live box (2026-08): all three real
        # conversations were frozen — 20 of 22, 10 of 13 and 8 of 10 turns arrived after their
        # session had been declared over, and the narrative the host injects as EARLIER CONTEXT
        # still described turn 2. The model then acted on a two-day-old snapshot and re-opened a
        # conversation that had long moved on.
        #
        # Re-picking on `max(turn) > ended_at` is self-limiting: consolidation re-closes with a
        # fresh `ended_at`, so a session comes back only once per new burst of turns, not once
        # per tick.
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT t.session_id AS id, t.scope AS scope, "
                "       min(t.created_at) AS started_at, max(t.created_at) AS last_activity "
                "FROM turns t LEFT JOIN sessions s ON s.id = t.session_id "
                "WHERE s.id IS NULL OR s.ended_at IS NULL OR t.created_at > s.ended_at "
                "GROUP BY t.session_id, t.scope "
                "HAVING max(t.created_at) < now() - make_interval(secs => %s) "
                "ORDER BY max(t.created_at) ASC LIMIT %s",
                (idle_seconds, limit))
            rows = await cur.fetchall()
        return [Session(id=str(r["id"]), scope=r["scope"], started_at=r["started_at"])
                for r in rows]

    async def recent_sessions(self, scope: str, *, limit: int = 5) -> list[Session]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id, scope, started_at, ended_at, summary FROM sessions "
                "WHERE scope = %s ORDER BY started_at DESC LIMIT %s", (scope, limit))
            rows = await cur.fetchall()
        return [Session(id=str(r["id"]), scope=r["scope"], started_at=r["started_at"],
                        ended_at=r["ended_at"], summary=r["summary"]) for r in rows]

    async def get_active_session(self, scope: str, *,
                                 within_seconds: int = 12 * 3600) -> Optional[Session]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id, scope, started_at, ended_at, summary FROM sessions "
                "WHERE scope = %s AND ended_at IS NULL "
                "AND started_at > now() - make_interval(secs => %s) "
                "ORDER BY started_at DESC LIMIT 1", (scope, within_seconds))
            row = await cur.fetchone()
        if not row:
            return None
        return Session(id=str(row["id"]), scope=row["scope"], started_at=row["started_at"],
                       ended_at=row["ended_at"], summary=row["summary"])

    # ── turns ────────────────────────────────────────────────────────────
    async def save_turn(self, turn: TurnRecord) -> None:
        _require_scope(turn.scope)
        user_input, response = turn.user_input, turn.response
        if self._mask_pii and turn.pii_types:
            user_input, response = _mask_pii(user_input), _mask_pii(response)
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO turns
                   (scope, session_id, turn_n, user_input, response, feedback,
                    goal, goal_status, sentiment, domains, pii_types)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (scope, session_id, turn_n) DO NOTHING""",
                (turn.scope, turn.session_id, turn.turn_n, user_input, response, turn.feedback,
                 turn.goal, turn.goal_status, turn.sentiment, turn.domains, turn.pii_types))

    async def update_turn_response(self, scope: str, session_id: str, turn_n: int,
                                   response: str) -> None:
        _require_scope(scope)
        if self._mask_pii:
            response = _mask_pii(response)
        async with self._conn() as conn:
            await conn.execute(
                "UPDATE turns SET response = %s WHERE scope = %s AND session_id = %s AND turn_n = %s",
                (response, scope, session_id, turn_n))

    async def load_turns(self, session_id: str, *, scope: str = "") -> list[TurnRecord]:
        cols = ("SELECT scope, session_id, turn_n, user_input, response, feedback, goal, "
                "goal_status, sentiment, domains, pii_types, created_at FROM turns ")
        async with self._conn() as conn:
            if scope:
                cur = await conn.execute(
                    cols + "WHERE session_id = %s AND scope = %s ORDER BY turn_n ASC",
                    (session_id, scope))
            else:
                cur = await conn.execute(
                    cols + "WHERE session_id = %s ORDER BY turn_n ASC", (session_id,))
            rows = await cur.fetchall()
        return [self._row_to_turn(r) for r in rows]

    async def turn_count(self, session_id: str, *, scope: str = "") -> int:
        async with self._conn() as conn:
            if scope:
                cur = await conn.execute(
                    "SELECT count(*) AS c FROM turns WHERE session_id = %s AND scope = %s",
                    (session_id, scope))
            else:
                cur = await conn.execute(
                    "SELECT count(*) AS c FROM turns WHERE session_id = %s", (session_id,))
            row = await cur.fetchone()
        return int(row["c"])

    # ── turn traces (own table) ──────────────────────────────────────────
    async def save_turn_trace(self, trace: "TurnTrace") -> None:
        _require_scope(trace.scope)
        async with self._conn() as conn:
            await conn.execute(
                # ``created_at`` is honoured when the caller sets it (a backfill, an import,
                # a test seeding history); absent, the column default stamps the row. The
                # in-memory adapter always did this — the Postgres one silently dropped it,
                # which made every imported trace "now" and any time-window read over them a
                # lie.
                "INSERT INTO turn_traces (scope, session_id, turn_n, trace, created_at) "
                "VALUES (%s, %s, %s, %s::jsonb, COALESCE(%s, now())) "
                "ON CONFLICT (scope, session_id, turn_n) DO UPDATE SET trace = EXCLUDED.trace",
                (trace.scope, trace.session_id, trace.turn_n, json.dumps(trace.trace),
                 trace.created_at))

    async def traces_for_session(self, session_id: str, *, scope: str = "") -> list["TurnTrace"]:
        async with self._conn() as conn:
            if scope:
                cur = await conn.execute(
                    "SELECT scope, session_id, turn_n, trace, created_at FROM turn_traces "
                    "WHERE session_id = %s AND scope = %s ORDER BY turn_n ASC", (session_id, scope))
            else:
                cur = await conn.execute(
                    "SELECT scope, session_id, turn_n, trace, created_at "
                    "FROM turn_traces WHERE session_id = %s ORDER BY turn_n ASC", (session_id,))
            rows = await cur.fetchall()
        return [TurnTrace(session_id=str(r["session_id"]), scope=r["scope"], turn_n=r["turn_n"],
                          trace=r["trace"] or {}, created_at=r["created_at"]) for r in rows]

    async def recent_turns(self, scope: str, *, limit: int = 5,
                           exclude_session: str = "") -> list[TurnRecord]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT scope, session_id, turn_n, user_input, response, feedback, goal, "
                "goal_status, sentiment, domains, pii_types, created_at "
                "FROM turns WHERE scope = %s AND session_id::text != %s "
                "ORDER BY created_at DESC, id DESC LIMIT %s", (scope, exclude_session, limit))
            rows = await cur.fetchall()
        return [self._row_to_turn(r) for r in rows]

    async def set_feedback(self, scope: str, session_id: str, turn_n: int, feedback: int) -> None:
        _require_scope(scope)
        async with self._conn() as conn:
            await conn.execute(
                "UPDATE turns SET feedback = %s WHERE scope = %s AND session_id = %s AND turn_n = %s",
                (feedback, scope, session_id, turn_n))

    @staticmethod
    def _subtree_like(scope_prefix: str) -> str:
        # match descendants ``prefix/…``; escape LIKE metacharacters (scope is opaque)
        esc = scope_prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return esc + "/%"

    # WHERE the scope IS the prefix OR a ``prefix/…`` descendant (subtree). ESCAPE '\' pairs the
    # escaping in _subtree_like. Not partition-pruned — an admin/maintenance read.
    _SUBTREE = "(scope = %s OR scope LIKE %s ESCAPE '\\')"

    async def admin_turns(self, scope_prefix: str, *, limit: int = 30,
                          offset: int = 0) -> "tuple[list[TurnRecord], int]":
        _require_scope(scope_prefix)
        like = self._subtree_like(scope_prefix)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT scope, session_id, turn_n, user_input, response, feedback, goal, "
                "goal_status, sentiment, domains, pii_types, created_at "
                f"FROM turns WHERE {self._SUBTREE} "
                "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                (scope_prefix, like, limit, offset))
            rows = await cur.fetchall()
            ccur = await conn.execute(
                f"SELECT count(*) AS c FROM turns WHERE {self._SUBTREE}", (scope_prefix, like))
            crow = await ccur.fetchone()
        total = crow["c"] if crow else 0
        return [self._row_to_turn(r) for r in rows], int(total)

    async def admin_scopes(self, scope_prefix: str) -> list[str]:
        _require_scope(scope_prefix)
        like = self._subtree_like(scope_prefix)
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT DISTINCT scope FROM turns WHERE {self._SUBTREE} ORDER BY scope",
                (scope_prefix, like))
            rows = await cur.fetchall()
        return [r["scope"] for r in rows]

    async def admin_traces(self, scope_prefix: str, *, since: Optional[datetime] = None,
                           limit: int = 1000, offset: int = 0) -> "tuple[list[TurnTrace], int]":
        _require_scope(scope_prefix)
        like = self._subtree_like(scope_prefix)
        where = f"{self._SUBTREE}" + (" AND created_at >= %s" if since is not None else "")
        params: tuple = (scope_prefix, like) + ((since,) if since is not None else ())
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT scope, session_id, turn_n, trace, created_at FROM turn_traces "
                f"WHERE {where} ORDER BY created_at DESC, session_id DESC, turn_n DESC "
                "LIMIT %s OFFSET %s", params + (limit, offset))
            rows = await cur.fetchall()
            ccur = await conn.execute(
                f"SELECT count(*) AS c FROM turn_traces WHERE {where}", params)
            crow = await ccur.fetchone()
        total = int(crow["c"]) if crow else 0
        return ([TurnTrace(session_id=str(r["session_id"]), scope=r["scope"], turn_n=r["turn_n"],
                           trace=r["trace"] or {}, created_at=r["created_at"]) for r in rows],
                total)

    @staticmethod
    def _row_to_turn(r: dict) -> TurnRecord:
        return TurnRecord(
            session_id=str(r["session_id"]), scope=r["scope"], turn_n=r["turn_n"],
            user_input=r["user_input"], response=r["response"], feedback=r["feedback"],
            goal=r["goal"], goal_status=r["goal_status"], sentiment=r["sentiment"],
            domains=list(r["domains"] or []), pii_types=list(r["pii_types"] or []),
            created_at=r["created_at"])

    # ── memories ─────────────────────────────────────────────────────────
    async def save_memory(self, memory: MemoryRecord) -> None:
        _require_scope(memory.scope)
        mid = memory.id or str(uuid4())
        async with self._conn() as conn:
            await conn.execute(
                """INSERT INTO memories (id, scope, category, content, confidence, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s::vector)
                   ON CONFLICT (scope, category, content) DO UPDATE SET
                       confidence = EXCLUDED.confidence,
                       embedding  = COALESCE(EXCLUDED.embedding, memories.embedding)""",
                (mid, memory.scope, memory.category, memory.content, memory.confidence,
                 _vec(memory.embedding)))

    async def load_memories(self, scope: str, *, query: Optional[RetrievalQuery] = None,
                            limit: int = 50,
                            weights: Optional[HybridWeights] = None) -> list[MemoryRecord]:
        _require_scope(scope)
        w = weights or HybridWeights()
        where = "WHERE scope = %s"
        params: list = [scope]
        if query and query.categories:
            where += " AND category = ANY(%s)"
            params.append(query.categories)

        q_emb = query.embedding if query else None
        q_txt = query.text if (query and query.text and query.text.strip()) else None
        tsq = f"plainto_tsquery('{self._ts}', %s)"

        if q_emb is not None and q_txt is not None:
            sql = (
                f"SELECT id, scope, category, content, confidence, feedback_score, created_at, "
                f"  ({w.vector} * (1.0 - (embedding <=> %s::vector)) "
                f"   + {w.lexical} * ts_rank_cd(tsv, {tsq}) "
                f"   + COALESCE(feedback_score, 0) * {w.feedback}) AS score "
                f"FROM memories {where} AND embedding IS NOT NULL "
                f"ORDER BY score DESC LIMIT %s")
            full = [_vec(q_emb), q_txt] + params + [limit]
        elif q_emb is not None:
            sql = (f"SELECT id, scope, category, content, confidence, feedback_score, created_at "
                   f"FROM memories {where} AND embedding IS NOT NULL "
                   f"ORDER BY (embedding <=> %s::vector) - COALESCE(feedback_score,0)*0.1 ASC LIMIT %s")
            full = params + [_vec(q_emb), limit]
        elif q_txt is not None:
            sql = (f"SELECT id, scope, category, content, confidence, feedback_score, created_at "
                   f"FROM memories {where} AND tsv @@ {tsq} "
                   f"ORDER BY ts_rank_cd(tsv, {tsq}) DESC LIMIT %s")
            full = params + [q_txt, q_txt, limit]
        else:
            sql = (f"SELECT id, scope, category, content, confidence, feedback_score, created_at "
                   f"FROM memories {where} ORDER BY created_at DESC LIMIT %s")
            full = params + [limit]

        async with self._conn() as conn:
            cur = await conn.execute(sql, full)
            rows = await cur.fetchall()
        return [MemoryRecord(scope=r["scope"], category=r["category"], content=r["content"],
                             confidence=r["confidence"], feedback_score=r["feedback_score"],
                             created_at=r["created_at"], id=str(r["id"])) for r in rows]

    async def scan_memories(self, scope: str, *, after_id: "Optional[str]" = None,
                            limit: int = 1000) -> list[MemoryRecord]:
        _require_scope(scope)
        sql = ("SELECT id, scope, category, content, confidence, feedback_score, created_at "
               "FROM memories WHERE scope = %s")
        params: list = [scope]
        if after_id is not None:
            sql += " AND id > %s"
            params.append(after_id)
        # ORDER BY the cursor column, so the next page resumes exactly where this one ended
        # regardless of what was inserted meanwhile.
        sql += " ORDER BY id ASC LIMIT %s"
        params.append(limit)
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [MemoryRecord(scope=r["scope"], category=r["category"], content=r["content"],
                             confidence=r["confidence"], feedback_score=r["feedback_score"],
                             created_at=r["created_at"], id=str(r["id"])) for r in rows]

    async def adjust_feedback_score(self, scope: str, query_text: str, delta: float,
                                    *, limit: int = 10) -> int:
        _require_scope(scope)
        tsq = f"plainto_tsquery('{self._ts}', %s)"
        async with self._conn() as conn:
            cur = await conn.execute(
                f"""UPDATE memories SET feedback_score =
                        GREATEST(-10, LEAST(10, COALESCE(feedback_score,0) + %s))
                    WHERE id IN (
                        SELECT id FROM memories WHERE scope = %s AND tsv @@ {tsq} LIMIT %s)""",
                (delta, scope, query_text, limit))
            return cur.rowcount

    async def memory_count(self, scope: str) -> int:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS c FROM memories WHERE scope = %s", (scope,))
            row = await cur.fetchone()
        return int(row["c"])

    async def delete_memories(self, scope: str, *, older_than: Optional[datetime] = None,
                              category: Optional[str] = None,
                              max_confidence: Optional[float] = None) -> int:
        _require_scope(scope)
        sql = "DELETE FROM memories WHERE scope = %s"
        params: list = [scope]
        if older_than is not None:
            sql += " AND created_at < %s"
            params.append(older_than)
        if category is not None:
            sql += " AND category = %s"
            params.append(category)
        if max_confidence is not None:
            sql += " AND confidence <= %s"
            params.append(max_confidence)
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            return cur.rowcount

    async def purge_scope(self, scope: str) -> int:
        _require_scope(scope)
        total = 0
        async with self._conn() as conn:
            # turn_traces + turns + sessions + memories all carry the scope column; drop them all.
            for table in ("turn_traces", "turns", "sessions", "memories"):
                cur = await conn.execute(
                    f"DELETE FROM {table} WHERE scope = %s", (scope,))
                total += cur.rowcount
        return total

    # ── concurrency: pg_advisory_lock keyed on the session id ────────────
    @asynccontextmanager
    async def session_lock(self, scope: str, session_id: str) -> AsyncIterator[None]:
        _require_scope(scope)
        digest = hashlib.sha256(f"{scope}:{session_id}".encode()).digest()
        lock_id = int.from_bytes(digest[:8], "big", signed=True)
        async with self._conn() as conn:
            await conn.execute("SELECT pg_advisory_lock(%s)", (lock_id,))
            try:
                yield
            finally:
                await conn.execute("SELECT pg_advisory_unlock(%s)", (lock_id,))


def _edge_from_row(scope: str, row: Any) -> GraphEdge:
    """One place turns a row into an edge. Written when the third read path appeared: three
    hand-built constructors is how one of them silently stops carrying a new column."""
    return GraphEdge(scope=scope, source=row["source"], target=row["target"],
                     relation=row["relation"], confidence=row["confidence"],
                     source_session=row["source_session"],
                     attributes=row.get("attributes") or {},
                     status=row.get("status") or EDGE_ACCEPTED)


class PostgresKnowledgeGraph(_PgBase):
    """Reference ``KnowledgeGraph`` — typed nodes + edges + recursive-CTE walk.

    The port speaks node *labels*; this adapter resolves them to the integer
    node ids the edge table joins on (creating bare nodes when needed).
    """

    async def upsert_node(self, node: GraphNode) -> int:
        _require_scope(node.scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                """INSERT INTO knowledge_nodes (scope, label, node_type, attributes, embedding)
                   VALUES (%s, %s, %s, %s::jsonb, %s::vector)
                   ON CONFLICT (scope, lower(label), node_type) DO UPDATE SET
                       attributes = knowledge_nodes.attributes || EXCLUDED.attributes,
                       embedding  = COALESCE(EXCLUDED.embedding, knowledge_nodes.embedding),
                       updated_at = now()
                   RETURNING id""",
                (node.scope, node.label, node.node_type, json.dumps(node.attributes),
                 _vec(node.embedding)))
            row = await cur.fetchone()
        return row["id"]

    async def _resolve_node_id(self, conn, scope: str, label: str) -> int:
        cur = await conn.execute(
            "SELECT id FROM knowledge_nodes WHERE scope = %s AND lower(label) = lower(%s) LIMIT 1",
            (scope, label))
        row = await cur.fetchone()
        if row:
            return row["id"]
        # Idempotent insert: two concurrent upsert_edge calls referencing a not-yet-existing
        # label both miss the SELECT above; a bare INSERT then races the uq(scope, lower(label),
        # node_type) index and the loser raises IntegrityError, failing that turn. ON CONFLICT
        # DO NOTHING makes the loser return no row → re-SELECT the winner's id.
        cur = await conn.execute(
            "INSERT INTO knowledge_nodes (scope, label) VALUES (%s, %s) "
            "ON CONFLICT (scope, lower(label), node_type) DO NOTHING RETURNING id",
            (scope, label))
        row = await cur.fetchone()
        if row:
            return row["id"]
        cur = await conn.execute(
            "SELECT id FROM knowledge_nodes WHERE scope = %s AND lower(label) = lower(%s) LIMIT 1",
            (scope, label))
        row = await cur.fetchone()
        return row["id"]

    async def upsert_edge(self, edge: GraphEdge) -> None:
        _require_scope(edge.scope)
        async with self._conn() as conn:
            src = await self._resolve_node_id(conn, edge.scope, edge.source)
            tgt = await self._resolve_node_id(conn, edge.scope, edge.target)
            await conn.execute(
                """INSERT INTO knowledge_edges
                   (scope, source_id, target_id, relation, confidence, source_session,
                    attributes, status)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                   ON CONFLICT (source_id, target_id, relation) DO UPDATE SET
                       confidence     = GREATEST(knowledge_edges.confidence, EXCLUDED.confidence),
                       source_session = EXCLUDED.source_session,
                       -- merge, never replace: a re-extraction must not wipe what a human
                       -- typed. And a PROPOSAL cannot modify a VERDICT in any field: the gate
                       -- covered `status` alone while attributes merged straight through, and
                       -- `_detail` renders attributes into the prompt — a caller that marked
                       -- the whole edge unreviewed had its relation held and its free text
                       -- SPOKEN. Reviewed means reviewed as it stood.
                       attributes     = CASE
                           WHEN knowledge_edges.status = 'accepted' AND EXCLUDED.status = 'proposed'
                           THEN knowledge_edges.attributes
                           ELSE knowledge_edges.attributes || EXCLUDED.attributes END,
                       -- a re-assertion PROMOTES a proposal and never demotes a verdict;
                       -- `rejected` is sticky on purpose (see the in-memory twin: this path
                       -- cannot tell a deliberate correction from the LLM re-emitting the same
                       -- edge, so `set_edge_status` is the only way back)
                       status         = CASE WHEN knowledge_edges.status = 'proposed'
                                             THEN EXCLUDED.status ELSE knowledge_edges.status END""",
                (edge.scope, src, tgt, edge.relation, edge.confidence, edge.source_session,
                 json.dumps(edge.attributes or {}, default=str),
                 sanitize_edge_status(edge.status)))

    async def find_node(self, scope: str, label: str) -> Optional[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id, scope, label, node_type, attributes, created_at, updated_at "
                "FROM knowledge_nodes WHERE scope = %s AND lower(label) = lower(%s) LIMIT 1",
                (scope, label))
            row = await cur.fetchone()
        if not row:
            return None
        return GraphNode(scope=row["scope"], label=row["label"], node_type=row["node_type"],
                         attributes=row["attributes"], id=row["id"],
                         created_at=row["created_at"], updated_at=row["updated_at"])

    async def find_nodes_by_embedding(self, scope: str, embedding: list[float],
                                      *, limit: int = 5) -> list[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id, scope, label, node_type, attributes FROM knowledge_nodes "
                "WHERE scope = %s AND embedding IS NOT NULL "
                "ORDER BY embedding <=> %s::vector LIMIT %s", (scope, _vec(embedding), limit))
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"]) for r in rows]

    async def walk(self, scope: str, start_label: str, *, max_depth: int = 2) -> list[GraphEdge]:
        _require_scope(scope)
        async with self._conn() as conn:
            # Emit the edge used at each hop, bounded by depth (mirrors the
            # in-memory BFS: only nodes at depth < max_depth expand their edges).
            cur = await conn.execute(
                """
                WITH RECURSIVE walk AS (
                    SELECT n.id AS node_id, 0 AS depth, NULL::bigint AS edge_id,
                           ARRAY[n.id] AS path
                    FROM knowledge_nodes n
                    WHERE n.scope = %s AND lower(n.label) = lower(%s)
                    UNION ALL
                    -- carry the visited-node path so a cyclic subgraph expands each node ONCE
                    -- (mirrors the in-memory walk's ``visited`` set); without this a cycle
                    -- re-expands every node per path → exponential intermediate rows.
                    SELECT nxt.node_id, w.depth + 1, nxt.edge_id, w.path || nxt.node_id
                    FROM walk w
                    JOIN LATERAL (
                        SELECT (CASE WHEN e.source_id = w.node_id THEN e.target_id
                                     ELSE e.source_id END) AS node_id, e.id AS edge_id
                        FROM knowledge_edges e
                        WHERE (e.source_id = w.node_id OR e.target_id = w.node_id)
                          AND e.scope = %s
                          -- an unreviewed edge must not decide what the walk can REACH either
                          AND e.status = 'accepted'
                    ) nxt ON true
                    WHERE w.depth < %s AND NOT (nxt.node_id = ANY(w.path))
                )
                SELECT DISTINCT e.id, sn.label AS source, tn.label AS target,
                       e.relation, e.confidence, e.source_session, e.attributes, e.status
                FROM walk w
                JOIN knowledge_edges e ON e.id = w.edge_id
                JOIN knowledge_nodes sn ON sn.id = e.source_id
                JOIN knowledge_nodes tn ON tn.id = e.target_id
                """,
                (scope, start_label, scope, max_depth))
            rows = await cur.fetchall()
        return [_edge_from_row(scope, r) for r in rows]

    async def pending_edges(self, scope: str, *, limit: int = 100) -> list[GraphEdge]:
        _require_scope(scope)
        if limit <= 0:          # a negative LIMIT raises here and truncated in memory; agree
            return []
        async with self._conn() as conn:
            cur = await conn.execute(
                """SELECT sn.label AS source, tn.label AS target, e.relation, e.confidence,
                          e.source_session, e.attributes, e.status
                   FROM knowledge_edges e
                   JOIN knowledge_nodes sn ON sn.id = e.source_id
                   JOIN knowledge_nodes tn ON tn.id = e.target_id
                   WHERE e.scope = %s AND e.status = %s
                   -- OLDEST first, and the direction is the point: with a `limit` and no
                   -- cursor, newest-first makes the oldest proposals — the ones a curator most
                   -- needs to clear — permanently unreachable, and the queue never drains. The
                   -- in-memory adapter returns insertion order, which is the same thing; a
                   -- review found the two disagreeing and the disagreement was invisible.
                   ORDER BY e.created_at ASC, e.id ASC
                   LIMIT %s""",
                (scope, EDGE_PROPOSED, limit))
            rows = await cur.fetchall()
        return [_edge_from_row(scope, r) for r in rows]

    async def set_edge_status(self, scope: str, source: str, target: str, relation: str,
                              status: str) -> bool:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                """UPDATE knowledge_edges e SET status = %s
                   FROM knowledge_nodes sn, knowledge_nodes tn
                   WHERE e.source_id = sn.id AND e.target_id = tn.id
                     AND e.scope = %s AND lower(sn.label) = lower(%s)
                     AND lower(tn.label) = lower(%s) AND e.relation = %s""",
                (require_edge_status(status), scope, source, target, relation))
        return bool(cur.rowcount)

    async def neighbors(self, scope: str, label: str) -> list[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                """SELECT DISTINCT nn.id, nn.scope, nn.label, nn.node_type, nn.attributes
                   FROM knowledge_nodes n
                   JOIN knowledge_edges e ON (e.source_id = n.id OR e.target_id = n.id)
                   JOIN knowledge_nodes nn
                     ON nn.id = CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END
                   WHERE n.scope = %s AND lower(n.label) = lower(%s)
                     -- an unreviewed edge still DISCLOSES its endpoint; see the in-memory twin
                     AND e.status = 'accepted'""", (scope, label))
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"]) for r in rows]

    async def get_node_context(self, scope: str, label: str) -> Optional[NodeContext]:
        _require_scope(scope)
        node = await self.find_node(scope, label)
        if node is None:
            return None
        edges = await self.walk(scope, label, max_depth=1)
        edges = [e for e in edges
                 if label.lower() in (e.source.lower(), e.target.lower())]
        return NodeContext(node=node, edges=edges, neighbors=await self.neighbors(scope, label))

    async def list_nodes(self, scope: str, *, node_type: Optional[str] = None,
                         limit: int = 100) -> list[GraphNode]:
        _require_scope(scope)
        sql = ("SELECT id, scope, label, node_type, attributes, created_at, updated_at "
               "FROM knowledge_nodes WHERE scope = %s")
        params: list = [scope]
        if node_type is not None:
            sql += " AND node_type = %s"
            params.append(node_type)
        sql += " ORDER BY id LIMIT %s"
        params.append(limit)
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"],
                          created_at=r["created_at"], updated_at=r["updated_at"])
                for r in rows]

    async def scan_nodes(self, scope: str, *, after_id: Optional[int] = None,
                         limit: int = 1000) -> list[GraphNode]:
        _require_scope(scope)
        sql = ("SELECT id, scope, label, node_type, attributes, created_at, updated_at "
               "FROM knowledge_nodes WHERE scope = %s")
        params: list = [scope]
        if after_id is not None:
            sql += " AND id > %s"
            params.append(after_id)
        sql += " ORDER BY id ASC LIMIT %s"
        params.append(limit)
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"],
                          created_at=r["created_at"], updated_at=r["updated_at"])
                for r in rows]

    async def delete_node(self, scope: str, label: str) -> bool:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM knowledge_nodes WHERE scope = %s AND lower(label) = lower(%s)", (scope, label))
            return cur.rowcount > 0   # edges cascade via FK ON DELETE CASCADE

    async def delete_edges_by_session(self, scope: str, session_id: str) -> int:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM knowledge_edges WHERE scope = %s AND source_session = %s",
                (scope, session_id))
            return cur.rowcount

    async def purge_scope(self, scope: str) -> int:
        _require_scope(scope)
        total = 0
        async with self._conn() as conn:
            # Drop edges first (deleting nodes would cascade them, but the explicit delete gives an
            # accurate count and covers any edge whose node was already gone).
            for table in ("knowledge_edges", "knowledge_nodes"):
                cur = await conn.execute(
                    f"DELETE FROM {table} WHERE scope = %s", (scope,))
                total += cur.rowcount
        return total
