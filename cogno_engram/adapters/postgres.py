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
    GraphEdge,
    GraphNode,
    HybridWeights,
    MemoryRecord,
    NodeContext,
    RetrievalQuery,
    Session,
    TurnRecord,
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
    if partition_by_scope:
        for tbl in ("turns", "memories"):
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
            created_at     timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source_id, target_id, relation)
        )
        """
    )
    # ── indexes (see engram-blueprint indexing strategy) ──
    stmts = [
        # Case-insensitive node identity (matches the in-memory adapter's dedup).
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_label_type "
        "ON knowledge_nodes (scope, lower(label), node_type)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_scope_time ON sessions (scope, started_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_turns_scope_time ON turns (scope, created_at DESC, id DESC)",
        "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, turn_n)",
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
                conn.row_factory = dict_row
                yield conn
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

    async def get_session(self, session_id: str) -> Optional[Session]:
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id, scope, started_at, ended_at, summary FROM sessions WHERE id = %s",
                (session_id,))
            row = await cur.fetchone()
        if not row:
            return None
        return Session(id=str(row["id"]), scope=row["scope"], started_at=row["started_at"],
                       ended_at=row["ended_at"], summary=row["summary"])

    async def close_session(self, session_id: str, *, summary: str = "") -> None:
        async with self._conn() as conn:
            await conn.execute(
                "UPDATE sessions SET ended_at = now(), summary = %s WHERE id = %s",
                (summary, session_id))

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

    async def load_turns(self, session_id: str) -> list[TurnRecord]:
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT scope, session_id, turn_n, user_input, response, feedback, goal, "
                "goal_status, sentiment, domains, pii_types, created_at "
                "FROM turns WHERE session_id = %s ORDER BY turn_n ASC", (session_id,))
            rows = await cur.fetchall()
        return [self._row_to_turn(r) for r in rows]

    async def turn_count(self, session_id: str) -> int:
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT count(*) AS c FROM turns WHERE session_id = %s", (session_id,))
            row = await cur.fetchone()
        return int(row["c"])

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
            "SELECT id FROM knowledge_nodes WHERE scope = %s AND label ILIKE %s LIMIT 1",
            (scope, label))
        row = await cur.fetchone()
        if row:
            return row["id"]
        cur = await conn.execute(
            "INSERT INTO knowledge_nodes (scope, label) VALUES (%s, %s) RETURNING id",
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
                   (scope, source_id, target_id, relation, confidence, source_session)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (source_id, target_id, relation) DO UPDATE SET
                       confidence     = GREATEST(knowledge_edges.confidence, EXCLUDED.confidence),
                       source_session = EXCLUDED.source_session""",
                (edge.scope, src, tgt, edge.relation, edge.confidence, edge.source_session))

    async def find_node(self, scope: str, label: str) -> Optional[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT id, scope, label, node_type, attributes FROM knowledge_nodes "
                "WHERE scope = %s AND label ILIKE %s LIMIT 1", (scope, label))
            row = await cur.fetchone()
        if not row:
            return None
        return GraphNode(scope=row["scope"], label=row["label"], node_type=row["node_type"],
                         attributes=row["attributes"], id=row["id"])

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
                    SELECT n.id AS node_id, 0 AS depth, NULL::bigint AS edge_id
                    FROM knowledge_nodes n
                    WHERE n.scope = %s AND n.label ILIKE %s
                    UNION ALL
                    SELECT CASE WHEN e.source_id = w.node_id THEN e.target_id ELSE e.source_id END,
                           w.depth + 1, e.id
                    FROM walk w
                    JOIN knowledge_edges e
                      ON (e.source_id = w.node_id OR e.target_id = w.node_id) AND e.scope = %s
                    WHERE w.depth < %s
                )
                SELECT DISTINCT e.id, sn.label AS source, tn.label AS target,
                       e.relation, e.confidence, e.source_session
                FROM walk w
                JOIN knowledge_edges e ON e.id = w.edge_id
                JOIN knowledge_nodes sn ON sn.id = e.source_id
                JOIN knowledge_nodes tn ON tn.id = e.target_id
                """,
                (scope, start_label, scope, max_depth))
            rows = await cur.fetchall()
        return [GraphEdge(scope=scope, source=r["source"], target=r["target"],
                          relation=r["relation"], confidence=r["confidence"],
                          source_session=r["source_session"]) for r in rows]

    async def neighbors(self, scope: str, label: str) -> list[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                """SELECT DISTINCT nn.id, nn.scope, nn.label, nn.node_type, nn.attributes
                   FROM knowledge_nodes n
                   JOIN knowledge_edges e ON (e.source_id = n.id OR e.target_id = n.id)
                   JOIN knowledge_nodes nn
                     ON nn.id = CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END
                   WHERE n.scope = %s AND n.label ILIKE %s""", (scope, label))
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
        sql = "SELECT id, scope, label, node_type, attributes FROM knowledge_nodes WHERE scope = %s"
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
                          attributes=r["attributes"], id=r["id"]) for r in rows]

    async def delete_node(self, scope: str, label: str) -> bool:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM knowledge_nodes WHERE scope = %s AND label ILIKE %s", (scope, label))
            return cur.rowcount > 0   # edges cascade via FK ON DELETE CASCADE

    async def delete_edges_by_session(self, scope: str, session_id: str) -> int:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "DELETE FROM knowledge_edges WHERE scope = %s AND source_session = %s",
                (scope, session_id))
            return cur.rowcount
