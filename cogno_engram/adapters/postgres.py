"""
cogno_engram.adapters.postgres — the reference Postgres + pgvector adapter.

Ported clean-room from the parent's ``memory/postgres_store.py`` +
``core/db_knowledge.py``, with the business identity (``tenant_id``/
``identity_id``) collapsed into a single opaque ``scope`` column. Implements
``MemoryStore`` (+ ``SupportsVectorSearch``), ``KnowledgeGraph`` and ``DocumentStore``
(``PostgresDocumentStore``, the ``kb_*`` tables) over one Postgres database:

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
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row

from cogno_engram import write_loss
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
    require_model,
    require_owner,
    require_profile,
    require_vector,
    sanitize_profiles,
    sanitize_reason,
)
from cogno_engram.trace_policy import TRACE_REVISION_WINDOW_S
from cogno_engram.folding import FOLD_FUNCTION_SQL, fold_label
from cogno_engram.types import (
    AUDIENCE_STAFF,
    AUDIENCE_TENANT,
    AUDIENCE_UNCLASSIFIED,
    sanitize_audience,
    EDGE_ACCEPTED,
    EDGE_PROPOSED,
    require_edge_status,
    sanitize_edge_status,
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

logger = logging.getLogger("cogno_engram.postgres")

# How many times ``save_turn`` re-reads and retries when it loses the race for a
# freshly allocated ``turn_n``. Each attempt is one round trip and each loss means a
# concurrent writer committed, so the budget bounds a real contention burst rather
# than a spin: under READ COMMITTED the first retry already sees the winner.
_ALLOC_ATTEMPTS = 5


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


logger = logging.getLogger("cogno_engram.postgres")


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


_SKIP_REMEDY = (
    "The rest of the schema was created. To change how `{tbl}` is partitioned, move the data "
    "deliberately (rename, recreate, INSERT SELECT, drop) — or leave it as it is, which is "
    "supported: partitioning is throughput, not correctness."
)


async def _partition_existing_table(conn, tbl: str, partitions: int) -> None:
    """Give ``tbl`` its HASH partitions, and NEVER raise because of the shape it already has.

    Partitioning is THROUGHPUT; every table and column `ensure_schema` creates after this loop
    is CORRECTNESS. An optimisation must not be fatal to a correctness step behind it — and this
    one was, twice, in production:

    * a database born FLAT (an older host, or ``partition_by_scope=False``) reached
      ``PARTITION OF`` with a plain table. `CREATE TABLE IF NOT EXISTS ... PARTITION BY HASH` is
      a NO-OP when the table exists — Postgres does not check the definition matches — so the
      error surfaced eleven statements before the knowledge graph. Measured 2026-08-25: a host
      running with no graph, `/health` reporting `stale`, and a log line about partitioning.
    * a database partitioned with a DIFFERENT modulus (4 children, host asking for 8) raised
      ``partition "turn_traces_p4" would overlap partition "turn_traces_p0"``. Measured
      2026-08-26 on the same box, immediately after the first fix shipped — `relkind` says
      PARTITIONED, it does not say WITH WHAT.

    So the shape is asked about first, for a message an operator can act on, and then the DDL
    itself is run defensively, because asking can never cover every shape: LIST/RANGE from a
    parent product, a different partition KEY, whatever the next database was created by. The
    two structural errors are downgraded to the same event; anything else (permissions, disk,
    a real bug) still raises, because those are not "this table has a history".

    Never converts. Moving data is the operator's decision, not the side effect of asking for a
    schema.
    """
    # `SELECT 1` + `is None`, never a value read by position or by name: this runs on BOTH row
    # factories — `migrate.py` hands it tuples, `PostgresStore._conn` hands it `dict_row` — and
    # the first cut read `kind[0]`, a `KeyError: 0` under dict_row that produced zero partitions
    # and no knowledge graph on a FRESH database: the guard breaking the healthy path worse than
    # the bug it came to fix. The ADD COLUMN guard inside `ensure_schema` — the one that asks
    # `SELECT 1 FROM information_schema.columns` before altering `knowledge_edges` — answers the
    # same way for the same reason. (Named by what it QUERIES: the first draft of this sentence
    # invented a `has_column` helper in an `_ensure_edge_audience` that does not exist, which is
    # how prose starts asserting symbols nobody can grep.)
    # `to_regclass` resolves through `search_path`, like the DDL does;
    # the first cut hardcoded `'public'::regnamespace` and answered "not flat" for a host whose
    # schema is not public.
    flat = await (await conn.execute(
        "SELECT 1 FROM pg_class WHERE oid = to_regclass(%s) AND relkind <> 'p'",
        (tbl,))).fetchone()
    if flat is not None:
        logger.error("stage=schema event=partitioning_skipped table=%s "
                     "reason=exists_unpartitioned remedy=%s", tbl, _SKIP_REMEDY.format(tbl=tbl))
        return

    # How many children it HAS, against how many we were asked for. Counting `pg_inherits` says
    # nothing about the strategy, which is the point: a table with N children when the caller
    # wants M is a real divergence whatever the strategy is, and the numbers belong in the log.
    # One row per child and COUNT THE ROWS — not `count(*)` read out of the row, which is the
    # row-factory sniff the rule above forbids and which this function had, twenty-six lines
    # below the rule, until the review put them side by side. A rule that the code beneath it
    # breaks is worse than no rule: the next reader trusts it.
    existing = len(await (await conn.execute(
        "SELECT 1 FROM pg_inherits WHERE inhparent = to_regclass(%s)", (tbl,))).fetchall())
    if existing and existing != partitions:
        logger.error("stage=schema event=partitioning_skipped table=%s "
                     "reason=exists_with_%d_partitions requested=%d remedy=%s",
                     tbl, existing, partitions, _SKIP_REMEDY.format(tbl=tbl))
        return

    for k in range(partitions):
        try:
            # A nested transaction so ONE refused statement cannot poison the caller's — psycopg
            # emits SAVEPOINT when already in a transaction and BEGIN when not, so this is right
            # for `migrate.py` (autocommit) and for a pooled store connection alike.
            async with conn.transaction():
                await conn.execute(
                    f"CREATE TABLE IF NOT EXISTS {tbl}_p{k} PARTITION OF {tbl} "
                    f"FOR VALUES WITH (MODULUS {partitions}, REMAINDER {k})")
        except (psycopg.errors.InvalidObjectDefinition,
                psycopg.errors.InvalidTableDefinition) as exc:
            # The shape the probes above could not name — LIST/RANGE, a different key, something
            # this version has not met. Same event, so an operator greps one string.
            logger.error("stage=schema event=partitioning_skipped table=%s "
                         "reason=incompatible_shape detail=%s remedy=%s",
                         tbl, str(exc).splitlines()[0], _SKIP_REMEDY.format(tbl=tbl))
            return


async def ensure_schema(conn, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                        ts_config: str = DEFAULT_TS_CONFIG,
                        partition_by_scope: bool = False, partitions: int = 8,
                        unaccent: bool = False) -> None:
    """Create the engram schema idempotently (extension + tables + indexes).

    ``unaccent`` applies to the DOCUMENT tables only (``kb_chunks``, see
    :func:`ensure_documents_schema`): their text search folds accents on both sides of the match.
    The ``memories`` table keeps ``ts_config`` as it is — changing the configuration of a
    generated column on a live table is a migration of its own, not a flag.

    No alembic required — this is the zero-friction path. Migrations can be
    layered on top by a host that wants versioned schema.

    ``partition_by_scope`` opts the high-volume ``turns`` and ``memories`` tables
    into HASH(scope) partitioning over a fixed number of buckets (``partitions``),
    the generic equivalent of the parent's LIST(tenant) partitioning — zero DDL
    per new scope. Every query carries ``scope`` so partition pruning applies.
    """
    ts_config = _validate_ts_config(ts_config)  # interpolated into the tsvector DDL
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # `unaccent` + o wrapper IMMUTABLE por baixo da identidade de nó: `José` e `Jose` são a mesma
    # pessoa (decisão de produto). Tem de vir ANTES dos índices — a UNIQUE de nós usa a função.
    await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    fold_before = await _installed_fold_definition(conn)
    await conn.execute(FOLD_FUNCTION_SQL)
    await _rebuild_index_if_fold_changed(conn, fold_before)

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
            voiced_by    text NOT NULL DEFAULT '',
            created_at   timestamptz NOT NULL DEFAULT now(),
            -- WHEN Tier-3 last consolidated this turn, or NULL for "not yet". The janitor's
            -- idempotence marker, and it lives HERE — on the row the consolidation READ —
            -- because the `sessions` row it used to live on is deleted by operations that
            -- deliberately spare `turns`. See `idle_sessions` for the loop that cost.
            consolidated_at timestamptz,
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
            first_heard_by text NOT NULL DEFAULT '',
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
            await _partition_existing_table(conn, tbl, partitions)
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
        f"""
        CREATE TABLE IF NOT EXISTS knowledge_edges (
            id             bigserial PRIMARY KEY,
            scope          text NOT NULL,
            source_id      bigint NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            target_id      bigint NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
            relation       text NOT NULL,
            confidence     real NOT NULL DEFAULT 1.0,
            source_session text NOT NULL DEFAULT '',
            attributes     jsonb NOT NULL DEFAULT '{{}}',
            status         text NOT NULL DEFAULT '{EDGE_ACCEPTED}',
            -- WHO MAY READ IT. '' = unclassified (staff only), 'tenant' = everyone in the
            -- tenant, 'identity:<id>' = that contact's own life. See `types.audience_can_read`.
            audience       text NOT NULL DEFAULT '',
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
    # The TABLE is part of the tuple because this is now the only place that knows how to add a
    # column without locking readers, and three tables need it. A second copy of the catalogue
    # read is a second chance to forget it.
    for table, column, ddl in (
        ("knowledge_edges", "attributes",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS attributes jsonb NOT NULL DEFAULT '{}'"),
        ("knowledge_edges", "status",
         f"ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT '{EDGE_ACCEPTED}'"),
        # The DEFAULT backfills every existing edge as UNCLASSIFIED, not as `tenant`: staff keeps
        # seeing them and no contact does. An upgrade must not hand a contact rows nobody has
        # classified — `maintenance.classify_edge_audience` is the deliberate act that assigns
        # owners, and until it runs the safe answer is "staff only".
        ("knowledge_edges", "audience",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS audience text NOT NULL DEFAULT ''"),
        # Who the contact was talking to. The DEFAULT backfills every existing row as BLANK —
        # deliberately "unknown", not a guess: rows written before this column existed were
        # produced by a persona nobody recorded, and inventing one would put a name in a field a
        # reader is meant to trust.
        ("turns", "voiced_by",
         "ALTER TABLE turns ADD COLUMN IF NOT EXISTS voiced_by text NOT NULL DEFAULT ''"),
        ("memories", "first_heard_by",
         "ALTER TABLE memories ADD COLUMN IF NOT EXISTS first_heard_by text NOT NULL DEFAULT ''"),
        # NULLABLE, and with NO backfill — the two together are the migration story. Every row
        # that predates the column reads "not yet consolidated", which is what the janitor's
        # predicate ALREADY said about it a moment before the upgrade: the marker only ever
        # SUPPRESSES a pick (`idle_sessions` ANDs it onto the old disjunction), so an unstamped
        # backlog cannot become a thundering herd of LLM re-reads on deploy. A DEFAULT now()
        # here would be the opposite mistake — it would silently declare the entire history
        # consolidated and freeze Tier-3 for every session that had not been.
        ("turns", "consolidated_at",
         "ALTER TABLE turns ADD COLUMN IF NOT EXISTS consolidated_at timestamptz"),
    ):
        cur = await conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = %s AND column_name = %s", (table, column))
        if await cur.fetchone() is None:
            await conn.execute(ddl)

    # ── indexes (see engram-blueprint indexing strategy) ──
    stmts = [
        # Identidade de nó insensível a caixa E a acento — `engram_fold`, a mesma função que o
        # adaptador in-memory corre em Python (`folding.fold_label`). Era `lower(label)`, que sob
        # um cluster `LC_COLLATE 'C'` nem sequer dobra maiúsculas acentuadas: `lower('JOSÉ')` dá
        # `'josÉ'` e o nó gravado como `josé` ficava inalcançável.
        #
        # NOME NOVO de propósito: `CREATE INDEX IF NOT EXISTS` com o nome antigo e definição nova
        # é um no-op SILENCIOSO — a armadilha que o `idx_turns_scope_pattern` já documentou. E o
        # CREATE vem antes do DROP: se o processo morrer entre os dois, fica-se com os dois
        # índices (a identidade mais apertada já em vigor) e não sem nenhum.
        #
        # ATENÇÃO À MIGRAÇÃO: numa base que já tenha `José` e `Jose` como nós SEPARADOS, este
        # CREATE FALHA com duplicate key — e é o comportamento correcto. Fundir nós de um grafo de
        # conhecimento automaticamente perderia arestas de um deles sem ninguém ver; a colisão tem
        # de ser resolvida por quem sabe qual dos dois é a pessoa. `python -m cogno_host.migrate`
        # levanta com os rótulos em conflito nomeados.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_fold_type "
        "ON knowledge_nodes (scope, engram_fold(label), node_type)",
        "DROP INDEX IF EXISTS uq_nodes_scope_label_type",
        # The curation queue reads one status within one scope; the prompt walk reads the other.
        "CREATE INDEX IF NOT EXISTS idx_edges_scope_status ON knowledge_edges (scope, status)",
        "CREATE INDEX IF NOT EXISTS idx_sessions_scope_time ON sessions (scope, started_at DESC)",
        # `admin_turns` e `admin_scopes` lêem uma SUBÁRVORE de escopo — `scope = %s OR scope
        # LIKE 'prefixo/%%'` — e o índice que aqui estava era btree COMUM. Pelo mesmo argumento
        # do `idx_turn_traces_scope_time`: num collation que não seja C (esta base é
        # `en_US.utf8`) um btree comum NÃO serve o ramo do LIKE, e o planeador nem o considera.
        # Isto era portanto o MESMO defeito que o índice dos traços corrigiu — e este ficheiro
        # citava-o como o irmão que "já tinha" o índice, o que estava ao contrário.
        #
        # SUBSTITUI em vez de acrescentar, e a escolha é medida. 200k linhas, tenant a ~10% da
        # tabela, medianas de 7-9 corridas (a leitura de amostra única mente aqui: uma corrida
        # dizia que o custo de escrita caía 13%, nove dizem que não se distingue):
        #
        #     .                       tamanho   escrita 20k          subárvore
        #     btree comum (o antigo)   24 MB    121 ms (117-142)     Seq Scan     18-24 ms
        #     os DOIS índices          34 MB    152 ms (133-165)     Bitmap Heap  10 ms
        #     só text_pattern_ops      24 MB    115 ms (103-163)     Bitmap Heap   8-12 ms
        #
        # Manter os dois custaria +10 MB e ~26% de escrita na tabela mais quente do schema, para
        # nada: o `text_pattern_ops` serve TAMBÉM o ramo `=` (está em `pg_amop`) e mantém o
        # Index Scan ordenado da consulta de igualdade+ordenação (`scope = %s ... ORDER BY
        # created_at DESC, id DESC`, medido 0,087 vs 0,094 ms). A ordem de saída do
        # `admin_scopes` é IDÊNTICA nas duas — o `ORDER BY scope` usa o collation da coluna,
        # não a opclass do índice.
        #
        # O NOME MUDA de propósito. `CREATE INDEX IF NOT EXISTS` com o nome antigo e definição
        # nova é um NO-OP silencioso: o nome existe, nada acontece, e o conserto subiria inerte
        # em toda a instalação existente. Nome novo + `DROP` do antigo é idempotente nos dois
        # sentidos (e um rollback de versão recria o antigo sozinho).
        "CREATE INDEX IF NOT EXISTS idx_turns_scope_pattern "
        "ON turns (scope text_pattern_ops, created_at DESC, id DESC)",
        # Depois do CREATE, nunca antes: se o processo morrer entre os dois, a instalação fica
        # com os dois índices (lenta a escrever, correcta a ler) e não sem nenhum. Dito aqui
        # porque NENHUM teste o guarda e isso não é esquecimento — medido, trocar a ordem
        # sobrevive à suíte, e sobrevive com razão: as duas ordens dão o mesmo estado final numa
        # passagem que termina. O que as separa é só a janela de falha a meio, que um teste não
        # simula sem matar o processo.
        "DROP INDEX IF EXISTS idx_turns_scope_time",
        "CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id, turn_n)",
        # PARTIAL, and it shrinks as history is consolidated: `idle_sessions` scans only
        # turns nobody has read into long-term memory yet, which on a steady-state box is
        # the last idle window rather than the whole table.
        "CREATE INDEX IF NOT EXISTS idx_turns_unconsolidated "
        "ON turns (session_id, scope) WHERE consolidated_at IS NULL",
        "CREATE INDEX IF NOT EXISTS idx_turn_traces_session ON turn_traces (session_id, turn_n)",
        # `admin_traces` lê uma SUBÁRVORE de escopo ordenada por tempo. O ramo `scope = %s` já
        # era servido pela UNIQUE `(scope, session_id, turn_n)`; o que NÃO tinha índice era o
        # ramo do LIKE — e é por isso que o `BitmapOr` do plano usa os dois. (A primeira versão
        # deste comentário dizia "não tinha índice nenhum que a servisse" e contradizia-se seis
        # linhas abaixo.)
        #
        # `text_pattern_ops` NÃO é decoração, e é a parte que uma correcção "óbvia" erra: num
        # collation que não seja C — esta base é `en_US.utf8` — um btree COMUM não serve
        # `LIKE 'prefixo/%'`, e o planeador nem o considera. Medido em 200k linhas, um tenant a
        # 0,025% da tabela:
        #
        #     sem índice              Parallel Seq Scan   12,8 ms
        #     btree comum             Parallel Seq Scan   13,0 ms   ← o índice nem é considerado
        #     text_pattern_ops        Bitmap Heap Scan     0,21 ms
        #
        # O padrão tem TRÊS consumidores e os três estão fechados: este, e o `admin_turns` /
        # `admin_scopes` pelo `idx_turns_scope_pattern` acima. O irmão que este comentário
        # nomeava — `idx_turns_scope_time`, btree comum — foi APOSENTADO por ter exactamente
        # este defeito; não o procure, já não existe.
        #
        # `created_at` no segundo lugar GANHA o seu lugar, e a medição contraria a leitura de
        # amostra única (uma corrida dizia "empate"; sete dizem outra coisa). Subárvore gorda
        # (20k linhas) com `since` selectivo, medianas de 7 corridas:
        #
        #     (scope)                 mediana 10,97 ms   índice 2208 kB
        #     (scope, created_at)     mediana  4,84 ms   índice 7560 kB   ← 2,3× mais rápido
        #
        # Custo: 3,4× o tamanho do índice e ~10% em insert (92,0 → 101,5 ms por 20k). Aceite.
        #
        # O `DESC`, esse, NUNCA é lido: um `BitmapOr` não preserva ordem de índice, e todos os
        # planos acabam num `Sort  Sort Key: created_at DESC` explícito. Fica por consistência
        # de forma com os irmãos, não por desempenho — DESC e ASC medem igual.
        "CREATE INDEX IF NOT EXISTS idx_turn_traces_scope_time "
        "ON turn_traces (scope text_pattern_ops, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_scope_time ON memories (scope, created_at DESC)",
        "CREATE INDEX IF NOT EXISTS idx_memories_tsv ON memories USING gin (tsv)",
        "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories "
        "USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS idx_nodes_embedding ON knowledge_nodes "
        "USING hnsw (embedding vector_cosine_ops)",
        "CREATE INDEX IF NOT EXISTS idx_edges_session ON knowledge_edges (scope, source_session)",
        # Every contact-scoped read filters on it. The column is added by the migration block
        # ABOVE — a first cut put the ALTER here, after this line, so on a database that
        # predates the column the index was built over a column that did not exist yet and
        # `ensure_schema` died with `UndefinedColumn`. The unit suite cannot see that: in
        # memory there is no DDL to order. The integration test for an existing database can,
        # and did.
        "CREATE INDEX IF NOT EXISTS idx_edges_audience ON knowledge_edges (scope, audience)",
        "CREATE INDEX IF NOT EXISTS idx_edges_source ON knowledge_edges (source_id)",
        "CREATE INDEX IF NOT EXISTS idx_edges_target ON knowledge_edges (target_id)",
    ]
    for stmt in stmts:
        try:
            await conn.execute(stmt)
        except psycopg.errors.UniqueViolation as cause:
            # SÓ violação de unicidade, e só neste índice. Um `except Exception` aqui rotulava
            # QUALQUER falha do CREATE — falta de ICU, permissões, função em falta — como
            # "colisão de rótulos", e mandava o operador fundir nós para resolver um problema que
            # não era esse. Um diagnóstico errado custa mais do que nenhum.
            if "uq_nodes_scope_fold_type" not in stmt:
                raise
            raise _label_collision_error(await _colliding_labels(conn), cause) from cause

    await ensure_documents_schema(conn, embedding_dim=embedding_dim, ts_config=ts_config,
                                  unaccent=unaccent)


# ── documents (see cogno_engram.documents) ────────────────────────────────────────────
#
# Five tables, created by `ensure_schema` (so a host whose migration delegates to it — e.g.
# `python -m cogno_host.migrate` — gets them with no new step) and ADDITIVE: nothing here
# alters a table that existed before. Why five and not the two a first sketch named:
#
#   kb_documents   one row per document: owner, title, who may read, which version is SERVED
#   kb_versions    one row per indexing ATTEMPT — the swap needs two states at once (the one
#                  being served and the one being built), so a version cannot be a column
#   kb_chunks      what a search reads; each row carries the model label it was embedded with
#   kb_originals   the uploaded bytes, in a table of their OWN so no search can reach them —
#                  the reader path never joins it, and a `SELECT *` over a version cannot drag
#                  ten megabytes along
#   kb_tombstones  what a delete or a purge removed: ids, versions, when, who — no content
#
# Every child cascades from its parent, so a delete of a document row removes its versions,
# chunks and originals in the SAME statement, and a late writer's insert finds no parent.
#
# NO vector index on kb_chunks, deliberately: a reader's search is bounded to one owner's
# served chunks (thousands, not millions) and scans them exactly; an HNSW index with that filter
# needs iterative scan to be correct and was not asked for by any measurement. Add it when one
# asks. The lexical half has its GIN index.
_UNACCENT_TOKENS = ("word", "hword", "hword_part")
_DICT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def documents_ts_config(ts_config: str = DEFAULT_TS_CONFIG, *, unaccent: bool = False) -> str:
    """The text-search configuration the document tables are built with AND queried with — one
    name, derived from the same two arguments on both sides, so the ``tsvector`` a chunk was
    indexed with and the ``tsquery`` a question is parsed with fold the same way.

    Without ``unaccent`` it is ``ts_config`` itself; with it, the derived
    ``cogno_<base>_unaccent`` that :func:`ensure_documents_schema` creates."""
    base = _validate_ts_config(ts_config)
    if not unaccent:
        return base
    return f"cogno_{base.replace('.', '_')}_unaccent"


async def _ensure_unaccent_config(conn, base: str) -> str:
    """Create ``cogno_<base>_unaccent`` IF IT DOES NOT EXIST: a COPY of ``base`` whose non-ASCII
    word tokens (``word``/``hword``/``hword_part``) pass through ``unaccent`` before the base's
    OWN dictionaries — so ``Sábado`` and ``sabado`` reach the stemmer as the same word, and a
    ``portuguese`` base still stems (``sábados`` → ``sab``). ASCII tokens carry no accent and keep
    the base mapping untouched.

    An existing configuration of that name is LEFT AS IT IS — the mapping is written once, in the
    statement that creates it, so a second migration cannot rewrite what the first indexed with.
    Two migrations racing to create it: the loser's ``DuplicateObject`` is the winner's success."""
    derived = documents_ts_config(base, unaccent=True)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS unaccent")
    exists = await (await conn.execute(
        "SELECT 1 FROM pg_ts_config c JOIN pg_namespace n ON n.oid = c.cfgnamespace "
        "WHERE c.cfgname = %s AND n.nspname = current_schema()", (derived,))).fetchone()
    if exists is not None:
        return derived
    rows = await (await conn.execute(
        "SELECT t.alias, array_agg(d.dictname::text ORDER BY m.mapseqno) "
        "FROM pg_ts_config_map m "
        "JOIN pg_ts_config c ON c.oid = m.mapcfg "
        "JOIN pg_ts_dict d ON d.oid = m.mapdict "
        "JOIN LATERAL ts_token_type(c.cfgparser) t ON t.tokid = m.maptokentype "
        "WHERE c.oid = %s::regconfig AND t.alias = ANY(%s) GROUP BY t.alias",
        (base, list(_UNACCENT_TOKENS)))).fetchall()
    mapping: dict[str, list[str]] = {}
    for row in rows:
        alias, dicts = (row["alias"], row["array_agg"]) if isinstance(row, dict) else (row[0], row[1])
        mapping[alias] = list(dicts or [])
    names = [d for ds in mapping.values() for d in ds]
    if not all(_DICT_NAME_RE.match(d) for d in names):
        raise ValueError(f"unexpected dictionary name in {base!r}: {names!r}")
    try:
        async with conn.transaction():
            await conn.execute(f"CREATE TEXT SEARCH CONFIGURATION {derived} (COPY = {base})")
            for alias in _UNACCENT_TOKENS:
                if mapping.get(alias):          # a token type the base ignores stays ignored
                    await conn.execute(
                        f"ALTER TEXT SEARCH CONFIGURATION {derived} ALTER MAPPING FOR {alias} "
                        f"WITH unaccent, {', '.join(mapping[alias])}")
    except psycopg.errors.DuplicateObject:
        pass
    return derived


async def _documents_tsv_config(conn) -> "str | None":
    """The configuration ``kb_chunks.tsv`` was GENERATED with, read from the catalogue — ``None``
    when there is no such column."""
    row = await (await conn.execute(
        "SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d "
        "JOIN pg_attribute a ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
        "WHERE d.adrelid = to_regclass('kb_chunks') AND a.attname = 'tsv'")).fetchone()
    if row is None:
        return None
    expr = list(row.values())[0] if isinstance(row, dict) else row[0]
    match = re.search(r"to_tsvector\('([^']+)'::regconfig", expr or "")
    return match.group(1) if match else None


async def rebuild_documents_tsv(conn, *, ts_config: str = DEFAULT_TS_CONFIG,
                                unaccent: bool = False) -> None:
    """Regenerate ``kb_chunks.tsv`` (and its index) with the configuration these arguments name —
    the one migration a CHANGE of ``ts_config``/``unaccent`` on an existing database needs.

    ``CREATE TABLE IF NOT EXISTS`` never touches a table that exists, so a store switched to
    another configuration would parse questions one way over chunks indexed another — nothing
    errors, words just stop matching. This drops the generated column and adds it back, which
    REWRITES the table under an ``ACCESS EXCLUSIVE`` lock: an operator's step, run when the flag
    changes, never on a boot."""
    config = (await _ensure_unaccent_config(conn, ts_config)) if unaccent \
        else documents_ts_config(ts_config)
    async with conn.transaction():
        await conn.execute("ALTER TABLE kb_chunks DROP COLUMN IF EXISTS tsv")
        await conn.execute(
            f"ALTER TABLE kb_chunks ADD COLUMN tsv tsvector "
            f"GENERATED ALWAYS AS (to_tsvector('{config}', content)) STORED")
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_chunks_tsv ON kb_chunks USING gin (tsv)")


async def ensure_documents_schema(conn, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                                  ts_config: str = DEFAULT_TS_CONFIG,
                                  unaccent: bool = False) -> None:
    """Create the document tables idempotently (``CREATE ... IF NOT EXISTS``, never an ALTER).

    ``kb_chunks.tsv`` is generated with :func:`documents_ts_config` of the same two arguments the
    store is built with. ``unaccent=True`` first creates the derived configuration (see
    :func:`_ensure_unaccent_config`). An EXISTING ``kb_chunks`` built with another configuration
    is not altered: it is logged as an ERROR naming :func:`rebuild_documents_tsv`, because a
    mismatch does not fail — it silently stops words from matching."""
    ts_config = _validate_ts_config(ts_config)
    config = (await _ensure_unaccent_config(conn, ts_config)) if unaccent else ts_config
    dim = int(embedding_dim)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for ddl in (
        """
        CREATE TABLE IF NOT EXISTS kb_documents (
            id             uuid PRIMARY KEY,
            owner_key      text NOT NULL,
            title          text NOT NULL,
            profiles       text[] NOT NULL DEFAULT '{}',
            media_type     text NOT NULL,
            active_version integer,
            created_at     timestamptz NOT NULL DEFAULT now(),
            updated_at     timestamptz NOT NULL DEFAULT now()
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS kb_versions (
            document_id  uuid NOT NULL REFERENCES kb_documents(id) ON DELETE CASCADE,
            version      integer NOT NULL,
            state        text NOT NULL DEFAULT '{KB_PROCESSING}',
            reason       text NOT NULL DEFAULT '',
            sha256       text NOT NULL,
            embed_model  text NOT NULL,
            size_bytes   bigint NOT NULL DEFAULT 0,
            pages        integer NOT NULL DEFAULT 0,
            chunks       integer NOT NULL DEFAULT 0,
            created_at   timestamptz NOT NULL DEFAULT now(),
            finished_at  timestamptz,
            PRIMARY KEY (document_id, version)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS kb_originals (
            document_id  uuid NOT NULL,
            version      integer NOT NULL,
            -- The owner rides with the bytes so `stored_original_bytes` counts them WITHOUT a
            -- join: a count through `kb_documents` could never see a leftover whose document
            -- row is gone, which is exactly what a purge check exists to catch.
            owner_key    text NOT NULL,
            data         bytea NOT NULL,
            PRIMARY KEY (document_id, version),
            FOREIGN KEY (document_id, version)
                REFERENCES kb_versions(document_id, version) ON DELETE CASCADE
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS kb_chunks (
            document_id  uuid NOT NULL,
            version      integer NOT NULL,
            ordinal      integer NOT NULL,
            heading_path text[] NOT NULL DEFAULT '{{}}',
            page         integer,
            content      text NOT NULL,
            embed_model  text NOT NULL,
            -- NOT NULL: a served version is always fully comparable under its own model (the
            -- all-or-none rule of `search`). A chunk without a vector cannot be stored.
            embedding    vector({dim}) NOT NULL,
            tsv          tsvector GENERATED ALWAYS AS (to_tsvector('{config}', content)) STORED,
            PRIMARY KEY (document_id, version, ordinal),
            FOREIGN KEY (document_id, version)
                REFERENCES kb_versions(document_id, version) ON DELETE CASCADE
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS kb_tombstones (
            id           bigserial PRIMARY KEY,
            owner_key    text NOT NULL,
            document_id  uuid NOT NULL,
            versions     integer[] NOT NULL DEFAULT '{}',
            kind         text NOT NULL,
            actor        text NOT NULL DEFAULT '',
            removed_at   timestamptz NOT NULL DEFAULT now()
        )
        """,
        # `text_pattern_ops`: the subtree read (`owner_key = %s OR owner_key LIKE 'p/%'`) is
        # served by it under any collation, and it serves plain equality too — the same
        # measurement `idx_turns_scope_pattern` records above.
        "CREATE INDEX IF NOT EXISTS idx_kb_documents_owner "
        "ON kb_documents (owner_key text_pattern_ops, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_kb_chunks_tsv ON kb_chunks USING gin (tsv)",
        "CREATE INDEX IF NOT EXISTS idx_kb_tombstones_owner "
        "ON kb_tombstones (owner_key text_pattern_ops, removed_at DESC)",
    ):
        await conn.execute(ddl)
    built_with = await _documents_tsv_config(conn)
    if built_with is not None and built_with != config:
        logger.error("stage=schema event=kb_ts_config_mismatch table=kb_chunks built_with=%s "
                     "requested=%s remedy=rebuild_documents_tsv(conn, ts_config=%r, unaccent=%r) "
                     "— until then questions are parsed with one configuration over chunks "
                     "indexed with another, and words silently stop matching",
                     built_with, config, ts_config, unaccent)


async def _installed_fold_definition(conn) -> "str | None":
    """A definição de `engram_fold` que está INSTALADA, ou None se ainda não existe."""
    try:
        cur = await conn.execute(
            "SELECT pg_get_functiondef(p.oid) FROM pg_proc p JOIN pg_namespace n "
            "  ON n.oid = p.pronamespace "
            " WHERE p.proname = 'engram_fold' AND p.pronargs = 1 LIMIT 1")
        row = await cur.fetchone()
    except Exception:                                # noqa: BLE001 — ausência não é erro
        return None
    if row is None:
        return None
    return row[0] if not isinstance(row, dict) else list(row.values())[0]


async def _rebuild_index_if_fold_changed(conn, previous: "str | None") -> None:
    """Se a regra de dobragem mudou, o índice que a GUARDA ficou a mentir — reconstrói.

    `uq_nodes_scope_fold_type` é um índice de EXPRESSÃO: guarda o resultado de `engram_fold`, não
    o rótulo. `CREATE OR REPLACE FUNCTION` troca a função e o Postgres **não reconstrói o índice
    nem avisa** — as chaves velhas ficam lá. Medido: depois de mudar a função, uma busca por
    índice devolve 0 e um seq scan sobre os mesmos dados devolve 1. É o defeito original
    (`find_node` não acha um nó que existe) a voltar através do TEMPO, e nenhum teste de base nova
    o apanha porque ali a função e o índice nascem juntos.

    CEITO CONHECIDO: `CREATE OR REPLACE FUNCTION` não pode mudar o NOME de um parâmetro. Uma
    versão futura que nomeie o argumento (`engram_fold(rotulo text)`) dá `InvalidFunctionDefinition`
    na própria criação — um passo ACIMA deste, fora do `except` da colisão — e aborta o schema com
    o erro do Postgres. É recuperável (`DROP FUNCTION` primeiro), mas quem mexer no
    `FOLD_FUNCTION_SQL` tem de saber.

    Só corre quando a definição MUDOU — num deploy normal `CREATE OR REPLACE` reinstala texto
    idêntico e isto é um no-op. Quando corre, `REINDEX` toma `ACCESS EXCLUSIVE` na tabela: é o
    preço de uma regra de identidade nova, e acontece uma vez por mudança dela, não por deploy.

    Se a dobragem nova fizer colidir rótulos que antes eram distintos, o `REINDEX` FALHA — e é o
    mesmo comportamento correcto da criação inicial, com o mesmo erro nomeado."""
    if previous is None:
        return                                       # instalação nova: nada a reconstruir
    current = await _installed_fold_definition(conn)
    if current is None or current == previous:
        return
    cur = await conn.execute(
        "SELECT 1 FROM pg_class WHERE relname = 'uq_nodes_scope_fold_type' AND relkind = 'i'")
    if await cur.fetchone() is None:
        return                                       # o índice ainda não existe; nasce já certo
    logger.warning(
        "event=fold_index_rebuilt index=uq_nodes_scope_fold_type "
        "reason=engram_fold_definition_changed")
    try:
        await conn.execute("REINDEX INDEX uq_nodes_scope_fold_type")
    except psycopg.errors.UniqueViolation as cause:
        raise _label_collision_error(await _colliding_labels(conn), cause) from cause

async def _colliding_labels(conn) -> "list[str]":
    """Os grupos de rótulos que a identidade nova funde — vazio se a leitura falhar.

    O erro do Postgres nomeia a CHAVE dobrada (`Key (scope, engram_fold(label), node_type)=(t,
    jose, PERSON) is duplicated`), que é precisamente o que o operador não precisa: ele quer saber
    QUE nós tem de fundir. Um diagnóstico não pode ser o motivo de a migração falhar de outra
    maneira, portanto qualquer erro AQUI degrada para lista vazia — o erro original propaga na
    mesma."""
    try:
        # ROLLBACK primeiro, e é o que faz esta função servir de todo: numa conexão SEM
        # autocommit — e `ensure_schema` é API pública, portanto recebe as duas — o CREATE que
        # acabou de falhar ENVENENOU a transacção, e toda consulta seguinte dá
        # `InFailedSqlTransaction`. O diagnóstico degradava para `[]` e o operador recebia
        # exactamente a chave dobrada que este módulo diz que ele não precisa. Em autocommit o
        # ROLLBACK é um no-op inofensivo.
        try:
            await conn.rollback()
        except Exception:                            # noqa: BLE001 — autocommit, ou já limpa
            pass
        cur = await conn.execute(
            "SELECT scope, node_type, string_agg(label, %s ORDER BY label) AS labels "
            "FROM knowledge_nodes GROUP BY scope, node_type, engram_fold(label) "
            "HAVING count(*) > 1 LIMIT 21", (" + ",))
        # nomeadas, não posicionais: esta conexão pode vir com `dict_row` (o adaptador usa-o) e
        # aí `r[0]` levanta `KeyError`. O diagnóstico morreria exactamente no caso em que faz
        # falta — a migração já falhou e é isto que diz ao operador o que fazer.
        rows = await cur.fetchall()
        def _field(r, i, name):
            return r[name] if isinstance(r, dict) else r[i]
        out = [f"{_field(r, 0, 'scope')} / {_field(r, 1, 'node_type')}: {_field(r, 2, 'labels')}"
                for r in rows[:20]]
        if len(rows) > 20:
            # truncar em silêncio manda o operador fundir 20, correr outra vez e falhar outra vez
            out.append("… e MAIS grupos além destes 20 — corra o relatório completo com "
                        "`cogno_engram.fold_migration.fold_collisions()`")
        return out
    except Exception:                                # noqa: BLE001
        return []


def _label_collision_error(groups: "list[str]", cause: Exception) -> RuntimeError:
    """A identidade de nó passou a ignorar acentos e esta base tem nós que agora colidem.

    Falhar é o comportamento CORRECTO, e a alternativa foi considerada e recusada: fundir os nós
    automaticamente escolheria um dos rótulos e mudaria as arestas do outro de dono, em silêncio,
    num grafo cujo propósito é dizer factos sobre pessoas. Qual dos dois `José` é a pessoa é
    conhecimento que esta função não tem."""
    if not groups:
        return RuntimeError(
            "a identidade de nó passa a ignorar acentos (`José` == `Jose`) e esta base tem nós "
            f"que agora colidem. Não consegui listá-los; o erro original foi: {cause}")
    return RuntimeError(
        "a identidade de nó passa a ignorar acentos (`José` == `Jose`) e esta base tem nós que "
        "agora colidem. Funda-os À MÃO antes de migrar — automaticamente não, porque escolher um "
        "dos rótulos muda as arestas do outro de dono em silêncio:\n  "
        + "\n  ".join(groups))


class _PgBase:
    """Shared connection plumbing for the Postgres adapters."""

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

    def __init__(self, *, dsn: Optional[str] = None, pool=None,
                 ts_config: str = DEFAULT_TS_CONFIG, mask_pii: bool = False,
                 trace_revision_window_s: float = TRACE_REVISION_WINDOW_S) -> None:
        if not dsn and pool is None:
            raise ValueError("provide either dsn= or pool=")
        self._dsn = dsn
        self._pool = pool
        self._ts = _validate_ts_config(ts_config)
        self._mask_pii = mask_pii
        self._trace_revision_window_s = float(trace_revision_window_s)

    @asynccontextmanager
    async def _conn(self) -> AsyncIterator[Any]:
        # Typed Any: the dict_row factory makes rows dicts at runtime, which the
        # static psycopg row type (tuple) doesn't reflect.
        if self._pool is not None:
            async with self._pool.connection() as conn:
                previous_rows = conn.row_factory
                conn.row_factory = dict_row
                try:
                    yield conn
                finally:
                    # restore the pooled connection's factory so a later borrower that expects
                    # tuple rows isn't handed dicts (a shared-pool consumer must not inherit this).
                    conn.row_factory = previous_rows
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
        """Close a session AND stamp the turns that close covers.

        **Two markers, written together, because one of them is deletable.** The `sessions`
        row is the summary's home and it is what `idle_sessions` used to consult alone; the
        `turns.consolidated_at` stamp is the same fact recorded on rows a scope cleanup keeps.
        The stamp carries the session's OWN `ended_at`, not a second `now()`, and covers only
        `created_at <= ended_at` — so while the `sessions` row exists the two markers agree
        turn for turn (see `idle_sessions`), and when it is gone the stamp still remembers.

        Both statements run in ONE transaction. `_conn()` is autocommit, so without the block
        a crash between them would leave the pair disagreeing — and BOTH orderings of the
        disagreement are wrong in a different way, which is why the answer is atomicity rather
        than a chosen order. Neither written → the next tick re-picks and re-consolidates,
        which costs tokens and loses nothing; that is the direction this whole marker fails in.

        NO TEST GUARDS THE TRANSACTION, and that is not an oversight — it was measured.
        Removing `conn.transaction()` leaves the whole suite green, and green with reason: the
        two statements always both run when the process survives, so what the block buys is
        only the window between them, which a test cannot open without killing the process
        mid-statement. Said here for the same reason the index CREATE/DROP order is said above.

        `RETURNING` is what makes the stamp honest about the cross-scope guard below: when a
        colliding id from another scope makes the conflict update a no-op, no row comes back,
        and stamping that scope's turns would silence a Tier-3 read nobody performed.
        """
        async with self._conn() as conn, conn.transaction():
            if scope:
                # UPSERT: a host that only save_turn()s has no sessions row to update — insert a
                # closed one (keyed by the same session id) so the janitor's idle scan skips it.
                # `sessions` is keyed by id alone, so a colliding id from ANOTHER scope would take
                # this scope's summary (the cross-scope write the read-side fix was written to
                # stop). Guard the conflict update with the scope.
                cur = await conn.execute(
                    "INSERT INTO sessions (id, scope, ended_at, summary) "
                    "VALUES (%s, %s, now(), %s) "
                    "ON CONFLICT (id) DO UPDATE SET ended_at = now(), summary = EXCLUDED.summary "
                    "WHERE sessions.scope = EXCLUDED.scope "
                    "RETURNING scope, ended_at",
                    (session_id, scope, summary))
            else:
                cur = await conn.execute(
                    "UPDATE sessions SET ended_at = now(), summary = %s WHERE id = %s "
                    "RETURNING scope, ended_at",
                    (summary, session_id))
            row = await cur.fetchone()
            if row is None:
                return          # nothing was closed → there is nothing to declare consolidated
            # The scope comes from the row we just wrote, never from the argument: the unscoped
            # branch has none to offer, and this table's ids are known to collide across scopes
            # on a live box. An unscoped `WHERE session_id = %s` would stamp a NEIGHBOUR's turns
            # as consolidated — the same cross-scope write the guard above exists to refuse.
            await conn.execute(
                "UPDATE turns SET consolidated_at = %s "
                "WHERE scope = %s AND session_id = %s "
                "  AND consolidated_at IS NULL AND created_at <= %s",
                (row["ended_at"], row["scope"], session_id, row["ended_at"]))

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
        # per tick. That clause is correct and stays; what follows is about the OTHER two.
        #
        # ── THE RULE THIS QUERY AND THE HOST'S PURGE CONTRACT COLLIDE OVER ──────────────────
        #
        # **`sessions` is bookkeeping, `turns` is the measurement, and a scope cleanup that
        # deletes the first while sparing the second re-arms this loop for ever.** The two
        # halves of that sentence are written in two repositories and neither used to know
        # about the other: here, pickability is derived from `turns`; in `cogno-host`,
        # `tests/unit/test_identity_purge_contract.py::test_the_purge_never_deletes_the_
        # measurement` forbids the identity purge from touching `turns`, because turns are what
        # the promise audit measures. Both rules are right. Nothing said which one governs
        # `sessions` — and that ABSENCE, not either rule, was the defect.
        #
        # Measured 2026-09-09, nobody provoking it: an authorised enumerated cleanup removed 11
        # scopes' memories, traces and `sessions` rows and deliberately left `turns` standing,
        # obeying the host rule to the letter. Nine minutes later ONE tick re-consolidated all
        # 10 surviving scopes — 5,258 tokens, one of them a scope whose identity had been gone
        # for 90 minutes returning with a byte-identical 407-token spend and writing four fresh
        # memories. The remediation manufactured what it was remediating, and would have done
        # so every tick for ever, because the "already handled" mark WAS the row the cleanup
        # deleted. Replayed read-only over the live box: wipe `sessions`, keep `turns`, and the
        # pre-2026-09-09 predicate returns 311 pickable sessions on every single tick.
        #
        # So the mark moved to where the cleanup cannot reach it: `turns.consolidated_at`,
        # written by `close_session` on the very rows the consolidation read. What a scope
        # cleanup may do is now stated in ONE sentence, in both repositories:
        #
        #     Delete `sessions` for a scope only together with that scope's `turns`
        #     (`purge_scope` does exactly that). Sparing `turns` is legitimate — they are the
        #     measurement — but then the `sessions` rows must be spared too, because on their
        #     own they are the record of work already done, not content.
        #
        # ── why the marker is a CONJUNCT and never a replacement ────────────────────────────
        #
        # `AND t.consolidated_at IS NULL` can only ever REMOVE a session from this result, never
        # add one, and three properties fall out of that asymmetry:
        #
        #  * **The legitimate re-pick is untouched.** `close_session` stamps exactly the turns
        #    with `created_at <= ended_at`, so while the `sessions` row exists the new conjunct
        #    and the old `t.created_at > s.ended_at` disjunct select the same turns, row for
        #    row. A genuine new burst is unstamped and re-arms the session as it always did.
        #  * **The upgrade is inert.** Every pre-existing turn reads NULL, i.e. exactly what the
        #    old predicate already assumed, so nothing re-consolidates that was not already due.
        #  * **A lost stamp costs tokens, never memory.** If the write is missing the session is
        #    picked again and consolidated again; the failure mode is a duplicate LLM read, the
        #    same direction the host janitor's orphan gate deliberately fails in.
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT t.session_id AS id, t.scope AS scope, "
                "       min(t.created_at) AS started_at, max(t.created_at) AS last_activity "
                "FROM turns t LEFT JOIN sessions s ON s.id = t.session_id "
                "WHERE (s.id IS NULL OR s.ended_at IS NULL OR t.created_at > s.ended_at) "
                "  AND t.consolidated_at IS NULL "
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
    async def save_turn(self, turn: TurnRecord) -> int:
        """Persist one turn; return the ``turn_n`` actually written (0 if none was).

        **The coordinate can be ALLOCATED here, and that is the point.** Pass
        ``turn.turn_n = ALLOCATE_TURN_N`` (any negative) and the number is
        chosen by this statement, as
        ``max(turn_n) + 1`` over the rows that already exist for
        ``(scope, session_id)`` — read and written inside ONE statement, so the
        read cannot go stale between the two.

        A caller that computes the next number itself and passes it in has a
        time-of-check/time-of-use hole that no amount of care closes: two workers
        serving the same session read the same maximum, and the loser's turn is
        silently discarded by the ``ON CONFLICT``. That is not hypothetical — it
        is the production incident this method was rewritten for, where a
        counter kept OUTSIDE this table replayed coordinates that were years of
        conversation old, and every replayed turn vanished without a trace.

        Allocation is still racy at the statement level, and deliberately so: two
        concurrent inserts compute the same ``max + 1``, one lands, the other
        conflicts and returns no row. The loser then **re-reads and retries**
        (``_ALLOC_ATTEMPTS``), which is what turns the race into two rows instead
        of one. Under READ COMMITTED the retry sees the winner's committed row,
        so it converges in one extra pass; the loop exists for the pathological
        case, not the ordinary one.

        A caller-pinned ``turn_n >= 0`` is honoured verbatim — a backfill, an
        import or a replay is re-stating history, not appending to it — and a
        collision there is reported as a loss rather than absorbed.

        Returns 0 when nothing was written. It never raises on a conflict: this
        runs after the contact already has their reply, and taking the turn down
        to report a bookkeeping failure trades a lost row for a lost turn.
        """
        _require_scope(turn.scope)
        user_input, response = turn.user_input, turn.response
        if self._mask_pii and turn.pii_types:
            user_input, response = _mask_pii(user_input), _mask_pii(response)
        payload = (user_input, response, turn.feedback, turn.goal, turn.goal_status,
                   turn.sentiment, turn.domains, turn.pii_types, turn.voiced_by)

        if turn.turn_n is not None and turn.turn_n >= 0:
            async with self._conn() as conn:
                cur = await conn.execute(
                    """INSERT INTO turns
                       (scope, session_id, turn_n, user_input, response, feedback,
                        goal, goal_status, sentiment, domains, pii_types, voiced_by)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (scope, session_id, turn_n) DO NOTHING
                       RETURNING turn_n""",
                    (turn.scope, turn.session_id, turn.turn_n) + payload)
                row = await cur.fetchone()
            if row is None:
                write_loss.record(write_loss.TURN_DISCARDED, scope=turn.scope,
                                  session=turn.session_id, turn=turn.turn_n,
                                  reason="coordinate_taken", pinned="true")
                return 0
            return int(row["turn_n"])

        # ``max(turn_n) + 1`` computed INSIDE the insert. The aggregate over a
        # filtered set always yields exactly one row (NULL → 0 for a new session),
        # so the SELECT feeds the INSERT unconditionally. Types are cast
        # explicitly: in a SELECT list a bare placeholder has no inferable type,
        # unlike the VALUES form above.
        for attempt in range(_ALLOC_ATTEMPTS):
            async with self._conn() as conn:
                cur = await conn.execute(
                    """INSERT INTO turns
                       (scope, session_id, turn_n, user_input, response, feedback,
                        goal, goal_status, sentiment, domains, pii_types, voiced_by)
                       SELECT %s::text, %s::uuid, COALESCE(max(t.turn_n), 0) + 1,
                              %s::text, %s::text, %s::smallint, %s::text, %s::text,
                              %s::text, %s::text[], %s::text[], %s::text
                         FROM turns t
                        WHERE t.scope = %s::text AND t.session_id = %s::uuid
                       ON CONFLICT (scope, session_id, turn_n) DO NOTHING
                       RETURNING turn_n""",
                    (turn.scope, turn.session_id) + payload
                    + (turn.scope, turn.session_id))
                row = await cur.fetchone()
            if row is not None:
                turn.turn_n = int(row["turn_n"])
                return turn.turn_n
        write_loss.record(write_loss.TURN_ALLOCATION_EXHAUSTED, scope=turn.scope,
                          session=turn.session_id, attempts=_ALLOC_ATTEMPTS)
        return 0

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
                "goal_status, sentiment, domains, pii_types, voiced_by, created_at FROM turns ")
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
    async def save_turn_trace(self, trace: "TurnTrace") -> bool:
        """Persist one turn trace; return whether it was stored.

        ``created_at`` is honoured when the caller sets it (a backfill, an import,
        a test seeding history); absent, the column default stamps the row. The
        in-memory adapter always did this — the Postgres one silently dropped it,
        which made every imported trace "now" and any time-window read over them a
        lie.

        **An older trace is never overwritten by a newer one, and ``created_at``
        never moves.** The upsert used to be an unconditional
        ``DO UPDATE SET trace = EXCLUDED.trace``, with ``created_at`` left out of
        the SET — so a trace arriving at an occupied coordinate replaced content
        that was days old while the row went on advertising the old date. The
        column did not merely go stale: it actively lied, and every read that
        trusted it (a time-window audit, an incident reconstruction) inherited the
        lie. Real ages had to be recovered from ``xmin``.

        The rule is a comparison against ``TRACE_REVISION_WINDOW_S``: a stored
        trace stays revisable for as long as the turn that produced it could still
        be writing (a correction loop, a re-voice, a backfill batch re-running its
        own rows), and is immutable afterwards. A write arriving days later at an
        occupied coordinate is, by construction, a DIFFERENT turn wearing an old
        turn's number. Keeping the stored row is the safe direction because it is
        the one already referenced: a ``turns`` row of the same age sits beside it,
        and replacing only the trace leaves the pair describing two different
        conversations — which is precisely the state the incident left behind.

        Refusing beats versioning here for a reason worth stating: a suffixed
        second version needs a schema migration and changes what
        ``traces_for_session`` returns — one row per coordinate is a promise every
        reader in the ecosystem already relies on, and a P0 that is racing a live
        data loss is the wrong moment to renegotiate it. Nothing is lost that was
        not already going to be lost, and the refusal is counted.
        """
        _require_scope(trace.scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "INSERT INTO turn_traces (scope, session_id, turn_n, trace, created_at) "
                "VALUES (%s, %s, %s, %s::jsonb, COALESCE(%s, now())) "
                "ON CONFLICT (scope, session_id, turn_n) DO UPDATE SET trace = EXCLUDED.trace "
                "  WHERE EXCLUDED.created_at <= turn_traces.created_at "
                "                            + make_interval(secs => %s) "
                "RETURNING turn_n",
                (trace.scope, trace.session_id, trace.turn_n, json.dumps(trace.trace),
                 trace.created_at, self._trace_revision_window_s))
            row = await cur.fetchone()
        if row is None:
            write_loss.record(write_loss.TRACE_OVERWRITE_REFUSED, scope=trace.scope,
                              session=trace.session_id, turn=trace.turn_n,
                              reason="stored_trace_is_older")
            return False
        return True

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
                "goal_status, sentiment, domains, pii_types, voiced_by, created_at "
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
                "goal_status, sentiment, domains, pii_types, voiced_by, created_at "
                f"FROM turns WHERE {self._SUBTREE} "
                "ORDER BY created_at DESC, id DESC LIMIT %s OFFSET %s",
                (scope_prefix, like, limit, offset))
            rows = await cur.fetchall()
            ccur = await conn.execute(
                f"SELECT count(*) AS c FROM turns WHERE {self._SUBTREE}", (scope_prefix, like))
            crow = await ccur.fetchone()
        total = crow["c"] if crow else 0
        return [self._row_to_turn(r) for r in rows], int(total)

    async def memory_scopes(self) -> list[str]:
        """Every scope with a memory row. NO `_require_scope` — see the port for the argument.

        `DISTINCT scope` over `memories`, not over `turns`: the two disagree, and the difference
        is the point. A scope can hold memories and no turns (a consolidated session whose rows
        were pruned), and — the case that matters — a scope can outlive the tenant that owned it.
        """
        async with self._conn() as conn:
            cur = await conn.execute("SELECT DISTINCT scope FROM memories ORDER BY scope")
            return [r["scope"] for r in await cur.fetchall()]

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
            voiced_by=r["voiced_by"], created_at=r["created_at"])

    # ── memories ─────────────────────────────────────────────────────────
    async def save_memory(self, memory: MemoryRecord) -> None:
        _require_scope(memory.scope)
        mid = memory.id or str(uuid4())
        async with self._conn() as conn:
            await conn.execute(
                # `first_heard_by` is written on INSERT and DELIBERATELY absent from the
                # DO UPDATE: the column answers "who did the contact tell this to FIRST", so the
                # second persona to hear the same fact must not overwrite the first. The
                # asymmetry IS the design — uniformising this list with the one above would
                # silently turn the field into "who mentioned it most recently".
                """INSERT INTO memories
                       (id, scope, category, content, confidence, embedding, first_heard_by)
                   VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                   ON CONFLICT (scope, category, content) DO UPDATE SET
                       confidence = EXCLUDED.confidence,
                       embedding  = COALESCE(EXCLUDED.embedding, memories.embedding)""",
                (mid, memory.scope, memory.category, memory.content, memory.confidence,
                 _vec(memory.embedding), memory.first_heard_by))

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
                f"SELECT id, scope, category, content, confidence, feedback_score, first_heard_by, created_at, "
                f"  ({w.vector} * (1.0 - (embedding <=> %s::vector)) "
                f"   + {w.lexical} * ts_rank_cd(tsv, {tsq}) "
                f"   + COALESCE(feedback_score, 0) * {w.feedback}) AS score "
                f"FROM memories {where} AND embedding IS NOT NULL "
                f"ORDER BY score DESC LIMIT %s")
            full = [_vec(q_emb), q_txt] + params + [limit]
        elif q_emb is not None:
            sql = (f"SELECT id, scope, category, content, confidence, feedback_score, first_heard_by, created_at "
                   f"FROM memories {where} AND embedding IS NOT NULL "
                   f"ORDER BY (embedding <=> %s::vector) - COALESCE(feedback_score,0)*0.1 ASC LIMIT %s")
            full = params + [_vec(q_emb), limit]
        elif q_txt is not None:
            sql = (f"SELECT id, scope, category, content, confidence, feedback_score, first_heard_by, created_at "
                   f"FROM memories {where} AND tsv @@ {tsq} "
                   f"ORDER BY ts_rank_cd(tsv, {tsq}) DESC LIMIT %s")
            full = params + [q_txt, q_txt, limit]
        else:
            sql = (f"SELECT id, scope, category, content, confidence, feedback_score, first_heard_by, created_at "
                   f"FROM memories {where} ORDER BY created_at DESC LIMIT %s")
            full = params + [limit]

        async with self._conn() as conn:
            cur = await conn.execute(sql, full)
            rows = await cur.fetchall()
        return [MemoryRecord(scope=r["scope"], category=r["category"], content=r["content"],
                             confidence=r["confidence"], feedback_score=r["feedback_score"],
                             first_heard_by=r["first_heard_by"],
                             created_at=r["created_at"], id=str(r["id"])) for r in rows]

    async def scan_memories(self, scope: str, *, after_id: "Optional[str]" = None,
                            limit: int = 1000) -> list[MemoryRecord]:
        _require_scope(scope)
        sql = ("SELECT id, scope, category, content, confidence, feedback_score, first_heard_by, created_at "
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
                             first_heard_by=r["first_heard_by"],
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
                              max_confidence: Optional[float] = None,
                              dry_run: bool = False) -> int:
        """Delete matching memories and return how many — or, with ``dry_run``, how many WOULD go.

        **ONE predicate, two verbs.** The filter is built once and only the leading clause
        changes. A caller that counted with its own query would be re-deriving the rule that
        decides a deletion, and the two would drift the day a filter is added — the exact shape
        of defect this codebase keeps finding. Here "what would go" and "what went" cannot
        disagree, because they are the same WHERE.

        ``dry_run`` exists because retention is irreversible and a first run must be readable
        before it is armed: nobody should learn what a 120-day rule removes by watching it
        remove it.
        """
        _require_scope(scope)
        head = "SELECT count(*) FROM memories" if dry_run else "DELETE FROM memories"
        sql = f"{head} WHERE scope = %s"
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
            if dry_run:
                row = await cur.fetchone()
                return int((row or {}).get("count") or 0)
            return cur.rowcount

    async def purge_scope(self, scope: str) -> int:
        """Erase every row this store holds for a scope. RTBF's engram half.

        **All four tables or none — and `sessions` may never be dropped without `turns`.**
        This method is already correct and this docstring is the reason it must stay that way:
        `idle_sessions` derives pickability from `turns` and is suppressed by a mark that
        `close_session` writes onto those same turns, so a cleanup that removes a scope's
        `sessions` rows and spares its `turns` leaves the janitor re-consolidating that scope
        for ever. Measured 2026-09-09 — the full account is in `idle_sessions`, and the same
        rule is written on the host side in
        `tests/unit/test_identity_purge_contract.py::test_the_purge_never_deletes_the_measurement`,
        whose "never delete the measurement" is the rule this one had been silently colliding
        with. Sparing `turns` is legitimate; sparing `turns` while deleting `sessions` is not.
        """
        _require_scope(scope)
        total = 0
        async with self._conn() as conn:
            # turn_traces + turns + sessions + memories all carry the scope column; drop them all.
            # `turns` is deleted alongside `sessions`, and the marker `close_session` stamped on
            # it goes with the row — a scope's consolidation bookkeeping outliving neither the
            # scope nor the turns it describes.
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


# ── audience predicates ──────────────────────────────────────────────────────
#
# Built from the constants rather than typed into the SQL, so the vocabulary has one source and
# a rename cannot leave a string literal behind. These are our own values, never user input.
#
# EDGE: staff sees everything; anyone sees a tenant fact; an identity sees its own.
# NOTE the `%s <> ''`, and it is a LEAK if it is missing. `audience_for(None)` returns `""` —
# which is what a contact with no identity yet (pre-registration) produces — and binding that
# raw makes `e.audience = %s` read `e.audience = ''`: TRUE for every UNCLASSIFIED row, i.e. all
# the legacy edges the migration has not reached. The in-memory adapter said False for the same
# input (`audience_can_read("", "")`), so the two stores disagreed in the direction that leaks.
# `tests/test_audience_parity.py` now pins them against the one pure rule.
_EDGE_VISIBLE = (f"(%s = '{AUDIENCE_STAFF}' "
                 f"OR e.audience = '{AUDIENCE_TENANT}' "
                 f"OR (%s <> '' AND e.audience = %s))")
# NODE: DERIVED — visible when some visible edge touches it. Staff short-circuits BEFORE the
# EXISTS, because an orphan node (no edges at all) must still be visible to staff; deriving it
# for staff too would hide every node `ingest_entities` created before any relation existed.
_NODE_VISIBLE = (f"""(%s = '{AUDIENCE_STAFF}' OR EXISTS (
        SELECT 1 FROM knowledge_edges e
        WHERE (e.source_id = n.id OR e.target_id = n.id) AND e.scope = n.scope
          AND (e.audience = '{AUDIENCE_TENANT}'
               OR (%s <> '' AND e.audience = %s))))""")


def _edge_from_row(scope: str, row: Any) -> GraphEdge:
    """One place turns a row into an edge. Written when the third read path appeared: three
    hand-built constructors is how one of them silently stops carrying a new column."""
    return GraphEdge(scope=scope, source=row["source"], target=row["target"],
                     relation=row["relation"], confidence=row["confidence"],
                     source_session=row["source_session"],
                     attributes=row.get("attributes") or {},
                     status=row.get("status") or EDGE_ACCEPTED,
                     audience=row.get("audience") or AUDIENCE_UNCLASSIFIED,
                     created_at=row.get("created_at"))


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
                   ON CONFLICT (scope, engram_fold(label), node_type) DO UPDATE SET
                       -- A grafia com DIACRÍTICOS ganha, e só sobe, nunca desce. Sem isto o
                       -- primeiro a chegar ficava para sempre — e como o contacto escreve o nome
                       -- sem acento metade das vezes, "Jose" chegava primeiro e a linha nunca mais
                       -- voltava a ser "José". Era o contrário exacto do que o `folding` promete:
                       -- o rótulo original é o que se guarda e se mostra; perder o acento no nome
                       -- de uma pessoa é o que esta funcionalidade existe para EVITAR.
                       -- Determinístico e sem oscilar: uma vez acentuado, fica.
                       label      = CASE
                           WHEN engram_fold(EXCLUDED.label)
                                  <> lower(EXCLUDED.label COLLATE "und-x-icu")
                            AND engram_fold(knowledge_nodes.label)
                                  =  lower(knowledge_nodes.label COLLATE "und-x-icu")
                           THEN EXCLUDED.label ELSE knowledge_nodes.label END,
                       attributes = knowledge_nodes.attributes || EXCLUDED.attributes,
                       embedding  = COALESCE(EXCLUDED.embedding, knowledge_nodes.embedding),
                       updated_at = now()
                   RETURNING id""",
                (node.scope, node.label, node.node_type, json.dumps(node.attributes),
                 _vec(node.embedding)))
            row = await cur.fetchone()
        return row["id"]

    async def _resolve_node_id(self, conn, scope: str, label: str) -> int:
        """O id do nó a que uma aresta se liga — criando-o se não existir.

        `ORDER BY (label = %s) DESC` e não `LIMIT 1` cru: a dobragem faz `José/PERSON` e
        `Jose/CONCEPT` casarem os dois, e um `LIMIT 1` sem ordem escolhia à SORTE — duas arestas
        pedidas para nós diferentes acabavam no mesmo, e qual delas ganhava dependia da ordem
        física das linhas. Preferir o rótulo exacto é determinístico e é o que quem chama quis
        dizer. Diferente do `_one_node_id`, que RECUSA quando é ambíguo: aqui a operação é criar
        uma ligação, não destruir nem mudar privacidade, portanto escolher o melhor palpite é
        preferível a falhar o turno."""
        cur = await conn.execute(
            "SELECT id FROM knowledge_nodes WHERE scope = %s AND engram_fold(label) = engram_fold(%s) "
            " ORDER BY (label = %s) DESC, id LIMIT 1",
            (scope, label, label))
        row = await cur.fetchone()
        if row:
            return row["id"]
        # Idempotent insert: two concurrent upsert_edge calls referencing a not-yet-existing
        # label both miss the SELECT above; a bare INSERT then races the uq(scope, engram_fold(label),
        # node_type) index and the loser raises IntegrityError, failing that turn. ON CONFLICT
        # DO NOTHING makes the loser return no row → re-SELECT the winner's id.
        cur = await conn.execute(
            "INSERT INTO knowledge_nodes (scope, label) VALUES (%s, %s) "
            "ON CONFLICT (scope, engram_fold(label), node_type) DO NOTHING RETURNING id",
            (scope, label))
        row = await cur.fetchone()
        if row:
            return row["id"]
        cur = await conn.execute(
            "SELECT id FROM knowledge_nodes WHERE scope = %s AND engram_fold(label) = engram_fold(%s) LIMIT 1",
            (scope, label))
        row = await cur.fetchone()
        return row["id"]

    async def upsert_edge(self, edge: GraphEdge) -> None:
        _require_scope(edge.scope)
        async with self._conn() as conn:
            src = await self._resolve_node_id(conn, edge.scope, edge.source)
            tgt = await self._resolve_node_id(conn, edge.scope, edge.target)
            await conn.execute(
                f"""INSERT INTO knowledge_edges
                   (scope, source_id, target_id, relation, confidence, source_session,
                    attributes, status, audience)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
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
                           WHEN knowledge_edges.status = '{EDGE_ACCEPTED}' AND EXCLUDED.status = '{EDGE_PROPOSED}'
                           THEN knowledge_edges.attributes
                           ELSE knowledge_edges.attributes || EXCLUDED.attributes END,
                       -- a re-assertion PROMOTES a proposal and never demotes a verdict;
                       -- `rejected` is sticky on purpose (see the in-memory twin: this path
                       -- cannot tell a deliberate correction from the LLM re-emitting the same
                       -- edge, so `set_edge_status` is the only way back)
                       status         = CASE WHEN knowledge_edges.status = '{EDGE_PROPOSED}'
                                             THEN EXCLUDED.status ELSE knowledge_edges.status END,
                       -- a re-assertion may NARROW the audience but never widen it: the same
                       -- reasoning as `status`, and the direction that cannot leak. An edge
                       -- already private to someone stays private even if a later writer
                       -- forgets to declare.
                       audience       = CASE WHEN knowledge_edges.audience = ''
                                             THEN EXCLUDED.audience ELSE knowledge_edges.audience END""",
                (edge.scope, src, tgt, edge.relation, edge.confidence, edge.source_session,
                 json.dumps(edge.attributes or {}, default=str),
                 sanitize_edge_status(edge.status), sanitize_audience(edge.audience)))

    async def find_node(self, scope: str, label: str, *,
                        audience: str) -> Optional[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT n.id, n.scope, n.label, n.node_type, n.attributes, n.created_at, "
                "n.updated_at FROM knowledge_nodes n "
                "WHERE n.scope = %s AND engram_fold(n.label) = engram_fold(%s) "
                f"AND {_NODE_VISIBLE} LIMIT 1",
                (scope, label, audience, audience, audience))
            row = await cur.fetchone()
        if not row:
            return None
        return GraphNode(scope=row["scope"], label=row["label"], node_type=row["node_type"],
                         attributes=row["attributes"], id=row["id"],
                         created_at=row["created_at"], updated_at=row["updated_at"])

    async def find_nodes_by_embedding(self, scope: str, embedding: list[float],
                                      *, audience: str, limit: int = 5,
                                      related_only: bool = False) -> list[GraphNode]:
        _require_scope(scope)
        # The EXISTS is the whole feature: a caller that will WALK from these nodes wants
        # candidates that can be walked from. An isolated node is a legitimate row — the node
        # list in the dashboard shows it, and staff may search it — but it spends one of the
        # caller's few slots and returns nothing. BOTH ends count: half the relations point AT
        # the person (``Rex OWNED_BY José``), so ``source_id`` alone would halve the recall.
        # `status = 'accepted'` is not decoration: `walk` (below) traverses ACCEPTED edges only,
        # so counting an unreviewed one here picks a node the caller cannot walk from — the very
        # thing this filter exists to prevent, and a PESSIMISATION, because it evicts a nearer
        # candidate for a farther one that also goes nowhere. Reachable by design: the host
        # writes proximity relations as PROPOSED (`propose_relations`), which is exactly the
        # class of edge a turn wants. Anything the walk will not traverse is not "related".
        related = (" AND EXISTS (SELECT 1 FROM knowledge_edges e "
                   "WHERE (e.source_id = n.id OR e.target_id = n.id) "
                   f"AND e.status = '{EDGE_ACCEPTED}')") if related_only else ""
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT n.id, n.scope, n.label, n.node_type, n.attributes "
                "FROM knowledge_nodes n WHERE n.scope = %s AND n.embedding IS NOT NULL "
                f"AND {_NODE_VISIBLE}{related} "
                "ORDER BY n.embedding <=> %s::vector LIMIT %s",
                (scope, audience, audience, audience, _vec(embedding), limit))
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"]) for r in rows]

    async def walk(self, scope: str, start_label: str, *, audience: str,
                   max_depth: int = 2) -> list[GraphEdge]:
        _require_scope(scope)
        async with self._conn() as conn:
            # Emit the edge used at each hop, bounded by depth (mirrors the
            # in-memory BFS: only nodes at depth < max_depth expand their edges).
            cur = await conn.execute(
                f"""
                WITH RECURSIVE walk AS (
                    SELECT n.id AS node_id, 0 AS depth, NULL::bigint AS edge_id,
                           ARRAY[n.id] AS path
                    FROM knowledge_nodes n
                    WHERE n.scope = %s AND engram_fold(n.label) = engram_fold(%s)
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
                          AND e.status = '{EDGE_ACCEPTED}'
                          -- ...and neither may an edge this reader is not allowed to see: an
                          -- invisible edge that still ROUTED the traversal would disclose the
                          -- neighbour it leads to.
                          AND {_EDGE_VISIBLE}
                    ) nxt ON true
                    WHERE w.depth < %s AND NOT (nxt.node_id = ANY(w.path))
                )
                SELECT DISTINCT e.id, sn.label AS source, tn.label AS target,
                       e.relation, e.confidence, e.source_session, e.attributes, e.status,
                       e.audience, e.created_at
                FROM walk w
                JOIN knowledge_edges e ON e.id = w.edge_id
                JOIN knowledge_nodes sn ON sn.id = e.source_id
                JOIN knowledge_nodes tn ON tn.id = e.target_id
                """,
                (scope, start_label, scope, audience, audience, audience, max_depth))
            rows = await cur.fetchall()
        return [_edge_from_row(scope, r) for r in rows]

    async def pending_edges(self, scope: str, *, audience: str,
                            limit: int = 100) -> list[GraphEdge]:
        _require_scope(scope)
        if limit <= 0:          # a negative LIMIT raises here and truncated in memory; agree
            return []
        async with self._conn() as conn:
            cur = await conn.execute(
                f"""SELECT sn.label AS source, tn.label AS target, e.relation, e.confidence,
                          e.source_session, e.attributes, e.status, e.audience,
                          e.created_at
                   FROM knowledge_edges e
                   JOIN knowledge_nodes sn ON sn.id = e.source_id
                   JOIN knowledge_nodes tn ON tn.id = e.target_id
                   WHERE e.scope = %s AND e.status = %s AND {_EDGE_VISIBLE}
                   -- OLDEST first, and the direction is the point: with a `limit` and no
                   -- cursor, newest-first makes the oldest proposals — the ones a curator most
                   -- needs to clear — permanently unreachable, and the queue never drains. The
                   -- in-memory adapter returns insertion order, which is the same thing; a
                   -- review found the two disagreeing and the disagreement was invisible.
                   ORDER BY e.created_at ASC, e.id ASC
                   LIMIT %s""",
                (scope, EDGE_PROPOSED, audience, audience, audience, limit))
            rows = await cur.fetchall()
        return [_edge_from_row(scope, r) for r in rows]

    async def set_edge_status(self, scope: str, source: str, target: str, relation: str,
                              status: str) -> bool:
        _require_scope(scope)
        async with self._conn() as conn:
            # por ID, não por rótulo dobrado: dois nós que dobram igual mas têm `node_type`
            # diferente coexistem legalmente, e um UPDATE por rótulo atingia os DOIS. No
            # `set_edge_audience` isso é uma mudança de PRIVACIDADE numa aresta que ninguém
            # escolheu.
            src = await self._one_node_id(conn, scope, source, op="set_edge_status")
            tgt = await self._one_node_id(conn, scope, target, op="set_edge_status")
            if src is None or tgt is None:
                return False
            cur = await conn.execute(
                "UPDATE knowledge_edges SET status = %s "
                " WHERE scope = %s AND source_id = %s AND target_id = %s AND relation = %s",
                (require_edge_status(status), scope, src, tgt, relation))
        return bool(cur.rowcount)

    async def set_edge_audience(self, scope: str, source: str, target: str, relation: str,
                                audience: str) -> bool:
        """Explicit re-classification — the only way back from a migration."""
        _require_scope(scope)
        async with self._conn() as conn:
            # por ID, não por rótulo dobrado: dois nós que dobram igual mas têm `node_type`
            # diferente coexistem legalmente, e um UPDATE por rótulo atingia os DOIS. No
            # `set_edge_audience` isso é uma mudança de PRIVACIDADE numa aresta que ninguém
            # escolheu.
            src = await self._one_node_id(conn, scope, source, op="set_edge_audience")
            tgt = await self._one_node_id(conn, scope, target, op="set_edge_audience")
            if src is None or tgt is None:
                return False
            cur = await conn.execute(
                "UPDATE knowledge_edges SET audience = %s "
                " WHERE scope = %s AND source_id = %s AND target_id = %s AND relation = %s",
                (sanitize_audience(audience), scope, src, tgt, relation))
        return bool(cur.rowcount)

    async def neighbors(self, scope: str, label: str, *, audience: str) -> list[GraphNode]:
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                f"""SELECT DISTINCT nn.id, nn.scope, nn.label, nn.node_type, nn.attributes
                   FROM knowledge_nodes n
                   JOIN knowledge_edges e ON (e.source_id = n.id OR e.target_id = n.id)
                   JOIN knowledge_nodes nn
                     ON nn.id = CASE WHEN e.source_id = n.id THEN e.target_id ELSE e.source_id END
                   WHERE n.scope = %s AND engram_fold(n.label) = engram_fold(%s)
                     -- an unreviewed edge still DISCLOSES its endpoint; see the in-memory twin
                     AND e.status = '{EDGE_ACCEPTED}'
                     AND {_EDGE_VISIBLE}""", (scope, label, audience, audience, audience))
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"]) for r in rows]

    async def get_node_context(self, scope: str, label: str, *,
                               audience: str) -> Optional[NodeContext]:
        _require_scope(scope)
        node = await self.find_node(scope, label, audience=audience)
        if node is None:
            return None
        edges = await self.walk(scope, label, audience=audience, max_depth=1)
        edges = [e for e in edges
                 if fold_label(label) in (fold_label(e.source), fold_label(e.target))]
        return NodeContext(node=node, edges=edges,
                           neighbors=await self.neighbors(scope, label, audience=audience))

    async def graph_stats(self, scope: str, *, audience: str, top: int = 5) -> GraphStats:
        """The whole dashboard summary in TWO aggregated reads, replacing ``1 + 3N``.

        The caller (the host's ``knowledge_stats``) listed every node and then asked
        ``get_node_context`` for each; that helper is ``find_node`` + ``walk`` + ``neighbors``,
        so the real cost was **1165 queries for the 388 nodes of the live box**, on every page
        open, growing with the graph. Nothing in the port could answer "how connected is each
        node" in bulk, which is why the caller had no better way to write it.

        **The semantics are the old ones, deliberately** — this is a cost change, not a meaning
        change, and a performance PR that quietly moves a number is the worst kind:

        * nodes counted by ``_NODE_VISIBLE`` — DERIVED for a non-staff reader, so an orphan is
          staff-only, exactly as ``list_nodes`` answered;
        * edges DISTINCT on ``(source_id, target_id, relation)`` and **ACCEPTED only**, because
          the old degree came from ``walk``, and ``walk`` traverses no other status;
        * degree counts both ends, since half the relations point AT the node.

        Two reads and not one because the by-type histogram and the degree ranking group by
        different things; forcing them together buys nothing and costs readability.
        """
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT n.node_type, count(*) AS c FROM knowledge_nodes n "
                f"WHERE n.scope = %s AND {_NODE_VISIBLE} GROUP BY n.node_type",
                (scope, audience, audience, audience))
            by_type = {r["node_type"]: int(r["c"]) for r in await cur.fetchall()}

            # `vis` is the DISTINCT visible+accepted edge set — the same set the old code
            # rebuilt as a Python `set` of `(source, target, relation)` across N contexts.
            # `deg` unrolls it to one row per endpoint so a LEFT JOIN gives every visible node
            # its degree INCLUDING zero (a node no visible edge touches is still staff-visible,
            # and dropping it here would silently shorten the ranking).
            cur = await conn.execute(
                f"""WITH vis AS (
                        SELECT DISTINCT e.source_id, e.target_id, e.relation
                          FROM knowledge_edges e
                         WHERE e.scope = %s AND e.status = %s AND {_EDGE_VISIBLE}
                    ), deg AS (
                        SELECT source_id AS nid FROM vis
                        UNION ALL
                        SELECT target_id AS nid FROM vis
                    )
                    SELECT n.id, n.scope, n.label, n.node_type, n.attributes,
                           count(d.nid) AS degree,
                           (SELECT count(*) FROM vis) AS total_edges
                      FROM knowledge_nodes n
                      LEFT JOIN deg d ON d.nid = n.id
                     WHERE n.scope = %s AND {_NODE_VISIBLE}
                     GROUP BY n.id, n.scope, n.label, n.node_type, n.attributes
                     -- `n.id` and not `n.label`: the previous version ranked over whatever
                     -- `list_nodes` returned, which is `ORDER BY id`. Ranking by label instead
                     -- reorders TIES, and `test_knowledge_walk_and_stats_shape` caught it — two
                     -- nodes of degree 1 swapped places. A tie-break is behaviour, and this
                     -- change is meant to cost less, not to mean anything different.
                     ORDER BY degree DESC, n.id
                     LIMIT %s""",
                (scope, EDGE_ACCEPTED, audience, audience, audience,
                 scope, audience, audience, audience, max(0, top)))
            rows = await cur.fetchall()

        # NO VISIBLE NODE ⟹ NO VISIBLE EDGE, and that is an invariant rather than a guess —
        # which is why this is a `0` and not a third query. Every accepted, visible edge has
        # endpoints that are nodes of this scope, and for a non-staff reader a visible edge is
        # exactly what MAKES its endpoints visible (`_NODE_VISIBLE` is derived from it); for
        # staff every node of the scope is visible, so an empty ranking means an empty scope.
        # Probed against real SQL on the three ways to get here — empty scope, orphans only, and
        # an accepted edge belonging to ANOTHER contact — and all three give zero edges:
        # `test_no_visible_node_means_no_visible_edge`. The first cut asked a third query "not
        # to report a zero we did not measure"; the zero is measured, by the invariant.
        total_edges = int(rows[0]["total_edges"]) if rows else 0
        return GraphStats(
            total_nodes=sum(by_type.values()), total_edges=total_edges, by_type=by_type,
            top_connected=[(GraphNode(scope=r["scope"], label=r["label"],
                                      node_type=r["node_type"],
                                      attributes=r["attributes"], id=r["id"]),
                            int(r["degree"])) for r in rows])

    async def list_nodes(self, scope: str, *, audience: str,
                         node_type: Optional[str] = None,
                         limit: int = 100) -> list[GraphNode]:
        _require_scope(scope)
        sql = ("SELECT n.id, n.scope, n.label, n.node_type, n.attributes, n.created_at, "
               "n.updated_at FROM knowledge_nodes n WHERE n.scope = %s "
               f"AND {_NODE_VISIBLE}")
        # TRÊS, e nesta ordem: `_NODE_VISIBLE` carrega `%s` três vezes — o curto-circuito de
        # staff, o teste `audience <> ''`, e a igualdade — e vem logo a seguir a `scope`. Dizia
        # "twice" nos três sítios: o comentário mentia sobre exactamente a coisa que faz alguém
        # contar mal os parâmetros ao mexer no predicado, e a lista ao lado (que está certa) tem
        # quatro elementos, não três.
        params: list = [scope, audience, audience, audience]
        if node_type is not None:
            sql += " AND n.node_type = %s"
            params.append(node_type)
        sql += " ORDER BY n.id LIMIT %s"
        params.append(limit)
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            rows = await cur.fetchall()
        return [GraphNode(scope=r["scope"], label=r["label"], node_type=r["node_type"],
                          attributes=r["attributes"], id=r["id"],
                          created_at=r["created_at"], updated_at=r["updated_at"])
                for r in rows]

    async def count_nodes(self, scope: str, *, audience: str,
                          label: Optional[str] = None,
                          node_type: Optional[str] = None) -> int:
        """How many nodes this scope holds, or how many carry ``label``.

        `engram_fold(label)` on both sides, matching `find_node` and the `walk` seed — the unique
        index `uq_nodes_scope_fold_type` is on `(scope, engram_fold(label), node_type)`, so this is the
        expression the planner already has an index for.
        """
        _require_scope(scope)
        sql = ("SELECT count(*) AS n FROM knowledge_nodes n WHERE n.scope = %s "
               f"AND {_NODE_VISIBLE}")
        # TRÊS, e nesta ordem: `_NODE_VISIBLE` carrega `%s` três vezes — o curto-circuito de
        # staff, o teste `audience <> ''`, e a igualdade — e vem logo a seguir a `scope`. Dizia
        # "twice" nos três sítios: o comentário mentia sobre exactamente a coisa que faz alguém
        # contar mal os parâmetros ao mexer no predicado, e a lista ao lado (que está certa) tem
        # quatro elementos, não três.
        params: list = [scope, audience, audience, audience]
        if label is not None:
            sql += " AND engram_fold(n.label) = engram_fold(%s)"
            params.append(label.strip())
        if node_type is not None:
            # EXACTLY the predicate `list_nodes` uses, so "how many" and "which ones" cannot
            # answer about different sets — the whole point of the parameter.
            sql += " AND n.node_type = %s"
            params.append(node_type)
        async with self._conn() as conn:
            cur = await conn.execute(sql, params)
            row = await cur.fetchone()
        return int(row["n"] if isinstance(row, dict) else row[0])

    async def scan_nodes(self, scope: str, *, audience: str,
                         after_id: Optional[int] = None,
                         limit: int = 1000) -> list[GraphNode]:
        _require_scope(scope)
        sql = ("SELECT n.id, n.scope, n.label, n.node_type, n.attributes, n.created_at, "
               "n.updated_at FROM knowledge_nodes n WHERE n.scope = %s "
               f"AND {_NODE_VISIBLE}")
        # TRÊS, e nesta ordem: `_NODE_VISIBLE` carrega `%s` três vezes — o curto-circuito de
        # staff, o teste `audience <> ''`, e a igualdade — e vem logo a seguir a `scope`. Dizia
        # "twice" nos três sítios: o comentário mentia sobre exactamente a coisa que faz alguém
        # contar mal os parâmetros ao mexer no predicado, e a lista ao lado (que está certa) tem
        # quatro elementos, não três.
        params: list = [scope, audience, audience, audience]
        if after_id is not None:
            sql += " AND n.id > %s"
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

    async def has_edges(self, scope: str, label: str) -> bool:
        """Every edge, any audience, any status — see the port for why it takes no audience."""
        _require_scope(scope)
        async with self._conn() as conn:
            cur = await conn.execute(
                """SELECT 1 FROM knowledge_edges e
                   JOIN knowledge_nodes n ON n.id IN (e.source_id, e.target_id)
                   WHERE e.scope = %s AND engram_fold(n.label) = engram_fold(%s) LIMIT 1""",
                (scope, label))
            return await cur.fetchone() is not None

    @staticmethod
    async def _one_node_id(conn, scope: str, label: str, *, op: str) -> "int | None":
        """O id do ÚNICO nó que este rótulo designa — ou None quando é ambíguo.

        A identidade dobrada ignora acento, mas o `node_type` NÃO faz parte do rótulo: `José`
        como PERSON e `Jose` como CONCEPT coexistem legalmente depois da migração — a suíte de
        integração descreve esse par como "exactamente como um tenant lá chega". Um comando por
        rótulo dobrado atinge portanto os DOIS, e foi medido a atingir: `delete_node('José')`
        apagava 2 linhas, com as arestas de ambas atrás por `ON DELETE CASCADE`.

        Desempate: o rótulo EXACTO ganha. Sem exacto e com mais de um candidato, devolve None —
        recusar é o que uma operação destrutiva ou de PRIVACIDADE tem de fazer, porque escolher
        por quem chamou é escolher errado metade das vezes, em silêncio."""
        cur = await conn.execute(
            "SELECT id, label FROM knowledge_nodes "
            " WHERE scope = %s AND engram_fold(label) = engram_fold(%s)", (scope, label))
        rows = await cur.fetchall()

        def _field(r, i, name):
            return r[name] if isinstance(r, dict) else r[i]

        if not rows:
            return None
        if len(rows) > 1:
            exact = [r for r in rows if _field(r, 1, "label") == label]
            if len(exact) != 1:
                logger.warning(
                    "event=node_reference_ambiguous op=%s scope=%s label=%s matches=%d "
                    "reason=fold_matches_several_node_types", op, scope, label, len(rows))
                return None
            rows = exact
        return int(_field(rows[0], 0, "id"))

    async def delete_node(self, scope: str, label: str) -> bool:
        """Apaga UM nó — nunca mais do que um, mesmo quando a dobragem casa com vários.

        O `cogno-ui` chama isto com um id que o host converte em rótulo, portanto o operador
        clicava num nó e perdia outro sem aviso. Ver `_one_node_id`."""
        _require_scope(scope)
        async with self._conn() as conn:
            victim_id = await self._one_node_id(conn, scope, label, op="delete_node")
            if victim_id is None:
                return False
            cur = await conn.execute(
                "DELETE FROM knowledge_nodes WHERE id = %s", (victim_id,))
            return cur.rowcount > 0   # edges cascade via FK ON DELETE CASCADE

    async def delete_edges_by_session(self, scope: str, session_id: str) -> int:
        _require_scope(scope)
        _require_session(session_id)
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


# ── the document store ────────────────────────────────────────────────────────────────

def _doc_uuid(document_id: object) -> Optional[str]:
    """``document_id`` as a canonical uuid string, or ``None`` when it is not one — an id that
    cannot exist answers "no such document", as it does in memory, instead of a driver error."""
    try:
        return str(UUID(str(document_id)))
    except (ValueError, TypeError, AttributeError):
        return None


_DOC_COLS = ("d.id, d.owner_key, d.title, d.profiles, d.media_type, d.active_version, "
             "d.created_at, d.updated_at")
# The version columns every record is built from. `has_original` is asked of the originals'
# PRIMARY KEY only — the bytes themselves are read by `get_original` and nowhere else.
_VERSION_SELECT = (
    "SELECT v.document_id, v.version, v.state, v.reason, v.sha256, v.embed_model, "
    "v.size_bytes, v.pages, v.chunks, v.created_at, v.finished_at, "
    "(o.document_id IS NOT NULL) AS has_original "
    "FROM kb_versions v LEFT JOIN kb_originals o "
    "  ON o.document_id = v.document_id AND o.version = v.version ")
# THE filter of the reader path — one string, so the three reads that serve a reader
# (`readable_documents`, the search itself, and the models the search reports) cannot drift:
# this owner, this profile among the published ones, the SERVED version, in state ready.
_SERVED = (f"JOIN kb_versions v ON v.document_id = d.id AND v.version = d.active_version "
           f"AND v.state = '{KB_READY}' "
           "WHERE d.owner_key = %s AND %s = ANY(d.profiles)")
_OWNER_SUBTREE = "(d.owner_key = %s OR d.owner_key LIKE %s ESCAPE '\\')"


def _version_from_row(r: Any) -> KbVersion:
    return KbVersion(document_id=str(r["document_id"]), version=int(r["version"]),
                     state=r["state"], sha256=r["sha256"], embed_model=r["embed_model"],
                     reason=r["reason"] or "", size_bytes=int(r["size_bytes"] or 0),
                     pages=int(r["pages"] or 0), chunks=int(r["chunks"] or 0),
                     has_original=bool(r["has_original"]), created_at=r["created_at"],
                     finished_at=r["finished_at"])


class PostgresDocumentStore(_PgBase):
    """Reference ``DocumentStore`` over the ``kb_*`` tables (``ensure_documents_schema``).

    Every write that must be whole — a version's number, the swap, a delete with its tombstone —
    runs in ONE transaction that first locks the document row (``FOR UPDATE``), so a concurrent
    delete either happens before (the write finds nothing and says so) or after (the delete
    cascades what the write made). No interleaving leaves a chunk without its document.
    """

    def __init__(self, *, dsn: Optional[str] = None, pool=None,
                 ts_config: str = DEFAULT_TS_CONFIG, unaccent: bool = False,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM,
                 max_original_bytes: int = DEFAULT_MAX_BYTES) -> None:
        super().__init__(dsn=dsn, pool=pool, ts_config=ts_config)
        # The SAME (ts_config, unaccent) the schema was built with: the name is derived by the
        # one function both sides call, so the question and the chunks fold alike.
        self._ts = documents_ts_config(ts_config, unaccent=unaccent)
        self.embedding_dim = int(embedding_dim)
        self.max_original_bytes = int(max_original_bytes)

    # ── rows → records ───────────────────────────────────────────────────
    async def _records(self, conn, rows: "list[Any]") -> "list[KbDocument]":
        # The version query runs EVEN WITH NO ROWS, on purpose: it is one indexed lookup of an
        # empty array, and skipping it made every version column invisible to the health probe
        # (which reads an owner that holds nothing). Measured by
        # `test_the_probe_passes_on_a_fresh_schema_and_fails_without_any_column`: with the early
        # return, dropping seven `kb_versions` columns left the probe green.
        cur = await conn.execute(
            _VERSION_SELECT + "WHERE v.document_id = ANY(%s::uuid[]) ORDER BY v.version",
            ([str(r["id"]) for r in rows],))
        versions: dict[str, list[KbVersion]] = {}
        for r in await cur.fetchall():
            versions.setdefault(str(r["document_id"]), []).append(_version_from_row(r))
        out = []
        for r in rows:
            vs = versions.get(str(r["id"]), [])
            active = next((v for v in vs if v.version == r["active_version"]), None)
            out.append(KbDocument(owner_key=r["owner_key"], id=str(r["id"]), title=r["title"],
                                  profiles=tuple(r["profiles"] or ()), media_type=r["media_type"],
                                  active=active, latest=vs[-1] if vs else None,
                                  created_at=r["created_at"], updated_at=r["updated_at"]))
        return out

    @staticmethod
    def _subtree_like(prefix: str) -> str:
        esc = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return esc + "/%"

    # ── management ───────────────────────────────────────────────────────
    async def create_document(self, owner_key: str, *, title: str, profiles, media_type: str,
                              max_documents: Optional[int] = None) -> KbDocument:
        require_owner(owner_key)
        if media_type not in VALID_MEDIA_TYPES:
            raise ValueError(f"unsupported media type {media_type!r}")
        doc_id = str(uuid4())
        async with self._conn() as conn:
            async with conn.transaction():
                if max_documents is not None:
                    # Serialise creations PER OWNER, then count: a count taken outside the lock
                    # lets two concurrent uploads both see room for one.
                    digest = hashlib.sha256(f"kb_documents:{owner_key}".encode()).digest()
                    await conn.execute("SELECT pg_advisory_xact_lock(%s)",
                                       (int.from_bytes(digest[:8], "big", signed=True),))
                    cur = await conn.execute(
                        "SELECT count(*) AS c FROM kb_documents WHERE owner_key = %s",
                        (owner_key,))
                    row = await cur.fetchone()
                    if int(row["c"]) >= int(max_documents):
                        raise DocumentLimitReached(
                            f"owner already holds {max_documents} documents")
                await conn.execute(
                    "INSERT INTO kb_documents (id, owner_key, title, profiles, media_type) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (doc_id, owner_key, " ".join(str(title or "").split()),
                     list(sanitize_profiles(profiles)), media_type))
                cur = await conn.execute(
                    f"SELECT {_DOC_COLS} FROM kb_documents d WHERE d.id = %s", (doc_id,))
                [record] = await self._records(conn, await cur.fetchall())
        return record

    async def get_document(self, owner_key: str, document_id: str) -> Optional[KbDocument]:
        require_owner(owner_key)
        doc = _doc_uuid(document_id)
        if doc is None:
            return None
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT {_DOC_COLS} FROM kb_documents d WHERE d.owner_key = %s AND d.id = %s",
                (owner_key, doc))
            records = await self._records(conn, await cur.fetchall())
        return records[0] if records else None

    async def list_documents(self, owner_key: str) -> "list[KbDocument]":
        require_owner(owner_key)
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT {_DOC_COLS} FROM kb_documents d WHERE d.owner_key = %s "
                "ORDER BY d.created_at, d.id", (owner_key,))
            return await self._records(conn, await cur.fetchall())

    async def set_profiles(self, owner_key: str, document_id: str, profiles) -> bool:
        require_owner(owner_key)
        doc = _doc_uuid(document_id)
        if doc is None:
            return False
        async with self._conn() as conn:
            cur = await conn.execute(
                "UPDATE kb_documents SET profiles = %s, updated_at = now() "
                "WHERE owner_key = %s AND id = %s",
                (list(sanitize_profiles(profiles)), owner_key, doc))
            return cur.rowcount > 0

    async def get_original(self, owner_key: str, document_id: str, *,
                           version: Optional[int] = None) -> Optional[bytes]:
        require_owner(owner_key)
        doc = _doc_uuid(document_id)
        if doc is None:
            return None
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT o.data FROM kb_originals o JOIN kb_documents d ON d.id = o.document_id "
                "WHERE d.owner_key = %s AND o.document_id = %s "
                "  AND o.version = COALESCE(%s::integer, d.active_version)",
                (owner_key, doc, version))
            row = await cur.fetchone()
        return bytes(row["data"]) if row is not None else None

    # ── ingestion ────────────────────────────────────────────────────────
    async def _lock_document(self, conn, owner_key: str, doc: str) -> Optional[Any]:
        cur = await conn.execute(
            "SELECT id, active_version FROM kb_documents WHERE owner_key = %s AND id = %s "
            "FOR UPDATE", (owner_key, doc))
        return await cur.fetchone()

    async def _version_record(self, conn, doc: str, version: int) -> Optional[KbVersion]:
        cur = await conn.execute(_VERSION_SELECT + "WHERE v.document_id = %s AND v.version = %s",
                                 (doc, version))
        row = await cur.fetchone()
        return _version_from_row(row) if row is not None else None

    async def begin_version(self, owner_key: str, document_id: str, *, sha256: str,
                            embed_model: str, size_bytes: int,
                            original: Optional[bytes] = None) -> Optional[KbVersion]:
        require_owner(owner_key)
        require_model(embed_model, self.embedding_dim)
        if original is not None and len(original) > self.max_original_bytes:
            raise OriginalTooLarge(f"original of {len(original)} bytes is over the "
                                   f"{self.max_original_bytes}-byte ceiling")
        doc = _doc_uuid(document_id)
        if doc is None:
            return None
        async with self._conn() as conn:
            async with conn.transaction():
                if await self._lock_document(conn, owner_key, doc) is None:
                    return None
                cur = await conn.execute(
                    "SELECT version, state FROM kb_versions "
                    "WHERE document_id = %s AND sha256 = %s AND embed_model = %s",
                    (doc, sha256, embed_model))
                same = await cur.fetchone()
                if same is not None:
                    number = int(same["version"])
                    if same["state"] != KB_READY:
                        await conn.execute(
                            f"UPDATE kb_versions SET state = '{KB_PROCESSING}', reason = '', "
                            "finished_at = NULL WHERE document_id = %s AND version = %s",
                            (doc, number))
                else:
                    cur = await conn.execute(
                        "SELECT COALESCE(max(version), 0) + 1 AS n FROM kb_versions "
                        "WHERE document_id = %s", (doc,))
                    number = int((await cur.fetchone())["n"])
                    await conn.execute(
                        "INSERT INTO kb_versions (document_id, version, state, sha256, "
                        "embed_model, size_bytes) VALUES (%s, %s, %s, %s, %s, %s)",
                        (doc, number, KB_PROCESSING, sha256, embed_model, int(size_bytes)))
                if original is not None:
                    await conn.execute(
                        "INSERT INTO kb_originals (document_id, version, owner_key, data) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (document_id, version) DO NOTHING",
                        (doc, number, owner_key, bytes(original)))
                return await self._version_record(conn, doc, number)

    async def add_chunks(self, owner_key: str, document_id: str, version: int,
                         chunks) -> bool:
        require_owner(owner_key)
        rows = [(c, require_vector(c.embedding, self.embedding_dim, what="chunk embedding"))
                for c in chunks]                          # validate ALL before staging any
        doc = _doc_uuid(document_id)
        if doc is None:
            return False
        async with self._conn() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT v.embed_model FROM kb_versions v "
                    "JOIN kb_documents d ON d.id = v.document_id "
                    f"WHERE d.owner_key = %s AND v.document_id = %s AND v.version = %s "
                    f"  AND v.state = '{KB_PROCESSING}' FOR SHARE OF v",
                    (owner_key, doc, int(version)))
                live = await cur.fetchone()
                if live is None:
                    return False
                # The chunk's model label is the VERSION's, copied here — never the caller's —
                # so a chunk cannot claim a model its version was not built with.
                async with conn.cursor() as c2:
                    await c2.executemany(
                        "INSERT INTO kb_chunks (document_id, version, ordinal, heading_path, "
                        "page, content, embed_model, embedding) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector) "
                        "ON CONFLICT (document_id, version, ordinal) DO UPDATE SET "
                        "heading_path = EXCLUDED.heading_path, page = EXCLUDED.page, "
                        "content = EXCLUDED.content, embed_model = EXCLUDED.embed_model, "
                        "embedding = EXCLUDED.embedding",
                        [(doc, int(version), int(c.ordinal), list(c.heading_path), c.page,
                          c.content, live["embed_model"], _vec(emb)) for c, emb in rows])
        return True

    async def commit_version(self, owner_key: str, document_id: str, version: int, *,
                             pages: int) -> str:
        require_owner(owner_key)
        doc = _doc_uuid(document_id)
        if doc is None:
            return COMMIT_DELETED
        number = int(version)
        async with self._conn() as conn:
            async with conn.transaction():
                row = await self._lock_document(conn, owner_key, doc)
                if row is None:
                    return COMMIT_DELETED
                cur = await conn.execute(
                    "SELECT state, (SELECT max(version) FROM kb_versions WHERE document_id = %s) "
                    "AS newest FROM kb_versions WHERE document_id = %s AND version = %s",
                    (doc, doc, number))
                v = await cur.fetchone()
                if v is None:
                    return COMMIT_SUPERSEDED
                if number != int(v["newest"]):
                    await conn.execute(
                        "DELETE FROM kb_versions WHERE document_id = %s AND version = %s",
                        (doc, number))
                    return COMMIT_SUPERSEDED
                if v["state"] == KB_READY and row["active_version"] == number:
                    return COMMIT_READY
                if v["state"] != KB_PROCESSING:
                    return COMMIT_SUPERSEDED
                await conn.execute(
                    f"UPDATE kb_versions SET state = '{KB_READY}', reason = '', "
                    "finished_at = now(), pages = %s, chunks = (SELECT count(*) FROM kb_chunks "
                    "  WHERE document_id = %s AND version = %s) "
                    "WHERE document_id = %s AND version = %s",
                    (int(pages), doc, number, doc, number))
                await conn.execute(
                    "UPDATE kb_documents SET active_version = %s, updated_at = now() WHERE id = %s",
                    (number, doc))
                # Older versions go with their chunks and originals (cascade) in the SAME
                # transaction — the swap is whole or it did not happen.
                await conn.execute(
                    "DELETE FROM kb_versions WHERE document_id = %s AND version < %s",
                    (doc, number))
        return COMMIT_READY

    async def fail_version(self, owner_key: str, document_id: str, version: int, *,
                           reason: str) -> bool:
        require_owner(owner_key)
        doc = _doc_uuid(document_id)
        if doc is None:
            return False
        async with self._conn() as conn:
            async with conn.transaction():
                if await self._lock_document(conn, owner_key, doc) is None:
                    return False
                cur = await conn.execute(
                    f"UPDATE kb_versions SET state = '{KB_ERROR}', reason = %s, finished_at = now() "
                    f"WHERE document_id = %s AND version = %s AND state <> '{KB_READY}'",
                    (sanitize_reason(reason), doc, int(version)))
                if cur.rowcount == 0:
                    return False
                for table in ("kb_chunks", "kb_originals"):
                    await conn.execute(
                        f"DELETE FROM {table} WHERE document_id = %s AND version = %s",
                        (doc, int(version)))
        return True

    # ── removal ──────────────────────────────────────────────────────────
    async def _remove(self, conn, rows: "list[Any]", kind: str, actor: str) -> int:
        """Delete these (already locked) documents and write one tombstone each — ids and version
        numbers only, read BEFORE the cascade takes them."""
        if not rows:
            return 0
        ids = [str(r["id"]) for r in rows]
        cur = await conn.execute(
            "SELECT document_id, array_agg(version ORDER BY version) AS versions "
            "FROM kb_versions WHERE document_id = ANY(%s::uuid[]) GROUP BY document_id", (ids,))
        versions = {str(r["document_id"]): list(r["versions"] or []) for r in await cur.fetchall()}
        await conn.execute("DELETE FROM kb_documents WHERE id = ANY(%s::uuid[])", (ids,))
        async with conn.cursor() as c2:
            await c2.executemany(
                "INSERT INTO kb_tombstones (owner_key, document_id, versions, kind, actor) "
                "VALUES (%s, %s, %s::integer[], %s, %s)",
                [(r["owner_key"], str(r["id"]), versions.get(str(r["id"]), []), kind,
                  str(actor or "")) for r in rows])
        return len(rows)

    async def delete_document(self, owner_key: str, document_id: str, *, actor: str = "") -> bool:
        require_owner(owner_key)
        doc = _doc_uuid(document_id)
        if doc is None:
            return False
        async with self._conn() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    "SELECT id, owner_key FROM kb_documents WHERE owner_key = %s AND id = %s "
                    "FOR UPDATE", (owner_key, doc))
                return await self._remove(conn, await cur.fetchall(), TOMBSTONE_DELETED,
                                          actor) > 0

    async def purge_owner_subtree(self, owner_prefix: str, *, actor: str = "") -> int:
        require_owner(owner_prefix)
        async with self._conn() as conn:
            async with conn.transaction():
                cur = await conn.execute(
                    f"SELECT d.id, d.owner_key FROM kb_documents d WHERE {_OWNER_SUBTREE} "
                    "FOR UPDATE", (owner_prefix, self._subtree_like(owner_prefix)))
                return await self._remove(conn, await cur.fetchall(), TOMBSTONE_PURGED, actor)

    async def tombstones(self, owner_prefix: str, *, limit: int = 100) -> "list[KbTombstone]":
        require_owner(owner_prefix)
        async with self._conn() as conn:
            cur = await conn.execute(
                "SELECT d.owner_key, d.document_id, d.versions, d.kind, d.actor, d.removed_at "
                f"FROM kb_tombstones d WHERE {_OWNER_SUBTREE} "
                "ORDER BY d.removed_at DESC, d.id DESC LIMIT %s",
                (owner_prefix, self._subtree_like(owner_prefix), int(limit)))
            rows = await cur.fetchall()
        return [KbTombstone(owner_key=r["owner_key"], document_id=str(r["document_id"]),
                            versions=tuple(int(x) for x in (r["versions"] or ())), kind=r["kind"],
                            actor=r["actor"] or "", removed_at=r["removed_at"]) for r in rows]

    async def prune_tombstones(self, *, before: datetime) -> int:
        async with self._conn() as conn:
            cur = await conn.execute("DELETE FROM kb_tombstones WHERE removed_at < %s", (before,))
            return cur.rowcount

    async def stored_original_bytes(self, owner_prefix: str) -> int:
        require_owner(owner_prefix)
        async with self._conn() as conn:
            # `octet_length` is answered from the value's header, without de-TOASTing the bytes.
            cur = await conn.execute(
                "SELECT COALESCE(sum(octet_length(d.data)), 0) AS n FROM kb_originals d "
                f"WHERE {_OWNER_SUBTREE}",
                (owner_prefix, self._subtree_like(owner_prefix)))
            row = await cur.fetchone()
        return int(row["n"])

    # ── the reader path ──────────────────────────────────────────────────
    async def readable_documents(self, owner_key: str, *, profile: str) -> "list[KbDocument]":
        require_owner(owner_key)
        profile = require_profile(profile)
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT {_DOC_COLS} FROM kb_documents d {_SERVED} ORDER BY d.created_at, d.id",
                (owner_key, profile))
            return await self._records(conn, await cur.fetchall())

    async def search(self, owner_key: str, *, profile: str, text: str,
                     vector: Optional[list[float]] = None, embed_model: Optional[str] = None,
                     limit: int = 5,
                     weights: Optional[HybridWeights] = None) -> KbSearchResult:
        require_owner(owner_key)
        profile = require_profile(profile)
        if vector is not None and embed_model is None:
            raise ValueError("a query vector needs the label of the model that made it")
        if embed_model is not None:
            require_model(embed_model, self.embedding_dim)
        query = require_vector(vector, self.embedding_dim, what="query vector") \
            if vector is not None else None
        w = weights or HybridWeights()
        hybrid_score(0.0, 0.0, vector_weight=w.vector, lexical_weight=w.lexical)   # validates
        q_txt = text if (text and text.strip()) else None
        # OR over the query's lexemes: a natural-language question rarely has EVERY word in the
        # passage that answers it, and `plainto_tsquery` ANDs them.
        tsq = (f"replace(plainto_tsquery('{self._ts}', %s)::text, ''' & ''', ''' | ''')::tsquery")
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT DISTINCT v.embed_model FROM kb_documents d {_SERVED}",
                (owner_key, profile))
            models = {r["embed_model"] for r in await cur.fetchall()}
            unavailable = sorted(models if query is None else models - {embed_model})
            use_vector = query is not None and not unavailable        # all or none

            params: list = []
            if use_vector:
                vexpr = "GREATEST(0.0, LEAST(1.0, 1.0 - (c.embedding <=> %s::vector)))"
                params.append(_vec(query))
            else:
                vexpr = "NULL::float8"
            if q_txt is not None:
                # normalisation 32 = rank / (rank + 1): ts_rank_cd is unbounded, the fusion needs
                # [0, 1]. A lexical-only candidate must MATCH (`@@`); with a vector, every served
                # chunk is a candidate and the vector ranks it.
                lmatch = f"(c.tsv @@ {tsq})"
                lexpr = f"CASE WHEN c.tsv @@ {tsq} THEN ts_rank_cd(c.tsv, {tsq}, 32) ELSE 0 END"
                params += [q_txt, q_txt, q_txt]              # lmatch, the CASE test, the rank
            else:
                lmatch, lexpr = "false", "0.0"
            params += [owner_key, profile]
            keep = "true" if use_vector else "s.lmatch"
            wv, wl = float(w.vector), float(w.lexical)
            order = ("(%s * s.vscore + %s * s.lscore) / (%s + %s)" if use_vector else "s.lscore")
            order_params = [wv, wl, wv, wl] if use_vector else []
            cur = await conn.execute(
                "SELECT * FROM ("
                "  SELECT c.document_id, c.version, c.ordinal, c.heading_path, c.page, c.content, "
                f"        c.embed_model, d.title, {vexpr} AS vscore, "
                f"        {lmatch} AS lmatch, {lexpr} AS lscore "
                "  FROM kb_chunks c "
                "  JOIN kb_documents d ON d.id = c.document_id AND d.active_version = c.version "
                f"  {_SERVED}"
                ") s "
                f"WHERE {keep} "
                f"ORDER BY {order} DESC, s.document_id, s.version, s.ordinal LIMIT %s",
                params + order_params + [max(0, int(limit))])
            rows = await cur.fetchall()
        hits = []
        for r in rows:
            vs = clamp_unit(float(r["vscore"])) if r["vscore"] is not None else None
            ls = clamp_unit(float(r["lscore"] or 0.0))
            doc = str(r["document_id"])
            hits.append(KbHit(
                id=chunk_id(doc, int(r["version"]), int(r["ordinal"])), document_id=doc,
                version=int(r["version"]), ordinal=int(r["ordinal"]), title=r["title"],
                heading_path=tuple(r["heading_path"] or ()), page=r["page"], content=r["content"],
                embed_model=r["embed_model"], vector_score=vs, lexical_score=ls,
                score=hybrid_score(vs, ls, vector_weight=wv, lexical_weight=wl)))
        hits.sort(key=hit_order_key)
        return KbSearchResult(hits=tuple(hits),
                              degradations=(KB_EMBED_SPACE_UNAVAILABLE,) if unavailable else (),
                              models_unavailable=tuple(unavailable))

    # ── maintenance ──────────────────────────────────────────────────────
    async def stale_documents(self, *, embed_model: str, limit: int = 100) -> "list[KbDocument]":
        require_model(embed_model, self.embedding_dim)
        async with self._conn() as conn:
            cur = await conn.execute(
                f"SELECT {_DOC_COLS} FROM kb_documents d JOIN kb_versions v "
                "  ON v.document_id = d.id AND v.version = d.active_version "
                "WHERE v.embed_model <> %s ORDER BY d.created_at, d.id LIMIT %s",
                (embed_model, int(limit)))
            return await self._records(conn, await cur.fetchall())
