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
import logging
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

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
    for column, ddl in (
        ("attributes",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS attributes jsonb NOT NULL DEFAULT '{}'"),
        ("status",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS status text NOT NULL DEFAULT 'accepted'"),
        # The DEFAULT backfills every existing edge as UNCLASSIFIED, not as `tenant`: staff keeps
        # seeing them and no contact does. An upgrade must not hand a contact rows nobody has
        # classified — `maintenance.classify_edge_audience` is the deliberate act that assigns
        # owners, and until it runs the safe answer is "staff only".
        ("audience",
         "ALTER TABLE knowledge_edges ADD COLUMN IF NOT EXISTS audience text NOT NULL DEFAULT ''"),
    ):
        cur = await conn.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = 'knowledge_edges' AND column_name = %s", (column,))
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
                """INSERT INTO knowledge_edges
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
                           WHEN knowledge_edges.status = 'accepted' AND EXCLUDED.status = 'proposed'
                           THEN knowledge_edges.attributes
                           ELSE knowledge_edges.attributes || EXCLUDED.attributes END,
                       -- a re-assertion PROMOTES a proposal and never demotes a verdict;
                       -- `rejected` is sticky on purpose (see the in-memory twin: this path
                       -- cannot tell a deliberate correction from the LLM re-emitting the same
                       -- edge, so `set_edge_status` is the only way back)
                       status         = CASE WHEN knowledge_edges.status = 'proposed'
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
                          AND e.status = 'accepted'
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
                     AND e.status = 'accepted'
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
                          label: Optional[str] = None) -> int:
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
