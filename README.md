# cogno-engram

**Persistence substrate for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — memory store, knowledge graph, conversation buffer, and sleep-time consolidation.

`cogno-engram` is the *body* to [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima)'s *mind*. Where `cogno-anima` is pure, infrastructure-agnostic cognition (no I/O), `cogno-engram` is the opinionated **substrate** that remembers: it persists conversation turns, consolidates them into long-term semantic memories, and threads them into a relational knowledge graph.

> Status: **alpha** — the contract (ports + types) and the zero-dependency in-memory adapter are in place. The Postgres/Redis reference adapters and the LLM-driven consolidation tiers are in build-out.

## Philosophy: ports, not a universal store

A single abstraction over relational + key-value + graph collapses to a lowest-common-denominator that throws away vector search and graph traversal. Instead, `cogno-engram` defines **three capability-scoped ports**, each backed by the storage engine that fits it:

| Port | Shape | Reference adapter |
| --- | --- | --- |
| `MemoryStore` | sessions / turns / memories + hybrid retrieval | Postgres + pgvector |
| `ConversationBuffer` | sliding short-term window (+ TTL) | Redis |
| `KnowledgeGraph` | typed nodes + directed edges + multi-hop walk | Postgres (recursive CTE) |

Vector search is an **optional capability** (`SupportsVectorSearch`) — a store without it degrades retrieval to lexical/chronological instead of breaking.

## Decoupled by an opaque `scope`

Every row is isolated by an opaque `scope` string. `cogno-engram` never interprets it — the **host** composes it (e.g. `"tenant/phone"`) and owns its meaning and any cross-scope aggregation. There is no `tenant_id`/`phone_id` in the schema; the substrate is reusable for any domain.

```python
from cogno_engram import InMemoryStore, MemoryRecord, RetrievalQuery

store = InMemoryStore()
session = await store.create_session(scope="acme/phone1")
await store.save_memory(MemoryRecord("acme/phone1", "preference", "likes oat milk", embedding=[...]))

hits = await store.load_memories(
    "acme/phone1",
    query=RetrievalQuery(text="what milk?", embedding=[...]),   # hybrid: vector + lexical + feedback
)
```

## `hypnos` — sleep-time consolidation (3-tier)

Named for the god of sleep: consolidation runs while a session "sleeps". As everywhere in Cogno, **engram provides the steps, the host runs the loop** — there is no daemon here.

- **Tier 1 — `micro_consolidate`** — synchronous, per-turn, **LLM-free** (goal transitions, sentiment spikes, PII leaks, new-domain interest).
- **Tier 2 — `periodic_consolidate`** — async, every N turns, LLM extraction (+ KG relations).
- **Tier 3 — `consolidate_session`** — async, on session close/idle, holistic LLM pass (+ feedback-driven KG pruning).

## Install

```bash
pip install cogno-engram                 # core: ports + in-memory adapter (zero deps)
pip install "cogno-engram[postgres]"     # Postgres + pgvector adapter
pip install "cogno-engram[redis]"        # Redis buffer adapter
```

The Tier-2/3 consolidation drives an LLM through a `cogno-anima` `LLMBackend` (host-injected).

## Postgres + pgvector adapter

The reference adapter implements `MemoryStore` (hybrid retrieval = `0.60·vector + 0.40·BM25 + 0.05·feedback`) and `KnowledgeGraph` (recursive-CTE multi-hop walk) over one database. Call `ensure_schema` once for the idempotent DDL (tables + indexes; no alembic required):

```python
import psycopg
from cogno_engram.adapters.postgres import PostgresStore, ensure_schema

async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
    await ensure_schema(conn)                 # CREATE TABLE/INDEX IF NOT EXISTS

store = PostgresStore(dsn=dsn, mask_pii=True)
hits = await store.load_memories("acme/phone1", query=RetrievalQuery(text="...", embedding=[...]))
```

## Reranking

`load_memories` returns relevance-ordered candidates; `rerank` refines them with a two-pass pipeline before you inject the top-k:

```python
from cogno_engram import rerank, RerankConfig

candidates = await store.load_memories("acme/phone1", query=q, limit=20)
top = rerank(candidates, query_text=q.text, top_k=5)   # sim + recency-decay + category boost
```

Pass 1 is pure (`sim·0.60 + recency·0.25 + category·0.15`, half-life and boosts configurable via `RerankConfig`). Pass 2 is an optional **host-injected** cross-encoder callable `(query, [content]) -> [score]` — so cogno-engram ships no heavy ML dependency.

## EngramBench

A self-contained quality harness (no DB, no model — deterministic) over the in-memory adapter, scoring the substrate's three jobs:

```bash
python3 cognobench.py                 # retrieval (hit@1) + consolidation + graph
python3 cognobench.py --only graph
python3 cognobench.py --min-score 100 # CI gate
```

## Testing

```bash
pip install -e ".[dev]"
python3 -m pytest -q                   # unit + bench-smoke (Postgres tests auto-skip)

# Run the integration suites against real services:
docker run -d --rm --name engram-pg -e POSTGRES_PASSWORD=postgres \
    -p 55432:5432 pgvector/pgvector:pg16
docker run -d --rm --name engram-redis -p 56379:6379 redis:7-alpine
ENGRAM_TEST_DSN=postgresql://postgres:postgres@localhost:55432/postgres \
ENGRAM_TEST_REDIS_URL=redis://localhost:56379/0 \
    python3 -m pytest tests/test_postgres_integration.py tests/test_redis_integration.py -q
```

## What lives in the host (not here)

Business identity (`tenants`/`identities`), billing/token ledgers, persona/domain schemas, feedback *capture* (emoji → ±1), persona switching, OTP/rate-limiting, and the consolidation **worker loop**. `cogno-engram` only knows `scope`, sessions/turns/memories, the graph, and the buffer.

## License

Apache-2.0 © Sudoers AI
