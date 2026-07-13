# cogno-engram

**Persistence substrate for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — memory store, knowledge graph, conversation buffer, and sleep-time consolidation.

`cogno-engram` is the *memory* to [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima)'s *mind*. Where `cogno-anima` is pure, infrastructure-agnostic cognition (no I/O), `cogno-engram` is the opinionated **substrate** that remembers: it persists conversation turns, consolidates them into long-term semantic memories, and threads them into a relational knowledge graph.

> Status: **alpha** — the contract (ports + types) and the zero-dependency in-memory adapter are in place. The Postgres/Redis reference adapters and the LLM-driven consolidation tiers are in build-out.

## Philosophy: ports, not a universal store

A single abstraction over relational + key-value + graph collapses to a lowest-common-denominator that throws away vector search and graph traversal. Instead, `cogno-engram` defines **three capability-scoped ports**, each backed by the storage engine that fits it:

| Port | Shape | Reference adapter |
| --- | --- | --- |
| `MemoryStore` | sessions / turns / memories + hybrid retrieval + per-turn traces | Postgres + pgvector |
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

The reference adapter implements `MemoryStore` (hybrid retrieval = `0.60·vector + 0.40·BM25 + 0.05·feedback`) and `KnowledgeGraph` (recursive-CTE multi-hop walk) over one database. Call `ensure_schema` once for the idempotent DDL (tables + indexes; no alembic required — re-running it additively creates any new table via `CREATE TABLE IF NOT EXISTS`):

The high-volume tables (`turns`, `memories`, `turn_traces`) opt into `HASH(scope)` partitioning. `turn_traces` is a dedicated table holding one **opaque JSONB** trace per turn (`save_turn_trace` / `traces_for_session`) — the host composes it (e.g. the pipeline's NER/EGO signals for an audit view); engram never interprets it.

```python
import psycopg
from cogno_engram.adapters.postgres import PostgresStore, ensure_schema

async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
    await ensure_schema(conn)                 # CREATE TABLE/INDEX IF NOT EXISTS

store = PostgresStore(dsn=dsn, mask_pii=True)
hits = await store.load_memories("acme/phone1", query=RetrievalQuery(text="...", embedding=[...]))
```

For scale, opt the high-volume tables into HASH(scope) partitioning (the generic
equivalent of the parent's LIST(tenant), with zero DDL per new scope):

```python
await ensure_schema(conn, partition_by_scope=True, partitions=8)
```

## Maintenance (sleep-time upkeep)

Like `hypnos`, the host schedules; engram does the work — keeping the substrate
bounded and consistent over time:

```python
from cogno_engram import maintenance

await maintenance.prune_memories(store, scope, older_than=timedelta(days=180), max_confidence=0.75)
await maintenance.reembed_memories(store, embedder, scope)      # after an embedding-model change
await maintenance.prune_orphan_nodes(kg, scope)                 # drop edgeless graph nodes
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
python3 cognobench.py                 # 5 deterministic dims: retrieval/buffer/consolidation/graph/lifecycle
python3 cognobench.py --only graph
python3 cognobench.py --min-score 100 # CI gate
# opt-in, model-dependent: hypnos Tier-2 extraction quality vs a real Ollama model
python3 cognobench.py --only llm_consolidation
# opt-in: the FULL edge-capture path (Tier-2 relation extraction → graph at kg_scope)
python3 cognobench.py --only graph_capture --model mistral:latest
# visualize every graph the bench built/captured as one self-contained HTML
python3 cognobench.py --only graph graph_capture --graph-html graphs.html
```

Dimensions: **retrieval** (hit@1, vector + BM25-only), **buffer** (sliding-window
retention), **consolidation** (Tier-1 micro), **graph** (multi-hop walk),
**lifecycle** (end-to-end: turns → Tier-1 → retrieval+rerank), and the opt-in
**llm_consolidation** (Tier-2 memory quality against Ollama) and **graph_capture**
(Tier-2 relation extraction against Ollama: hard invariants — no dangling edges,
valid confidence, session-tagged, graph rows only at `kg_scope` — plus soft
entity-connectivity checks). Case distributions are modelled on the parent's real
data (goal-heavy memories, BM25-dominant retrieval, NEEDS/PREFERS graph) — all
synthetic.

## The Cogno ecosystem

`cogno-engram` is one organ of **[Cogno](https://github.com/sudoers-ai)** — a family of
small, composable, Apache-2.0 libraries that together form a complete
conversational-agent platform. Each library owns a single concern and stays
infra-agnostic; a **host** assembles them into a running agent:

![The Cogno ecosystem](docs/assets/cogno-ecosystem.svg)

The open-source libraries are the organs; the **host is the body** that joins
them. Our reference host — `cogno-host`, with its `cogno-ui` dashboard — is the
private product layer, but it holds no special powers: everything it does rides
on the public seams documented in each library's `docs/HOST_INTEGRATION.md`, so
you can assemble a body of your own.

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
