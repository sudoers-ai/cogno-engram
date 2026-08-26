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

## Edge curation — who asserted it decides whether it is spoken

A graph edge becomes a sentence the agent states about a person **as if it knew**. "Your son Pedro" is either a kindness or an invention, and nothing downstream can tell which — so the difference rides on the edge:

```python
GraphEdge(scope, "José", "Pedro", "PARENT_OF", attributes={"age": 8})            # asserted
GraphEdge(scope, "José", "Rex", "OWNS_PET", status="proposed")                   # waiting
```

| | |
| --- | --- |
| `walk()` | returns **`accepted` only**, and has no flag to say otherwise |
| traversal | a proposal is **skipped**, not merely filtered from the result |
| `format_graph_context` | repeats the filter at the last step before text |
| `pending_edges` / `set_edge_status` | the curation queue and the verdict |

The detail is what reaches the prompt, not just the store:

```
[Knowledge Graph]
- José --[PARENT_OF]--> Pedro (age: 8; note: joga futebol no sábado)
```

Bounded per edge and newline-flattened — the value comes from a person typing into an admin field, and a line break inside a bullet turns one fact into what reads as two.

The missing flag is deliberate: a walk feeds the prompt, and "show me the unreviewed ones too" is a curation question, not a retrieval one. Skipping the traversal is the half that is easy to miss — filter only the result and a proposal still decides what the walk can *reach*, leaking the same unverified claim one hop further away.

**Graph audience.** Every read that can return contact data takes a required `audience` keyword: `AUDIENCE_STAFF` for a tenant/staff read, `audience_for(identity_id)` inside a contact's turn. The filter is on the EDGE — `knowledge_nodes` is unique on `(scope, lower(label), node_type)`, so a node is one row per tenant and cannot carry it — and node visibility is derived from the edges the reader may see. `""` means unclassified: staff sees it, no contact does. Required, not defaulted, because with an optional argument forgetting it returns everything. `has_edges` is the one read with no audience: it answers the orphan question for `prune_orphan_nodes`, which deletes.

`hypnos.periodic_consolidate(propose_relations=True)` makes Tier 2 propose instead of assert. **Opt-in**, because flipping the default would silently empty the graph block of every host already running. It also takes a **predicate** — `propose_relations(source, target, relation) -> bool` — because "review everything or review nothing" is the wrong granularity: the edges that become a sentence about a PERSON ("your wife Maria") are a small, nameable class, while the rest ("the clinic accepts Unimed") are domain facts a walk should keep stating. All-or-nothing forces a host to choose between speaking unreviewed claims about someone's family and losing its whole knowledge block. Same seam as `edge_filter`. A predicate that raises yields `proposed`, never `accepted`, and one of the wrong shape (wrong arity, or `async`) is refused at wiring time rather than silently holding every edge. It receives the relation **as the model emitted it**, so normalise before comparing — a miss stamps the edge `accepted`, which fails open. Turning it on does **not** demote rows already `accepted`: `upsert_edge` only promotes, so existing edges keep being spoken until a host migrates them with `set_edge_status`. Re-asserting an edge merges its attributes and may *promote* a proposal, but never demotes a verdict — a review the next LLM pass could expire is a review nobody would do. `rejected` is **sticky**: `upsert_edge` cannot tell a deliberate correction from the LLM re-emitting the same edge, and it defaults to `accepted`, so promoting from `rejected` would resurrect every rejected edge on the next Tier-2 run. `set_edge_status` is the way back, and undoing a human verdict taking a human is the point.

`pending_edges` returns **oldest first** in both adapters: with a `limit` and no cursor, newest-first would make the oldest proposals — the ones a curator most needs to clear — permanently unreachable, and the queue would never drain.

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

# Run the integration suites against real services. The Postgres suites pick their own
# destination — `engram_test`, on the server `COGNO_PG_DSN` names or on libpq's defaults —
# and skip when nothing is listening there. They DROP TABLE, so the database name is never
# taken from you: it is always `engram_test`. `ENGRAM_TEST_DSN` overrides, and a name
# without "test" in it is refused at collection.
docker run -d --rm --name engram-pg -e POSTGRES_PASSWORD=postgres \
    -e POSTGRES_DB=engram_test -p 5432:5432 pgvector/pgvector:pg16
docker run -d --rm --name engram-redis -p 56379:6379 redis:7-alpine
ENGRAM_TEST_REDIS_URL=redis://localhost:56379/0 \
    python3 -m pytest tests/test_postgres_integration.py tests/test_redis_integration.py -q
```

## What lives in the host (not here)

Business identity (`tenants`/`identities`), billing/token ledgers, persona/domain schemas, feedback *capture* (emoji → ±1), persona switching, OTP/rate-limiting, and the consolidation **worker loop**. `cogno-engram` only knows `scope`, sessions/turns/memories, the graph, and the buffer.

## License

Apache-2.0 © Sudoers AI
