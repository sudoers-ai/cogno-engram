# cogno-engram

**Persistence substrate for the [Cogno](https://github.com/sudoers-ai/cogno-anima) cognitive pipeline** — memory store, knowledge graph, conversation buffer, and sleep-time consolidation.

`cogno-engram` is the *memory* to [`cogno-anima`](https://github.com/sudoers-ai/cogno-anima)'s *mind*. Where `cogno-anima` is pure, infrastructure-agnostic cognition (no I/O), `cogno-engram` is the opinionated **substrate** that remembers: it persists conversation turns, consolidates them into long-term semantic memories, and threads them into a relational knowledge graph.

> Status: **alpha** — the contract (ports + types) and the zero-dependency in-memory adapter are in place. The Postgres/Redis reference adapters and the LLM-driven consolidation tiers are in build-out.

## Philosophy: ports, not a universal store

A single abstraction over relational + key-value + graph collapses to a lowest-common-denominator that throws away vector search and graph traversal. Instead, `cogno-engram` defines **four capability-scoped ports**, each backed by the storage engine that fits it:

| Port | Shape | Reference adapter |
| --- | --- | --- |
| `MemoryStore` | sessions / turns / memories + hybrid retrieval + per-turn traces | Postgres + pgvector |
| `ConversationBuffer` | sliding short-term window (+ TTL) | Redis |
| `KnowledgeGraph` | typed nodes + directed edges + multi-hop walk | Postgres (recursive CTE) |
| `DocumentStore` | published documents: versions, chunks, hybrid search | Postgres + pgvector |

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

## Documents — published text, versioned, searched on demand

`DocumentStore` keeps text somebody WROTE to be read (a manual, a syllabus, a price list),
uploaded as Markdown or PDF. It shares no table and no read with memories or the graph. The
rules live in one module, `cogno_engram.documents`:

- **An opaque owner.** `owner_key` is to documents what `scope` is to the rest — the host
  composes it, engram never parses it — with the same subtree rule: `purge_owner_subtree("t1")`
  removes `t1` and every `t1/…`, never `t10`.
- **Opaque reader labels, no wildcard.** A document is published to `profiles`; every reader
  call (`search`, `readable_documents`) takes `profile` as a REQUIRED keyword and a blank one
  is refused. Only the SERVED version in state `ready` is ever read.
- **One model per version, never mixed.** Every version records the `embed_model` label
  (`embed_model_label("ollama:nomic-embed-text:latest", 768)`) it was indexed with. A search is
  handed one vector plus that label; if any readable chunk has another label (a global model
  swap not yet re-indexed) or there is no vector, the whole search is lexical and says so with
  `kb_embed_space_unavailable` — never a cosine across models.
- **Raw scores in [0, 1], no floor.** `vector_score` (`1 − cosine distance`, cut to [0, 1], or
  `None` when not measured), `lexical_score` (Postgres: `ts_rank_cd` normalisation 32), and
  `score` = `0.6·v + 0.4·l` renormalised — `score == lexical_score` exactly on a lexical search.
  Ties break by `(document, version, ordinal)`. Which score is relevant is the caller's floor.
- **Versions swap atomically.** `ingest()` indexes in the background (extract → chunk by
  heading, ~2000 **characters**, 15% overlap, heading path on every chunk → embed → stage →
  swap); the previous version answers until the new one is ready and keeps answering if it
  fails. It returns the embedder's own usage (`embedding_tokens`, `embedding_calls`) and takes
  a `gate` (refuse before embedding: zero calls) and a `pace` (tokens per minute).
- **Or in two steps, when the cost must be confirmed first.** `prepare()` extracts, chunks and
  ESTIMATES, and parks the version in `awaiting_confirmation` — no embedder, no gate, nothing
  spent; a bad file fails HERE. `estimated_tokens` and `expires_at` (prepare's clock + 24 h) are
  persisted on the version. `commit()` claims the draft atomically (two confirmations embed it
  once), hands the gate that same estimate, embeds and swaps; a second commit is `unchanged`.
  `expire_drafts(now=…)` — the sweep a host runs on its tick — turns every unconfirmed draft
  past its expiry into `error`/`expired`, removes its draft and its stored original, and leaves a
  tombstone. `discard_draft(...)` does the same NOW, on the uploader's request (an upload made
  by mistake must not keep its original for a day), and only to a draft. `interrupt_stale(
  older_than=…)` — the other tick sweep — ends every `processing` version whose `claimed_at` is
  older (a crashed prepare, a commit that died after claiming its draft) as
  `error`/`interrupted`, so no version stays `processing` beyond its process. `ingest()` is
  `prepare()` + `commit()` back to back.
- **Delete is immediate and leaves a tombstone.** The originals of every version go with it
  (they live in their own table, which no search joins); a job finishing after the delete
  writes nothing.
- **PDF extraction is not here.** `TextExtractor` is a structural Protocol; the PDF
  implementation (separate process, deadline, no network, text layer only) is in `cogno-vox`.

```python
from cogno_engram import InMemoryDocumentStore, embed_model_label
from cogno_engram.ingest import ingest

model = embed_model_label("ollama:nomic-embed-text:latest", 768)
docs = InMemoryDocumentStore()
doc = await docs.create_document("acme/secretary", title="Student guide",
                                 profiles=["GUEST"], media_type="text/markdown")
outcome = await ingest(docs, "acme/secretary", doc.id, data=markdown_bytes,
                       embedder=embedder, embed_model=model)
hits = await docs.search("acme/secretary", profile="GUEST", text="saturday hours",
                         vector=await embedder.embed("saturday hours"), embed_model=model)
```

The Postgres adapter (`PostgresDocumentStore`) adds six tables through `ensure_schema`
(`kb_documents`, `kb_versions`, `kb_chunks`, `kb_drafts`, `kb_originals`, `kb_tombstones`) —
additive; the two-step columns on `kb_versions` (`estimated_tokens`, `expires_at`) are added to
an existing table by the same catalogue-checked `ADD COLUMN` the edge columns use.

**Accents.** `ensure_schema(conn, ts_config="portuguese", unaccent=True)` builds the document
text search over a derived configuration, `cogno_portuguese_unaccent` (a copy of the base whose
non-ASCII word tokens pass through `unaccent` before the base's own stemmer), and
`PostgresDocumentStore(ts_config="portuguese", unaccent=True)` parses questions with the SAME
one — so «sabado» finds «Sábado», «Sábado» finds «sabado», and «Sábados» still stems to it. It
applies to `kb_chunks` only; `memories` keeps its configuration. **Changing `ts_config` or
`unaccent` on an EXISTING database does not re-index anything by itself**: `CREATE TABLE IF NOT
EXISTS` leaves the generated `tsv` as it was, words silently stop matching, and `ensure_schema`
logs `event=kb_ts_config_mismatch`. The migration is `rebuild_documents_tsv(conn, ts_config=…,
unaccent=…)`, which rewrites `kb_chunks` under an exclusive lock — an operator's step, not a boot's. `documents_probe(store, embed_model=...)` runs every read against
an owner that holds nothing, for a host's health check: a missing table or column fails it.
**A purge cannot reach the database's backups** — they keep an original until their own
retention expires; that retention is the operator's decision.

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

## Lexical relevance — `cogno_engram.lexical`

A retrieval that fetches by proximity always has a nearest, so it always returns something.
`cogno_engram.lexical` is the part that says whether what came back is ABOUT the question — and,
when nothing is, says *nothing relevant* instead of handing over the nearest noise:

```python
from cogno_engram import lexical

asked = lexical.variants(canonical_query, contact_text)      # the rewrite + the contact's words
pool = (lexical.graph_candidates(walks, limit=lexical.MAX_CANDIDATES + 1)
        + lexical.memory_candidates(records))                # your reads, your audience rules
decision, picked = lexical.decide(lexical.rank(pool, asked), failed=sources_that_broke)
payload = lexical.render(canonical_query, picked)            # each result under a citable id
```

* **One tokenizer** (`tokens` + `STOPWORDS`): the accent/case fold (`textfold.fold`, below),
  Portuguese plural fold, function words dropped. Hand the SAME callable to a ranking and to
  every floor over it.
* **One score** (`relevance`): the share of the question's meaning-carrying words a candidate
  carries, the better of the query variants; a one-hop inheritance along a graph walk
  (`HOP_DECAY`); a deterministic tie-break.
* **One floor** (`RELEVANCE_FLOOR`, a point on this lexical scale — the default was calibrated
  by a consumer over its own labelled set; pass your own) and a closed decision:
  `relevant | nothing_relevant | error` — a source that broke is never reported as one that holds
  nothing.
* **Content-free ids** (`edge:<node>.<n>`, `mem:<id>`): what a reply cites and a trace may keep.
* **A cost bound by count** (`MAX_CANDIDATES`): ranking is CPU on your event loop; over ~3 MB
  of section-sized candidates the capped ranking is ~16× cheaper than the uncapped one (~22 ms
  against ~350 ms on a quiet box). `tests/test_lexical_cost.py` proves the cap mechanically and
  asserts that RATIO, timed intercalated — never a number of milliseconds, which is the load's.
  The SIZE of each candidate is yours to cut.

It fetches nothing and decides nothing about who may read what — candidates arrive already
fetched with your audience and your scopes. Why lexical and not a vector distance: a floor over
a cosine is a number about the embedder, while the share of folded words is the same number in a
unit test and in production. The price is stated in the module: a paraphrase that shares no word
with the question scores zero.

### One fold — `cogno_engram.textfold`

`fold(text, *, punctuation=False, apostrophes=False, collapse_whitespace=False, strip=False)` is
the accent and case fold for every lexicon you compare against: NFKD → combining marks dropped →
`casefold`, in that order, idempotent over all of Unicode, with each consumer's extra step as a
keyword it has to SAY. Use it rather than a private copy — a copy that drifts moves a match with
no red anywhere. It is NOT a key fold: the graph's node identity is `folding.fold_label`, which
must agree with Postgres `unaccent` and deliberately differs.

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

Business identity (`tenants`/`identities`), billing/token ledgers, persona/domain schemas, feedback *capture* (emoji → ±1), persona switching, OTP/rate-limiting, and the consolidation **worker loop**. `cogno-engram` only knows `scope`, sessions/turns/memories, the graph, and the buffer — and, for documents, an opaque `owner_key`, opaque reader `profiles`, and the usage an ingestion spent (who pays for it, which profiles exist and which upload limits a plan has are the host's).

## License

Apache-2.0 © Sudoers AI
