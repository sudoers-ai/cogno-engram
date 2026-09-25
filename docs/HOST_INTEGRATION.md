# Host integration

`cogno-engram` is the **memory substrate**. The *host* owns orchestration and
business identity; `cogno-anima` owns cognition. This guide shows how a host
wires the three together so a conversation can **perceive → route → execute →
remember**.

```
            ┌──────────── host (your code) ────────────┐
 user ─▶    │  scope = compose(tenant, user)            │
            │  ① recall:   memories = store.load_memories(scope, query)        │  ── cogno-engram
            │  ② cognition: ctx = anima pipeline(... + memories)               │  ── cogno-anima
            │  ③ persist:  store.save_turn(turn); buffer.push(turn)            │  ── cogno-engram
            │  ④ micro:    hypnos.micro_consolidate(turn) → save_memory        │  ── cogno-engram
 reply ◀─   │  … every N turns: hypnos.periodic_consolidate(...)              │
            │  … on idle/close: hypnos.consolidate_session(...)               │
            └───────────────────────────────────────────┘
```

## The boundary

| Concern | Owner |
| --- | --- |
| Perception / routing / execution / voicing | **cogno-anima** |
| Sessions, turns, memories, knowledge graph, short-term buffer | **cogno-engram** |
| Sleep-time consolidation (the *steps*) | **cogno-engram** (`hypnos`) |
| The consolidation **worker loop**, cadence, billing | **host** |
| Business identity (`tenant`/`user`), composing `scope` | **host** |
| Atomicity (when to commit / `session_lock`) | **host** |
| Feedback *capture* (emoji → ±1) | **host** (engram only honours it) |

## What the host provides

- A **`scope`** string per request — engram isolates every row by it (e.g.
  `f"{tenant_id}/{user_id}"`). It is opaque; engram never parses it.
- A **`cogno-anima` `LLMBackend` + `Embedder`** — passed into `hypnos`
  consolidation and used to embed memories. (duck-typed; engram imports neither.)
- The **orchestration**: recall → cognition → persist → consolidate.

## Recall: inject memories into cognition

Before running the pipeline, fetch relevant long-term memories and the recent
window, and feed them to the host's persona/context:

```python
memories = await store.load_memories(
    scope, query=RetrievalQuery(text=user_text, embedding=await embedder.embed(user_text)), limit=20)
top = rerank(memories, query_text=user_text, top_k=5)          # recency + category + (optional CE)
window = await buffer.window(scope, session_id, size=10)        # short-term episodic context
# → render `top` + `window` into the anima EGO/SUPEREGO prompts (host's job)
```

## Persist + Tier-1 (every turn)

```python
turn = TurnRecord(session_id, scope, turn_n, user_text, response=reply,
                  goal=ctx.id_result.active_goal, goal_status=ctx.id_result.goal_status,
                  sentiment=ctx.intent.sentiment, domains=ctx.intent.domains,
                  pii_types=ctx.intent.pii)                      # host maps anima → flat signals
async with store.session_lock(scope, session_id):               # host decides when to hold it
    await store.save_turn(turn)
await buffer.push(scope, session_id, turn)
for m in hypnos.micro_consolidate(turn, prev_turn):             # LLM-free
    await store.save_memory(m)
```

## Sleep-time consolidation (host schedules)

```python
# every N turns (background)
await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=session_id,
                                  embedder=embedder, kg=kg)
# on session idle/close (the "janitor" loop — host owns the loop, engram does the work)
await hypnos.consolidate_session(store, backend, session=session, kg=kg, embedder=embedder)
```

## Edge curation — the host owns the verdict

A graph edge becomes a sentence the agent states about a person **as if it knew**. So the host
decides what may be spoken, and engram enforces it:

```python
# an LLM extraction PROPOSES (opt-in; default is still assert, so nothing changes on upgrade)
await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid, kg=kg,
                                  # bool, or a predicate per relation: hold the ones that
                                  # become a sentence about a PERSON, keep domain facts
                                  # `accepted` so the staff block stays populated.
                                  # NOTE the `.upper()`: the predicate receives the relation
                                  # as the MODEL emitted it, unnormalised. Comparing it raw
                                  # makes `spouse_of` miss the set, and a miss stamps the edge
                                  # `accepted` — failing OPEN, on exactly the class this is
                                  # meant to hold back.
                                  # from cogno_engram import VALID_PROXIMITY_RELATIONS
                                  propose_relations=lambda s, t, r: (
                                      (r or "").upper() in VALID_PROXIMITY_RELATIONS))

# the host's curation UI reads the queue and writes the verdict
for e in await kg.pending_edges(scope):
    ...
await kg.set_edge_status(scope, "José", "Pedro", "PARENT_OF", "accepted")
```
### Migrating rows that are already `accepted`

Turning the predicate on changes what the NEXT extraction stamps. It does not touch what is
already stored: `upsert_edge` only ever promotes a status, so every proximity edge an earlier
run wrote as `accepted` stays `accepted` and keeps being walked, and `pending_edges` cannot even
list them (it lists proposals, which is exactly what these are not).

Migrating them is a deliberate act on live data. It is **reversible** — `set_edge_status(...,
"accepted")` puts any row back — and **idempotent**: `walk` returns only `accepted` edges, so a
second run finds nothing left to demote.

```python
from cogno_engram import AUDIENCE_STAFF, VALID_PROXIMITY_RELATIONS
from cogno_engram.types import EDGE_PROPOSED

async def demote_extracted_proximity(kg, scope: str, *, dry_run: bool = True) -> int:
    """Send the LLM-extracted proximity edges back for review. Returns how many were demoted.

    Only edges an EXTRACTION asserted are touched — `source_session` is non-empty for those and
    empty for anything a human or an admin API wrote, and a human's note must not be demoted by
    a migration. Run once with `dry_run=True` and read the count before running it for real.
    """
    seen, demoted = set(), 0
    after = None
    while True:
        nodes = await kg.scan_nodes(scope, audience=AUDIENCE_STAFF,
                                    after_id=after, limit=500)
        if not nodes:
            break
        after = nodes[-1].id
        for node in nodes:
            for e in await kg.walk(scope, node.label,
                                   audience=AUDIENCE_STAFF, max_depth=1):
                key = (e.source, e.target, e.relation)
                if key in seen:
                    continue                    # a walk reaches an edge from both endpoints
                seen.add(key)
                if e.relation.upper() not in VALID_PROXIMITY_RELATIONS:
                    continue                    # domain fact: the staff block keeps it
                if not (e.source_session or "").strip():
                    continue                    # a person wrote it; not ours to demote
                demoted += 1
                if not dry_run:
                    await kg.set_edge_status(scope, e.source, e.target, e.relation,
                                             EDGE_PROPOSED)
    return demoted
```

`scope` is the **tenant** scope the graph is written at (`tenant_of(...)` on the host side), not
the per-identity one. After it runs, the contact block goes quiet for those relations until
somebody reviews them — which is the point, and what makes it worth reading the dry-run count
first.

Three things the host does **not** have to remember:

* `walk()` returns `accepted` edges only and has **no flag** to say otherwise — a proposal is
  also skipped by the TRAVERSAL, so it cannot decide what the walk reaches;
* `neighbors()` and `get_node_context()` obey the same rule (an unreviewed edge still discloses
  its endpoint, and `NodeContext` hands both fields to one caller);
* `format_graph_context` repeats the filter at the last step before the text becomes a prompt.

`rejected` is **sticky**: `upsert_edge` cannot tell a deliberate correction from the LLM
re-emitting the same edge and defaults to `accepted`, so only `set_edge_status` reverses a human
verdict. Re-asserting merges `attributes` and may promote a proposal, never demotes one.

`pending_edges` returns **oldest first** so a bounded queue drains.

`count_nodes(scope, label=…)` answers "how many nodes carry this label" without paging. A host
anchoring anything on a node — *"this is the contact's own node"* — needs it: `walk` seeds on
`lower(label)` and `knowledge_nodes` has no unique constraint on `(scope, label)` alone, so a
label can name more than one node and the walk will expand from all of them.

## Documents — the host decides who, engram keeps what

`DocumentStore` (`cogno_engram.documents`) is the store for text a tenant publishes to its
contacts. The split with the host, piece by piece:

| Concern | Owner |
| --- | --- |
| Tables, versions + atomic swap, chunking, hybrid search, the model guard, tombstones | **cogno-engram** |
| `ingest()` — extract → chunk → gate → embed → stage → swap, returning the usage | **cogno-engram** |
| PDF text extraction (separate process, deadline, no network, text layer only) | **cogno-vox** (`TextExtractor`) |
| What `owner_key` is (e.g. `f"{tenant}/{persona}"`), which `profiles` exist, a reader's profile | **host** |
| Upload API/UI, the job runner (one at a time, never on a turn), plan limits | **host** |
| Writing `IngestOutcome.embedding_tokens` to the tenant's ledger; the budget `gate` | **host** |
| The tool that searches, its relevance floor, its description | the skill (cortex) |

```python
from cogno_engram import documents_probe, embed_model_label
from cogno_engram.adapters.postgres import PostgresDocumentStore
from cogno_engram.ingest import (TokensPerMinute, commit, discard_draft, expire_drafts, ingest,
                                 interrupt_stale, prepare)

docs = PostgresDocumentStore(dsn=DSN, ts_config="portuguese", unaccent=True)
# ...built by ensure_schema(conn, ts_config="portuguese", unaccent=True) — the SAME two values
model = embed_model_label(embed_spec(), embed_dimensions()) # the ONE platform embedder
owner = f"{tenant_id}/{persona_id}"                         # opaque to engram

# upload (API): the row exists at once, state `processing`
doc = await docs.create_document(owner, title=title, profiles=["GUEST"],
                                 media_type="application/pdf", max_documents=50)
# the job (background, never inside a turn)
out = await ingest(docs, owner, doc.id, data=pdf_bytes, embedder=embedder, embed_model=model,
                   extractor=pdf_extractor,                  # cogno-vox, extra `pdf`
                   gate=lambda n: budget.allows(tenant_id, n),   # refuse → zero embed calls
                   pace=TokensPerMinute(20_000))            # yield to live conversations
ledger.record(tenant_id, stage="kb_ingest", tokens=out.embedding_tokens)  # host's business

# …or in TWO steps, when the uploader must confirm the cost first
draft = await prepare(docs, owner, doc.id, data=pdf_bytes, embed_model=model,
                      extractor=pdf_extractor)              # no embedder, no gate: spends nothing
# draft.status == "awaiting_confirmation"; show draft.estimated_tokens, until draft.expires_at
done = await commit(docs, owner, doc.id, draft.version, embedder=embedder, embed_model=model,
                    gate=lambda n: budget.allows(tenant_id, n),   # n == draft.estimated_tokens
                    pace=TokensPerMinute(20_000))
ledger.record(tenant_id, stage="kb_ingest", tokens=done.embedding_tokens)
# on the host's tick: every unconfirmed draft past expires_at → error/expired + tombstone
await expire_drafts(docs, now=clock())
# …and every `processing` version whose worker is gone (claimed_at older than 30 min)
await interrupt_stale(docs, older_than=clock() - timedelta(minutes=30))
# the uploader withdraws a draft: its original goes NOW, not in 24 h
await discard_draft(docs, owner, doc.id, draft.version, actor=admin_id)  # "discarded"|"not_a_draft"|"missing"

# the admin's "view" (management path, no profile, never on a turn): the text as the assistant
# reads it — the served version, or a draft BEFORE its cost is confirmed
text = await docs.version_text(owner, doc.id, version=draft.version, page=None, after=None)
# text.state in ("ready", "awaiting_confirmation"); text.chunks[i].text has the path OFF;
# continue with after=text.next_after while text.has_more; None → nothing readable

# a turn (the reader path): profile is REQUIRED, and it is the reader's, not the model's
res = await docs.search(owner, profile=identity_role, text=original_text,
                        vector=query_vector, embed_model=model)
# res.degradations == ("kb_embed_space_unavailable",) → the search was lexical only
```

- **Health.** Call `documents_probe(docs, embed_model=model)` from `/health`: it runs every read
  the store serves against an owner that holds nothing, and raises what the database raises. A
  table or column the migration never created fails it — a test drops every column of every
  `kb_*` table in turn and requires the probe to fail. (The graph probe could not see the
  engram's `turns` schema once, and the janitor failed silently for days behind a green
  `/health`; this is the same check for these tables, owned here so it moves with the pin.)
- **Outline.** `readable_documents` fills `KbDocument.sections` — the section headings of the
  ACTIVE version, in document order, at the first `heading_path` depth with two distinct
  headings (at most `MAX_SECTIONS_PER_DOCUMENT`) — so a host can say what a document COVERS
  without a search. They are the tenant's text taken from the CONTENT, so sanitise them like
  titles before they reach a prompt, and treat them as possibly holding personal data (a title
  can be refused at upload; a heading inside the file never was).
- **Accents.** Pass `unaccent=True` (and the same `ts_config`) to BOTH `ensure_schema` and
  `PostgresDocumentStore` — the tables and the questions must fold alike, and the derived
  configuration name comes from the one function both call (`documents_ts_config`). Changing
  either on an EXISTING database needs `rebuild_documents_tsv(conn, ts_config=…, unaccent=…)`:
  the generated `tsv` is not recomputed by a migration re-run, and `ensure_schema` logs
  `event=kb_ts_config_mismatch` until it is. With no documents yet the rebuild is instant; do it
  in the same deploy that first sets the flag.
- **Migration.** `ensure_schema` creates the five `kb_*` tables (`CREATE ... IF NOT EXISTS`,
  additive). A host whose migration delegates to it gets them with no new step; a pin bump
  that includes this change needs that migration run on the live database before the first
  upload, like any schema change. The same holds for the columns added later by a
  catalogue-checked `ADD COLUMN` (`kb_versions.estimated_tokens`/`expires_at`/`claimed_at`,
  `kb_tombstones.reason`): `tombstones()` reads `reason`, so until the migration runs it — and
  `documents_probe` — fail on the old table.
- **Two steps.** `prepare` records everything the FILE can get wrong before anything is spent;
  `commit` is the only step that calls the embedder. `commit` outcomes a host must handle:
  `ready`, `unchanged` (a repeat — bill nothing: `embedding_tokens` is 0), `not_prepared`,
  `expired` (past `expires_at` and not yet swept — nothing changed), `error`, `deleted`,
  `superseded`. The expiry is the ENGRAM's rule: call `expire_drafts(docs, now=clock())` on the
  tick and read `KbVersion.state`/`reason`; do not re-implement the 24 h in the host.
  `pending_drafts(owner)` lists what waits for confirmation (no text). `discard_draft` is the
  privacy half of the same rule (condition (e): a delete takes effect at once): it applies to a
  draft only and answers `not_a_draft` for anything else — a served version is removed with
  `delete_document`, never by discarding. `interrupt_stale` is the second tick sweep: a
  `processing` version is judged by `claimed_at` (when it last entered `processing` — begun, or
  claimed by a commit), never by `created_at`, so a draft confirmed a minute ago is not ended
  for having been uploaded an hour ago.
- **Purge.** A tenant purge calls `purge_owner_subtree(tenant_prefix)`: every document under
  the prefix goes, with the chunks and the stored originals of every version, and one
  tombstone per document (ids, versions, when, who — no title, no text). `prune_tombstones`
  bounds their retention. **Backups are out of reach:** a database backup keeps an original
  until the backup's own retention expires — record that in the operator's data policy.
- **Reading a version's text.** `version_text` is the management read behind a "view what is in
  this document" screen, and it answers what the ASSISTANT reads — the chunks, with their 15%
  overlap — not the original file. Readable: the served version and a draft in
  `awaiting_confirmation` (read from the draft itself: nothing is embedded to show it). `None`
  for every other version; to tell "exists or existed, not served" from "never existed", compare
  the number with `get_document(...).latest.version` — numbers are assigned increasing, never
  reused, and the latest attempt is never the one a removal takes (a test pins it). The passage
  comes with its heading path OFF (`KbTextChunk.text`); do not strip it in the host — the rule
  lives beside the chunker (`chunking.chunk_text`). `page` is the PDF page and is ignored on a
  version without pages; `after` is an exclusive ordinal cursor; `limit` is cut to
  `VERSION_TEXT_MAX_LIMIT`. Postgres reads the version and its slice in ONE snapshot
  (`REPEATABLE READ, READ ONLY`, no lock) and slices a draft's chunks inside the database. It is
  part of `documents_probe`.
- **The removal trail.** `tombstones(owner_prefix)` (newest first) is where every removal of a
  version's content is recorded — `deleted`, `purged`, `expired`, `discarded`, `interrupted`,
  `superseded` and `failed`. `superseded` is the commit's own: ONE per commit naming every
  version it removed (the versions the swap replaced, or a version whose commit arrived after a
  newer one was begun), with no actor — nobody asked for it, it is a side effect of an upload.
  `failed` is `fail_version`'s: the version ended in `error`, its chunks, draft and original
  went, and the stone carries the version's `reason` (the closed alphabet — never the detail a
  failure raised); `reason` is blank on every other kind. Only the SERVED version (and a draft
  or a version still being built) keeps its original: a host that offers "download the
  original" offers it for those, and reads an older version's absence from the trail, never
  re-extracts it.
- **Model swap.** A global embedder change leaves every version on the old label; until it is
  re-indexed, searches are lexical and marked. `stale_documents(embed_model=new)` lists the work
  and `reindex()` rebuilds one document from its stored original.

## Feedback-driven quality

The host captures reactions and writes the signal; engram honours it:

```python
await store.set_feedback(scope, session_id, turn_n, -1)         # host: emoji 👎 → -1
# Tier-2/3 then exclude disliked turns; consolidate_session prunes that session's
# KG edges; store.adjust_feedback_score() boosts/penalises hybrid ranking.
```

## Relevance over what you retrieved — `cogno_engram.lexical`

Recall and graph walks fetch by PROXIMITY, so they always return something. When the host has to
say whether any of it is ABOUT the question — and say *nothing relevant* when none is — it
composes its own reads with the engine; engram never decides who may read what:

```python
from cogno_engram import lexical

asked = lexical.variants(canonical_query, contact_text)       # rewrite + the contact's own words
pool = lexical.graph_candidates(walks, limit=lexical.MAX_CANDIDATES + 1,
                                baseline_nodes=2)             # host: nodes its OLD path walks
pool += lexical.memory_candidates(records, limit=lexical.MAX_CANDIDATES + 1 - len(pool))
pool += my_own_candidates                                     # host: its documents, one per section
decision, picked = lexical.decide(lexical.rank(pool[:lexical.MAX_CANDIDATES], asked),
                                  failed=sources_that_raised, floor=my_floor)
```

The host owns: the reads (each with the reader's audience and scope), the SIZE of what it turns
into candidates (the count bound cannot see bytes), any wall-clock ceiling (ranking is CPU on the
host's event loop), and the floor's CALIBRATION — `RELEVANCE_FLOOR` is a default measured by one
host over its labelled set; a host with its own set pins the constant to what its set picks, so
an engram bump that moves the optimum is red on the host, at the bump. `tokens`/`STOPWORDS` are
the ONE tokenizer: hand the same object to every ranking and every floor over it.

## Folding text for a lexicon — `cogno_engram.textfold`

`textfold.fold` is the accent/case fold for every lexicon a host matches against (NFKD → marks →
`casefold`, the extra steps as keywords). A host that already had one should RE-EXPORT this one
(`from cogno_engram.textfold import fold`) and pin identity with `is`, rather than keep a copy in
step. It is not a key fold: node identity is `folding.fold_label`, which must agree with Postgres
`unaccent` and deliberately differs.

## Swapping adapters

The ports are infrastructure-agnostic. Dev uses the zero-dependency in-memory
adapters; production swaps the constructors only:

```python
# dev
store, buffer, kg = InMemoryStore(), InMemoryBuffer(), InMemoryGraph()
# prod
store = PostgresStore(dsn=DSN, mask_pii=True)
kg = PostgresKnowledgeGraph(dsn=DSN)
buffer = RedisConversationBuffer(redis_url=REDIS_URL)
```

See `examples/host_min.py` for a runnable host wiring anima + engram end to end.
