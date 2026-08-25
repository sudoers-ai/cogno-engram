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
                                  propose_relations=True)

# the host's curation UI reads the queue and writes the verdict
for e in await kg.pending_edges(scope):
    ...
await kg.set_edge_status(scope, "José", "Pedro", "PARENT_OF", "accepted")
```

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

## Feedback-driven quality

The host captures reactions and writes the signal; engram honours it:

```python
await store.set_feedback(scope, session_id, turn_n, -1)         # host: emoji 👎 → -1
# Tier-2/3 then exclude disliked turns; consolidate_session prunes that session's
# KG edges; store.adjust_feedback_score() boosts/penalises hybrid ranking.
```

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
