# Changelog

## Unreleased

- `MemoryStore.admin_traces(scope_prefix, *, since=None, limit=1000, offset=0)` — the
  turn traces of a scope SUBTREE, newest-first, with an inclusive `since` and a total.
  `traces_for_session` reads one session; an audit over a tenant's whole history (or a
  janitor pass over "everything since T") had to enumerate sessions through the
  300-row `admin_turns` window. Same subtree semantics and pagination shape as
  `admin_turns`; implemented on the Postgres and in-memory adapters. **Contract change:**
  a third-party `MemoryStore` adapter must add the method to satisfy the Protocol.
- `PostgresStore.save_turn_trace` now honours `TurnTrace.created_at` when set (the
  in-memory adapter always did); absent, the column default stamps the row as before.
  A backfilled or imported trace no longer reads as "now", so a `since` window over it
  means what it says.

## 0.1.1 — 2026-08-02

Maintenance ops for an embedding-model switch.

- `reembed_knowledge_nodes`: only `memories` had a re-embed op, so a graph node
  left in the old vector space became silently unreachable by
  `find_nodes_by_embedding` — re-embedding one store and not the other left the
  system half-migrated in a way nothing reported.
- Both re-embed ops now prefer the embedder's `embed_batch` when it offers one,
  falling back to sequential `embed`. Re-embedding is the bulk operation by
  definition, and against a metered provider the difference is latency and
  rate-limit headroom.

## 0.1.0 — 2026-07-25

First public release on PyPI.

Persistence substrate for the Cogno cognitive pipeline — memory store, knowledge graph, conversation buffer, and sleep-time consolidation (hypnos)
