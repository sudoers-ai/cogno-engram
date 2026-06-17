# Examples

## `host_min.py` — anima + engram, end to end

A minimal host that wires **cogno-anima** (cognition) and **cogno-engram**
(memory) so a conversation can perceive → route → remember. It demonstrates the
full memory loop:

1. **Recall** — `load_memories` + `rerank` + the short-term buffer window, injected
   into the persona voice prompt.
2. **Cognition** — the anima pipeline (NOUMENO → NER → ID → EGO → SUPEREGO).
3. **Persist** — `save_turn` + `buffer.push` + Tier-1 `micro_consolidate`.
4. **Consolidate** — `hypnos.periodic_consolidate` (sleep-time).

The demo states a preference in turn 1, consolidates, and shows turn 2 recalling
it.

### Run

```bash
pip install -e ../cogno-core          # cogno-anima (sibling repo), editable
ollama pull mistral:latest            # and a local Ollama at :11434
python3 examples/host_min.py
```

Uses the in-memory engram adapters (no DB needed). For production, swap the three
constructors in `MemoryHost.__init__` for `PostgresStore` / `PostgresKnowledgeGraph`
/ `RedisConversationBuffer` — see `docs/HOST_INTEGRATION.md`.
