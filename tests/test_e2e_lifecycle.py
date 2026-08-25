"""
End-to-end memory lifecycle through the whole engram substrate (in-memory adapters
+ a scripted stub LLM, so it runs in CI with no DB and no model).

Exercises the SEAM the per-component tests don't: a full session flowing through
buffer → save_turn → Tier-1 micro → Tier-2 periodic (memories + KG) → retrieval →
rerank → Tier-3 final (summary + close), with feedback-driven pruning.
"""

import json

from cogno_engram import hypnos, rerank
from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore
from cogno_engram.types import AUDIENCE_STAFF, RetrievalQuery, TurnRecord


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "stub"

    async def generate(self, system, prompt):
        return (self.responses.pop(0) if self.responses else "{}"), 7, 5


class StubEmbedder:
    async def embed(self, text):
        # crude bag-of-chars vector so shared words correlate
        v = [0.0] * 8
        for ch in text.lower():
            v[ord(ch) % 8] += 1.0
        return v


SCOPE = "acme/phone1"


async def test_full_memory_lifecycle():
    store = InMemoryStore()
    buffer = InMemoryBuffer()
    kg = InMemoryGraph()
    emb = StubEmbedder()

    # ── 1. a session of turns: persist + buffer + Tier-1 micro per turn ──
    session = await store.create_session(SCOPE)
    turns = [
        TurnRecord(session.id, SCOPE, 1, "bom dia", response="oi!", sentiment="NEUTRAL"),
        TurnRecord(session.id, SCOPE, 2, "quero agendar um corte", response="claro",
                   goal="agendar corte", goal_status="NEW", domains=["SCHEDULING"]),
        TurnRecord(session.id, SCOPE, 3, "meu email é joao@x.com", response="anotado",
                   pii_types=["EMAIL"]),
        TurnRecord(session.id, SCOPE, 4, "prefiro pagar no pix", response="ok",
                   goal="agendar corte", goal_status="COMPLETED"),
    ]
    prev = None
    for t in turns:
        await store.save_turn(t)
        await buffer.push(SCOPE, session.id, t)
        for m in hypnos.micro_consolidate(t, prev):
            await store.save_memory(m)
        prev = t

    # short-term window holds the recent turns
    window = await buffer.window(SCOPE, session.id, size=3)
    assert [w.turn_n for w in window] == [2, 3, 4]

    # Tier-1 produced: started+completed goal, pii leak, domain interest
    micro = await store.load_memories(SCOPE)
    cats = {m.category for m in micro}
    assert {"goal", "pii", "preference"} <= cats

    # ── 2. Tier-2 periodic: LLM memories + KG relations ──
    backend = ScriptedBackend([
        json.dumps({"preference": ["prefere pagar no pix"], "fact": ["quer agendar um corte"]}),
        json.dumps({"nodes": [{"label": "João", "type": "PERSON"}, {"label": "Corte", "type": "EVENT"}],
                    "edges": [{"source": "João", "target": "Corte", "relation": "WANTS", "confidence": 0.9}]}),
    ])
    new_mems = await hypnos.periodic_consolidate(
        store, backend, scope=SCOPE, session_id=session.id, embedder=emb, kg=kg)
    assert any("pix" in m.content for m in new_mems)
    edges = await kg.walk(SCOPE, "João", max_depth=1, audience=AUDIENCE_STAFF)
    assert [e.relation for e in edges] == ["WANTS"] and edges[0].source_session == session.id

    # ── 3. retrieval + rerank surfaces the payment preference ──
    hits = await store.load_memories(
        SCOPE, query=RetrievalQuery(text="prefere pagar", embedding=await emb.embed("prefere pagar")),
        limit=10)
    top = rerank(hits, query_text="prefere pagar", top_k=3)
    assert any("pix" in m.content for m in top)

    # ── 4. Tier-3 final: summary + close ──
    final_backend = ScriptedBackend([json.dumps({"fact": ["cliente recorrente"]})])
    result = await hypnos.consolidate_session(store, final_backend, session=session, kg=kg, embedder=emb)
    assert "Domains: SCHEDULING" in result.summary
    assert (await store.get_session(session.id)).ended_at is not None
    assert any("recorrente" in m.content for m in result.memories)


async def test_lifecycle_feedback_prunes_graph_on_close():
    store = InMemoryStore()
    kg = InMemoryGraph()
    session = await store.create_session(SCOPE)
    # a turn that gets a thumbs-down, and a KG edge asserted by this session
    await store.save_turn(TurnRecord(session.id, SCOPE, 1, "ruim", feedback=-1))
    from cogno_engram.types import GraphEdge
    await kg.upsert_edge(GraphEdge(SCOPE, "X", "Y", "REL", source_session=session.id))

    await hypnos.consolidate_session(store, ScriptedBackend([]), session=session, kg=kg)
    assert await kg.walk(SCOPE, "X", max_depth=1, audience=AUDIENCE_STAFF) == []   # pruned because a turn was disliked
