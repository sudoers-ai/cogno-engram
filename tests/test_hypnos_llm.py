"""Tier-2/3 hypnos consolidation, driven by stub backend/embedder (no anima, no model)."""

import json

from cogno_engram import hypnos
from cogno_engram.adapters.in_memory import InMemoryGraph, InMemoryStore
from cogno_engram.types import TurnRecord


class ScriptedBackend:
    """Returns canned responses in call order; mimics anima's LLMBackend tuple."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "stub"
        self.calls = []

    async def generate(self, system, prompt):
        self.calls.append((system, prompt))
        text = self.responses.pop(0) if self.responses else "{}"
        return text, 7, 5


class StubEmbedder:
    async def embed(self, text):
        # deterministic tiny vector keyed on length + first char
        return [float(len(text) % 5), float(ord(text[0]) % 7) if text else 0.0]


def _mem_json(**cats):
    return json.dumps(cats)


async def _seed(store, scope, sid, turns):
    for t in turns:
        await store.save_turn(t)


# ── Tier 2 ────────────────────────────────────────────────────────────────────

async def test_periodic_extracts_and_saves_memories():
    store = InMemoryStore()
    scope, sid = "acme/p", "s1"
    await _seed(store, scope, sid, [
        TurnRecord(sid, scope, 1, "quero juntar dinheiro pra viagem"),
        TurnRecord(sid, scope, 2, "prefiro pagar no pix"),
    ])
    backend = ScriptedBackend([_mem_json(preference=["prefere pagar no pix"],
                                         goal=["juntar dinheiro para viagem"])])
    mems = await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid,
                                             extract_relations=False)
    assert {m.category for m in mems} == {"preference", "goal"}
    assert all(m.confidence == 0.8 for m in mems)
    stored = await store.load_memories(scope)
    assert len(stored) == 2


async def test_periodic_embeds_when_embedder_given():
    store = InMemoryStore()
    scope, sid = "acme/p", "s1"
    await _seed(store, scope, sid, [TurnRecord(sid, scope, 1, "oi")])
    backend = ScriptedBackend([_mem_json(fact=["mora em Berlim"])])
    await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid,
                                      embedder=StubEmbedder(), extract_relations=False)
    [m] = await store.load_memories(scope)
    assert m.embedding is not None


async def test_periodic_excludes_disliked_turns_from_transcript():
    store = InMemoryStore()
    scope, sid = "acme/p", "s1"
    await store.save_turn(TurnRecord(sid, scope, 1, "boa mensagem", feedback=1))
    await store.save_turn(TurnRecord(sid, scope, 2, "RESPOSTA RUIM", feedback=-1))
    backend = ScriptedBackend([_mem_json(fact=["x"])])
    await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid,
                                      extract_relations=False)
    _, prompt = backend.calls[0]
    assert "RESPOSTA RUIM" not in prompt and "boa mensagem" in prompt


async def test_periodic_malformed_json_is_tolerant():
    store = InMemoryStore()
    scope, sid = "acme/p", "s1"
    await _seed(store, scope, sid, [TurnRecord(sid, scope, 1, "oi")])
    backend = ScriptedBackend(["not json at all"])
    mems = await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid,
                                             extract_relations=False)
    assert mems == []


async def test_malformed_json_logs_warning(caplog):
    import logging
    store = InMemoryStore()
    scope, sid = "acme/p", "s1"
    await _seed(store, scope, sid, [TurnRecord(sid, scope, 1, "oi")])
    backend = ScriptedBackend(["not json at all"])
    with caplog.at_level(logging.WARNING, logger="cogno_engram.hypnos"):
        await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid,
                                          extract_relations=False)
    assert any("event=parse_failed kind=memories" in r.message
               and r.levelno == logging.WARNING for r in caplog.records)


async def test_periodic_extracts_relations_into_graph():
    store = InMemoryStore()
    kg = InMemoryGraph()
    scope, sid = "acme/p", "s1"
    await _seed(store, scope, sid, [TurnRecord(sid, scope, 1, "o cachorro do José é o Rex")])
    backend = ScriptedBackend([
        _mem_json(fact=["o José tem um cachorro"]),
        json.dumps({"nodes": [{"label": "José", "type": "PERSON"}, {"label": "Rex", "type": "ANIMAL"}],
                    "edges": [{"source": "José", "target": "Rex", "relation": "OWNS", "confidence": 0.9}]}),
    ])
    await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=sid, kg=kg)
    edges = await kg.walk(scope, "José", max_depth=1)
    assert [e.relation for e in edges] == ["OWNS"]
    assert edges[0].source_session == sid     # tagged for later pruning


async def test_periodic_empty_session_returns_empty():
    store = InMemoryStore()
    backend = ScriptedBackend([_mem_json(fact=["x"])])
    mems = await hypnos.periodic_consolidate(store, backend, scope="s", session_id="missing")
    assert mems == [] and backend.calls == []   # no LLM call when no turns


# ── Tier 3 ────────────────────────────────────────────────────────────────────

async def test_final_writes_summary_and_closes_session():
    store = InMemoryStore()
    session = await store.create_session("acme/p")
    await store.save_turn(TurnRecord(session.id, "acme/p", 1, "oi", domains=["FINANCE"]))
    backend = ScriptedBackend([_mem_json(fact=["mora em SP"], preference=["gosta de pix"])])
    result = await hypnos.consolidate_session(store, backend, session=session)
    assert len(result.memories) == 2
    assert "Domains: FINANCE" in result.summary
    reloaded = await store.get_session(session.id)
    assert reloaded.ended_at is not None and reloaded.summary == result.summary


async def test_final_prunes_graph_on_disliked_turns():
    store = InMemoryStore()
    kg = InMemoryGraph()
    session = await store.create_session("acme/p")
    await store.save_turn(TurnRecord(session.id, "acme/p", 1, "ok", feedback=-1))
    # an edge asserted by this session must be pruned because a turn was disliked
    from cogno_engram.types import GraphEdge
    await kg.upsert_edge(GraphEdge("acme/p", "A", "B", "R", source_session=session.id))
    backend = ScriptedBackend([_mem_json()])
    await hypnos.consolidate_session(store, backend, session=session, kg=kg)
    assert await kg.walk("acme/p", "A", max_depth=1) == []


async def test_final_no_active_turns_still_closes():
    store = InMemoryStore()
    session = await store.create_session("acme/p")
    await store.save_turn(TurnRecord(session.id, "acme/p", 1, "x", feedback=-1))
    backend = ScriptedBackend([])
    result = await hypnos.consolidate_session(store, backend, session=session)
    assert result.memories == []
    assert (await store.get_session(session.id)).ended_at is not None
