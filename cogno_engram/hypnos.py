"""
cogno_engram.hypnos — sleep-time memory consolidation (3-tier).

Named for Hypnos, the god of sleep: consolidation runs while a session "sleeps"
(idle/closing), turning episodic turns into semantic long-term memories. As with
the rest of the project, **engram provides the steps, the host runs the loop** —
there is no daemon here. The host's worker calls these functions on its own
cadence (and owns billing the LLM cost).

Three complementary tiers (ported from the parent's ConsolidationManager):

  * Tier 1 — ``micro_consolidate``  : synchronous, per-turn, **LLM-free**
                                       (goal transitions, sentiment spikes,
                                       PII leaks, new-domain interest).
  * Tier 2 — ``periodic_consolidate``: async, every N turns, LLM extraction
                                       (+ optional KG relation extraction).
  * Tier 3 — ``consolidate_session`` : async, on session close/idle, holistic
                                       LLM pass (+ feedback-driven KG pruning).

Tiers 2 and 3 take a ``cogno_anima`` ``LLMBackend`` and (optionally) an
``Embedder``, both **duck-typed and host-injected** — engram never imports
cogno-anima at runtime. The backend is called as
``await backend.generate(system, prompt) -> (text, tokens_in, tokens_out)``;
the embedder as ``await embedder.embed(text) -> list[float]``.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

from cogno_engram.ports import KnowledgeGraph, MemoryStore
from cogno_engram.types import (
    DEFAULT_CONFIDENCE,
    VALID_NODE_TYPES,
    GraphEdge,
    GraphNode,
    MemoryRecord,
    Session,
    SessionSummary,
    TurnRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cogno_anima.llm.base import Embedder, LLMBackend


# ── Prompts (defaults; every consolidation fn accepts overrides) ──────────────

DEFAULT_PERIODIC_SYSTEM = """You are a memory analyst. Given a batch of recent \
conversation turns, extract NEW, durable insights about the user.

Return ONLY a JSON object (use empty lists if nothing new):
{
  "preference": ["new preference"],
  "fact": ["new fact about the user"],
  "goal": ["updated or new goal"],
  "behavior": ["communication pattern observed"]
}

Rules:
- Only NEW information that generalises beyond a single turn; look for patterns.
- One concise sentence per item.
- Respond with the JSON object only."""

DEFAULT_PERIODIC_PROMPT = "Analyse these {count} recent turns and extract new insights:\n\n{transcript}"

DEFAULT_FINAL_SYSTEM = """You are a memory analyst. Given a COMPLETE conversation \
transcript, extract durable structured memories about the user.

Return ONLY a JSON object (use empty lists if nothing found):
{
  "preference": ["..."],
  "fact": ["name, location, profession, relationships..."],
  "goal": ["things the user wanted but did not finish"],
  "behavior": ["how they communicate"]
}

Respond with the JSON object only."""

DEFAULT_FINAL_PROMPT = "Analyse this complete conversation ({turn_count} turns):\n\n{transcript}"

DEFAULT_RELATION_SYSTEM = """You extract a knowledge graph from a conversation. \
Return ONLY JSON:
{
  "nodes": [{"label": "José", "type": "PERSON"}],
  "edges": [{"source": "José", "target": "Rex", "relation": "OWNS", "confidence": 0.9}]
}
Allowed node types: PERSON, ANIMAL, PLACE, ORG, CONCEPT, OBJECT, EVENT.
Respond with the JSON object only."""

DEFAULT_RELATION_PROMPT = "Extract entities and relations from this conversation:\n\n{transcript}"


# ── Tier 1 — micro (deterministic, LLM-free) ──────────────────────────────────

def micro_consolidate(turn: TurnRecord, prev: Optional[TurnRecord] = None) -> list[MemoryRecord]:
    """Tier 1 — deterministic, LLM-free consolidation of a single turn.

    Reads the flat cognition signals the host mapped onto the turn (goal /
    goal_status / sentiment / domains / pii_types) and emits memories for the
    structural changes since the previous turn. Pure and cheap — same nature as
    cogno-anima's ID/Drift heuristics, but it writes I/O so it lives here.
    """
    mems: list[MemoryRecord] = []
    scope = turn.scope

    if turn.goal and turn.goal_status:
        changed = prev is None or prev.goal != turn.goal or prev.goal_status != turn.goal_status
        if changed:
            status = turn.goal_status.upper()
            if status == "NEW":
                mems.append(MemoryRecord(scope, "goal", f"Started goal: {turn.goal}",
                                         DEFAULT_CONFIDENCE["goal_started"]))
            elif status == "COMPLETED":
                mems.append(MemoryRecord(scope, "goal", f"Completed goal: {turn.goal}",
                                         DEFAULT_CONFIDENCE["goal_completed"]))
            elif status == "ABANDONED":
                mems.append(MemoryRecord(scope, "goal", f"Abandoned goal: {turn.goal}",
                                         DEFAULT_CONFIDENCE["goal_abandoned"]))

    if turn.sentiment.upper() == "FRUSTRATED" and (prev is None or prev.sentiment.upper() != "FRUSTRATED"):
        mems.append(MemoryRecord(scope, "sentiment", "User became frustrated.",
                                 DEFAULT_CONFIDENCE["sentiment"]))

    if turn.pii_types:
        mems.append(MemoryRecord(scope, "pii", f"User exposed PII: {', '.join(turn.pii_types)}",
                                 DEFAULT_CONFIDENCE["pii"]))

    prev_domains = set(prev.domains) if prev else set()
    for domain in turn.domains:
        if domain not in prev_domains:
            mems.append(MemoryRecord(scope, "preference", f"Interested in domain: {domain}",
                                     DEFAULT_CONFIDENCE["preference"]))

    return mems


# ── shared helpers for the LLM tiers ──────────────────────────────────────────

async def _generate(backend: Any, system: str, prompt: str) -> str:
    resp = await backend.generate(system, prompt)
    return resp[0] if isinstance(resp, tuple) else resp


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_memories(text: str, scope: str, confidence: float) -> list[MemoryRecord]:
    """Parse a ``{category: [item, ...]}`` JSON object into MemoryRecords.

    Tolerant: malformed JSON yields no memories (consolidation must not crash the
    host's background worker over a bad generation).
    """
    try:
        data = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    out: list[MemoryRecord] = []
    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and item.strip():
                out.append(MemoryRecord(scope, str(category), item.strip(), confidence))
    return out


def _transcript(turns: list[TurnRecord]) -> str:
    lines = []
    for t in turns:
        goal = f" [goal: {t.goal}]" if t.goal else ""
        lines.append(f"Turn {t.turn_n}: {t.user_input}\n  sentiment={t.sentiment or 'NEUTRAL'} "
                     f"domains={t.domains}{goal}\n  assistant: {t.response}")
    return "\n".join(lines)


async def _embed_and_save(store: MemoryStore, mems: list[MemoryRecord], embedder: Any) -> None:
    for m in mems:
        if embedder is not None:
            vec = await embedder.embed(m.content)
            m.embedding = vec or None
        await store.save_memory(m)


async def _extract_relations(kg: KnowledgeGraph, backend: Any, *, scope: str, session_id: str,
                             turns: list[TurnRecord], embedder: Any,
                             system: str, prompt: str) -> int:
    """LLM relation extraction → upsert nodes + edges (source_session tagged)."""
    text = await _generate(backend, system, prompt.format(transcript=_transcript(turns)))
    try:
        data = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return 0
    if not isinstance(data, dict):
        return 0

    for node in data.get("nodes", []) or []:
        if not isinstance(node, dict) or not node.get("label"):
            continue
        ntype = str(node.get("type", "CONCEPT")).upper()
        ntype = ntype if ntype in VALID_NODE_TYPES else "CONCEPT"
        gn = GraphNode(scope, str(node["label"]), ntype)
        if embedder is not None:
            gn.embedding = (await embedder.embed(gn.label)) or None
        await kg.upsert_node(gn)

    edges = 0
    for edge in data.get("edges", []) or []:
        if not isinstance(edge, dict) or not (edge.get("source") and edge.get("target") and edge.get("relation")):
            continue
        try:
            conf = float(edge.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        await kg.upsert_edge(GraphEdge(scope, str(edge["source"]), str(edge["target"]),
                                       str(edge["relation"]), conf, source_session=session_id))
        edges += 1
    return edges


# ── Tier 2 — periodic (LLM) ───────────────────────────────────────────────────

async def periodic_consolidate(
    store: MemoryStore,
    backend: "LLMBackend",
    *,
    scope: str,
    session_id: str,
    batch_n: int = 10,
    embedder: Optional["Embedder"] = None,
    kg: Optional[KnowledgeGraph] = None,
    extract_relations: bool = True,
    system: str = DEFAULT_PERIODIC_SYSTEM,
    prompt: str = DEFAULT_PERIODIC_PROMPT,
    confidence: float = DEFAULT_CONFIDENCE["llm"],
    relation_system: str = DEFAULT_RELATION_SYSTEM,
    relation_prompt: str = DEFAULT_RELATION_PROMPT,
) -> list[MemoryRecord]:
    """Tier 2 — periodic LLM extraction over the last ``batch_n`` (non-disliked) turns.

    Extracts durable memories (embedding + upserting them) and, if a graph is
    given, KG relations tagged with ``session_id`` (for later feedback pruning).
    """
    turns = [t for t in await store.load_turns(session_id) if t.feedback != -1][-batch_n:]
    if not turns:
        return []
    text = await _generate(backend, system, prompt.format(count=len(turns), transcript=_transcript(turns)))
    mems = _parse_memories(text, scope, confidence)
    await _embed_and_save(store, mems, embedder)
    if extract_relations and kg is not None:
        await _extract_relations(kg, backend, scope=scope, session_id=session_id, turns=turns,
                                 embedder=embedder, system=relation_system, prompt=relation_prompt)
    return mems


# ── Tier 3 — final (LLM) ──────────────────────────────────────────────────────

async def consolidate_session(
    store: MemoryStore,
    backend: "LLMBackend",
    *,
    session: Session,
    kg: Optional[KnowledgeGraph] = None,
    embedder: Optional["Embedder"] = None,
    close: bool = True,
    system: str = DEFAULT_FINAL_SYSTEM,
    prompt: str = DEFAULT_FINAL_PROMPT,
    confidence: float = DEFAULT_CONFIDENCE["goal_abandoned"],   # 0.85
) -> SessionSummary:
    """Tier 3 — holistic consolidation at session close/idle.

    Loads all turns (dropping disliked ones), runs a holistic LLM pass, writes a
    summary to the session, and prunes KG edges asserted by this session when any
    turn was disliked (feedback-driven cleanup).
    """
    turns = await store.load_turns(session.id)
    active = [t for t in turns if t.feedback != -1]
    disliked = len(turns) - len(active)

    mems: list[MemoryRecord] = []
    if active:
        text = await _generate(backend, system,
                               prompt.format(turn_count=len(active), transcript=_transcript(active)))
        mems = _parse_memories(text, session.scope, confidence)
        await _embed_and_save(store, mems, embedder)

    domains = sorted({d for t in active for d in t.domains})
    summary = (f"Session with {len(turns)} turns. "
               f"Domains: {', '.join(domains) or 'general'}. Memories: {len(mems)}.")

    if disliked and kg is not None:
        await kg.delete_edges_by_session(session.scope, session.id)
    if close:
        await store.close_session(session.id, summary=summary)

    return SessionSummary(session_id=session.id, summary=summary, memories=mems)
