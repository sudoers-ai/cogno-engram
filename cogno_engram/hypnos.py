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

Tiers 2 and 3 take a ``cogno_synapse`` ``LLMBackend`` and (optionally) an
``Embedder``, both **duck-typed and host-injected** — engram never imports
cogno-synapse at runtime. The backend is called as
``await backend.generate(system, prompt) -> (text, tokens_in, tokens_out)``;
the embedder as ``await embedder.embed(text) -> list[float]``.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from cogno_engram.ports import KnowledgeGraph, MemoryStore
from cogno_engram.types import (
    EDGE_ACCEPTED,
    EDGE_PROPOSED,
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
    from cogno_synapse import Embedder, LLMBackend

logger = logging.getLogger("cogno_engram.hypnos")


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
  "summary": "2-3 sentence recap: what was discussed, what was resolved, what was left open",
  "preference": ["..."],
  "fact": ["name, location, profession, relationships..."],
  "goal": ["things the user wanted but did not finish"],
  "behavior": ["how they communicate"]
}

Rules for "summary": topics and outcomes only — NO exact dates, times, amounts or \
identifiers (those must come from live data, never from a recap).

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
        logger.warning("stage=hypnos event=parse_failed kind=memories scope=%s", scope)
        return []
    if not isinstance(data, dict):
        logger.warning("stage=hypnos event=parse_failed kind=memories reason=not_object scope=%s", scope)
        return []
    out: list[MemoryRecord] = []
    for category, items in data.items():
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, str) and item.strip():
                out.append(MemoryRecord(scope, str(category), item.strip(), confidence))
    return out


def _parse_narrative(text: str) -> str:
    """The optional ``"summary"`` string from the Tier-3 JSON (narrative session recap).

    Tolerant like ``_parse_memories``: malformed JSON or a missing/non-string key
    yields ``""`` (callers fall back to the stats line)."""
    try:
        data = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return ""
    summary = data.get("summary") if isinstance(data, dict) else None
    return summary.strip() if isinstance(summary, str) else ""


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
                             system: str, prompt: str,
                             edge_filter: Optional[Callable[[str, str, str], bool]] = None,
                             status: "Union[str, Callable[[str, str, str], str]]" =
                             EDGE_ACCEPTED) -> int:
    """LLM relation extraction → upsert nodes + edges (source_session tagged).

    ``edge_filter(source, target, relation) -> bool`` (optional) vetoes edges before upsert
    — the caller decides which relations are worth persisting (e.g. a host dropping volatile
    domain state that belongs to a live read, not the durable graph). Default keeps all."""
    text = await _generate(backend, system, prompt.format(transcript=_transcript(turns)))
    try:
        data = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        logger.warning("stage=hypnos event=parse_failed kind=relations scope=%s", scope)
        return 0
    if not isinstance(data, dict):
        logger.warning("stage=hypnos event=parse_failed kind=relations reason=not_object scope=%s", scope)
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
        src, tgt, rel = str(edge["source"]), str(edge["target"]), str(edge["relation"])
        if edge_filter is not None and not edge_filter(src, tgt, rel):
            continue                          # caller vetoed (e.g. volatile domain state)
        try:
            conf = float(edge.get("confidence", 1.0))
        except (TypeError, ValueError):
            conf = 1.0
        # `status` is a value (every edge the same) or a decision PER RELATION. The failure
        # mode lives with the decision, in `_status_rule` — not duplicated here, where a second
        # guard would be a branch the public path can never reach.
        st = status(src, tgt, rel) if callable(status) else status
        await kg.upsert_edge(GraphEdge(scope, src, tgt, rel, conf, source_session=session_id,
                                          status=st))
        edges += 1
    return edges


# ── Tier 2 — periodic (LLM) ───────────────────────────────────────────────────

def _accepts_three(params: "Any") -> bool:
    """Can this signature take three positional arguments?"""
    kinds = [p.kind for p in params.values()]
    if any(k is inspect.Parameter.VAR_POSITIONAL for k in kinds):
        return True
    # A REQUIRED keyword-only parameter is as fatal as a wrong arity: the call site passes three
    # positionals and nothing else, so `lambda s, t, r, *, flag` raises on every single edge.
    if any(p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty
           for p in params.values()):
        return False
    positional = [k for k in kinds if k in (inspect.Parameter.POSITIONAL_ONLY,
                                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    required = [p for p in params.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
                and p.default is inspect.Parameter.empty]
    return len(positional) >= 3 and len(required) <= 3


def _status_rule(
    propose: "Union[bool, Callable[[str, str, str], bool]]",
) -> "Union[str, Callable[[str, str, str], str]]":
    """`propose_relations` (bool or predicate) → what `_extract_relations` should stamp.

    A bool keeps the original meaning exactly (every edge the same), so nothing changes for a
    host that never passes a predicate. A callable is forwarded, and the per-edge decision plus
    its failure mode live at the single site that applies it.
    """
    if not callable(propose):
        return EDGE_PROPOSED if propose else EDGE_ACCEPTED

    # Validate the callable ONCE, here, and refuse it loudly. Both mistakes below are silent
    # and identical in effect if left to the per-edge guard: every edge becomes `proposed`,
    # so the host's graph block quietly EMPTIES — which is the regression the opt-in default
    # exists to prevent, arriving through the option meant to avoid it. A wrong predicate is a
    # programming error at wiring time; it must not be discovered as a slow leak in prod.
    # `iscoroutinefunction` is False for an OBJECT whose `__call__` is async, and such an object
    # is just as callable — and just as silently wrong, since the call returns a truthy coroutine.
    if inspect.iscoroutinefunction(propose) or \
            inspect.iscoroutinefunction(getattr(propose, "__call__", None)):
        raise TypeError("propose_relations must be a sync predicate: a coroutine function is "
                        "callable, so every edge would be judged on a truthy coroutine object")
    try:
        params = inspect.signature(propose).parameters
    except (TypeError, ValueError):      # builtins/C callables expose no signature
        params = None
    if params is not None and not _accepts_three(params):
        raise TypeError("propose_relations must take (source, target, relation); got "
                        f"{tuple(params)} — a wrong arity would raise on every edge and hold "
                        "the whole graph for review")

    def _decide(source: str, target: str, relation: str) -> str:
        # A predicate that raises yields `proposed`, never `accepted`: an edge nobody could
        # classify waits for a human rather than being spoken as fact. Defaulting the other way
        # would let one broken classifier publish exactly the class it exists to hold back.
        try:
            hold = propose(source, target, relation)
        except Exception:  # noqa: BLE001 — an unclassifiable edge is held, not lost
            logger.warning("stage=hypnos event=status_predicate_failed relation=%s", relation,
                           exc_info=True)
            return EDGE_PROPOSED
        return EDGE_PROPOSED if hold else EDGE_ACCEPTED

    return _decide


async def periodic_consolidate(
    store: MemoryStore,
    backend: "LLMBackend",
    *,
    scope: str,
    session_id: str,
    batch_n: int = 10,
    embedder: Optional["Embedder"] = None,
    kg: Optional[KnowledgeGraph] = None,
    kg_scope: Optional[str] = None,
    extract_relations: bool = True,
    system: str = DEFAULT_PERIODIC_SYSTEM,
    prompt: str = DEFAULT_PERIODIC_PROMPT,
    confidence: float = DEFAULT_CONFIDENCE["llm"],
    relation_system: str = DEFAULT_RELATION_SYSTEM,
    relation_prompt: str = DEFAULT_RELATION_PROMPT,
    edge_filter: Optional[Callable[[str, str, str], bool]] = None,
    propose_relations: "Union[bool, Callable[[str, str, str], bool]]" = False,
) -> list[MemoryRecord]:
    """Tier 2 — periodic LLM extraction over the last ``batch_n`` (non-disliked) turns.

    Extracts durable memories (embedding + upserting them) and, if a graph is
    given, KG relations tagged with ``session_id`` (for later feedback pruning).
    The graph is often shared at a BROADER scope than the memories (e.g. the host
    keeps memories per identity but the knowledge graph per tenant) — pass
    ``kg_scope`` to aim the graph writes there; default is ``scope``.

    ``edge_filter(source, target, relation) -> bool`` (optional) vetoes relations before they
    are persisted — the caller owns the domain judgement of what belongs in the durable graph
    (e.g. dropping volatile state like an appointment/status that a live tool read should own).

    ``propose_relations`` writes the extracted edges as ``proposed`` instead of ``accepted``, so
    they wait for a human before a walk can speak them (``KnowledgeGraph.pending_edges`` /
    ``set_edge_status``). **Opt-in, and deliberately so.** An edge here becomes a sentence the
    agent states about a person as if it knew — "your son Pedro" is either a kindness or an
    invention — which argues for review; but flipping the default would, on the next deploy,
    silently empty the graph block of every host already running, with nothing in the logs
    saying why. The host that has somewhere to review them turns it on.

    **It also takes a PREDICATE** — ``propose_relations(source, target, relation) -> bool`` —
    because "review everything or review nothing" is the wrong granularity for the thing being
    protected. The edges that become a sentence about a PERSON ("your wife Maria") are a small,
    nameable class; the rest ("the clinic accepts Unimed") are domain facts a walk should keep
    stating. All-or-nothing forces a host to choose between speaking unreviewed claims about
    someone's family and losing its whole knowledge block — and the first host to meet that
    choice took the first option without noticing. The predicate lets proximity wait for a human
    while domain relations stay ``accepted``. Same shape and same seam as ``edge_filter``, which
    already decides per relation what is worth persisting at all.

    **Turning it on does NOT reach rows that are already there.** ``upsert_edge`` only ever
    PROMOTES a status (a review nobody would redo must not expire), so an edge already stamped
    ``accepted`` stays ``accepted`` when the next extraction re-emits it. A host that flips this
    on believing its existing proximity edges now wait for a human is wrong, and wrong in the
    direction that matters: it keeps speaking every one of them. Migrating them is a deliberate
    act — walk the affected nodes and ``set_edge_status(..., "proposed")`` — and there is no
    enumeration shortcut, because ``pending_edges`` lists proposals, which is precisely what
    these are not.
    """
    # Scope the read: session_id is a host-derived uuid5 that can collide across scopes, and the
    # extract below is written back into THIS scope — an un-scoped read would fold another tenant's
    # turns into this tenant's memories.
    turns = [t for t in await store.load_turns(session_id, scope=scope)
             if t.feedback != -1][-batch_n:]
    if not turns:
        return []
    text = await _generate(backend, system, prompt.format(count=len(turns), transcript=_transcript(turns)))
    mems = _parse_memories(text, scope, confidence)
    await _embed_and_save(store, mems, embedder)
    if extract_relations and kg is not None:
        await _extract_relations(kg, backend, scope=kg_scope or scope, session_id=session_id,
                                 turns=turns, embedder=embedder,
                                 system=relation_system, prompt=relation_prompt,
                                 edge_filter=edge_filter,
                                 status=_status_rule(propose_relations))
    logger.info("stage=hypnos tier=2 event=periodic_done turns=%d extracted=%d",
                len(turns), len(mems))
    return mems


# ── Tier 3 — final (LLM) ──────────────────────────────────────────────────────

async def consolidate_session(
    store: MemoryStore,
    backend: "LLMBackend",
    *,
    session: Session,
    kg: Optional[KnowledgeGraph] = None,
    kg_scope: Optional[str] = None,
    embedder: Optional["Embedder"] = None,
    close: bool = True,
    system: str = DEFAULT_FINAL_SYSTEM,
    prompt: str = DEFAULT_FINAL_PROMPT,
    confidence: float = DEFAULT_CONFIDENCE["goal_abandoned"],   # 0.85
) -> SessionSummary:
    """Tier 3 — holistic consolidation at session close/idle.

    Loads all turns (dropping disliked ones), runs a holistic LLM pass, writes a
    summary to the session, and prunes KG edges asserted by this session when any
    turn was disliked (feedback-driven cleanup). ``kg_scope`` mirrors Tier 2:
    when the graph lives at a broader scope than the session (host keeps it per
    tenant), pass it so the feedback pruning hits the edges Tier 2 actually
    wrote; default is ``session.scope``.
    """
    turns = await store.load_turns(session.id, scope=session.scope)   # isolate to this session's scope
    active = [t for t in turns if t.feedback != -1]
    disliked = len(turns) - len(active)

    mems: list[MemoryRecord] = []
    narrative = ""
    if active:
        text = await _generate(backend, system,
                               prompt.format(turn_count=len(active), transcript=_transcript(active)))
        mems = _parse_memories(text, session.scope, confidence)
        narrative = _parse_narrative(text)
        await _embed_and_save(store, mems, embedder)

    # The narrative recap (same LLM call) is what a host can inject as EARLIER CONTEXT on the
    # next session; the stats line is the fallback when the model omitted/garbled it.
    domains = sorted({d for t in active for d in t.domains})
    summary = narrative or (f"Session with {len(turns)} turns. "
                            f"Domains: {', '.join(domains) or 'general'}. Memories: {len(mems)}.")

    if disliked and kg is not None:
        # The prune REFUSES a blank session id (a blank is not a wildcard). Check it HERE rather
        # than catching the raise: by this line the LLM pass has run and the memories are saved,
        # and `close_session` is still ahead — letting it out would leave the session OPEN and
        # the janitor would re-consolidate it every tick, duplicating memories. And an
        # `except ValueError` around the call would also swallow `_require_scope`'s, reporting a
        # bad SCOPE as a blank session id — a wrong diagnosis is worse than none.
        if not (session.id or "").strip():
            logger.warning("stage=hypnos tier=3 event=prune_skipped reason=blank_session_id "
                           "scope=%s — feedback pruning needs a real session id", session.scope)
        else:
            await kg.delete_edges_by_session(kg_scope or session.scope, session.id)
    if close:
        # scope the WRITE too: the sessions table is keyed by id alone, so without it a colliding
        # id from another tenant would receive this scope's LLM-generated summary.
        await store.close_session(session.id, summary=summary, scope=session.scope)

    logger.info("stage=hypnos tier=3 event=session_done turns=%d memories=%d disliked=%d",
                len(turns), len(mems), disliked)
    return SessionSummary(session_id=session.id, summary=summary, memories=mems)
