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

Tiers 2 and 3 take a ``cogno_anima`` ``LLMBackend`` (host-injected) and a default
prompt that the host may override. They are declared here as the contract; the
LLM-driven bodies land in the D1 build-out phase.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from cogno_engram.ports import KnowledgeGraph, MemoryStore
from cogno_engram.types import DEFAULT_CONFIDENCE, MemoryRecord, Session, SessionSummary, TurnRecord

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cogno_anima.llm.base import LLMBackend


DEFAULT_PERIODIC_PROMPT = (
    "Extract durable user facts, preferences, pending goals, and behaviour from "
    "the conversation transcript below. Return strict JSON."
)
DEFAULT_FINAL_PROMPT = (
    "Holistically summarise the whole session: persistent sentiment, profile "
    "updates, completed and abandoned goals. Return strict JSON."
)


def micro_consolidate(turn: TurnRecord, prev: Optional[TurnRecord] = None) -> list[MemoryRecord]:
    """Tier 1 — deterministic, LLM-free consolidation of a single turn.

    Reads the flat cognition signals the host mapped onto the turn (goal /
    goal_status / sentiment / domains / pii_types) and emits memories for the
    structural changes since the previous turn. Pure and cheap — same nature as
    cogno-anima's ID/Drift heuristics, but it writes I/O so it lives here.
    """
    mems: list[MemoryRecord] = []
    scope = turn.scope

    # Goal lifecycle transitions.
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

    # Sentiment spike into frustration (edge-triggered).
    if turn.sentiment.upper() == "FRUSTRATED" and (prev is None or prev.sentiment.upper() != "FRUSTRATED"):
        mems.append(MemoryRecord(scope, "sentiment", "User became frustrated.",
                                 DEFAULT_CONFIDENCE["sentiment"]))

    # PII exposure — record the fact (never the value).
    if turn.pii_types:
        mems.append(MemoryRecord(scope, "pii", f"User exposed PII: {', '.join(turn.pii_types)}",
                                 DEFAULT_CONFIDENCE["pii"]))

    # New domain interest.
    prev_domains = set(prev.domains) if prev else set()
    for domain in turn.domains:
        if domain not in prev_domains:
            mems.append(MemoryRecord(scope, "preference", f"Interested in domain: {domain}",
                                     DEFAULT_CONFIDENCE["preference"]))

    return mems


async def periodic_consolidate(
    store: MemoryStore,
    backend: "LLMBackend",
    *,
    scope: str,
    session_id: str,
    batch_n: int = 10,
    extract_relations: bool = True,
    kg: Optional[KnowledgeGraph] = None,
    prompt: str = DEFAULT_PERIODIC_PROMPT,
) -> list[MemoryRecord]:
    """Tier 2 — periodic LLM extraction over the last ``batch_n`` turns.

    Excludes turns with negative feedback, asks the LLM for durable memories,
    embeds + upserts them, and (optionally) extracts KG relations. LLM body
    lands in the build-out phase.
    """
    raise NotImplementedError("Tier-2 LLM consolidation is implemented in the D1 build-out phase")


async def consolidate_session(
    store: MemoryStore,
    backend: "LLMBackend",
    *,
    session: Session,
    kg: Optional[KnowledgeGraph] = None,
    prompt: str = DEFAULT_FINAL_PROMPT,
) -> SessionSummary:
    """Tier 3 — holistic consolidation at session close/idle.

    Loads all turns (dropping disliked ones), runs a holistic LLM pass, writes
    the summary to the session, and prunes KG edges asserted by disliked turns.
    LLM body lands in the build-out phase.
    """
    raise NotImplementedError("Tier-3 LLM consolidation is implemented in the D1 build-out phase")
