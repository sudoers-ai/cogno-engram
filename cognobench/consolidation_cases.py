"""Synthetic consolidation cases for hypnos Tier-1 (micro, LLM-free).

Each case feeds a turn (+ optional previous turn) and asserts the set of
memories micro_consolidate emits — category + a content substring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ConsolidationCase:
    id: str
    turn: dict
    prev: Optional[dict] = None
    expect: list[tuple[str, str]] = field(default_factory=list)   # (category, substr)
    expect_empty: bool = False


def _t(**kw) -> dict:
    base = dict(session_id="sess", scope="bench", turn_n=1, user_input="x")
    base.update(kw)
    return base


CASES: list[ConsolidationCase] = [
    ConsolidationCase(
        id="goal_started",
        turn=_t(goal="agendar consulta", goal_status="NEW"),
        expect=[("goal", "Started goal: agendar consulta")],
    ),
    ConsolidationCase(
        id="goal_completed",
        turn=_t(goal="registrar despesa", goal_status="COMPLETED"),
        expect=[("goal", "Completed goal: registrar despesa")],
    ),
    ConsolidationCase(
        id="goal_abandoned",
        turn=_t(goal="reservar mesa", goal_status="ABANDONED"),
        expect=[("goal", "Abandoned goal: reservar mesa")],
    ),
    ConsolidationCase(
        id="goal_unchanged_silent",
        turn=_t(goal="g", goal_status="ONGOING"),
        prev=_t(goal="g", goal_status="ONGOING"),
        expect_empty=True,
    ),
    ConsolidationCase(
        id="frustration_edge",
        turn=_t(sentiment="FRUSTRATED"),
        prev=_t(sentiment="NEUTRAL"),
        expect=[("sentiment", "frustrated")],
    ),
    ConsolidationCase(
        id="pii_leak",
        turn=_t(pii_types=["EMAIL", "PHONE"]),
        expect=[("pii", "EMAIL")],
    ),
    ConsolidationCase(
        id="new_domain",
        turn=_t(domains=["FINANCE", "HEALTH"]),
        prev=_t(domains=["FINANCE"]),
        expect=[("preference", "HEALTH")],
    ),
    ConsolidationCase(
        id="combined_signals",
        turn=_t(goal="pagar conta", goal_status="NEW", sentiment="FRUSTRATED",
                pii_types=["NATIONAL_ID"]),
        prev=_t(sentiment="NEUTRAL"),
        expect=[("goal", "Started goal: pagar conta"),
                ("sentiment", "frustrated"),
                ("pii", "NATIONAL_ID")],
    ),
]
