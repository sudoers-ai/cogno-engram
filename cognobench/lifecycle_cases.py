"""End-to-end lifecycle cases: turns → Tier-1 micro → retrieval → rerank.

Deterministic (no LLM): each case feeds a multi-turn session, lets Tier-1
consolidation form memories, then queries — asserting the expected memory is
retrievable in the top-3 after reranking. Synthetic PT-BR data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LifecycleCase:
    id: str
    turns: list[dict]            # each: user_input + optional goal/goal_status/sentiment/domains/pii_types
    query: str
    expect_retrieved: str        # substring expected in a top-3 memory
    description: str = ""


CASES: list[LifecycleCase] = [
    LifecycleCase(
        id="goal_recall",
        turns=[
            {"user_input": "bom dia"},
            {"user_input": "quero agendar corte de cabelo", "goal": "agendar corte de cabelo",
             "goal_status": "NEW", "domains": ["SCHEDULING"]},
        ],
        query="agendar corte",
        expect_retrieved="agendar corte de cabelo",
        description="a started goal is recalled later",
    ),
    LifecycleCase(
        id="completed_goal_recall",
        turns=[
            {"user_input": "registra despesa de 50", "goal": "registrar despesa", "goal_status": "NEW"},
            {"user_input": "pronto", "goal": "registrar despesa", "goal_status": "COMPLETED"},
        ],
        query="completed goal registrar despesa",
        expect_retrieved="Completed goal: registrar despesa",
    ),
    LifecycleCase(
        id="domain_interest_recall",
        turns=[{"user_input": "como anda a bolsa?", "domains": ["FINANCE"]}],
        query="interested domain FINANCE",
        expect_retrieved="FINANCE",
    ),
    LifecycleCase(
        id="pii_exposure_recall",
        turns=[{"user_input": "meu email é joao@x.com", "pii_types": ["EMAIL"]}],
        query="user exposed PII EMAIL",
        expect_retrieved="EMAIL",
    ),
    LifecycleCase(
        id="frustration_recall",
        turns=[
            {"user_input": "ok", "sentiment": "NEUTRAL"},
            {"user_input": "isso de novo não!", "sentiment": "FRUSTRATED"},
        ],
        query="user became frustrated",
        expect_retrieved="frustrated",
    ),
    LifecycleCase(
        id="goal_amid_distractors",
        turns=[
            {"user_input": "oi", "domains": ["SOCIAL"]},
            {"user_input": "quero reservar uma mesa pra hoje", "goal": "reservar mesa",
             "goal_status": "NEW", "domains": ["RESTAURANT"]},
            {"user_input": "meu cpf é 111", "pii_types": ["NATIONAL_ID"]},
        ],
        query="reservar mesa",
        expect_retrieved="reservar mesa",
        description="the goal must win over domain/PII distractors",
    ),
]
