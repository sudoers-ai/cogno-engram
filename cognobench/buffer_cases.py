"""Short-term buffer retention cases (mirrors the parent's memory_cases focus).

The parent's memory benchmark targets the ConversationBuffer: entity retention,
PII carryover, domain continuity, and behaviour under load. These cases check
the sliding-window semantics — which terms survive in the window of a given size.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BufferCase:
    id: str
    turns: list[str]                                  # user_input per turn
    window_size: int
    expect_present: list[str] = field(default_factory=list)   # substrings in the window
    expect_absent: list[str] = field(default_factory=list)    # substrings dropped by the window
    description: str = ""


CASES: list[BufferCase] = [
    BufferCase(
        id="entity_retention",
        turns=["preciso de informações sobre Python e Django",
               "qual deles é melhor para APIs?"],
        window_size=10,
        expect_present=["Python", "Django"],
        description="entities mentioned earlier remain in the window",
    ),
    BufferCase(
        id="pii_carryover",
        turns=["meu cartão é 4111 1111 1111 1111", "pode confirmar?"],
        window_size=10,
        expect_present=["4111", "confirmar"],
        description="a PII turn stays in the window for the follow-up",
    ),
    BufferCase(
        id="domain_continuity",
        turns=["quanto tá o dólar hoje?", "e o euro?"],
        window_size=10,
        expect_present=["dólar", "euro"],
    ),
    BufferCase(
        id="buffer_under_load",
        turns=[f"mensagem número {n}" for n in range(1, 9)],   # 8 turns
        window_size=3,
        expect_present=["número 6", "número 7", "número 8"],
        expect_absent=["número 1", "número 5"],
        description="under load the window keeps the most recent turns only",
    ),
    BufferCase(
        id="cross_reference_within_window",
        turns=["o cliente se chama Maria", "ela trabalha em SP", "qual o nome dela?"],
        window_size=10,
        expect_present=["Maria", "nome dela"],
        description="a distant entity is still referencible within the window",
    ),
    BufferCase(
        id="fresh_after_farewell",
        turns=["registra 50 do almoço", "valeu!", "tchau", "oi de novo", "quanto tá o dólar?"],
        window_size=2,
        expect_present=["dólar"],
        expect_absent=["almoço"],
        description="recent context dominates after a topic reset",
    ),
]
