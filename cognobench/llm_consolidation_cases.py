"""Cases for the gated LLM-consolidation dimension (hypnos Tier-2 quality).

Each case is a synthetic PT-BR session transcript; we run the real LLM
extraction and check the expected durable facts/preferences surface in the
produced memories (soft, model-dependent). No real user data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LlmConsolidationCase:
    id: str
    turns: list[tuple[str, str]]          # (user_input, assistant_response)
    expect_contains: list[str] = field(default_factory=list)   # substrings in the memories
    description: str = ""


CASES: list[LlmConsolidationCase] = [
    LlmConsolidationCase(
        id="payment_preference",
        turns=[
            ("Oi, tudo bem?", "Tudo ótimo! Como posso ajudar?"),
            ("Prefiro pagar tudo no pix, nunca uso boleto.", "Anotado, sempre pix."),
            ("Pode registrar uma despesa de 50 do almoço?", "Registrei R$50 do almoço."),
        ],
        expect_contains=["pix"],
        description="extracts the payment preference",
    ),
    LlmConsolidationCase(
        id="profile_facts",
        turns=[
            ("Meu nome é Maria e eu moro em São Paulo.", "Prazer, Maria!"),
            ("Trabalho como dentista.", "Legal!"),
        ],
        # The default prompt extracts to canonical English; "dentist" survives,
        # the name is often generalised — use a language-robust expectation.
        expect_contains=["dentist"],
        description="extracts a profile fact (profession)",
    ),
    LlmConsolidationCase(
        id="pending_goal",
        turns=[
            ("Quero juntar dinheiro para uma viagem ao Japão.", "Ótima meta!"),
            ("Mas ainda não comecei a poupar.", "Podemos planejar."),
        ],
        expect_contains=["japan"],   # proper noun survives the English extraction
        description="extracts an unfinished goal",
    ),
    LlmConsolidationCase(
        id="dietary_restriction",
        turns=[
            ("Sou intolerante a lactose.", "Vou lembrar disso."),
            ("Pode sugerir um restaurante?", "Claro, com opções sem lactose."),
        ],
        expect_contains=["lactose"],
        description="extracts a dietary restriction",
    ),
]
