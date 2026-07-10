"""Cases for the gated graph-capture dimension (hypnos Tier-2 relation extraction).

Each case is a synthetic entity-rich PT-BR session; we run the REAL capture path
(``periodic_consolidate`` with ``extract_relations=True`` + a graph + ``kg_scope``)
against a live model and check that the expected entity connections are reachable
in the resulting graph (soft, model-dependent — relation NAMES vary by model, so
we assert label-to-label reachability, not exact relation strings). Hard,
model-independent invariants (valid confidence, session tagging, kg_scope
segregation) are checked by the dimension itself. No real user data.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GraphCaptureCase:
    id: str
    turns: list[tuple[str, str]]                 # (user_input, assistant_response)
    expect_connected: list[tuple[str, str]] = field(default_factory=list)
    #   (start_label, reachable_label) — case-insensitive substring match on node
    #   labels, reachability within a 2-hop walk from start.
    description: str = ""


CASES: list[GraphCaptureCase] = [
    GraphCaptureCase(
        id="owner_pet",
        turns=[
            ("Oi, meu nome é José e meu cachorro se chama Rex.",
             "Olá José! Prazer em conhecer você e o Rex."),
            ("O Rex é um Pastor Alemão de 3 anos.", "Anotado: Rex, Pastor Alemão."),
        ],
        expect_connected=[("josé", "rex")],
        description="owner↔pet edge captured from the conversation",
    ),
    GraphCaptureCase(
        id="employment_place",
        turns=[
            ("Eu trabalho na Acme, em São Paulo.", "Legal, anotado."),
            ("Sou gerente de contas lá.", "Perfeito, gerente de contas na Acme."),
        ],
        expect_connected=[("acme", "são paulo")],
        description="org↔place edge captured",
    ),
    GraphCaptureCase(
        id="family_pet_chain",
        turns=[
            ("Minha esposa Maria tem uma gata chamada Mimi.", "Que fofa a Mimi!"),
            ("A Mimi é siamesa.", "Anotado: Mimi, siamesa."),
        ],
        expect_connected=[("maria", "mimi")],
        description="family member↔pet edge captured",
    ),
]
