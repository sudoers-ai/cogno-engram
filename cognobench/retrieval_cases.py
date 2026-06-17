"""Synthetic retrieval cases (PT-BR, finance/vet/scheduling flavoured).

Each case stores a small corpus WITH distractors, then queries it — so hit@1
actually tests ranking discrimination, not just recall. All data is synthetic;
no real user content. The expected hit is matched as a substring of the
top-ranked memory's content.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievalCase:
    id: str
    memories: list[tuple[str, str]]      # (category, content)
    query: str
    expect_top: str                      # substring expected at rank 1
    use_embedding: bool = True           # False → BM25-only (reflects real data: ~0% embedded)
    description: str = ""


CASES: list[RetrievalCase] = [
    RetrievalCase(
        id="finance_payment_pref",
        memories=[
            ("preference", "o cliente prefere pagar via pix"),
            ("preference", "o cliente gosta de café sem açúcar"),
            ("fact", "o cliente tem uma conta no banco digital"),
        ],
        query="como o cliente prefere pagar",
        expect_top="pix",
        description="surface-token overlap should surface the payment preference",
    ),
    RetrievalCase(
        id="vet_pet_name",
        memories=[
            ("fact", "o cachorro do cliente se chama Rex"),
            ("fact", "o cliente mora em São Paulo"),
            ("preference", "o cliente prefere atendimento pela manhã"),
        ],
        query="qual o nome do cachorro",
        expect_top="Rex",
    ),
    RetrievalCase(
        id="schedule_recurring",
        memories=[
            ("preference", "o cliente costuma agendar corte de cabelo toda sexta"),
            ("fact", "o cliente pagou a última fatura em dia"),
            ("sentiment", "o cliente ficou satisfeito com o último atendimento"),
        ],
        query="quando o cliente costuma agendar",
        expect_top="sexta",
    ),
    RetrievalCase(
        id="finance_budget",
        memories=[
            ("fact", "o orçamento mensal do cliente é de 3000 reais"),
            ("fact", "o cliente tem dois filhos"),
            ("preference", "o cliente prefere receber relatórios por email"),
        ],
        query="qual o orçamento mensal",
        expect_top="3000",
    ),
    RetrievalCase(
        id="vet_allergy",
        memories=[
            ("fact", "o gato do cliente tem alergia a frango"),
            ("fact", "o gato do cliente é da raça siamês"),
            ("preference", "o cliente prefere ração premium"),
        ],
        query="o gato tem alguma alergia",
        expect_top="alergia a frango",
    ),
    RetrievalCase(
        id="contact_preference",
        memories=[
            ("preference", "o cliente prefere ser contatado por whatsapp"),
            ("fact", "o cliente trabalha como dentista"),
            ("preference", "o cliente gosta de promoções"),
        ],
        query="como o cliente prefere ser contatado",
        expect_top="whatsapp",
    ),
    RetrievalCase(
        id="finance_goal",
        memories=[
            ("goal", "o cliente quer juntar dinheiro para uma viagem"),
            ("goal", "o cliente quer quitar o cartão de crédito"),
            ("fact", "o cliente recebe salário no quinto dia útil"),
        ],
        query="qual a meta de viagem do cliente",
        expect_top="viagem",
    ),
    RetrievalCase(
        id="restaurant_restriction",
        memories=[
            ("fact", "o cliente é intolerante a lactose"),
            ("preference", "o cliente prefere mesa perto da janela"),
            ("fact", "o cliente costuma pedir vinho tinto"),
        ],
        query="o cliente tem restrição alimentar de lactose",
        expect_top="lactose",
    ),
    # ── BM25-only (no embedding) — reflects production reality: in the parent's
    #    PG dump only ~2/626 memories carried an embedding, so retrieval leans on
    #    BM25/chronological. These cases retrieve with NO query embedding. ──
    RetrievalCase(
        id="bm25_only_goal_recall",
        memories=[
            ("goal", "o cliente quer parcelar a fatura do cartão"),
            ("goal", "o cliente quer abrir uma conta poupança"),
            ("fact", "o cliente tem dois cartões de crédito"),
        ],
        query="parcelar fatura cartão",
        expect_top="parcelar a fatura",
        use_embedding=False,
        description="BM25-only path (the common case in production)",
    ),
    RetrievalCase(
        id="bm25_only_pii_recall",
        memories=[
            ("pii", "o cliente informou o CPF durante o cadastro"),
            ("preference", "o cliente prefere atendimento por email"),
            ("goal", "o cliente quer atualizar o endereço"),
        ],
        query="cliente informou CPF cadastro",
        expect_top="CPF",
        use_embedding=False,
    ),
]
