"""``prune_memories(dry_run=True)`` — ler o número ANTES de o rasgo acontecer.

`prune_memories` existe e está testada **desde sempre, e ninguém a chama**: varrido no host, com
controlo positivo (a mesma varredura encontra `from cogno_engram import maintenance` no
`reembed.py`, portanto sabe achar). A memória só cresce; nada sai por idade.

Ligá-la é barato — a função difícil está feita. O que faltava era poder **aprovar** a regra:
retenção é irreversível, e ninguém deve descobrir o que uma regra de 120 dias remove **vendo-a
remover**.

**UM predicado, dois verbos.** Contar com uma consulta à parte seria re-derivar a regra que
decide um apagamento, e as duas divergiriam no dia em que um filtro fosse acrescentado — a forma
de defeito que este repositório passa a vida a encontrar. Aqui *"o que iria"* e *"o que foi"* não
podem discordar, porque são o mesmo `WHERE`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from cogno_engram.adapters.in_memory import InMemoryStore
from cogno_engram.maintenance import prune_memories
from cogno_engram.types import MemoryRecord

pytestmark = pytest.mark.asyncio

SCOPE = "t/ret"
NOW = datetime(2026, 8, 27, tzinfo=timezone.utc)


async def _store():
    store = InMemoryStore()
    velha = NOW - timedelta(days=200)
    nova = NOW - timedelta(days=10)
    await store.save_memory(MemoryRecord(scope=SCOPE, content="palpite velho",
                                         category="preference", confidence=0.4,
                                         created_at=velha))
    await store.save_memory(MemoryRecord(scope=SCOPE, content="facto velho",
                                         category="fact", confidence=1.0, created_at=velha))
    await store.save_memory(MemoryRecord(scope=SCOPE, content="palpite novo",
                                         category="preference", confidence=0.4,
                                         created_at=nova))
    return store


async def _conteudos(store):
    return sorted(m.content for m in await store.load_memories(SCOPE, limit=100))


async def test_a_dry_run_counts_and_touches_NOTHING():
    store = await _store()
    antes = await _conteudos(store)
    n = await prune_memories(store, SCOPE, older_than=timedelta(days=120),
                             max_confidence=0.75, now=NOW, dry_run=True)
    assert n == 1, "só o palpite VELHO é candidato"
    assert await _conteudos(store) == antes, "um ensaio que apaga não é um ensaio"


async def test_the_dry_run_number_is_EXACTLY_what_the_real_run_removes():
    """O que o par existe para garantir: a contagem e o corte são o MESMO predicado.

    Um teste que só verificasse "o ensaio devolve um número" passaria com os dois a discordar —
    e a discordância só apareceria em produção, depois de apagar.
    """
    store = await _store()
    previsto = await prune_memories(store, SCOPE, older_than=timedelta(days=120),
                                    max_confidence=0.75, now=NOW, dry_run=True)
    apagado = await prune_memories(store, SCOPE, older_than=timedelta(days=120),
                                   max_confidence=0.75, now=NOW)
    assert previsto == apagado == 1


async def test_a_CONFIRMED_fact_survives_its_own_age():
    """`max_confidence` não é decoração: sem ele isto apaga um facto confirmado por ser velho,
    que é perda de dados vestida com a palavra "limpeza"."""
    store = await _store()
    await prune_memories(store, SCOPE, older_than=timedelta(days=120),
                         max_confidence=0.75, now=NOW)
    assert await _conteudos(store) == ["facto velho", "palpite novo"], "o facto de confiança 1.0 tinha de ficar"


async def test_WITHOUT_the_ceiling_the_confirmed_fact_would_go():
    """O controlo do teste acima: prova que ele mede o tecto, e não que o facto era novo."""
    store = await _store()
    n = await prune_memories(store, SCOPE, older_than=timedelta(days=120), now=NOW,
                             dry_run=True)
    assert n == 2, "sem tecto, o facto confirmado entra na conta — é por isso que o tecto existe"


async def test_the_default_is_still_to_DELETE():
    """`dry_run` tem de ser opt-in: um default que só conta transformaria em silêncio todos os
    chamadores existentes em ensaios, e a faxina pararia sem ninguém dar por isso."""
    store = await _store()
    await prune_memories(store, SCOPE, older_than=timedelta(days=120), max_confidence=0.75,
                         now=NOW)
    assert "palpite velho" not in await _conteudos(store)


async def test_the_PORTS_default_is_also_to_delete():
    """O default do ADAPTADOR, e não só o do ajudante — apanhado por uma mutação sobrevivente.

    Trocar `dry_run: bool = False` para `True` no adaptador passava os dez testes acima, porque
    o `prune_memories` passa sempre o valor explicitamente e nunca cai no default. Mas
    `delete_memories` é **porta pública**: um chamador que a use directamente veria a faxina
    parar em silêncio — a limpeza a devolver números certos e a não limpar nada.

    O par de asserções é o que discrimina: o default apaga, e o `dry_run=True` explícito não.
    """
    store = await _store()
    n = await store.delete_memories(SCOPE, older_than=NOW - timedelta(days=120),
                                    max_confidence=0.75)
    assert n == 1 and "palpite velho" not in await _conteudos(store), (
        "o default da PORTA tem de apagar — um default que só conta transforma todo o "
        "chamador existente num ensaio, sem uma linha de aviso")

    store2 = await _store()
    antes = await _conteudos(store2)
    n2 = await store2.delete_memories(SCOPE, older_than=NOW - timedelta(days=120),
                                      max_confidence=0.75, dry_run=True)
    assert n2 == 1 and await _conteudos(store2) == antes
