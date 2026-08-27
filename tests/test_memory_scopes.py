"""``memory_scopes`` — a enumeração que a retenção precisa e que a loja recusa a toda a gente.

`_require_scope` recusa prefixo vazio, **e tem razão**: uma base isolada por scope não entrega a
loja inteira a quem pede nada. Esta é a excepção, e é ARGUMENTADA e não assumida.

**A razão é a razão de a retenção existir.** A poda não pode ser guiada por tenant, porque
**o scope cujo tenant desapareceu é precisamente o que mais precisa de ser podado** — ninguém o
possui, ninguém vai pedi-lo, e mais nada o visita.

Medido na caixa viva a 27/08/2026: **14 scopes têm memórias, 13 têm tenant vivo**, e o que falta é
`default/guest` com **29 memórias de VISITANTES** — pessoas que nunca se registaram, falaram uma
vez e foram embora — **com categorias que incluem `pii`**. **A regra salta exactamente quem tem
menos base para ser retido.**
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryStore
from cogno_engram.maintenance import memory_scopes
from cogno_engram.types import MemoryRecord

pytestmark = pytest.mark.asyncio


async def _store(*scopes: str):
    store = InMemoryStore()
    for scope in scopes:
        await store.save_memory(MemoryRecord(scope=scope, content="x", category="fact",
                                             confidence=1.0))
    return store


async def test_ve_o_scope_ORFAO_que_a_enumeracao_por_tenant_nao_ve():
    """O caso que motivou o método, e o único que interessa.

    `default/guest` não tem linha em `tenants`, logo uma varredura guiada por tenant nunca o
    visita — e é o que tem a base mais fraca para reter seja o que for.
    """
    store = await _store("acme/ana", "acme/bruno", "default/guest")
    assert "default/guest" in await memory_scopes(store)


async def test_a_lista_e_ESTAVEL_e_sem_repetidos():
    """Uma varredura periódica que muda de ordem entre corridas faz diffs de log ilegíveis, e
    um scope repetido faz a poda visitá-lo duas vezes — no modo armado, contá-lo-ia a dobrar."""
    store = await _store("t/b", "t/a", "t/b", "t/a", "t/c")
    scopes = await memory_scopes(store)
    assert scopes == ["t/a", "t/b", "t/c"] == sorted(set(scopes))


async def test_uma_loja_VAZIA_devolve_lista_vazia_e_nao_levanta():
    """Zero é uma resposta legítima — um deploy novo não tem memórias, e a varredura tem de
    correr na mesma e dizer zero, em vez de rebentar e parecer uma avaria."""
    assert await memory_scopes(InMemoryStore()) == []


async def test_NAO_pede_scope_e_e_esse_o_ponto():
    """A assinatura é a decisão. Se alguém lhe acrescentar um `scope` para "ser consistente com
    as irmãs", o método deixa de responder à pergunta para que foi escrito — e o órfão volta a
    ser invisível, que é o defeito que ele existe para fechar.
    """
    import inspect

    from cogno_engram.ports import MemoryStore

    params = set(inspect.signature(MemoryStore.memory_scopes).parameters) - {"self"}
    assert params == set(), (
        f"`memory_scopes` ganhou parâmetros {params} — se passar a exigir um scope, deixa de "
        f"poder ver o scope cujo tenant desapareceu, que é a única razão de existir")


async def test_devolve_CHAVES_e_nunca_conteudo():
    """A FRONTEIRA, e é ela que torna este método defensável em vez de uma excepção ao guarda.

    `_require_scope` existe para impedir uma LEITURA ATRAVÉS de scopes — que uma consulta
    devolva conteúdo de vários contactos porque alguém passou vazio. **Enumerar CHAVES não é
    isso**: são duas perguntas diferentes, e o guarda só responde à primeira.

    O que mantém as duas separadas é absoluto: **identificadores à saída, NUNCA conteúdo.** Nem
    uma memória, nem um rótulo, nem uma contagem por categoria. **O dia em que isto devolver
    conteúdo é um buraco**, e este teste é o que o diz.
    """
    store = InMemoryStore()
    segredo = "o CPF do contacto é 123.456.789-09"
    await store.save_memory(MemoryRecord(scope="acme/ana", content=segredo, category="pii",
                                         confidence=1.0))
    await store.save_memory(MemoryRecord(scope="default/guest", content="mora em Berlim",
                                         category="fact", confidence=1.0))

    scopes = await memory_scopes(store)

    assert scopes == ["acme/ana", "default/guest"], "as chaves saem"
    achatado = repr(scopes)
    assert segredo not in achatado and "Berlim" not in achatado, (
        "conteúdo de memória saiu pela enumeração — isto deixou de ser uma lista de chaves e "
        "passou a ser a leitura através de scopes que o `_require_scope` existe para impedir")
    assert all(isinstance(s, str) for s in scopes), (
        "a saída tem de ser texto simples: um objecto rico é por onde o conteúdo volta a entrar")
    # CONTROLO POSITIVO: o segredo ESTÁ mesmo na loja, logo a asserção acima mede a fronteira
    # e não uma loja vazia.
    guardadas = await store.load_memories("acme/ana", limit=10)
    assert any(segredo in m.content for m in guardadas)
