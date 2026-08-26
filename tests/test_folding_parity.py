"""A tabela de transliteração é uma CÓPIA — e este ficheiro é o que a impede de apodrecer.

`cogno_engram/folding.py::_TRANSLIT` foi DERIVADA do dicionário do `unaccent` do Postgres, que
vive noutro processo e pode mudar de versão sem ninguém deste lado saber. Uma cópia que ninguém
re-deriva é uma cópia que diverge em silêncio — e aqui divergir significa os dois adaptadores
darem respostas DIFERENTES à mesma pergunta, que é o defeito exacto que este trabalho veio fechar.

Corre só com `ENGRAM_TEST_DSN`, porque a única forma honesta de verificar a cópia é perguntar ao
original.
"""

from __future__ import annotations

import os
import unicodedata

import pytest

psycopg = pytest.importorskip("psycopg")

from cogno_engram.adapters.postgres import ensure_schema         # noqa: E402
from cogno_engram.folding import fold_label                      # noqa: E402

DSN = os.environ.get("ENGRAM_TEST_DSN", "")

#: O alfabeto que o acordo COBRE, dito em voz alta: Latin-1 Supplement + Latin Extended-A, onde
#: vivem os nomes pt/es/en/de/fr/it. Fora dele os dois lados podem divergir (medido: sigma final
#: grego dá `σ` no Python e `ς` no Postgres) — fazer os dois concordarem em todo o Unicode exigiria
#: portar o dicionário inteiro do `unaccent`, e um domínio maior do que o produto tem não paga a
#: cópia. Um tecto declarado não é o mesmo que um tecto esquecido.
def _alfabeto_coberto() -> "list[str]":
    letras = [chr(c) for c in range(0xC0, 0x180)
              if unicodedata.category(chr(c)).startswith("L")]
    return list(dict.fromkeys(letras + list("ßœŒæÆøØðÐþÞıİ")))


async def _pg():
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(conn, embedding_dim=8)      # cria a extensão + `engram_fold`
    return conn


@pytest.mark.asyncio
async def test_python_and_postgres_fold_the_SAME_over_the_covered_alphabet():
    """A asserção que dá sentido a esta lib: os dois adaptadores respondem igual.

    Se um dia falhar, NÃO se ajusta o teste — ou o `unaccent` mudou de dicionário (e a
    `_TRANSLIT` tem de ser re-derivada), ou alguém tocou numa das metades sozinho."""
    conn = await _pg()
    alfabeto = _alfabeto_coberto()
    cur = await conn.execute(
        "SELECT v, engram_fold(v) FROM unnest(%s::text[]) v", (alfabeto,))
    do_banco = {linha[0]: linha[1] for linha in await cur.fetchall()}
    await conn.close()

    divergem = {c: (fold_label(c), do_banco[c])
                for c in alfabeto if fold_label(c) != do_banco[c]}
    assert not divergem, (
        f"os dois adaptadores dobram diferente em {len(divergem)} de {len(alfabeto)} "
        f"caracteres — o mesmo rótulo encontraria nós diferentes conforme o adaptador: "
        f"{ {c: f'py={p!r} pg={g!r}' for c, (p, g) in list(divergem.items())[:8]} }")


@pytest.mark.asyncio
async def test_the_TRANSLIT_table_still_matches_the_postgres_dictionary():
    """Mais apertado que o de cima: não basta o resultado FINAL bater — cada entrada da tabela
    tem de ser a que o Postgres produz, e nenhuma entrada pode estar a mais.

    Uma entrada a mais é pior que uma a menos: passa despercebida (o resultado bate por acaso) e
    depois muda o comportamento no dia em que o caractere aparece a sério num nome."""
    from cogno_engram.folding import _TRANSLIT

    conn = await _pg()
    chaves = sorted(_TRANSLIT)
    cur = await conn.execute(
        "SELECT v, engram_fold(v) FROM unnest(%s::text[]) v", (chaves,))
    do_banco = {linha[0]: linha[1] for linha in await cur.fetchall()}
    await conn.close()

    erradas = {c: (_TRANSLIT[c], do_banco[c]) for c in chaves if _TRANSLIT[c] != do_banco[c]}
    assert not erradas, f"entradas que já não batem com o `unaccent`: {erradas}"

    # e nenhuma entrada supérflua: se o NFD sozinho já lá chegava, a linha é ruído que esconde
    # uma mudança futura do dicionário
    def so_nfd(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", s.casefold())
                       if not unicodedata.combining(c))

    supérfluas = [c for c in chaves if so_nfd(c) == do_banco[c]]
    assert not supérfluas, (
        f"entradas que o NFD já resolvia sozinho — ruído na tabela: {supérfluas}")


@pytest.mark.asyncio
async def test_a_base_with_COLLIDING_nodes_fails_the_migration_by_NAME():
    """A migração numa base que já tem `José` e `Jose` como nós SEPARADOS.

    Falhar é o comportamento correcto — fundir automaticamente escolheria um dos rótulos e mudaria
    as arestas do outro de dono, em silêncio, num grafo cujo propósito é dizer factos sobre
    pessoas. Qual dos dois é a pessoa não é conhecimento que o `ensure_schema` tenha.

    O que este teste fixa é a segunda metade: o erro tem de NOMEAR os rótulos. O do Postgres diz
    `Key (scope, engram_fold(label), node_type)=(t, jose, PERSON) is duplicated` — a chave dobrada,
    que é exactamente o que o operador não precisa de saber."""
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(conn, embedding_dim=8)

    # o estado de uma base ANTIGA: dois nós que a identidade nova funde. Entram por SQL directo
    # porque o adaptador — já com a identidade nova — recusaria o segundo, que é o ponto.
    escopo = f"t/{os.urandom(4).hex()}"
    await conn.execute("DROP INDEX IF EXISTS uq_nodes_scope_fold_type")
    await conn.execute(
        "INSERT INTO knowledge_nodes (scope, label, node_type) VALUES (%s,%s,%s), (%s,%s,%s)",
        (escopo, "José", "PERSON", escopo, "Jose", "PERSON"))

    with pytest.raises(RuntimeError) as caiu:
        await ensure_schema(conn, embedding_dim=8)
    await conn.execute("DELETE FROM knowledge_nodes WHERE scope = %s", (escopo,))
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_fold_type "
        "ON knowledge_nodes (scope, engram_fold(label), node_type)")
    await conn.close()

    msg = str(caiu.value)
    assert "José" in msg and "Jose" in msg, (
        f"o erro não nomeou os rótulos em conflito — o operador fica com a chave dobrada e "
        f"sem saber que nós fundir:\n{msg}")
    assert "À MÃO" in msg, "tem de dizer que a fusão é manual, e porquê"


@pytest.mark.asyncio
async def test_the_OLD_case_only_index_is_gone_and_the_new_one_is_there():
    """A migração do índice, e é a metade que a mutação mostrou desprotegida.

    `uq_nodes_scope_label_type` era `(scope, lower(label), node_type)`. Deixá-lo de pé ao lado do
    novo não dá erro nenhum — dá uma identidade MAIS APERTADA a viver por baixo: `José` e `Jose`
    passariam a colidir na UNIQUE nova, mas o índice antigo continuaria a distingui-los e a base
    pagaria escrita por um índice que já não decide nada. Medido: apagar o `DROP` sobrevivia à
    suíte inteira.

    Corre `ensure_schema` sobre uma base que JÁ tem o índice antigo — o estado real de qualquer
    instalação existente — e duas vezes, para a idempotência."""
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(conn, embedding_dim=8)

    # recria à mão o índice que este commit aposenta: sem isto a asserção seria VÁCUA, porque
    # numa base nova ele nunca chega a existir
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_label_type "
        "ON knowledge_nodes (scope, lower(label), node_type)")
    for _ in range(2):
        await ensure_schema(conn, embedding_dim=8)

    cur = await conn.execute(
        "SELECT x.indexrelid::regclass::text FROM pg_index x "
        "  JOIN pg_class t ON t.oid = x.indrelid WHERE t.relname = 'knowledge_nodes'")
    indices = {r[0] for r in await cur.fetchall()}
    await conn.close()

    assert "uq_nodes_scope_label_type" not in indices, (
        "o índice antigo (só-caixa) sobreviveu — a base paga escrita por um índice que já não "
        "decide identidade, e a UNIQUE nova fica com uma vizinha que discorda dela")
    assert "uq_nodes_scope_fold_type" in indices, f"o índice novo não está lá: {indices}"
