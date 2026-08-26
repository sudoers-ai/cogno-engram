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
def _covered_alphabet() -> "list[str]":
    letters = [chr(c) for c in range(0xC0, 0x180)
              if unicodedata.category(chr(c)).startswith("L")]
    return list(dict.fromkeys(letters + list("ßœŒæÆøØðÐþÞıİ")))


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
    alphabet = _covered_alphabet()
    cur = await conn.execute(
        "SELECT v, engram_fold(v) FROM unnest(%s::text[]) v", (alphabet,))
    from_db = {row[0]: row[1] for row in await cur.fetchall()}
    await conn.close()

    diverge = {c: (fold_label(c), from_db[c])
                for c in alphabet if fold_label(c) != from_db[c]}
    assert not diverge, (
        f"os dois adaptadores dobram diferente em {len(diverge)} de {len(alphabet)} "
        f"caracteres — o mesmo rótulo encontraria nós diferentes conforme o adaptador: "
        f"{ {c: f'py={p!r} pg={g!r}' for c, (p, g) in list(diverge.items())[:8]} }")


@pytest.mark.asyncio
async def test_the_TRANSLIT_table_still_matches_the_postgres_dictionary():
    """Mais apertado que o de cima: não basta o resultado FINAL bater — cada entrada da tabela
    tem de ser a que o Postgres produz, e nenhuma entrada pode estar a mais.

    Uma entrada a mais é pior que uma a menos: passa despercebida (o resultado bate por acaso) e
    depois muda o comportamento no dia em que o caractere aparece a sério num nome."""
    from cogno_engram.folding import _TRANSLIT

    conn = await _pg()
    keys = sorted(_TRANSLIT)
    cur = await conn.execute(
        "SELECT v, engram_fold(v) FROM unnest(%s::text[]) v", (keys,))
    from_db = {row[0]: row[1] for row in await cur.fetchall()}
    await conn.close()

    wrong = {c: (_TRANSLIT[c], from_db[c]) for c in keys if _TRANSLIT[c] != from_db[c]}
    assert not wrong, f"entradas que já não batem com o `unaccent`: {wrong}"

    # e nenhuma entrada supérflua: se o NFD sozinho já lá chegava, a linha é ruído que esconde
    # uma mudança futura do dicionário
    def _nfd_only(s: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFD", s.casefold())
                       if not unicodedata.combining(c))

    superfluous = [c for c in keys if _nfd_only(c) == from_db[c]]
    assert not superfluous, (
        f"entradas que o NFD já resolvia sozinho — ruído na tabela: {superfluous}")


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
    scope_ = f"t/{os.urandom(4).hex()}"
    await conn.execute("DROP INDEX IF EXISTS uq_nodes_scope_fold_type")
    await conn.execute(
        "INSERT INTO knowledge_nodes (scope, label, node_type) VALUES (%s,%s,%s), (%s,%s,%s)",
        (scope_, "José", "PERSON", scope_, "Jose", "PERSON"))

    with pytest.raises(RuntimeError) as raised:
        await ensure_schema(conn, embedding_dim=8)
    await conn.execute("DELETE FROM knowledge_nodes WHERE scope = %s", (scope_,))
    await conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_fold_type "
        "ON knowledge_nodes (scope, engram_fold(label), node_type)")
    await conn.close()

    msg = str(raised.value)
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


@pytest.mark.asyncio
async def test_the_two_sides_agree_on_whole_LABELS_not_just_characters():
    """A paridade por caractere não chega, e a lacuna é concreta.

    Os testes acima comparam caractere a caractere. Um defeito que só apareça na COMPOSIÇÃO —
    normalização em ordem diferente, um `strip` a mais de um lado, uma sequência de dois
    combinantes — sobrevive a todos eles e parte um nome inteiro (`Łódź`, `São Gonçalo do Amarante`)
    sem tocar em nenhum caractere isolado.

    Os rótulos abaixo são o domínio a sério: nomes pt-BR/es com acento, o nome do próprio dono nas
    duas grafias em que a base viva o tem, e as formas Unicode compostas E decompostas do mesmo
    nome — que é o caso que um teste por caractere nunca constrói."""
    conn = await _pg()
    labels = [
        "José", "Jose", "JOSÉ", "josé",
        "Vinicius Vale", "Vinícius Vale",          # o par que existe na base viva
        "Hernani", "Hernaní",
        "São Gonçalo do Amarante", "Sao Goncalo do Amarante",
        "Łódź", "Añez", "Müller", "D'Ávila", "Conceição",
        "André Castro", "Clínica veterinária", "Funcionário",
        unicodedata.normalize("NFC", "José"),      # composta
        unicodedata.normalize("NFD", "José"),      # decomposta — mesmo nome, bytes diferentes
        "  José  ", "JOSÉ MARIA da SILVA",
    ]
    cur = await conn.execute(
        "SELECT v, engram_fold(v) FROM unnest(%s::text[]) v", (labels,))
    from_db = {row[0]: row[1] for row in await cur.fetchall()}
    await conn.close()

    diverge = {r: (fold_label(r), from_db[r]) for r in labels if fold_label(r) != from_db[r]}
    assert not diverge, (
        f"os dois adaptadores dobram RÓTULOS diferente — o mesmo nome encontraria nós diferentes "
        f"conforme o adaptador: { {r: f'py={p!r} pg={g!r}' for r, (p, g) in diverge.items()} }")

    # e o que o produto PROMETE: as formas do mesmo nome caem todas na mesma chave
    for grupo in (["José", "Jose", "JOSÉ", "josé",
                   unicodedata.normalize("NFD", "José")],
                  ["Vinicius Vale", "Vinícius Vale"],
                  ["São Gonçalo do Amarante", "Sao Goncalo do Amarante"]):
        keys = {fold_label(r) for r in grupo}
        assert len(keys) == 1, f"{grupo} devia ser uma só pessoa/lugar, deu {keys}"


@pytest.mark.asyncio
async def test_a_CHANGED_fold_rule_rebuilds_the_index_that_stores_it(monkeypatch):
    """O defeito original a voltar através do TEMPO — e o que nenhuma base nova apanha.

    `uq_nodes_scope_fold_type` é um índice de EXPRESSÃO: guarda o RESULTADO de `engram_fold`, não
    o rótulo. No dia em que a tabela de transliteração crescer — e o módulo já prevê que cresça —
    o `CREATE OR REPLACE FUNCTION` do deploy troca a função e o Postgres **não reconstrói o índice
    nem avisa**. As chaves velhas ficam lá e `find_node` deixa de achar um nó que existe.

    O cenário é encenado como acontece: instalação com a regra v1, índice construído com v1, e
    depois um deploy que instala v2 — trocando o `FOLD_FUNCTION_SQL` que o `ensure_schema` usa,
    que é literalmente o que um `_TRANSLIT` novo faria.

    **A primeira versão deste teste não discriminava**: mudava a função à mão e deixava o
    `ensure_schema` repor a original, portanto o índice — construído com a original — voltava a
    bater por acidente e apagar a reconstrução sobrevivia. Encenar a v2 e DEIXÁ-LA é o que torna
    a asserção capaz de falhar."""
    from cogno_engram.adapters import postgres as pgmod

    conn = await _pg()
    scope_ = f"drift/{os.urandom(4).hex()}"
    await conn.execute(
        "INSERT INTO knowledge_nodes (scope, label, node_type) VALUES (%s,%s,%s)",
        (scope_, "Ørsted", "PERSON"))

    async def _found_via_index(key: str) -> int:
        await conn.execute("SET enable_seqscan = off")
        cur = await conn.execute(
            "SELECT count(*) FROM knowledge_nodes "
            " WHERE scope = %s AND engram_fold(label) = %s", (scope_, key))
        n = int((await cur.fetchone())[0])
        await conn.execute("SET enable_seqscan = on")
        return n

    assert await _found_via_index("orsted") == 1, "o nó tinha de ser achável com a regra v1"

    # o deploy da v2: `Ø` passa a dobrar para `oe`, como faria uma entrada nova em `_TRANSLIT`
    v2 = ('CREATE OR REPLACE FUNCTION engram_fold(text) RETURNS text '
          '  LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS '
          '$$ SELECT replace(public.unaccent(\'public.unaccent\', '
          '                  lower($1 COLLATE "und-x-icu")), \'o\', \'oe\') $$')
    monkeypatch.setattr(pgmod, "FOLD_FUNCTION_SQL", v2)
    await ensure_schema(conn, embedding_dim=8)

    achou = await _found_via_index("oersted")
    await conn.execute("DELETE FROM knowledge_nodes WHERE scope = %s", (scope_,))
    await conn.execute("REINDEX INDEX uq_nodes_scope_fold_type")
    await conn.close()

    assert achou == 1, (
        "o índice ficou com as chaves da regra ANTIGA: a dobragem mudou, o Postgres não "
        "reconstruiu nem avisou, e uma busca pelo valor novo não acha um nó que existe — o "
        "defeito original, através do tempo")


# NÃO há aqui um teste a fixar que uma falha NÃO-colisão escapa sem ser rotulada como colisão, e
# a ausência é medida, não esquecida. Tentei encenar três falhas do `CREATE UNIQUE INDEX` que não
# fossem `UniqueViolation` e as três são inalcançáveis por construção:
#
#   * função com volatilidade errada (`STABLE`) — o `ensure_schema` reinstala a função certa
#     ANTES de chegar ao índice, portanto a via repara-se sozinha;
#   * nome já ocupado por uma tabela — `CREATE INDEX IF NOT EXISTS` aceita-o em silêncio (NOTICE,
#     sem erro), medido;
#   * ICU em falta — falha na criação da FUNÇÃO, um passo acima, e nunca chega ao `except`.
#
# Consequência assumida: apagar o `psycopg.errors.` do `except` (voltando a `Exception`) sobrevive
# à suíte. O estreitamento continua certo — o erro de colisão manda o operador FUNDIR NÓS, que é
# destrutivo, e rotular qualquer falha assim mandava-o destruir dados para resolver outra coisa —
# mas quem o remover não é apanhado aqui. Registado em vez de disfarçado.


@pytest.mark.asyncio
@pytest.mark.parametrize("err,labelled_as_collision", [
    ("UniqueViolation", True),
    ("InsufficientPrivilege", False),
    ("LockNotAvailable", False),
])
async def test_only_a_UNIQUE_violation_is_reported_as_a_label_collision(err, labelled_as_collision):
    """O erro de colisão manda o operador FUNDIR NÓS — destrutivo. Rotular qualquer falha assim
    manda-o destruir dados para resolver outra coisa.

    **Eu tinha declarado esta propriedade como não-testável e estava ERRADO.** Escrevi que as
    falhas não-`UniqueViolation` do `CREATE INDEX` eram inalcançáveis, depois de tentar três vias
    que de facto não chegam lá (função com volatilidade errada — o `ensure_schema` repara antes;
    nome ocupado — `IF NOT EXISTS` engole; ICU em falta — falha um passo acima). A revisão
    encontrou a quarta: privilégios. Uma falta de privilégio no próprio `CREATE INDEX` propaga.
    Registar a lacuna foi certo; concluir que era intransponível foi cedo demais.

    O erro é INJECTADO na declaração exacta, através de um espião que delega tudo o resto — assim
    o que corre é o `except` verdadeiro do `ensure_schema`, não uma imitação dele."""
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    import contextlib

    cls = getattr(psycopg.errors, err)

    class _Spy:
        """Delega tudo; levanta na declaração do índice do fold."""

        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        async def execute(self, sql, params=None, *a, **kw):
            if isinstance(sql, str) and "uq_nodes_scope_fold_type" in sql and "CREATE" in sql:
                raise cls(f"injectado: {err}")
            return await self._conn.execute(sql, params, *a, **kw)

    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(conn, embedding_dim=8)
    scope_ = f"inj/{os.urandom(4).hex()}"
    await conn.execute("DROP INDEX IF EXISTS uq_nodes_scope_fold_type")
    await conn.execute(
        "INSERT INTO knowledge_nodes (scope, label, node_type) VALUES (%s,%s,%s), (%s,%s,%s)",
        (scope_, "José", "PERSON", scope_, "Jose", "PERSON"))

    with pytest.raises(Exception) as raised:            # noqa: B017 — a MENSAGEM é o teste
        await ensure_schema(_Spy(conn), embedding_dim=8)

    await conn.execute("DELETE FROM knowledge_nodes WHERE scope = %s", (scope_,))
    with contextlib.suppress(Exception):
        await ensure_schema(conn, embedding_dim=8)
    await conn.close()

    msg = str(raised.value)
    if labelled_as_collision:
        assert "ignorar acentos" in msg and "José" in msg, (
            f"uma violação de unicidade É colisão e tinha de nomear os rótulos:\n{msg}")
    else:
        assert "ignorar acentos" not in msg and "À MÃO" not in msg, (
            f"{err} não é colisão — o operador seria mandado FUNDIR NÓS, que é destrutivo, "
            f"para resolver outra coisa:\n{msg}")
        assert err.lower()[:8] in msg.lower() or "injectado" in msg, (
            f"o erro real tem de sobreviver, não ser substituído:\n{msg}")


@pytest.mark.asyncio
async def test_the_diagnostic_names_the_labels_on_a_NON_autocommit_connection_too():
    """`ensure_schema` é API pública e recebe as duas espécies de conexão.

    Sem autocommit, o `CREATE INDEX` que falha ENVENENA a transacção: toda consulta seguinte dá
    `InFailedSqlTransaction`, o diagnóstico degradava para lista vazia, e o operador recebia
    exactamente a chave dobrada (`Key (scope, engram_fold(label), node_type)=(t, jose, PERSON)`)
    que este módulo diz que ele não precisa. A mensagem era honesta — dizia "não consegui
    listá-los" — e inútil.

    Também com `dict_row`, que é o que o adaptador usa: as duas condições juntas são as do caminho
    real."""
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    from psycopg.rows import dict_row

    prep = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(prep, embedding_dim=8)
    scope_ = f"tx/{os.urandom(4).hex()}"
    await prep.execute("DROP INDEX IF EXISTS uq_nodes_scope_fold_type")
    await prep.execute(
        "INSERT INTO knowledge_nodes (scope, label, node_type) VALUES (%s,%s,%s), (%s,%s,%s)",
        (scope_, "José", "PERSON", scope_, "Jose", "PERSON"))
    await prep.close()

    conn = await psycopg.AsyncConnection.connect(DSN, row_factory=dict_row)   # SEM autocommit
    with pytest.raises(RuntimeError) as raised:
        await ensure_schema(conn, embedding_dim=8)
    await conn.rollback()
    await conn.close()

    cleanup = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await cleanup.execute("DELETE FROM knowledge_nodes WHERE scope = %s", (scope_,))
    await ensure_schema(cleanup, embedding_dim=8)
    await cleanup.close()

    msg = str(raised.value)
    assert "José" in msg and "Jose" in msg, (
        f"sob transacção envenenada o diagnóstico perdeu os rótulos e o operador fica com a "
        f"chave dobrada:\n{msg}")
    assert "Não consegui listá-los" not in msg, "degradou quando podia ter respondido"
