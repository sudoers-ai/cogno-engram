"""O escape do LIKE, testado SEM banco — porque com banco ele não é testado.

`PostgresStore._subtree_like` constrói o padrão que decide quem está na subárvore de um scope. O
scope é OPACO por contrato, portanto pode conter `%`, `_` e `\\` — e cada um deles, sem escape,
vira curinga dentro de um `LIKE`.

**Medido: apagar o escaping deixava 179 unitários VERDES e produzia seis vazamentos reais entre
tenants** (prefixo `_` puxa todo scope de um caractere; prefixo `%` puxa tudo). Os testes que
apanhariam isso vivem na suíte de integração, que **pula sem a variável de ambiente do DSN de teste** — e uma rede que
só existe quando alguém exporta uma variável de ambiente não é rede na máquina de quem desenvolve.

Este ficheiro testa a função PURA. Não precisa de banco, corre sempre, e mata a mutação.
"""

from __future__ import annotations

import re

import pytest

from cogno_engram.adapters.postgres import PostgresStore


def _like_to_regex(pattern: str) -> "re.Pattern[str]":
    """Traduz um `LIKE ... ESCAPE '\\'` para regex, para se poder PERGUNTAR ao padrão o que ele
    casa sem levantar um Postgres. `\\x` é literal; `%` é `.*`; `_` é `.`; o resto é literal."""
    out, i = [], 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            out.append(re.escape(pattern[i + 1]))
            i += 2
        elif c == "%":
            out.append(".*")
            i += 1
        elif c == "_":
            out.append(".")
            i += 1
        else:
            out.append(re.escape(c))
            i += 1
    return re.compile("".join(out) + r"\Z")


@pytest.mark.parametrize("prefix,intruso", [
    ("t_a", "tXa/u1"),        # `_` casaria QUALQUER caractere
    ("t_a", "t-a/u1"),
    ("100%", "1000/u1"),      # `%` casaria QUALQUER sequência
    ("100%", "100pct/u1"),
    ("a%b", "aXXb/u1"),
    ("c\\d", "cXd/u1"),       # a própria barra de escape
    ("_", "Z/u1"),            # prefixo que é SÓ o curinga
    ("%", "qualquer/coisa"),  # o pior caso: puxaria TUDO
])
def test_a_metacharacter_in_the_scope_does_not_become_a_wildcard(prefix, intruso):
    padrao = _like_to_regex(PostgresStore._subtree_like(prefix))
    assert not padrao.match(intruso), (
        f"o padrão de {prefix!r} casa {intruso!r} — um scope vizinho entra na subárvore, e o "
        f"conteúdo aqui é o TRAÇO COMPLETO do turno de outro tenant"
    )


@pytest.mark.parametrize("prefix", ["t_a", "100%", "a%b", "c\\d", "_", "%", "acme"])
def test_the_real_descendants_still_match(prefix):
    """A gêmea. Escapar demais é tão defeito quanto escapar de menos — só que silencioso: o
    padrão deixa de casar os filhos legítimos e a leitura devolve vazio sem erro nenhum."""
    padrao = _like_to_regex(PostgresStore._subtree_like(prefix))
    assert padrao.match(prefix + "/u1"), f"o filho legítimo de {prefix!r} ficou de fora"
    assert padrao.match(prefix + "/a/b"), "neto legítimo ficou de fora"


def test_the_prefix_itself_is_not_matched_by_the_LIKE_branch():
    """O próprio prefixo entra pelo ramo `scope = %s`, não pelo LIKE — se o LIKE também o
    casasse, a fronteira estaria a ser feita em dois sítios e um deles poderia mudar sozinho."""
    padrao = _like_to_regex(PostgresStore._subtree_like("acme"))
    assert not padrao.match("acme")


def test_the_subtree_predicate_is_PARENTHESISED():
    """Os parênteses do `_SUBTREE` não são estilo: são o que faz um `AND` acrescentado ligar-se
    à disjunção INTEIRA.

    Sem eles, `scope = %s OR scope LIKE %s ... AND created_at >= %s` liga o `AND` só ao ramo do
    LIKE — e a linha que está NO prefixo escapa à janela do `since`. Medido: a mutação produz
    `['B_novo_filho', 'A_velho_no_prefixo']` onde o código certo devolve só o primeiro.

    É teste de FORMA, e digo-o em voz alta: o teste de comportamento vive na suíte de integração
    e **pula sem a variável de ambiente do DSN de teste**, portanto não protege a máquina de quem desenvolve. Aqui a
    forma é o mecanismo — o `AND` é concatenado a esta string, e a precedência do SQL faz o
    resto."""
    sql = PostgresStore._SUBTREE
    assert sql.startswith("(") and sql.endswith(")"), (
        "o predicado de subárvore perdeu os parênteses — um `AND` acrescentado passa a ligar-se "
        "só ao ramo do LIKE, e a linha no próprio prefixo ignora o `since`"
    )
    assert " OR " in sql, "sem a disjunção não há o que parentizar, e este teste vira decorativo"
