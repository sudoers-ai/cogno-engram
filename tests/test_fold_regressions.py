"""As duas regressões que a dobragem introduziu — e que a revisão adversarial encontrou.

Ambas são consequências de "a identidade passa a ignorar acento" que ninguém pediu e que ninguém
teria visto até doer: uma perde o acento do nome de uma pessoa, a outra apaga um nó a mais.
"""

from __future__ import annotations

import os

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import AUDIENCE_STAFF, GraphNode

psycopg = pytest.importorskip("psycopg")

from cogno_engram.adapters.postgres import (          # noqa: E402
    PostgresKnowledgeGraph,
    ensure_schema,
)

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path

DSN = resolve_test_dsn()      # ENGRAM_TEST_DSN, else `engram_test` on the local server


async def _pg_graph():
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(conn, embedding_dim=8)
    await conn.close()
    return PostgresKnowledgeGraph(dsn=DSN)


async def _both_adapters():
    """Os DOIS, para o mesmo teste — é a discordância entre eles que estas regressões produzem."""
    return [("in-memory", InMemoryGraph()), ("postgres", await _pg_graph())]


@pytest.mark.asyncio
async def test_the_ACCENTED_spelling_wins_and_never_loses_it_again():
    """O rótulo original é o que se guarda e se mostra — é a promessa do `folding`.

    Medido antes do conserto: `Jose` chega primeiro, `José` depois, e a linha fica `Jose` PARA
    SEMPRE. E como o próprio PR argumenta que "o contacto escreve o nome sem acento metade das
    vezes", a grafia sem acento é a que chega primeiro com mais frequência — portanto o defeito
    não é raro, é o caso comum. Antes desta funcionalidade eram duas linhas e a acentuada existia.

    Sobe e não desce: uma vez acentuado, um `Jose` posterior não o rebaixa. Determinístico, senão
    o nome no painel oscilava a cada turno."""
    for which, kg in await _both_adapters():
        scope_ = f"acc/{os.urandom(4).hex()}"
        await kg.upsert_node(GraphNode(scope_, "Jose", "PERSON"))       # sem acento primeiro
        await kg.upsert_node(GraphNode(scope_, "José", "PERSON"))       # com acento depois
        found = await kg.find_node(scope_, "jose", audience=AUDIENCE_STAFF)
        assert found is not None and found.label == "José", (
            f"[{which}] a grafia acentuada não subiu: ficou {found.label if found else None!r}")

        await kg.upsert_node(GraphNode(scope_, "JOSE", "PERSON"))       # e não volta a descer
        found = await kg.find_node(scope_, "jose", audience=AUDIENCE_STAFF)
        assert found.label == "José", f"[{which}] o acento foi perdido por um upsert posterior"


@pytest.mark.asyncio
async def test_deleting_one_node_does_not_delete_its_fold_TWIN():
    """`José` como PERSON e `Jose` como CONCEPT coexistem legalmente depois da migração — a
    UNIQUE inclui `node_type`. Mas `delete_node` recebe só um RÓTULO.

    Medido antes do conserto: `delete_node('José')` apagava os DOIS, e as arestas de ambos iam
    atrás por `ON DELETE CASCADE`. O `cogno-ui` chama isto com um id que o host converte em
    rótulo, portanto o operador clicava num nó e perdia outro, sem aviso."""
    kg = await _pg_graph()
    scope_ = f"del/{os.urandom(4).hex()}"
    await kg.upsert_node(GraphNode(scope_, "José", "PERSON"))
    await kg.upsert_node(GraphNode(scope_, "Jose", "CONCEPT"))

    deleted = await kg.delete_node(scope_, "José")
    left = await kg.list_nodes(scope_, audience=AUDIENCE_STAFF)

    assert deleted is True, "o rótulo EXACTO existe, portanto tinha de apagar esse"
    assert [n.node_type for n in left] == ["CONCEPT"], (
        f"apagar o PERSON levou o CONCEPT atrás: sobrou {[(n.label, n.node_type) for n in left]}")


@pytest.mark.asyncio
async def test_an_AMBIGUOUS_delete_refuses_instead_of_guessing():
    """Sem rótulo exacto e com vários candidatos, recusa. Escolher por quem chamou é escolher
    errado metade das vezes, em silêncio — e isto apaga arestas em cascata."""
    kg = await _pg_graph()
    scope_ = f"amb/{os.urandom(4).hex()}"
    await kg.upsert_node(GraphNode(scope_, "José", "PERSON"))
    await kg.upsert_node(GraphNode(scope_, "José", "CONCEPT"))

    deleted = await kg.delete_node(scope_, "Jose")        # não bate exacto com nenhum
    left = await kg.list_nodes(scope_, audience=AUDIENCE_STAFF)

    assert deleted is False, "devia ter recusado em vez de escolher"
    assert len(left) == 2, f"recusou e apagou na mesma: {left}"


@pytest.mark.asyncio
async def test_set_edge_audience_does_not_reclassify_a_twins_edge():
    """A mesma família, mas o que muda aqui é uma decisão de PRIVACIDADE — quem pode ler a
    aresta. Alargá-la para um nó que ninguém escolheu é expor (ou esconder) dados por engano."""
    from cogno_engram.types import GraphEdge

    kg = await _pg_graph()
    scope_ = f"aud/{os.urandom(4).hex()}"
    await kg.upsert_node(GraphNode(scope_, "José", "PERSON"))
    await kg.upsert_node(GraphNode(scope_, "Jose", "CONCEPT"))
    await kg.upsert_node(GraphNode(scope_, "Rex", "PET"))
    await kg.upsert_edge(GraphEdge(scope_, "José", "Rex", "OWNS", audience="tenant"))
    await kg.upsert_edge(GraphEdge(scope_, "Jose", "Rex", "OWNS", audience="tenant"))

    await kg.set_edge_audience(scope_, "José", "Rex", "OWNS", "identity:x")

    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    cur = await conn.execute(
        "SELECT sn.node_type, e.audience FROM knowledge_edges e "
        "  JOIN knowledge_nodes sn ON sn.id = e.source_id WHERE e.scope = %s", (scope_,))
    by_kind = {r[0]: r[1] for r in await cur.fetchall()}
    await conn.close()

    assert by_kind.get("PERSON") == "identity:x", "a aresta escolhida não mudou"
    assert by_kind.get("CONCEPT") == "tenant", (
        f"a aresta do GÉMEO mudou de audiência sem ninguém a escolher: {by_kind}")
