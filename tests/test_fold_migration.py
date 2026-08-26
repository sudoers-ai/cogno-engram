"""A ferramenta de fusão — e o `ON DELETE CASCADE` que ela existe para não disparar.

Em Postgres as arestas referenciam `source_id`/`target_id` com `ON DELETE CASCADE`. O comando
óbvio para resolver uma colisão — apagar o nó duplicado — **leva as arestas dele consigo, sem
aviso**. É por isso que a fusão reponta primeiro e apaga depois, e é por isso que o primeiro teste
deste ficheiro conta arestas antes e depois.
"""

from __future__ import annotations

import os

import pytest

psycopg = pytest.importorskip("psycopg")

from cogno_engram.adapters.postgres import ensure_schema                       # noqa: E402
from cogno_engram.fold_migration import (                                      # noqa: E402
    fold_collisions,
    merge_fold_collisions,
)

DSN = os.environ.get("ENGRAM_TEST_DSN", "")


async def _seed_colliding_base(*, extra_edges=()):
    """Uma base no estado ANTES da migração: `José` e `Jose` como nós separados, com arestas.

    Entram por SQL directo porque o adaptador — já com a identidade nova — recusaria o segundo,
    que é precisamente a condição que esta ferramenta existe para desfazer.

    QUEM CHAMA TEM DE LIMPAR — use o `base` (fixture) em vez desta função directamente. A versão
    sem limpeza deixava a base com nós em colisão E sem a UNIQUE, e o ficheiro seguinte por ordem
    alfabética corria `ensure_schema` contra esse estado e rebentava: 4 vermelhos que não eram
    dele. Estado partilhado entre ficheiros de teste é a forma de defeito mais barata de criar e a
    mais cara de diagnosticar."""
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    for t in ("knowledge_edges", "knowledge_nodes", "turn_traces", "memories", "turns",
              "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {t} CASCADE")
    await ensure_schema(conn, embedding_dim=8)
    await conn.execute("DROP INDEX IF EXISTS uq_nodes_scope_fold_type")

    ids = {}
    for label in ("José", "Jose", "Rex", "Maria"):
        cur = await conn.execute(
            "INSERT INTO knowledge_nodes (scope, label, node_type) VALUES (%s,%s,%s) "
            "RETURNING id", ("t", label, "PERSON"))
        ids[label] = (await cur.fetchone())[0]
    # `José` tem 2 arestas, `Jose` tem 1 — o sobrevivente é o mais conectado
    pattern = [("José", "Rex", "OWNS_PET"), ("José", "Maria", "MARRIED_TO"),
              ("Jose", "Maria", "WORKS_WITH")]
    for s, t, r in list(pattern) + list(extra_edges):
        await conn.execute(
            "INSERT INTO knowledge_edges (scope, source_id, target_id, relation) "
            "VALUES (%s,%s,%s,%s) ON CONFLICT DO NOTHING", ("t", ids[s], ids[t], r))
    return conn, ids


@pytest.fixture(autouse=True)
async def _restore_base():
    """Depois de CADA teste, a base volta ao estado migrado — corra ele verde ou vermelho.

    Sem isto, este ficheiro deixava nós em colisão e a base SEM a UNIQUE, e o ficheiro seguinte
    por ordem alfabética (`test_folding_parity.py`) corria `ensure_schema` contra esse estado e
    rebentava: 4 vermelhos que não eram dele. Estado partilhado entre ficheiros de teste é a forma
    de defeito mais barata de criar e a mais cara de diagnosticar — o vermelho aparece longe da
    causa e no ficheiro de outra pessoa.

    `autouse` e não fábrica: uma fixture que devolve uma função para o teste chamar convida a ser
    chamada errado (foi, e o pytest recusou-a). Esta não devolve nada, portanto não há forma de a
    usar mal."""
    yield
    if not DSN:
        return
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    try:
        await conn.execute("DELETE FROM knowledge_nodes WHERE scope = %s", ("t",))
        await conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_nodes_scope_fold_type "
            "ON knowledge_nodes (scope, engram_fold(label), node_type)")
    except Exception:                                # noqa: BLE001 — a limpeza nunca é a falha
        pass
    finally:
        await conn.close()


async def _edges(conn) -> set:
    cur = await conn.execute(
        "SELECT sn.label, e.relation, tn.label FROM knowledge_edges e "
        "  JOIN knowledge_nodes sn ON sn.id = e.source_id "
        "  JOIN knowledge_nodes tn ON tn.id = e.target_id")
    return {tuple(r) for r in await cur.fetchall()}


@pytest.mark.asyncio
async def test_the_naive_DELETE_would_lose_edges_and_the_merge_does_not():
    """O teste que justifica a ferramenta inteira, e mede os dois caminhos lado a lado."""
    conn, ids = await _seed_colliding_base()
    before = await _edges(conn)
    assert len(before) == 3

    # (1) o comando óbvio: apagar o duplicado
    await conn.execute("BEGIN")
    await conn.execute("DELETE FROM knowledge_nodes WHERE id = %s", (ids["Jose"],))
    depois_do_delete = await _edges(conn)
    await conn.execute("ROLLBACK")
    assert len(depois_do_delete) == 2, (
        "o DELETE ingénuo devia ter levado a aresta de `Jose` por CASCADE — se não levou, o "
        "esquema mudou e este teste deixou de medir o perigo que descreve")

    # (2) a ferramenta
    merges = await merge_fold_collisions(conn, dry_run=False)
    after = await _edges(conn)
    await conn.close()

    assert len(merges) == 1 and merges[0].nodes_deleted == 1
    assert len(after) == 3, f"a fusão perdeu arestas: {before - after}"
    assert ("José", "WORKS_WITH", "Maria") in after, (
        "a aresta que era de `Jose` tinha de passar para o sobrevivente, não desaparecer")


@pytest.mark.asyncio
async def test_dry_run_is_the_DEFAULT_and_changes_nothing():
    """Uma ferramenta que apaga nós de um grafo tem de exigir que alguém escreva `dry_run=False`."""
    conn, _ = await _seed_colliding_base()
    antes_n = (await (await conn.execute("SELECT count(*) FROM knowledge_nodes")).fetchone())[0]
    antes_e = await _edges(conn)

    merges = await merge_fold_collisions(conn)                  # sem dry_run explícito
    depois_n = (await (await conn.execute("SELECT count(*) FROM knowledge_nodes")).fetchone())[0]
    depois_e = await _edges(conn)
    await conn.close()

    assert (antes_n, antes_e) == (depois_n, depois_e), "o default mexeu na base"
    assert merges and merges[0].nodes_deleted == 1, "mas tem de RELATAR o que faria"


@pytest.mark.asyncio
async def test_the_dry_run_numbers_match_what_apply_actually_does():
    """Um relatório que não corresponde à aplicação é pior do que não haver relatório: o operador
    decide sobre números e a aplicação faz outra coisa."""
    conn, _ = await _seed_colliding_base()
    planned = (await merge_fold_collisions(conn, dry_run=True))[0]
    applied = (await merge_fold_collisions(conn, dry_run=False))[0]
    await conn.close()

    assert (planned.edges_repointed, planned.duplicate_edges_removed,
            planned.self_loops_removed, planned.nodes_deleted) ==\
           (applied.edges_repointed, applied.duplicate_edges_removed,
            applied.self_loops_removed, applied.nodes_deleted)


@pytest.mark.asyncio
async def test_an_edge_that_would_DUPLICATE_is_dropped_not_crashed_on():
    """Se os dois nós têm a MESMA relação para o mesmo alvo, repontar violaria a UNIQUE — e
    rebentar a meio de uma fusão deixa o grafo em estado misto, pior do que não ter começado."""
    conn, _ = await _seed_colliding_base(extra_edges=[("Jose", "Rex", "OWNS_PET")])
    merge = (await merge_fold_collisions(conn, dry_run=False))[0]
    edges = await _edges(conn)
    await conn.close()

    assert merge.duplicate_edges_removed == 1
    assert ("José", "OWNS_PET", "Rex") in edges
    assert len([a for a in edges if a[1] == "OWNS_PET"]) == 1, "ficou duplicada"


@pytest.mark.asyncio
async def test_an_edge_BETWEEN_the_two_would_become_a_self_loop_and_is_dropped():
    """`José --CASADO_COM--> Jose` é artefacto da duplicação, não facto sobre a pessoa."""
    conn, _ = await _seed_colliding_base(extra_edges=[("José", "Jose", "SAME_AS")])
    merge = (await merge_fold_collisions(conn, dry_run=False))[0]
    edges = await _edges(conn)
    await conn.close()

    assert merge.self_loops_removed == 1
    assert not [a for a in edges if a[0] == a[2]], f"ficou um laço: {edges}"


@pytest.mark.asyncio
async def test_the_lost_label_survives_as_an_ALIAS():
    """Perder a grafia é perder informação: saber que a pessoa já se escreveu das duas maneiras é
    um facto sobre ela — e é o que permite desfazer a fusão à mão."""
    conn, _ = await _seed_colliding_base()
    await merge_fold_collisions(conn, dry_run=False)
    cur = await conn.execute(
        "SELECT attributes->'aliases' FROM knowledge_nodes WHERE label = %s", ("José",))
    aliases = (await cur.fetchone())[0]
    await conn.close()
    assert aliases == ["Jose"], f"o rótulo perdido não ficou guardado: {aliases}"


@pytest.mark.asyncio
async def test_after_the_merge_the_UNIQUE_index_can_finally_be_created():
    """O objectivo de tudo isto: a migração que falhava passa a correr."""
    conn, _ = await _seed_colliding_base()
    await merge_fold_collisions(conn, dry_run=False)
    await ensure_schema(conn, embedding_dim=8)                 # não levanta
    cur = await conn.execute(
        "SELECT count(*) FROM pg_index x JOIN pg_class i ON i.oid = x.indexrelid "
        " WHERE i.relname = 'uq_nodes_scope_fold_type'")
    existe = (await cur.fetchone())[0]
    await conn.close()
    assert existe == 1


@pytest.mark.asyncio
async def test_the_report_names_the_survivor_and_the_losers():
    """O operador decide sobre isto — tem de conseguir lê-lo."""
    conn, _ = await _seed_colliding_base()
    collisions = await fold_collisions(conn)
    await conn.close()

    assert len(collisions) == 1
    text = str(collisions[0])
    assert "José" in text and "Jose" in text and "arestas" in text, text
    assert collisions[0].survivor.label == "José", "o mais conectado tinha de sobreviver"
