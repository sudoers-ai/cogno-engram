"""Fundir os nós que a identidade nova faz colidir — com o operador ao volante.

`cogno_engram/folding.py` passou a identidade de nó a ignorar caixa E acento: `José` e `Jose`
são a mesma pessoa. Numa base que já corria, isso significa que linhas hoje DISTINTAS passam a
ser a mesma — e a UNIQUE nova recusa-se a nascer enquanto elas existirem, de propósito.

**Porque é que o `ensure_schema` não funde sozinho, e porque é que isto existe na mesma.** Fundir
automaticamente escolheria um dos rótulos e mudaria as arestas do outro de dono, em silêncio, num
grafo cujo propósito é dizer factos sobre pessoas. Mas parar aí deixa o operador com um traceback
e um `psql` — e no cluster de outra pessoa isso é pior do que a fusão automática, porque o SQL que
ele vai escrever à pressa é exactamente o SQL perigoso que este módulo evita. A resposta certa não
é "não fundir", é **"fundir com o operador ao volante"**: um relatório que se lê antes, e uma
aplicação que exige `dry_run=False` escrito à mão.

**O perigo concreto, medido e não imaginado.** Em Postgres as arestas referenciam
`source_id`/`target_id` com `ON DELETE CASCADE`. Um `DELETE` do nó duplicado — o comando óbvio,
o que qualquer um escreveria — **leva as arestas dele consigo, sem aviso**. Fundir é portanto
REPONTAR primeiro e apagar depois, nunca o contrário.

**Não passa pela porta `KnowledgeGraph`, e não é preguiça.** A porta endereça nós por RÓTULO, e
depois da dobragem `delete_node(scope, "Jose")` casa com as DUAS linhas em conflito — a porta
deixou de as saber distinguir, que é precisamente a condição que este módulo existe para
resolver. Só `id` as separa, e `id` é do adaptador.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NoEmConflito:
    """Uma das linhas que a identidade nova funde."""

    id: int
    label: str
    grau: int                       # quantas arestas lhe tocam — o critério de sobrevivência
    attributes: dict = field(default_factory=dict)


@dataclass
class Colisao:
    """Um grupo `(scope, node_type)` cujos rótulos dobram para a mesma chave."""

    scope: str
    node_type: str
    dobrado: str
    nos: "list[NoEmConflito]"

    @property
    def sobrevivente(self) -> NoEmConflito:
        """O mais conectado; empate desfeito pelo `id` mais BAIXO — o mais antigo.

        Determinístico de propósito: o operador lê o relatório e vê exactamente o que a aplicação
        vai fazer. Um critério que dependesse da ordem de leitura daria relatórios que não
        correspondem à aplicação, que é a única coisa pior do que não ter relatório."""
        return max(self.nos, key=lambda n: (n.grau, -n.id))

    @property
    def perdidos(self) -> "list[NoEmConflito]":
        s = self.sobrevivente
        return [n for n in self.nos if n.id != s.id]

    def __str__(self) -> str:
        s = self.sobrevivente
        outros = ", ".join(f"{n.label!r} (id={n.id}, {n.grau} arestas)" for n in self.perdidos)
        return (f"{self.scope} / {self.node_type}: fica {s.label!r} "
                f"(id={s.id}, {s.grau} arestas) <- {outros}")


_COLISOES_SQL = """
WITH g AS (
    SELECT n.id, n.scope, n.node_type, n.label, n.attributes,
           engram_fold(n.label) AS dobrado,
           (SELECT count(*) FROM knowledge_edges e
             WHERE e.source_id = n.id OR e.target_id = n.id) AS grau
      FROM knowledge_nodes n
     WHERE (%(scope)s::text IS NULL OR n.scope = %(scope)s)
)
SELECT scope, node_type, dobrado,
       json_agg(json_build_object('id', id, 'label', label, 'grau', grau,
                                  'attributes', attributes) ORDER BY id) AS nos
  FROM g
 GROUP BY scope, node_type, dobrado
HAVING count(*) > 1
 ORDER BY scope, node_type, dobrado
"""


async def fold_collisions(conn, *, scope: "str | None" = None) -> "list[Colisao]":
    """O relatório — SÓ LEITURA, e é o que se lê antes de decidir seja o que for.

    Corre contra uma base que ainda NÃO tem a UNIQUE nova (se já a tem, não há colisões por
    construção). Requer a função `engram_fold`, que o `ensure_schema` cria antes dos índices."""
    cur = await conn.execute(_COLISOES_SQL, {"scope": scope})
    fora: "list[Colisao]" = []
    for linha in await cur.fetchall():
        escopo, tipo, dobrado, nos = (linha[0], linha[1], linha[2], linha[3]) \
            if not isinstance(linha, dict) else \
            (linha["scope"], linha["node_type"], linha["dobrado"], linha["nos"])
        fora.append(Colisao(escopo, tipo, dobrado, [
            NoEmConflito(n["id"], n["label"], n["grau"], n.get("attributes") or {})
            for n in nos]))
    return fora


@dataclass
class Fusao:
    """O que uma fusão FEZ (ou faria, em dry-run) — em factos contáveis, não em prosa."""

    colisao: Colisao
    arestas_repontadas: int = 0
    arestas_duplicadas_removidas: int = 0
    auto_arestas_removidas: int = 0
    nos_apagados: int = 0
    aliases: "list[str]" = field(default_factory=list)


async def merge_fold_collisions(conn, *, dry_run: bool = True,
                                scope: "str | None" = None) -> "list[Fusao]":
    """Funde cada colisão no seu sobrevivente. **`dry_run=True` por omissão, e é lei.**

    O default seguro aqui é o oposto do habitual: uma ferramenta que apaga nós de um grafo tem de
    exigir que alguém escreva `dry_run=False`, porque o custo de correr sem querer é perder
    ligações que ninguém sabe que existiam.

    A ordem importa e está medida: REPONTAR as arestas do perdido para o sobrevivente, remover as
    que passariam a duplicar (`UNIQUE (source_id, target_id, relation)`) ou a apontar para si
    próprias, guardar o rótulo perdido como alias, e SÓ ENTÃO apagar a linha. Apagar primeiro
    levaria as arestas atrás por `ON DELETE CASCADE` — silenciosamente.

    Em `dry_run` os números são CONTADOS pelas mesmas consultas, não estimados: o relatório diz o
    que a aplicação fará, ou não serve para decidir."""
    fusoes: "list[Fusao]" = []
    for col in await fold_collisions(conn, scope=scope):
        f = Fusao(colisao=col)
        vivo = col.sobrevivente
        for perdido in col.perdidos:
            f.aliases.append(perdido.label)
            # (a) arestas do perdido que, repontadas, colidiriam com uma que o sobrevivente já tem
            dup = await _conta_ou_apaga_duplicadas(conn, vivo.id, perdido.id, dry_run)
            f.arestas_duplicadas_removidas += dup
            # (b) arestas entre o perdido e o sobrevivente — repontadas virariam laço sobre si
            auto = await _conta_ou_apaga_auto(conn, vivo.id, perdido.id, dry_run)
            f.auto_arestas_removidas += auto
            # (c) o resto muda de dono
            f.arestas_repontadas += await _conta_ou_reponta(conn, vivo.id, perdido.id, dry_run)
            if not dry_run:
                await _guarda_alias(conn, vivo.id, perdido.label)
                await conn.execute("DELETE FROM knowledge_nodes WHERE id = %s", (perdido.id,))
            f.nos_apagados += 1
        fusoes.append(f)
    return fusoes


async def _conta_ou_apaga_duplicadas(conn, vivo: int, perdido: int, dry_run: bool) -> int:
    """Arestas do perdido cuja versão repontada JÁ existe no sobrevivente.

    Sem isto o `UPDATE` rebenta na `UNIQUE (source_id, target_id, relation)` — e rebentar a meio
    de uma fusão deixa o grafo em estado misto, que é pior do que não ter começado."""
    sql = """
        SELECT {alvo} FROM knowledge_edges e
         WHERE (e.source_id = %(perdido)s OR e.target_id = %(perdido)s)
           AND EXISTS (
               SELECT 1 FROM knowledge_edges o
                WHERE o.relation = e.relation
                  AND o.source_id = CASE WHEN e.source_id = %(perdido)s
                                         THEN %(vivo)s ELSE e.source_id END
                  AND o.target_id = CASE WHEN e.target_id = %(perdido)s
                                         THEN %(vivo)s ELSE e.target_id END
                  AND o.id <> e.id)
    """
    p = {"perdido": perdido, "vivo": vivo}
    if dry_run:
        cur = await conn.execute(sql.format(alvo="count(*)"), p)
        linha = await cur.fetchone()
        return int(linha[0] if not isinstance(linha, dict) else list(linha.values())[0])
    cur = await conn.execute(
        f"DELETE FROM knowledge_edges WHERE id IN ({sql.format(alvo='e.id')}) RETURNING id", p)
    return len(await cur.fetchall())


async def _conta_ou_apaga_auto(conn, vivo: int, perdido: int, dry_run: bool) -> int:
    """Arestas ENTRE os dois nós: repontadas, ficariam a apontar para si próprias.

    Um `José --CASADO_COM--> Jose` é artefacto da duplicação, não facto sobre a pessoa."""
    onde = ("(source_id = %(perdido)s AND target_id = %(vivo)s) OR "
            "(source_id = %(vivo)s AND target_id = %(perdido)s)")
    p = {"perdido": perdido, "vivo": vivo}
    if dry_run:
        cur = await conn.execute(f"SELECT count(*) FROM knowledge_edges WHERE {onde}", p)
        linha = await cur.fetchone()
        return int(linha[0] if not isinstance(linha, dict) else list(linha.values())[0])
    cur = await conn.execute(
        f"DELETE FROM knowledge_edges WHERE {onde} RETURNING id", p)
    return len(await cur.fetchall())


async def _conta_ou_reponta(conn, vivo: int, perdido: int, dry_run: bool) -> int:
    """O resto das arestas do perdido muda de dono."""
    p = {"perdido": perdido, "vivo": vivo}
    if dry_run:
        cur = await conn.execute(
            "SELECT count(*) FROM knowledge_edges "
            " WHERE source_id = %(perdido)s OR target_id = %(perdido)s", p)
        linha = await cur.fetchone()
        return int(linha[0] if not isinstance(linha, dict) else list(linha.values())[0])
    cur = await conn.execute(
        "UPDATE knowledge_edges "
        "   SET source_id = CASE WHEN source_id = %(perdido)s THEN %(vivo)s ELSE source_id END, "
        "       target_id = CASE WHEN target_id = %(perdido)s THEN %(vivo)s ELSE target_id END "
        " WHERE source_id = %(perdido)s OR target_id = %(perdido)s RETURNING id", p)
    return len(await cur.fetchall())


async def _guarda_alias(conn, vivo: int, perdido: str) -> None:
    """O rótulo que desaparece fica no sobrevivente, em `attributes.aliases`.

    Perder a grafia é perder informação: `Vinícius Vale` e `Vinicius Vale` são a mesma pessoa, mas
    saber que ela já se escreveu das duas maneiras é um facto sobre ela — e é o que permite
    desfazer a fusão à mão se alguém decidir que foi errada."""
    await conn.execute(
        "UPDATE knowledge_nodes "
        "   SET attributes = jsonb_set(coalesce(attributes,'{}'::jsonb), '{aliases}', "
        "       (coalesce(attributes->'aliases','[]'::jsonb) || to_jsonb(%s::text)), true) "
        " WHERE id = %s AND NOT coalesce(attributes->'aliases','[]'::jsonb) @> to_jsonb(%s::text)",
        (perdido, vivo, perdido))
