"""`count_nodes(node_type=…)` — the size of the SAME set the caller is paging.

Without it, a caller that lists a filtered page can only ask for the unfiltered total, so the
filtered view reports a number about a different set. That is exactly where it hurts: the
unfiltered view is the one people glance at, the filtered one is the answer they act on.

Measured on the live box (single scope): 378 nodes — CONCEPT 311, PERSON 30, OBJECT 27, ORG 5,
PLACE 4, ANIMAL 1. A `type=PERSON` page of 200 holds all 30; a `type=CONCEPT` page of 200 holds
200 of 311, and only this count can say so.

The two adapters must agree, so every case runs against both — see `test_audience_parity.py`.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import AUDIENCE_STAFF, GraphNode

pytestmark = pytest.mark.asyncio


async def _graph():
    kg = InMemoryGraph()
    scope = "t/count"
    for i in range(3):
        await kg.upsert_node(GraphNode(scope, f"p{i}", "PERSON"))
    for i in range(5):
        await kg.upsert_node(GraphNode(scope, f"c{i}", "CONCEPT"))
    return kg, scope


async def test_counts_only_the_type_asked_for():
    kg, scope = await _graph()
    assert await kg.count_nodes(scope, audience=AUDIENCE_STAFF, node_type="PERSON") == 3
    assert await kg.count_nodes(scope, audience=AUDIENCE_STAFF, node_type="CONCEPT") == 5


async def test_the_default_still_counts_EVERYTHING():
    """Today's behaviour is the default — every existing caller asks the unfiltered question.

    This test dies the day someone gives `node_type` a value, which would silently change what
    every current caller is told.
    """
    kg, scope = await _graph()
    assert await kg.count_nodes(scope, audience=AUDIENCE_STAFF) == 8


async def test_a_type_the_scope_does_not_have_counts_ZERO_not_everything():
    """The failure mode worth naming: an ignored filter answers the TOTAL, which reads as a
    healthy number. Zero is the honest answer and it is the one that looks like a problem."""
    kg, scope = await _graph()
    assert await kg.count_nodes(scope, audience=AUDIENCE_STAFF, node_type="ANIMAL") == 0


async def test_the_count_matches_what_list_nodes_returns():
    """THE INVARIANT, and the reason the parameter exists: the count and the list must answer
    about the SAME set. A count that narrows differently from the list is worse than no count.
    """
    kg, scope = await _graph()
    for tp in ("PERSON", "CONCEPT", "ANIMAL", None):
        listed = await kg.list_nodes(scope, audience=AUDIENCE_STAFF, node_type=tp, limit=1000)
        counted = await kg.count_nodes(scope, audience=AUDIENCE_STAFF, node_type=tp)
        assert counted == len(listed), f"count and list disagree for {tp}"


async def test_label_and_type_narrow_TOGETHER():
    """Both filters at once: neither may quietly win. `p0` is a PERSON, so a PERSON+p0 count is
    1 and a CONCEPT+p0 count is 0 — if either parameter were dropped, one of the two breaks."""
    kg, scope = await _graph()
    assert await kg.count_nodes(scope, audience=AUDIENCE_STAFF,
                                label="p0", node_type="PERSON") == 1
    assert await kg.count_nodes(scope, audience=AUDIENCE_STAFF,
                                label="p0", node_type="CONCEPT") == 0


async def test_the_type_match_is_EXACT_because_the_list_is():
    """O `count` e o `list` têm de dobrar o tipo da MESMA maneira — hoje, nenhum dobra.

    **A premissa deste teste mudou entre ser escrito e ser aberto, e vale dizê-lo.** Quando o
    escrevi (26/08) o `node_type` NÃO era normalizado na escrita, e `Rex/PERSON` + `Rex/person`
    eram duas linhas sob o índice único `(scope, engram_fold(label), node_type)`. Entretanto a
    main passou a normalizar na fronteira — `GraphNode(..., "person").node_type == "PERSON"` —
    e **esse caminho deixou de ser alcançável pelo construtor.**

    Por que o teste FICA, em vez de cair com a premissa:

    * a normalização é de HOJE, e **as linhas escritas antes dela continuam na base** — o que
      já lá está com caixa diferente não foi reescrito por ninguém;
    * o construtor não é o único caminho: uma migração, um `INSERT` cru ou um restauro entram
      por baixo dele;
    * e a invariante que interessa nunca foi sobre a caixa: é **o count e o list responderem
      sobre o MESMO conjunto**. Se um dia um deles dobrar e o outro não, esta asserção morre.

    Um teste cuja premissa foi consertada por outra pessoa não fica automaticamente inútil —
    fica com uma razão diferente, e escrevê-la é o trabalho.
    """
    kg = InMemoryGraph()
    scope = "t/exact"
    await kg.upsert_node(GraphNode(scope, "canonical", "PERSON"))
    await kg.upsert_node(GraphNode(scope, "outro", "CONCEPT"))

    for tp in ("PERSON", "person", "CONCEPT"):
        counted = await kg.count_nodes(scope, audience=AUDIENCE_STAFF, node_type=tp)
        listed = await kg.list_nodes(scope, audience=AUDIENCE_STAFF, node_type=tp, limit=100)
        assert counted == len(listed), f"count e list divergem para {tp!r}"


async def test_the_boundary_normalises_so_the_mixed_case_row_is_no_longer_reachable():
    """O CONTROLO da explicação acima: se o construtor voltar a NÃO normalizar, este morre e
    obriga a rever o teste anterior — em vez de o deixar com uma razão que caducou."""
    assert GraphNode("t/x", "sloppy", "person").node_type == "PERSON"
