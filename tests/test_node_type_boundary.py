"""``node_type`` is case-normalised at the BOUNDARY — prevention, with the measurement that
makes it prevention and not repair.

The Postgres unique index is ``(scope, engram_fold(label), node_type)``: the LABEL half is
folded, the TYPE half was not. ``Rex/PERSON`` and ``Rex/person`` would be two rows for one
thing — the same shape as ``José``/``Jose``, with half the work already done. **A half-folded
identity is worse than a raw one, because it looks solved.**

Two convenience paths (``graph_context.ingest_entities`` and ``hypnos``) already upper-cased
correctly. Neither is the DOOR: the documented way in is to build a ``GraphNode`` and call
``upsert_node``, and this is a public library whose callers are not only our own host.

**Measured on the live box, 2026-08-27: ZERO rows outside upper case, ZERO pairs differing only
in the type's case** — with a POSITIVE CONTROL (the same query shape finds 10 groups of
same-label/different-type, so it knows how to find things). Nothing to migrate, which is why
this is cheap today. With one lower-case row stored the answer would invert.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import AUDIENCE_STAFF, GraphNode

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("given,want", [
    ("person", "PERSON"), ("Person", "PERSON"), ("  person  ", "PERSON"),
    ("PERSON", "PERSON"), ("pErSoN", "PERSON"),
])
async def test_the_case_is_folded_at_construction(given, want):
    assert GraphNode("t", "Rex", given).node_type == want


@pytest.mark.parametrize("empty", ["", "   ", None])
async def test_an_empty_type_falls_back_to_the_default(empty):
    assert GraphNode("t", "Rex", empty).node_type == "CONCEPT"


async def test_an_UNKNOWN_type_is_folded_but_NOT_coerced():
    """Case only, and the line matters.

    Coercing an unknown value to ``CONCEPT`` here would rewrite it on the way OUT of the
    database — ``__post_init__`` also runs when the adapters build a node from a row. That is a
    different, lossy decision from folding case, and it belongs to the write helpers, which
    already do it against ``VALID_NODE_TYPES``.
    """
    assert GraphNode("t", "Rex", "sasquatch").node_type == "SASQUATCH"


async def test_two_CASINGS_are_ONE_node_and_not_two():
    """The defect this prevents, end to end: the index folds the label and not the type."""
    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode("t/n", "Rex", "PERSON"))
    await kg.upsert_node(GraphNode("t/n", "Rex", "person"))
    nodes = await kg.list_nodes("t/n", audience=AUDIENCE_STAFF, limit=100)
    assert len(nodes) == 1, (
        f"duas grafias do mesmo tipo criaram {len(nodes)} nós — é o José/Jose com metade "
        f"do trabalho feito: {[(n.label, n.node_type) for n in nodes]}")
    assert nodes[0].node_type == "PERSON"


async def test_a_node_READ_BACK_is_unchanged_when_the_store_is_already_clean():
    """The half that makes this safe TODAY, and the one that would make it wrong tomorrow.

    Normalising at construction also normalises what the adapters build FROM A ROW. That is a
    no-op while every stored row is already upper case — measured — and it is the reason the
    zero has to be re-checked immediately before merging rather than quoted from an earlier
    hour: the graph grows with every conversation.
    """
    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode("t/r", "Acme", "ORG"))
    got = await kg.list_nodes("t/r", audience=AUDIENCE_STAFF, limit=10)
    assert [(n.label, n.node_type) for n in got] == [("Acme", "ORG")]


async def test_it_never_raises_on_a_type_that_is_not_a_string():
    """A node whose type is unusable must still be a node: the graph losing a row is worse than
    the row carrying an odd label."""
    class Odd:
        def __str__(self):
            raise RuntimeError("não sou texto")

    assert GraphNode("t", "Rex", Odd()).node_type == "CONCEPT"
