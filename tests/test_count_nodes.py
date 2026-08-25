"""Counting nodes is a QUERY, not a page.

`list_nodes` is `ORDER BY id LIMIT n` — no label filter, no offset. A caller asking "is this
label unique in this scope?" over it gets the right answer only while the tenant stays smaller
than the page, and a homonym created past the cut is invisible. That is a live defect: the host
had to refuse to answer whenever the page came back full (`contact_context_unprovable`), because
the alternative was speaking a stranger's facts.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import GraphNode

SCOPE = "tenant:acme"


@pytest.fixture
def kg() -> InMemoryGraph:
    return InMemoryGraph()


async def test_it_counts_the_scope(kg):
    for label in ("José", "Maria", "Rex"):
        await kg.upsert_node(GraphNode(SCOPE, label, "PERSON"))
    await kg.upsert_node(GraphNode("tenant:other", "José", "PERSON"))

    assert await kg.count_nodes(SCOPE) == 3          # the other tenant is not ours
    assert await kg.count_nodes("tenant:other") == 1


async def test_it_counts_a_LABEL_case_insensitively(kg):
    """`find_node` and the `walk` seed both compare `lower(label)`, so a case-SENSITIVE count
    would answer a different question from the one the caller is about to act on."""
    await kg.upsert_node(GraphNode(SCOPE, "José", "PERSON"))
    assert await kg.count_nodes(SCOPE, label="JOSÉ") == 1
    assert await kg.count_nodes(SCOPE, label="  josé  ") == 1
    assert await kg.count_nodes(SCOPE, label="Maria") == 0


async def test_it_answers_beyond_any_page(kg):
    """The point of the whole thing: 600 nodes, and the count is still exact — `list_nodes`
    would have stopped at its limit and told the caller a comfortable lie."""
    for i in range(600):
        await kg.upsert_node(GraphNode(SCOPE, f"Contato {i}", "PERSON"))
    await kg.upsert_node(GraphNode(SCOPE, "Maria", "PERSON"))

    assert await kg.count_nodes(SCOPE) == 601
    assert await kg.count_nodes(SCOPE, label="Maria") == 1
    assert len(await kg.list_nodes(SCOPE, limit=500)) == 500       # the page, for contrast


async def test_an_empty_scope_counts_zero_not_an_error(kg):
    assert await kg.count_nodes(SCOPE) == 0
    assert await kg.count_nodes(SCOPE, label="ninguém") == 0


async def test_the_DOUBLE_collapses_a_homonym_the_real_store_keeps():
    """**A divergence, recorded rather than fixed here — and it is the case that matters.**

    Postgres's unique index is `(scope, lower(label), node_type)`, so `José/PERSON` and
    `José/CONCEPT` are two rows; `walk` seeds on the LABEL and would expand from both. The
    in-memory double keys on `(scope, lower(label))` alone and keeps ONE — so a test of the
    uniqueness guard passes here while the same graph in production has two nodes.

    Measured 2026-08-25 against a real Postgres: `count_nodes(label="José")` → **2** there, **1**
    here. Not fixed in this PR: the key is read in eight places and `find_node`/`delete_node` do
    not take a `node_type`, so it is a refactor of the double, not a line. Pinned so that when
    someone does it, this test fails and says why.
    """
    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode(SCOPE, "José", "PERSON"))
    await kg.upsert_node(GraphNode(SCOPE, "José", "CONCEPT"))

    assert await kg.count_nodes(SCOPE, label="José") == 1, (
        "the double now keeps both node types — Postgres always did. If this is the fix, delete "
        "this test and the note in the docstring above; the host's uniqueness guard gets more "
        "accurate, not less.")
