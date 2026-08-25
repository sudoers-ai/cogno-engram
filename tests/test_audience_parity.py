"""The two adapters must answer the SAME question — pinned against the one pure rule.

They did not. The Postgres predicate bound the caller's audience raw, so `audience=""` — what
`audience_for(None)` returns for a contact with no identity yet — read as `e.audience = ''` and
matched **every unclassified row**: the whole legacy graph, before the migration reaches it. The
in-memory adapter said `False` for the same input. Two stores disagreeing, in the direction that
leaks, on the invariant this feature IS.

`audience_can_read` is the rule; these tests make each adapter agree with it rather than with
whatever its own SQL or comprehension happens to do.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import (
    AUDIENCE_STAFF,
    AUDIENCE_TENANT,
    AUDIENCE_UNCLASSIFIED,
    GraphEdge,
    GraphNode,
    audience_can_read,
    audience_for,
)

TENANT = "acme"
A, B = audience_for("aaa"), audience_for("bbb")

# reader × stored-row, and the empty reader is the case that leaked
MATRIX = [
    (AUDIENCE_STAFF, AUDIENCE_UNCLASSIFIED), (AUDIENCE_STAFF, AUDIENCE_TENANT),
    (AUDIENCE_STAFF, A), (A, A), (A, B), (A, AUDIENCE_TENANT), (A, AUDIENCE_UNCLASSIFIED),
    (AUDIENCE_UNCLASSIFIED, AUDIENCE_UNCLASSIFIED), (AUDIENCE_UNCLASSIFIED, A),
    (AUDIENCE_UNCLASSIFIED, AUDIENCE_TENANT),
]


@pytest.mark.parametrize("reader,row", MATRIX)
async def test_the_in_memory_walk_agrees_with_the_rule(reader, row):
    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode(TENANT, "Ana", "PERSON"))
    await kg.upsert_node(GraphNode(TENANT, "Maria", "PERSON"))
    await kg.upsert_edge(GraphEdge(TENANT, "Ana", "Maria", "SPOUSE_OF", audience=row))
    got = bool(await kg.walk(TENANT, "Ana", audience=reader, max_depth=2))
    assert got == audience_can_read(reader, row), (reader, row)


def test_an_EMPTY_reader_sees_only_tenant_facts():
    """The exact case that leaked. `audience_for(None)` is `""` — a contact before registration
    — and it must NOT match the unclassified rows, which are the entire legacy graph."""
    assert audience_can_read(AUDIENCE_UNCLASSIFIED, AUDIENCE_UNCLASSIFIED) is False
    assert audience_can_read(AUDIENCE_UNCLASSIFIED, A) is False
    assert audience_can_read(AUDIENCE_UNCLASSIFIED, AUDIENCE_TENANT) is True


@pytest.mark.parametrize("raw", ["lixo", "identity:", "  ", None, 0, "IDENTITY:x", ["a"]])
def test_garbage_sanitizes_to_unclassified_never_to_something_readable(raw):
    """A cheap test the first cut did not have: removing the guard in `sanitize_audience`
    survived the whole suite."""
    from cogno_engram.types import sanitize_audience

    assert sanitize_audience(raw) == AUDIENCE_UNCLASSIFIED


async def test_the_extractor_stamps_the_contact_it_came_from():
    """Without this, every edge Tier 2 writes is born unclassified — staff-only — and the
    contact loses their own life the moment the host starts passing an audience to the reads.
    The migration docstring said "the rule the writers now follow"; they did not."""
    from cogno_engram.hypnos import periodic_consolidate
    from cogno_engram.adapters.in_memory import InMemoryStore
    from cogno_engram.types import TurnRecord

    class _Backend:
        model = "stub"

        async def generate(self, system, prompt):
            return ('{"nodes": [{"label": "Ana", "type": "PERSON"},'
                    '           {"label": "Maria", "type": "PERSON"}],'
                    ' "edges": [{"source": "Ana", "target": "Maria", '
                    '            "relation": "SPOUSE_OF"}]}'), 1, 1

    store, kg = InMemoryStore(), InMemoryGraph()
    session = await store.create_session(TENANT)
    await store.save_turn(TurnRecord(scope=TENANT, session_id=session.id, turn_n=1,
                                     user_input="minha esposa Maria", response="ok"))
    await periodic_consolidate(store, _Backend(), scope=TENANT, session_id=session.id,
                               kg=kg, audience=A)
    assert [e.audience for e in kg._edges] == [A]
    assert await kg.walk(TENANT, "Ana", audience=A, max_depth=2)          # the owner hears it
    assert not await kg.walk(TENANT, "Ana", audience=B, max_depth=2)      # nobody else does


async def test_a_node_with_only_a_PROPOSAL_is_not_pruned():
    """`prune_orphan_nodes` deletes. `has_edges` ignores status as well as audience — a first
    cut used `neighbors`, which returns accepted only, so a node whose single edge was waiting
    for review looked unattached and was destroyed. Measured 2 vs 0 before this existed."""
    from cogno_engram.maintenance import prune_orphan_nodes
    from cogno_engram.types import EDGE_PROPOSED

    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode(TENANT, "Ana", "PERSON"))
    await kg.upsert_node(GraphNode(TENANT, "Maria", "PERSON"))
    await kg.upsert_edge(GraphEdge(TENANT, "Ana", "Maria", "SPOUSE_OF",
                                   audience=A, status=EDGE_PROPOSED))
    assert await prune_orphan_nodes(kg, TENANT) == 0
    assert await kg.find_node(TENANT, "Ana", audience=AUDIENCE_STAFF) is not None


async def test_a_classification_can_be_UNDONE():
    """`upsert_edge` is narrow-never-widen, so re-writing cannot take a classification back —
    and the migration promotes `''` to `tenant`, the widest step there is. Without an explicit
    setter that would be a one-way door on live data."""
    kg = InMemoryGraph()
    await kg.upsert_node(GraphNode(TENANT, "Clinica", "PERSON"))
    await kg.upsert_node(GraphNode(TENANT, "Unimed", "PERSON"))
    await kg.upsert_edge(GraphEdge(TENANT, "Clinica", "Unimed", "ACCEPTS",
                                   audience=AUDIENCE_TENANT))
    assert await kg.set_edge_audience(TENANT, "Clinica", "Unimed", "ACCEPTS",
                                      AUDIENCE_UNCLASSIFIED) is True
    assert [e.audience for e in kg._edges] == [AUDIENCE_UNCLASSIFIED]
    assert await kg.set_edge_audience(TENANT, "x", "y", "z", A) is False


async def test_the_migration_reaches_a_PROPOSAL_too():
    """`walk` returns accepted only. A proposal written before the column would stay `''` and,
    once a human accepted it, be invisible to the very contact it is about."""
    from cogno_engram.maintenance import classify_edge_audience
    from cogno_engram.types import EDGE_ACCEPTED, EDGE_PROPOSED

    kg = InMemoryGraph()
    for lbl in ("Ana", "Maria"):
        await kg.upsert_node(GraphNode(TENANT, lbl, "PERSON"))
    await kg.upsert_edge(GraphEdge(TENANT, "Ana", "Maria", "SPOUSE_OF",
                                   source_session="sA", status=EDGE_PROPOSED))
    got = await classify_edge_audience(kg, TENANT, identity_of_session=lambda s: "aaa",
                                       dry_run=False)
    assert got["identity"] == 1
    await kg.set_edge_status(TENANT, "Ana", "Maria", "SPOUSE_OF", EDGE_ACCEPTED)
    assert "Maria" in repr(await kg.walk(TENANT, "Ana", audience=A, max_depth=2))
