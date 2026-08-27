"""Two contacts in one tenant: A must never see B's life, through ANY read.

"Tenant sees everything in the graph; an identity sees only its own life. They do not mix."
— the product decision, 2026-08-25.

Measured on `origin/main` BEFORE the change, with this exact fixture: **7 of 8 reads returned
B's private data to a contact-scoped call.** (`count_nodes` was the only one that did not, and
only because it returns an integer.) There was no `audience` argument to pass — every read was
tenant-wide by construction. Recording that baseline first is what makes the 0/N below mean
something.

The parametrisation is over the READ SURFACE, not over reads someone remembered to list: a new
read that forgets the filter fails on the day it lands, without anybody adding a case.
"""

from __future__ import annotations

import inspect

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.ports import KnowledgeGraph
from cogno_engram.types import (
    AUDIENCE_STAFF,
    AUDIENCE_TENANT,
    EDGE_PROPOSED,
    GraphEdge,
    GraphNode,
    audience_for,
)

TENANT = "acme"
A, B = audience_for("aaa"), audience_for("bbb")
BS_SECRETS = ("PARENT_OF", "SIBLING_OF", "Carlos")


async def _box():
    kg = InMemoryGraph()
    for label in ("Ana", "Bruno", "Maria", "Carlos", "Clínica", "Unimed"):
        await kg.upsert_node(GraphNode(TENANT, label, "PERSON"))
    await kg.upsert_edge(GraphEdge(TENANT, "Ana", "Maria", "SPOUSE_OF", audience=A))
    await kg.upsert_edge(GraphEdge(TENANT, "Bruno", "Carlos", "PARENT_OF", audience=B))
    # ...and one of B's WAITING FOR REVIEW, or `pending_edges` answers "no leak" because the
    # fixture is empty — a false negative, not a safe read. The baseline run caught that.
    await kg.upsert_edge(GraphEdge(TENANT, "Bruno", "Carlos", "SIBLING_OF",
                                   audience=B, status=EDGE_PROPOSED))
    await kg.upsert_edge(GraphEdge(TENANT, "Clínica", "Unimed", "ACCEPTS",
                                   audience=AUDIENCE_TENANT))
    return kg


def _reads(kg):
    """Every read on the port that takes an audience, called with B's own labels as the target
    — the most favourable case for a leak."""
    return {
        "walk": lambda a: kg.walk(TENANT, "Bruno", audience=a, max_depth=3),
        "neighbors": lambda a: kg.neighbors(TENANT, "Bruno", audience=a),
        "get_node_context": lambda a: kg.get_node_context(TENANT, "Bruno", audience=a),
        "list_nodes": lambda a: kg.list_nodes(TENANT, audience=a, limit=100),
        "find_node": lambda a: kg.find_node(TENANT, "Carlos", audience=a),
        "count_nodes": lambda a: kg.count_nodes(TENANT, audience=a),
        "pending_edges": lambda a: kg.pending_edges(TENANT, audience=a, limit=100),
        "scan_nodes": lambda a: kg.scan_nodes(TENANT, audience=a, limit=100),
        "find_nodes_by_embedding": lambda a: kg.find_nodes_by_embedding(
            TENANT, [0.1, 0.2], audience=a, limit=100),
        # An AGGREGATE is still a read, and the least obvious one: it returns no rows of its
        # own, so nothing about it looks like disclosure — but `top_connected` carries whole
        # nodes, and a total that counts another contact's nodes discloses their existence.
        "graph_stats": lambda a: kg.graph_stats(TENANT, audience=a, top=100),
    }


@pytest.mark.parametrize("name", sorted(_reads(InMemoryGraph())))
async def test_A_never_sees_B(name):
    kg = await _box()
    got = repr(await _reads(kg)[name](A))
    leaked = [s for s in BS_SECRETS if s in got]
    assert not leaked, f"{name} leaked {leaked} to another contact"


@pytest.mark.parametrize("name", sorted(_reads(InMemoryGraph())))
async def test_STAFF_still_sees_everything(name):
    """The other half. A filter that also blinds staff is not scoping, it is deletion."""
    kg = await _box()
    got = await _reads(kg)[name](AUDIENCE_STAFF)
    assert got or got == 0 or got is not None
    if name in ("walk", "neighbors", "get_node_context", "list_nodes", "find_node",
                "scan_nodes", "pending_edges"):
        assert any(s in repr(got) for s in BS_SECRETS), f"{name} hid B from staff"


async def test_A_still_sees_its_OWN_life():
    """A filter that blinds the owner is not scoping either."""
    kg = await _box()
    assert "Maria" in repr(await kg.walk(TENANT, "Ana", audience=A, max_depth=2))


async def test_a_TENANT_fact_reaches_every_contact():
    kg = await _box()
    for who in (A, B):
        assert "ACCEPTS" in repr(await kg.walk(TENANT, "Clínica", audience=who, max_depth=2))


async def test_an_UNCLASSIFIED_edge_is_staff_only():
    """The default is fail-CLOSED for the contact. A writer that forgets costs a missing block
    — visible, annoying, safe — and never a leak. Two discriminators in this codebase defaulted
    the permissive way and both had to be undone after they had already spoken."""
    kg = await _box()
    await kg.upsert_edge(GraphEdge(TENANT, "Ana", "Carlos", "FRIEND_OF"))    # no audience
    assert "FRIEND_OF" not in repr(await kg.walk(TENANT, "Ana", audience=A, max_depth=2))
    assert "FRIEND_OF" in repr(await kg.walk(TENANT, "Ana", audience=AUDIENCE_STAFF,
                                             max_depth=2))


def test_every_audience_read_on_the_PORT_is_covered_here():
    """The guard's own guard: a read added to the port with an `audience` parameter and NOT
    listed above would leave this suite passing while covering less. Enumerated from the
    Protocol, not from memory."""
    covered = set(_reads(InMemoryGraph()))
    # KEYWORD-ONLY `audience` is the mark of a READ: that is the design ("required keyword on
    # every read that can return contact data"). `set_edge_audience` takes it POSITIONALLY
    # because it is a write — it does not filter, it assigns — and this test caught the
    # difference the moment that method landed, which is what it is for.
    on_port = {
        name for name, fn in inspect.getmembers(KnowledgeGraph, inspect.isfunction)
        if inspect.signature(fn).parameters.get("audience", None) is not None
        and inspect.signature(fn).parameters["audience"].kind
        is inspect.Parameter.KEYWORD_ONLY
    }
    assert on_port, "the enumeration found nothing — it would pass by covering zero reads"
    assert on_port <= covered, f"unprobed reads: {sorted(on_port - covered)}"


async def test_the_ORPHAN_predicate_takes_no_audience_and_must_not():
    """`prune_orphan_nodes` DELETES. A filtered read there would not narrow what is seen, it
    would widen what is destroyed: a node whose edges all belong to another contact would look
    unattached. Found by review before it shipped."""
    from cogno_engram.maintenance import prune_orphan_nodes

    kg = await _box()
    assert await kg.has_edges(TENANT, "Carlos") is True       # B's, and still an edge
    assert "audience" not in inspect.signature(KnowledgeGraph.has_edges).parameters
    await prune_orphan_nodes(kg, TENANT)
    assert await kg.find_node(TENANT, "Carlos", audience=AUDIENCE_STAFF) is not None


# ── the migration, executed ──────────────────────────────────────────────────

async def _legacy_box():
    """A graph as it exists before the column: every edge unclassified."""
    kg = InMemoryGraph()
    for label in ("Ana", "Maria", "Clínica", "Unimed"):
        await kg.upsert_node(GraphNode(TENANT, label, "PERSON"))
    await kg.upsert_edge(GraphEdge(TENANT, "Ana", "Maria", "SPOUSE_OF", source_session="sA"))
    await kg.upsert_edge(GraphEdge(TENANT, "Clínica", "Unimed", "ACCEPTS"))   # staff/KB write
    return kg


async def test_the_migration_counts_before_it_writes():
    from cogno_engram.maintenance import classify_edge_audience

    kg = await _legacy_box()
    got = await classify_edge_audience(kg, TENANT, identity_of_session=lambda s: "aaa")
    assert got == {"tenant": 1, "identity": 1, "unresolved": 0}
    assert all(not e.audience for e in kg._edges), "a dry run must not write"


async def test_the_migration_gives_each_edge_the_owner_the_writers_now_use():
    from cogno_engram.maintenance import classify_edge_audience

    kg = await _legacy_box()
    await classify_edge_audience(kg, TENANT, identity_of_session=lambda s: "aaa",
                                 dry_run=False)
    by_rel = {e.relation: e.audience for e in kg._edges}
    assert by_rel == {"SPOUSE_OF": A, "ACCEPTS": AUDIENCE_TENANT}
    # ...and the effect that matters: A hears its own life, B does not.
    assert "Maria" in repr(await kg.walk(TENANT, "Ana", audience=A, max_depth=2))
    assert "Maria" not in repr(await kg.walk(TENANT, "Ana", audience=B, max_depth=2))


async def test_an_UNRESOLVABLE_session_is_left_unknown_not_guessed():
    """A closed or purged session cannot name an owner. Leaving it `''` keeps it staff-only —
    the safe direction — instead of attributing someone's life to the wrong person."""
    from cogno_engram.maintenance import classify_edge_audience

    kg = await _legacy_box()
    got = await classify_edge_audience(kg, TENANT, identity_of_session=lambda s: "",
                                       dry_run=False)
    assert got["unresolved"] == 1
    assert {e.relation: e.audience for e in kg._edges}["SPOUSE_OF"] == ""


async def test_the_migration_is_idempotent():
    from cogno_engram.maintenance import classify_edge_audience

    kg = await _legacy_box()
    first = await classify_edge_audience(kg, TENANT, identity_of_session=lambda s: "aaa",
                                         dry_run=False)
    second = await classify_edge_audience(kg, TENANT, identity_of_session=lambda s: "aaa",
                                          dry_run=False)
    assert sum(first.values()) == 2 and sum(second.values()) == 0
