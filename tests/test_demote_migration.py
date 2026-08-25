"""The migration recipe in `docs/HOST_INTEGRATION.md`, executed.

Turning the predicate on changes what the NEXT extraction stamps; it does not touch rows that
are already `accepted`, because `upsert_edge` only promotes. A host that flips the flag believing
its existing proximity edges now wait for a human is wrong in the direction that matters — it
keeps speaking every one of them — so the docs carry a migration, and a documented migration
nobody ran is a guess.

This runs the recipe verbatim against the real in-memory adapter.
"""

from __future__ import annotations

from cogno_engram import VALID_PROXIMITY_RELATIONS
from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import AUDIENCE_STAFF, EDGE_ACCEPTED, EDGE_PROPOSED, GraphEdge, GraphNode

SCOPE = "acme"


async def demote_extracted_proximity(kg, scope: str, *, dry_run: bool = True) -> int:
    """Copied from `docs/HOST_INTEGRATION.md` — keep the two in step."""
    seen, demoted = set(), 0
    after = None
    while True:
        nodes = await kg.scan_nodes(scope, after_id=after, limit=500, audience=AUDIENCE_STAFF)
        if not nodes:
            break
        after = nodes[-1].id
        for node in nodes:
            for e in await kg.walk(scope, node.label, max_depth=1, audience=AUDIENCE_STAFF):
                key = (e.source, e.target, e.relation)
                if key in seen:
                    continue
                seen.add(key)
                if e.relation.upper() not in VALID_PROXIMITY_RELATIONS:
                    continue
                if not (e.source_session or "").strip():
                    continue
                demoted += 1
                if not dry_run:
                    await kg.set_edge_status(scope, e.source, e.target, e.relation,
                                             EDGE_PROPOSED)
    return demoted


async def _box():
    """A graph in the state the live deployment is actually in."""
    kg = InMemoryGraph()
    for label in ("José", "Maria", "Pedro", "Clínica", "Unimed"):
        await kg.upsert_node(GraphNode(SCOPE, label, "PERSON"))
    # extracted proximity — the rows the incident produced
    await kg.upsert_edge(GraphEdge(SCOPE, "José", "Maria", "SPOUSE_OF", source_session="s1"))
    await kg.upsert_edge(GraphEdge(SCOPE, "José", "Pedro", "PARENT_OF", source_session="s2"))
    # a human's note: same class of relation, no stamp
    await kg.upsert_edge(GraphEdge(SCOPE, "José", "Pedro", "NICKNAME_OF"))
    # a domain fact the staff block must keep
    await kg.upsert_edge(GraphEdge(SCOPE, "Clínica", "Unimed", "ACCEPTS", source_session="s3"))
    return kg


def _status(kg, relation: str) -> str:
    return next(e.status for e in kg._edges if e.relation == relation)


async def test_the_dry_run_counts_without_writing():
    """Read the count before changing live data — the recipe says so, so it has to be true."""
    kg = await _box()
    assert await demote_extracted_proximity(kg, SCOPE, dry_run=True) == 2
    assert all(e.status == EDGE_ACCEPTED for e in kg._edges), "dry run must not write"


async def test_it_demotes_exactly_the_extracted_proximity_edges():
    kg = await _box()
    assert await demote_extracted_proximity(kg, SCOPE, dry_run=False) == 2
    assert _status(kg, "SPOUSE_OF") == EDGE_PROPOSED
    assert _status(kg, "PARENT_OF") == EDGE_PROPOSED
    assert _status(kg, "NICKNAME_OF") == EDGE_ACCEPTED, "a person wrote it; not ours to demote"
    assert _status(kg, "ACCEPTS") == EDGE_ACCEPTED, "a domain fact the staff block keeps"


async def test_it_is_idempotent():
    """`walk` returns only `accepted`, so a second pass has nothing left to find. A migration
    someone runs twice by accident must not be a different migration."""
    kg = await _box()
    first = await demote_extracted_proximity(kg, SCOPE, dry_run=False)
    second = await demote_extracted_proximity(kg, SCOPE, dry_run=False)
    assert (first, second) == (2, 0)


async def test_it_is_reversible():
    """The claim the deploy note rests on: `set_edge_status` puts any row back."""
    kg = await _box()
    await demote_extracted_proximity(kg, SCOPE, dry_run=False)
    assert await kg.set_edge_status(SCOPE, "José", "Maria", "SPOUSE_OF", EDGE_ACCEPTED)
    assert _status(kg, "SPOUSE_OF") == EDGE_ACCEPTED


async def test_the_contact_block_goes_quiet_and_a_review_brings_it_back():
    """The effect the migration exists for, end to end through `walk`."""
    kg = await _box()
    await demote_extracted_proximity(kg, SCOPE, dry_run=False)
    assert {e.relation for e in await kg.walk(SCOPE, "José", max_depth=1, audience=AUDIENCE_STAFF)} == {"NICKNAME_OF"}
    await kg.set_edge_status(SCOPE, "José", "Maria", "SPOUSE_OF", EDGE_ACCEPTED)
    assert "SPOUSE_OF" in {e.relation for e in await kg.walk(SCOPE, "José", max_depth=1, audience=AUDIENCE_STAFF)}
