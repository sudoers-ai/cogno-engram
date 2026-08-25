"""A blank session id is not a wildcard.

`delete_edges_by_session(scope, "")` matches every edge whose `source_session` is empty — which
is exactly the class nothing automated writes: notes a HUMAN or an admin API put there. One
disliked turn arriving with a blank id would erase them all, and `DELETE ... WHERE
source_session = ''` reads as entirely ordinary in a log.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph
from cogno_engram.types import GraphEdge

SCOPE = "acme"


async def _graph():
    kg = InMemoryGraph()
    await kg.upsert_edge(GraphEdge(SCOPE, "José", "Maria", "SPOUSE_OF"))          # human note
    await kg.upsert_edge(GraphEdge(SCOPE, "José", "Rex", "OWNS", source_session="s1"))
    return kg


@pytest.mark.parametrize("blank", ["", "   ", "\t"])
async def test_a_blank_session_is_REFUSED_not_treated_as_every_human_note(blank):
    kg = await _graph()
    with pytest.raises(ValueError, match="not a wildcard"):
        await kg.delete_edges_by_session(SCOPE, blank)
    assert len(kg._edges) == 2, "nothing may have been deleted"


async def test_a_real_session_still_prunes_only_its_own():
    """The positive control — the guard must not break the feature it protects."""
    kg = await _graph()
    assert await kg.delete_edges_by_session(SCOPE, "s1") == 1
    assert [e.relation for e in kg._edges] == ["SPOUSE_OF"]
