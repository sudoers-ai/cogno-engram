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


def test_the_POSTGRES_guard_exists_too():
    """The twin. `tests/test_prune_wildcard.py` exercises the in-memory adapter, and the guard
    that protects the PRODUCTION adapter was the untested one — deleting it left the suite
    green. The validator is module-level and needs no database, so the twin costs nothing."""
    from cogno_engram.adapters.postgres import _require_session as pg_require

    for blank in ("", "   ", "\t"):
        with pytest.raises(ValueError, match="not a wildcard"):
            pg_require(blank)
    assert pg_require("s1") == "s1"


async def test_tier_3_does_not_split_its_WRITE_over_a_blank_id(caplog):
    """The guard must protect the human notes without aborting a consolidation mid-way.

    By the prune, `consolidate_session` has already run the LLM pass and saved the memories, and
    `close_session` is still ahead — so a raise escaping here leaves the session OPEN and the
    janitor re-consolidates it on every tick, duplicating memories.
    """
    from cogno_engram.adapters.in_memory import InMemoryStore
    from cogno_engram.hypnos import consolidate_session
    from cogno_engram.types import Session, TurnRecord

    store, kg = InMemoryStore(), InMemoryGraph()
    from datetime import datetime, timezone
    session = Session(id="", scope=SCOPE,             # a host-built Session, blank id
                      started_at=datetime.now(timezone.utc))
    await store.save_turn(TurnRecord(scope=SCOPE, session_id="", turn_n=1,
                                     user_input="oi", response="ok", feedback=-1))

    class _Backend:
        model = "stub"

        async def generate(self, system, prompt):
            return "{}", 1, 1

    with caplog.at_level("WARNING"):
        out = await consolidate_session(store, _Backend(), session=session, kg=kg, close=False)
    assert out is not None, "the consolidation must complete"
    assert "prune_skipped" in caplog.text
