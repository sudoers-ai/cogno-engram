"""Tier 2 can propose SOME relations and accept others.

`propose_relations` was all-or-nothing, and that is the wrong granularity for what it protects.
The edges that become a sentence about a PERSON ("your wife Maria") are a small, nameable class;
the rest ("the clinic accepts Unimed") are domain facts a walk should keep stating. All-or-
nothing forces a host to choose between speaking unreviewed claims about someone's family and
losing its whole knowledge block — and the first host to meet that choice took the first option
without noticing, for months, in production.

The predicate is the same shape and the same seam as `edge_filter`, which already decides per
relation what is worth persisting at all.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters.in_memory import InMemoryGraph, InMemoryStore
from cogno_engram.hypnos import _status_rule, periodic_consolidate
from cogno_engram.types import EDGE_ACCEPTED, EDGE_PROPOSED, TurnRecord

SCOPE = "acme/jose"

_PROXIMITY = {"SPOUSE_OF", "PARENT_OF", "CHILD_OF", "SIBLING_OF"}


def _is_proximity(source: str, target: str, relation: str) -> bool:
    return relation.upper() in _PROXIMITY


# ── the rule itself, in isolation ────────────────────────────────────────────

def test_a_bool_keeps_its_original_meaning_exactly():
    """Nothing may change for a host that never passes a predicate."""
    assert _status_rule(False) == EDGE_ACCEPTED
    assert _status_rule(True) == EDGE_PROPOSED


def test_a_predicate_decides_per_relation():
    rule = _status_rule(_is_proximity)
    assert rule("José", "Maria", "SPOUSE_OF") == EDGE_PROPOSED
    assert rule("Clínica", "Unimed", "ACCEPTS") == EDGE_ACCEPTED


def test_a_predicate_that_raises_HOLDS_the_edge_rather_than_speaking_it():
    """The direction of the failure is the whole point: an edge nobody could classify waits for
    a human. Defaulting to `accepted` would let a broken predicate publish exactly the class the
    predicate exists to hold back."""
    def boom(s, t, r):
        raise RuntimeError("classifier down")

    assert _status_rule(boom)("José", "Maria", "SPOUSE_OF") == EDGE_PROPOSED


# ── end to end through the extractor ─────────────────────────────────────────

class _Backend:
    """Returns one proximity edge and one domain edge, every time."""

    model = "stub"

    async def generate(self, system, prompt):
        return ('{"nodes": [{"label": "José", "type": "PERSON"},'
                '           {"label": "Maria", "type": "PERSON"},'
                '           {"label": "Unimed", "type": "CONCEPT"}],'
                ' "edges": [{"source": "José", "target": "Maria", "relation": "SPOUSE_OF"},'
                '           {"source": "José", "target": "Unimed", "relation": "USES"}]}'), 1, 1


async def _run(propose):
    store, kg = InMemoryStore(), InMemoryGraph()
    session = await store.create_session(SCOPE)
    await store.save_turn(TurnRecord(scope=SCOPE, session_id=session.id, turn_n=1,
                                     user_input="minha esposa Maria", response="ok"))
    await periodic_consolidate(store, _Backend(), scope=SCOPE, session_id=session.id,
                               kg=kg, propose_relations=propose)
    return {e.relation: e.status for e in kg._edges}


async def test_proximity_waits_for_a_human_while_the_domain_edge_is_spoken():
    got = await _run(_is_proximity)
    assert got == {"SPOUSE_OF": EDGE_PROPOSED, "USES": EDGE_ACCEPTED}


@pytest.mark.parametrize("flag,expected", [(False, EDGE_ACCEPTED), (True, EDGE_PROPOSED)])
async def test_the_bool_still_applies_to_EVERY_edge(flag, expected):
    got = await _run(flag)
    assert set(got.values()) == {expected}, got


# ── a wrong predicate is a WIRING error, refused loudly ──────────────────────

def test_a_predicate_of_the_wrong_ARITY_is_refused_at_wiring_time():
    """`lambda r: ...` is the natural mistake — the docs talk about relations — and left to the
    per-edge guard it raises on EVERY edge, is swallowed, and stamps everything `proposed`. The
    host's graph block then empties silently: the very regression the opt-in default exists to
    prevent, arriving through the option meant to avoid it."""
    with pytest.raises(TypeError, match="source, target, relation"):
        _status_rule(lambda relation: True)


def test_an_ASYNC_predicate_is_refused_too():
    """A coroutine function is `callable`, so the call returns a coroutine object — always
    truthy — and every edge becomes `proposed`, with a `RuntimeWarning` as the only clue."""
    async def _is_proximity_async(s, t, r):
        return True

    with pytest.raises(TypeError, match="sync predicate"):
        _status_rule(_is_proximity_async)


@pytest.mark.parametrize("ok", [
    lambda s, t, r: True,
    lambda *a: True,
    lambda s, t, r, extra=None: True,
])
def test_shapes_that_DO_work_are_not_refused(ok):
    """The positive control: the arity check must not reject a legitimate predicate."""
    assert _status_rule(ok)("a", "b", "SPOUSE_OF") == EDGE_PROPOSED


def test_a_callable_whose_signature_cannot_be_READ_is_allowed_through(monkeypatch):
    """Some C callables expose no signature. Refusing those would break a legitimate caller for
    the checker's convenience, so they fall through to the per-edge guard instead."""
    import inspect as _inspect

    import cogno_engram.hypnos as hypnos

    def _no_signature(_obj):
        raise ValueError("no signature for builtin")

    monkeypatch.setattr(hypnos.inspect, "signature", _no_signature)
    monkeypatch.setattr(hypnos.inspect, "iscoroutinefunction",
                        _inspect.iscoroutinefunction)
    rule = _status_rule(lambda s, t, r: True)          # shape is fine; it just cannot be read
    assert rule("a", "b", "SPOUSE_OF") == EDGE_PROPOSED
