import pytest

from cogno_engram import hypnos
from cogno_engram.types import TurnRecord


def _turn(**kw):
    base = dict(session_id="sess", scope="s", turn_n=1, user_input="x")
    base.update(kw)
    return TurnRecord(**base)


def test_goal_new_emits_started_memory():
    mems = hypnos.micro_consolidate(_turn(goal="agendamento", goal_status="NEW"))
    assert len(mems) == 1
    assert mems[0].category == "goal" and mems[0].content == "Started goal: agendamento"
    assert mems[0].confidence == 0.9


def test_goal_completed_is_absolute_confidence():
    mems = hypnos.micro_consolidate(_turn(goal="agendamento", goal_status="COMPLETED"))
    assert mems[0].content == "Completed goal: agendamento" and mems[0].confidence == 1.0


def test_goal_unchanged_emits_nothing():
    prev = _turn(goal="g", goal_status="ONGOING")
    cur = _turn(goal="g", goal_status="ONGOING")
    assert hypnos.micro_consolidate(cur, prev) == []


def test_frustration_is_edge_triggered():
    prev = _turn(sentiment="NEUTRAL")
    cur = _turn(sentiment="FRUSTRATED")
    mems = hypnos.micro_consolidate(cur, prev)
    assert any(m.category == "sentiment" for m in mems)
    # already frustrated → not re-emitted
    assert hypnos.micro_consolidate(_turn(sentiment="FRUSTRATED"),
                                    _turn(sentiment="FRUSTRATED")) == []


def test_pii_leak_recorded_as_fact_not_value():
    mems = hypnos.micro_consolidate(_turn(pii_types=["EMAIL", "PHONE"]))
    pii = [m for m in mems if m.category == "pii"]
    assert pii and pii[0].confidence == 1.0 and "EMAIL" in pii[0].content


def test_new_domain_becomes_preference():
    prev = _turn(domains=["FINANCE"])
    cur = _turn(domains=["FINANCE", "HEALTH"])
    mems = hypnos.micro_consolidate(cur, prev)
    prefs = [m for m in mems if m.category == "preference"]
    assert [m.content for m in prefs] == ["Interested in domain: HEALTH"]


async def test_tier2_and_tier3_are_declared_but_unimplemented():
    # the contract exists (signatures); LLM bodies land in the build-out phase
    with pytest.raises(NotImplementedError):
        await hypnos.periodic_consolidate(None, None, scope="s", session_id="x")
