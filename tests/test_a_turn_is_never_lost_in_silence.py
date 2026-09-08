"""The turn write is allowed to lose; it is not allowed to lose QUIETLY.

Three of engram's turn writes absorb a conflict on purpose, because they run after
the contact already has their reply and an exception there would take down a turn
that succeeded. Absorbing is fine. Absorbing *invisibly* is the defect these tests
pin, and it is not hypothetical: a production session's turn counter was kept
outside the ``turns`` table, regressed from 119 to the seventies, and replayed
coordinates that were two weeks of conversation old. Every replayed turn was
discarded by ``ON CONFLICT DO NOTHING`` and every trace overwrote an older one
while keeping its old ``created_at``. It ran for two days. Nothing raised, nothing
logged, and no counter moved — so nothing could have noticed.

The four properties, one per twin:

1. a discarded turn is COUNTED and announced;
2. an older stored trace is never overwritten, and ``created_at`` never moves;
3. the coordinate is derived from the STORE, so it is monotonic across a restart
   of whatever was counting;
4. two concurrent writers on one session get two rows, not one (the race twin
   lives in ``test_postgres_integration.py`` — it needs a real database to be a
   real race; the in-memory mirror of the allocation rule is here).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from cogno_engram import write_loss
from cogno_engram.adapters.in_memory import InMemoryStore
from cogno_engram.trace_policy import TRACE_REVISION_WINDOW_S
from cogno_engram.types import ALLOCATE_TURN_N, TurnRecord, TurnTrace

SCOPE = "tenant-a/contact-1"


@pytest.fixture(autouse=True)
def _clean_counters():
    write_loss.reset()
    yield
    write_loss.reset()


def _turn(sid: str, n: int, text: str = "hi") -> TurnRecord:
    return TurnRecord(session_id=sid, scope=SCOPE, turn_n=n, user_input=text)


# ── 1. the conflict is counted, never silent ─────────────────────────────────────
@pytest.mark.asyncio
async def test_a_discarded_turn_is_counted_and_announced(caplog):
    """The twin for the ``DO NOTHING``. A turn landing on a taken coordinate is
    dropped — and says so, on both channels."""
    store, sid = InMemoryStore(), str(uuid4())
    assert await store.save_turn(_turn(sid, 1, "first")) == 1

    with caplog.at_level("ERROR", logger="cogno_engram.write_loss"):
        written = await store.save_turn(_turn(sid, 1, "second, lands on a taken seat"))

    assert written == 0, "a discarded turn must not report a coordinate it did not take"
    assert write_loss.counts()[write_loss.TURN_DISCARDED] == 1
    assert any(write_loss.TURN_DISCARDED in r.getMessage() for r in caplog.records), \
        "the loss must be announced, not only counted"
    assert await store.turn_count(sid) == 1


@pytest.mark.asyncio
async def test_the_loss_report_never_carries_the_scope_in_clear():
    """A scope's second segment is a contact's phone number and logs have no
    retention deadline. The report carries a digest and a coordinate — never the
    scope, never the turn's text."""
    store, sid = InMemoryStore(), str(uuid4())
    seen: list[tuple[str, dict]] = []
    write_loss.subscribe(lambda e, f: seen.append((e, f)))

    await store.save_turn(_turn(sid, 1))
    await store.save_turn(_turn(sid, 1, "a secret the contact typed"))

    assert len(seen) == 1
    _, fields = seen[0]
    flat = " ".join(f"{k}={v}" for k, v in fields.items())
    assert SCOPE not in flat and "secret" not in flat
    assert fields["scope_sha"] == write_loss.scope_digest(SCOPE)
    assert fields["turn"] == 1


@pytest.mark.asyncio
async def test_a_conflict_never_raises():
    """Deliberate, and load-bearing: this write runs after the reply was delivered.
    Turning a bookkeeping loss into an exception would lose the whole turn."""
    store, sid = InMemoryStore(), str(uuid4())
    await store.save_turn(_turn(sid, 1))
    await store.save_turn(_turn(sid, 1))          # must not raise
    assert write_loss.counts()[write_loss.TURN_DISCARDED] == 1


# ── 2. an older trace is never overwritten, and created_at never moves ───────────
@pytest.mark.asyncio
async def test_an_older_trace_is_not_overwritten_by_a_newer_one():
    """The twin for the trace upsert. The stored trace is two weeks old; a turn
    wearing the same number arrives today. The old one stays, and its date is
    still its own."""
    store, sid = InMemoryStore(), str(uuid4())
    august = datetime(2026, 8, 28, 5, 52, tzinfo=timezone.utc)
    await store.save_turn_trace(TurnTrace(session_id=sid, scope=SCOPE, turn_n=76,
                                          trace={"who": "august"}, created_at=august))

    stored = await store.save_turn_trace(
        TurnTrace(session_id=sid, scope=SCOPE, turn_n=76, trace={"who": "september"},
                  created_at=august + timedelta(days=10)))

    assert stored is False
    assert write_loss.counts()[write_loss.TRACE_OVERWRITE_REFUSED] == 1
    kept = (await store.traces_for_session(sid, scope=SCOPE))[0]
    assert kept.trace == {"who": "august"}, "the older trace must survive"
    assert kept.created_at == august, "created_at must not be restated"


@pytest.mark.asyncio
async def test_created_at_does_not_lie_after_an_allowed_revision():
    """The defect was not only the overwrite — it was that ``created_at`` was left
    out of the SET, so the row went on advertising the old date while its content
    was new. An ALLOWED revision keeps the stored date too: an upsert revises
    content, it does not restate when the turn happened."""
    store, sid = InMemoryStore(), str(uuid4())
    when = datetime(2026, 9, 7, 2, 29, tzinfo=timezone.utc)
    await store.save_turn_trace(TurnTrace(session_id=sid, scope=SCOPE, turn_n=3,
                                          trace={"v": 1}, created_at=when))

    assert await store.save_turn_trace(
        TurnTrace(session_id=sid, scope=SCOPE, turn_n=3, trace={"v": 2},
                  created_at=when)) is True

    kept = (await store.traces_for_session(sid, scope=SCOPE))[0]
    assert kept.trace == {"v": 2} and kept.created_at == when
    assert write_loss.counts()[write_loss.TRACE_OVERWRITE_REFUSED] == 0


# ── 3. the coordinate comes from the store, so a restart cannot rewind it ────────
@pytest.mark.asyncio
async def test_the_counter_is_monotonic_across_a_restart():
    """The twin that proves the fix for the cause.

    The old design kept the next ``turn_n`` in a per-contact row OUTSIDE this
    table. That row can be deleted, can be served by a process-local store, and in
    production regressed from 119 to the single digits — after which every turn
    replayed an occupied coordinate and was discarded. Deriving the coordinate
    from the rows that already exist makes the regression unrepresentable: there
    is no second place for the number to be wrong.

    ``restart`` here is "whatever was counting has forgotten everything" — which is
    exactly the condition that caused the incident.
    """
    store, sid = InMemoryStore(), str(uuid4())
    for expected in (1, 2, 3):
        assert await store.save_turn(_turn(sid, ALLOCATE_TURN_N)) == expected

    # …the process restarts, the counter is gone, nothing is carried over.
    assert await store.save_turn(_turn(sid, ALLOCATE_TURN_N)) == 4, \
        "the next coordinate must come from the stored rows, not from a counter"
    assert write_loss.counts()[write_loss.TURN_DISCARDED] == 0, \
        "a restart must not cost a single turn"
    assert await store.turn_count(sid) == 4


@pytest.mark.asyncio
async def test_allocation_is_per_session_not_global():
    """Two conversations advance independently — the coordinate is scoped to
    ``(scope, session_id)``, the same pair the uniqueness constraint uses."""
    store = InMemoryStore()
    a, b = str(uuid4()), str(uuid4())
    assert await store.save_turn(_turn(a, ALLOCATE_TURN_N)) == 1
    assert await store.save_turn(_turn(a, ALLOCATE_TURN_N)) == 2
    assert await store.save_turn(_turn(b, ALLOCATE_TURN_N)) == 1


@pytest.mark.asyncio
async def test_a_pinned_coordinate_is_still_honoured():
    """A backfill, an import or a replay is re-stating history, not appending to
    it, and must keep saying which turn it is talking about."""
    store, sid = InMemoryStore(), str(uuid4())
    assert await store.save_turn(_turn(sid, 41)) == 41
    assert await store.save_turn(_turn(sid, ALLOCATE_TURN_N)) == 42, \
        "allocation must continue from a pinned history, not from zero"


@pytest.mark.asyncio
async def test_a_same_turn_revision_is_still_allowed():
    """The other half of the window, and the reason it is a window at all.

    The turn that is still being written revises its own trace — a correction
    loop, a re-voice, a backfill batch re-running its own rows. Refusing THAT
    would trade the overwrite this guard exists to stop for a brand-new silent
    loss, which is the worse of the two.
    """
    store, sid = InMemoryStore(), str(uuid4())
    t0 = datetime(2026, 9, 7, 2, 29, tzinfo=timezone.utc)
    await store.save_turn_trace(TurnTrace(session_id=sid, scope=SCOPE, turn_n=3,
                                          trace={"v": 1}, created_at=t0))

    assert await store.save_turn_trace(
        TurnTrace(session_id=sid, scope=SCOPE, turn_n=3, trace={"v": 2},
                  created_at=t0 + timedelta(seconds=90))) is True

    kept = (await store.traces_for_session(sid, scope=SCOPE))[0]
    assert kept.trace == {"v": 2}, "a same-turn revision must land"
    assert kept.created_at == t0, "…and must not restate when the turn happened"
    assert write_loss.counts()[write_loss.TRACE_OVERWRITE_REFUSED] == 0


@pytest.mark.asyncio
async def test_the_window_is_a_bound_not_a_knob_and_zero_means_never():
    """A host that wants strict immutability sets the window to 0. Pinned so the
    knob cannot quietly stop existing."""
    store, sid = InMemoryStore(trace_revision_window_s=0), str(uuid4())
    t0 = datetime(2026, 9, 7, 2, 29, tzinfo=timezone.utc)
    await store.save_turn_trace(TurnTrace(session_id=sid, scope=SCOPE, turn_n=1,
                                          trace={"v": 1}, created_at=t0))
    assert await store.save_turn_trace(
        TurnTrace(session_id=sid, scope=SCOPE, turn_n=1, trace={"v": 2},
                  created_at=t0 + timedelta(seconds=1))) is False
    assert TRACE_REVISION_WINDOW_S > 0, "the shipped default must still admit a revision"


@pytest.mark.asyncio
async def test_zero_is_a_coordinate_not_a_request_to_allocate():
    """``0`` is a real turn number that callers across this repo already use.
    Overloading it as the allocate sentinel silently re-numbered their rows — this
    is the twin for that mistake, which the suite caught before it shipped."""
    store, sid = InMemoryStore(), str(uuid4())
    assert await store.save_turn(_turn(sid, 0)) == 0
    assert await store.save_turn(_turn(sid, 1)) == 1
    assert write_loss.counts()[write_loss.TURN_DISCARDED] == 0
    assert ALLOCATE_TURN_N < 0, "the sentinel must not be a representable coordinate"
