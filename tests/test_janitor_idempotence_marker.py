"""The janitor's "already handled" mark, and the cleanup that used to erase it.

**The defect, in one sentence:** pickability was derived from ``turns`` while the mark that
said "already consolidated" was a row in ``sessions`` — so a scope cleanup that deleted
``sessions`` and spared ``turns`` made every touched scope pickable again, tick after tick,
for ever.

And such a cleanup is *obeying a correct rule*. ``cogno-host``'s
``tests/unit/test_identity_purge_contract.py::test_the_purge_never_deletes_the_measurement``
forbids the identity purge from touching ``turns``, because turns are what the promise audit
measures. ``purge_scope`` escaped the trap only by deleting both. Nothing anywhere said which
of the two rules governed ``sessions`` — and that ABSENCE was the defect, not either rule.

Measured 2026-09-09, nobody provoking it: an authorised enumerated cleanup removed 11 scopes'
memories, traces and ``sessions`` rows and deliberately left ``turns`` standing. Nine minutes
later ONE janitor tick re-consolidated all 10 surviving scopes — 5,258 tokens, one of them a
scope whose identity had been gone for 90 minutes, returning with a byte-identical 407-token
spend and writing four fresh memories. Replayed read-only against the live box: wipe
``sessions``, keep ``turns``, and the old predicate returns **311** pickable sessions on every
tick, for ever. With the mark on the turns it returns **0**, and a simulated new burst still
returns 310 — which is the pair of numbers this file turns into tests.

The mark now rides on ``turns.consolidated_at`` (a set keyed by the same UNIQUE triple in the
in-memory adapter), written by ``close_session`` on the very rows the consolidation read. It is
ANDed onto the old disjunction, never substituted for it, so it can only ever REMOVE a session
from the result — the legitimate re-pick and the deploy-day backlog are untouched by
construction, and a lost stamp costs a duplicate LLM read rather than a contact's memory.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timedelta, timezone

import pytest

from cogno_engram.adapters.in_memory import InMemoryStore
from cogno_engram.types import MemoryRecord, TurnRecord


@pytest.fixture
def store():
    return InMemoryStore()


def _now():
    return datetime.now(timezone.utc)


async def _consolidated_session(store, sid: str, scope: str, *, age_h: int = 3):
    """A session that went idle and WAS consolidated — the janitor's normal end state."""
    await store.save_turn(TurnRecord(sid, scope, 0, "oi",
                                     created_at=_now() - timedelta(hours=age_h)))
    await store.save_memory(MemoryRecord(scope, "fact", "prefere manha"))
    await store.close_session(sid, summary="turno 1", scope=scope)
    assert await store.idle_sessions(idle_seconds=1800) == []


def _scope_cleanup_sparing_turns(store, scope: str) -> None:
    """The enumerated cleanup that caused the incident, reproduced exactly.

    Memories, traces and ``sessions`` rows go; ``turns`` deliberately stay, because they are
    the measurement and the host contract forbids deleting them. This is the operation that
    must stop re-arming the janitor — NOT an operation that must stop existing.
    """
    for sid in [s for s, sess in store._sessions.items() if sess.scope == scope]:
        del store._sessions[sid]
    store._memories = [m for m in store._memories if m.scope != scope]
    store._traces = [t for t in store._traces if t.scope != scope]


# ── TWIN 1: the loop ─────────────────────────────────────────────────────────────────────

async def test_a_cleanup_that_deletes_sessions_and_spares_turns_does_not_re_arm_the_janitor(store):
    """THE defect. The scope's ``sessions`` row is gone, its turns remain, and the janitor
    must NOT pick it up — not on this tick, not on any later one.

    Before the mark moved, ``s.id IS NULL`` fired unconditionally here and the session came
    back on every single tick: a fresh LLM consolidation, fresh memories written under a scope
    whose contact had been deleted, for ever.

    Mutation: delete the ``if (t.scope, t.session_id, t.turn_n) in self._consolidated:
    continue`` guard from ``InMemoryStore.idle_sessions`` (in Postgres: drop
    ``AND t.consolidated_at IS NULL`` from the WHERE) and this dies — the ten ticks below
    return the session ten times.
    """
    await _consolidated_session(store, "sess-purged", "t/u1")
    _scope_cleanup_sparing_turns(store, "t/u1")

    assert await store.load_turns("sess-purged") != [], (
        "the premise: the cleanup spared the turns, exactly as the host contract requires")
    assert await store.get_session("sess-purged") is None, (
        "the premise: the cleanup took the sessions row, which used to be the only mark")

    for tick in range(10):
        assert await store.idle_sessions(idle_seconds=1800) == [], (
            f"tick {tick}: a scope whose sessions row was deleted was picked up again — "
            f"the janitor is consolidating a contact who no longer exists")


# ── TWIN 2: the re-pick that must NOT regress ────────────────────────────────────────────

async def test_a_genuine_new_burst_after_consolidation_still_re_arms_the_session(store):
    """The clause the fix was forbidden to break.

    ``t.created_at > s.ended_at`` is deliberate and correct — a contact whose session id never
    rotates must come back for Tier-3 when they write again, or long-term memory freezes at
    the first quiet spell (measured 2026-08: 20 of 22, 10 of 13 and 8 of 10 turns arrived after
    their session had been declared over). The mark must kill the SPURIOUS re-pick without
    killing this one, and it does so because ``close_session`` stamps only the turns it
    actually covered (``created_at <= ended_at``).

    Mutation: make the mark SESSION-grained instead of TURN-grained — skip a session as soon
    as ANY of its turns is stamped, rather than filtering turn by turn — and this dies. That
    is not a strawman: it is the shape the old ``sessions``-row mark had, and adopting it here
    would re-introduce the 2026-08 frozen-memory bug through the new column.

    (The first draft of this docstring named a different mutation — dropping the
    ``created_at <= ended`` watermark — and that mutation SURVIVED, because the returning
    contact's turn is written after the close and was never a candidate for its stamp. The
    watermark is real but it is a different claim, and it gets its own test below.)
    """
    await _consolidated_session(store, "sess-grow", "t/u")
    closed = await store.get_session("sess-grow")
    closed.ended_at = _now() - timedelta(hours=3)     # backdate: the close really was 3h ago

    await store.save_turn(TurnRecord("sess-grow", "t/u", 1, "voltei",
                                     created_at=_now() - timedelta(minutes=45)))

    assert [s.id for s in await store.idle_sessions(idle_seconds=1800)] == ["sess-grow"], (
        "a genuine new burst must still bring the session back for consolidation")

    # …and it is still SELF-LIMITING: once per burst, not once per tick.
    await store.close_session("sess-grow", summary="turnos 1-2", scope="t/u")
    for _ in range(3):
        assert await store.idle_sessions(idle_seconds=1800) == []


async def test_the_new_burst_survives_the_cleanup_too(store):
    """The two twins met: the ``sessions`` row is gone AND the contact wrote again.

    The unstamped turn is the only thing that decides it now, which is the point of moving the
    mark — the old predicate could not tell this case from twin 1 at all, and answered "pick"
    to both. Answering "skip" to both would have been the blind spot traded for the loop.
    """
    await _consolidated_session(store, "sess-both", "t/u2")
    await store.save_turn(TurnRecord("sess-both", "t/u2", 1, "voltei",
                                     created_at=_now() - timedelta(minutes=45)))
    _scope_cleanup_sparing_turns(store, "t/u2")

    assert [s.id for s in await store.idle_sessions(idle_seconds=1800)] == ["sess-both"]


# ── the marker is a CONJUNCT, never a replacement ────────────────────────────────────────

async def test_an_unstamped_backlog_is_not_a_thundering_herd(store):
    """Deploy day: every turn that predates the column reads "not consolidated".

    If the mark had REPLACED the old disjunction, the whole history would have become pickable
    at once and the box would have spent an LLM read per session it had already read. Because
    it is ANDed on, a closed session with nothing newer stays quiet on the old clause alone.

    Mutation: replace the WHERE's disjunction with the mark (i.e. make the mark the only test)
    and this dies.
    """
    await _consolidated_session(store, "sess-old", "t/u3")
    store._consolidated.clear()                  # the pre-migration world: no stamps anywhere

    for _ in range(3):
        assert await store.idle_sessions(idle_seconds=1800) == [], (
            "a closed session with no newer turns must stay quiet on the sessions row alone")


async def test_the_mark_can_only_ever_remove_a_session_from_the_result(store):
    """Stated as a property over the two predicates rather than one scenario: whatever the
    stamps say, the marked result is a SUBSET of the unmarked one. That is what makes "the
    legitimate re-pick is preserved" a claim about the mechanism instead of about examples."""
    for i, mins in enumerate((200, 150, 100, 20)):
        await store.save_turn(TurnRecord(f"s{i}", f"t/u{i}", 0, "q",
                                         created_at=_now() - timedelta(minutes=mins)))
    await store.close_session("s1", summary="done", scope="t/u1")

    with_mark = {s.id for s in await store.idle_sessions(idle_seconds=1800)}
    store._consolidated.clear()
    without_mark = {s.id for s in await store.idle_sessions(idle_seconds=1800)}
    assert with_mark <= without_mark and without_mark, (
        "the mark added a session the old predicate did not select — it is a filter, not a rule")


# ── TWIN 3: the supported path is unchanged ──────────────────────────────────────────────

async def test_purge_scope_behaves_exactly_as_before(store):
    """RTBF through the supported path: everything goes, and nothing is left to re-arm.

    ``delete_identity`` → ``purge_scope`` was never affected by this defect — it deletes
    ``turns`` alongside ``sessions``, which is precisely why it escaped. The mark dies with
    the row it was stamped on (a COLUMN of it, in Postgres), so no bookkeeping outlives the
    contact it describes.
    """
    await _consolidated_session(store, "sess-rtbf", "t/u9")
    await store.save_turn(TurnRecord("sess-other", "t/keep", 0, "hi",
                                     created_at=_now() - timedelta(minutes=45)))

    removed = await store.purge_scope("t/u9")
    assert removed >= 3                                     # session + turn + memory
    assert await store.load_turns("sess-rtbf") == []
    assert await store.get_session("sess-rtbf") is None
    assert not [k for k in store._consolidated if k[0] == "t/u9"], (
        "the mark must not outlive the turns it was stamped on")
    # the neighbour is untouched, and still pickable
    assert [s.id for s in await store.idle_sessions(idle_seconds=1800)] == ["sess-other"]


# ── the rule, made executable on this side of the collision ──────────────────────────────

def test_purge_scope_may_never_drop_sessions_without_turns():
    """The engram half of the rule the two repositories collide over, as a guard that RUNS.

    Prose in one repository rots the day somebody reads the other, so the sentence is written
    in both (``idle_sessions``/``purge_scope`` here,
    ``tests/unit/test_identity_purge_contract.py`` in ``cogno-host``) and pinned by a test on
    each side. This one: no edit to ``purge_scope`` may leave it deleting ``sessions`` while
    sparing ``turns``, because that is the exact shape that re-arms the janitor.

    Pinned on the SOURCE and not on behaviour, deliberately — the behaviour of a correct
    ``purge_scope`` is already covered above, and what this guards against is the next edit
    that "spares the measurement" without knowing what else that spares.

    Mutation: remove ``"turns"`` from the table tuple in
    ``PostgresStore.purge_scope`` and this dies.
    """
    from cogno_engram.adapters.postgres import PostgresStore
    src = inspect.getsource(PostgresStore.purge_scope)
    body = src.split('for table in (', 1)[1].split(')', 1)[0]
    tables = {t.strip().strip('"\'') for t in body.split(',') if t.strip()}
    assert "sessions" in tables, "the tuple stopped naming sessions — re-read this test"
    assert "turns" in tables, (
        "purge_scope deletes `sessions` without deleting `turns`: every scope it touches "
        "becomes permanently re-pickable by `idle_sessions`. Delete both, or neither.")


# ── what the stamp is allowed to cover ───────────────────────────────────────────────────

async def test_close_session_declares_consolidated_only_the_turns_it_covered(store):
    """The watermark, pinned on the MECHANISM because that is where it acts.

    ``close_session`` stamps ``created_at <= ended_at`` and nothing beyond it. This is what
    makes the equivalence claim exact: while the ``sessions`` row exists the mark and the old
    ``t.created_at > s.ended_at`` disjunct select the same turns, so the conjunct can be added
    without changing a single verdict. A turn dated past the close — a backfill or an import
    re-stating history with caller-supplied timestamps, which ``save_turn`` explicitly
    supports — was not read by this consolidation and must not be declared read by it.

    Asserted on the stamp itself rather than on a later pick, deliberately: a future-dated turn
    is not idle yet, so a behavioural twin would have to rewrite the very field the watermark
    compares in order to observe it, and would then be testing the rewrite.

    Mutation: drop ``and turn.created_at <= ended`` from ``InMemoryStore.close_session``
    (in Postgres: ``AND created_at <= %s`` from the stamp UPDATE) and this dies.
    """
    await store.save_turn(TurnRecord("sess-w", "t/u", 0, "covered",
                                     created_at=_now() - timedelta(hours=2)))
    await store.save_turn(TurnRecord("sess-w", "t/u", 1, "dated past the close",
                                     created_at=_now() + timedelta(hours=2)))
    await store.close_session("sess-w", summary="one turn", scope="t/u")

    assert ("t/u", "sess-w", 0) in store._consolidated, "the covered turn was not stamped"
    assert ("t/u", "sess-w", 1) not in store._consolidated, (
        "a turn dated after the close was declared consolidated by it — the consolidation "
        "never read it, and the mark now says nobody needs to")


async def test_closing_one_scope_does_not_stamp_a_colliding_scopes_turns(store):
    """Session ids collide across scopes, and this is not hypothetical: on the live box one
    id spans two scopes today (311 ``(session_id, scope)`` groups over 310 distinct ids).

    ``close_session`` therefore takes the scope from the row it actually closed — Postgres
    reads it back with ``RETURNING`` — and never from the ``session_id`` alone. Stamping by id
    would mark a NEIGHBOUR contact's turns as consolidated, silently ending their Tier-3
    memory: the same cross-scope write the summary guard already refuses, arriving through the
    new column instead.

    What this test does NOT claim: that the *pick* side already separates colliding scopes.
    It does not — ``idle_sessions`` LEFT JOINs ``sessions`` on the id alone, so one scope's
    close currently suppresses the other's group in BOTH adapters. That is a pre-existing
    defect of the join, older than this change and deliberately not touched by it; the
    assertions below are about the STAMP, which this change owns, and about the one downstream
    consequence that follows from the stamp alone.

    Mutation: compare ``turn.scope`` against anything but the CLOSED session's scope in
    ``InMemoryStore.close_session`` (in Postgres: drop ``scope = %s`` from the stamp UPDATE,
    or bind the argument instead of the returned value) and this dies.
    """
    await store.save_turn(TurnRecord("shared-id", "t/mine", 0, "meu",
                                     created_at=_now() - timedelta(minutes=45)))
    await store.save_turn(TurnRecord("shared-id", "t/neighbour", 0, "alheio",
                                     created_at=_now() - timedelta(minutes=45)))

    await store.close_session("shared-id", summary="mine only", scope="t/mine")

    assert ("t/mine", "shared-id", 0) in store._consolidated
    assert ("t/neighbour", "shared-id", 0) not in store._consolidated, (
        "closing one scope's session declared a colliding scope's turns consolidated")
    # The consequence that IS this change's: once the cleanup takes the shared `sessions` row,
    # the mark is all that is left — and it must speak for one scope only. Mine stays quiet
    # because it was consolidated; the neighbour, who never was, comes back.
    _scope_cleanup_sparing_turns(store, "t/mine")
    picked = await store.idle_sessions(idle_seconds=1800)
    assert [s.scope for s in picked] == ["t/neighbour"], (
        "a scope that was never consolidated must not be silenced by its neighbour's close")
