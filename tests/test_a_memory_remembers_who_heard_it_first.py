"""A memory records WHICH PERSONA the contact was talking to — and never re-attributes it.

The product this pins: what a contact told the secretary must not come out of the bookkeeper's
mouth as if the bookkeeper had heard it. The chosen answer is ATTRIBUTION (the memory says where
it came from), not isolation (the memory is hidden), and the reason is measured rather than
preferred — isolating by scope would need a second definition of the scope key, and the identity
purge matches `scope = %s` by EQUALITY, so rows under a persona segment would survive a
right-to-be-forgotten delete.

Two fields, because they answer two questions: `TurnRecord.voiced_by` is WHO SPOKE that turn,
`MemoryRecord.first_heard_by` is WHO THE CONTACT TOLD IT TO FIRST. They diverge on the second
turn, which is why the memory field carries "first" in its name.
"""

from __future__ import annotations


import pytest

from cogno_engram.adapters.in_memory import InMemoryStore
from cogno_engram.hypnos import _parse_memories, _sole_voice, micro_consolidate
from cogno_engram.types import MemoryRecord, TurnRecord

SCOPE = "acme/+5511999"
SECRETARY, BOOKKEEPER = "persona-secretary", "persona-bookkeeper"


def _turn(n: int, *, voice: str, goal: str = "book a slot") -> TurnRecord:
    return TurnRecord(session_id="s1", scope=SCOPE, turn_n=n, user_input="oi",
                      goal=goal, goal_status="NEW", domains=["scheduling"],
                      voiced_by=voice)


# ── the carry-through: without it the field is inert ──────────────────────────
def test_tier1_carries_the_turns_voice_into_every_memory_it_emits():
    """The memories are built FROM the turn, so this is the only place the voice can enter.

    A `first_heard_by` column with nothing writing it is the defect this repo has catalogued as
    a fix born inert: it would ship, read as done, and attribute nothing.
    """
    mems = micro_consolidate(_turn(1, voice=SECRETARY))

    assert mems, "the fixture must actually emit memories, else this asserts nothing"
    assert {m.first_heard_by for m in mems} == {SECRETARY}


def test_a_turn_nobody_attributed_leaves_the_field_blank_rather_than_guessing():
    assert {m.first_heard_by for m in micro_consolidate(_turn(1, voice=""))} == {""}


# ── the design decision: FIRST heard, never re-attributed ─────────────────────
@pytest.mark.asyncio
async def test_the_second_persona_to_hear_a_fact_does_not_take_the_credit():
    """`(scope, category, content)` is the unique key, so the same fact told to two personas is
    ONE row — and a single-valued column can only answer one question about it.

    The answer is the FIRST voice. The alternative that looks like a fix — putting the persona in
    the unique key — fragments the same memory per persona, which is the isolation this design
    rejected, arriving through the back door.
    """
    store = InMemoryStore()
    await store.save_memory(MemoryRecord(SCOPE, "preference", "likes mornings",
                                         first_heard_by=SECRETARY))
    await store.save_memory(MemoryRecord(SCOPE, "preference", "likes mornings",
                                         confidence=0.9, first_heard_by=BOOKKEEPER))

    rows = await store.load_memories(SCOPE)
    assert len(rows) == 1, "same (scope, category, content) must stay one row"
    assert rows[0].first_heard_by == SECRETARY
    assert rows[0].confidence == pytest.approx(0.9), (
        "the OTHER fields must still upsert — a test that only proves nothing changed would "
        "also pass against a store that ignored the second write entirely")


# ── above tier 1 there is no single turn: say unknown, never guess ────────────
def test_a_session_that_changed_persona_attributes_the_memory_to_NEITHER():
    """The LLM tiers summarise a whole session, and a session can switch persona partway.

    Blank is the honest answer. Picking the first or the last voice would put a guess into a
    field readers are meant to trust — and a guess is worse than the "unknown" the column
    already defaults to.
    """
    mixed = [_turn(1, voice=SECRETARY), _turn(2, voice=BOOKKEEPER)]
    assert _sole_voice(mixed) == ""

    uniform = [_turn(1, voice=SECRETARY), _turn(2, voice=SECRETARY)]
    assert _sole_voice(uniform) == SECRETARY, (
        "and the positive half: when the session HAS one voice, it is a fact, not a guess")

    assert _sole_voice([_turn(1, voice="")]) == ""


def test_ONE_unrecorded_turn_is_enough_to_withhold_the_attribution():
    """The MIXED case — some turns carry a voice, some do not — and the one the predicate was
    originally silent about.

    Skipping the blanks would answer `secretary` for a session whose OTHER turn nobody recorded,
    which could as easily have been the bookkeeper. That is a claim about the unrecorded turn, and
    it is the same guess the whole field exists to refuse.

    A predicate that answers the pure case and was never asked about the mixed one is not defined,
    it is UNDEFINED — and the next reader takes the pure answer for the rule. So it is pinned in
    both directions here.
    """
    assert _sole_voice([_turn(1, voice=SECRETARY), _turn(2, voice="")]) == ""
    assert _sole_voice([_turn(1, voice=""), _turn(2, voice=SECRETARY)]) == "", (
        "order must not decide it — a set-based answer that depended on position would be a "
        "second bug wearing the first one's answer")
    assert _sole_voice([]) == ""

    # ...and the positive control, so this cannot pass by refusing everything.
    assert _sole_voice([_turn(1, voice=SECRETARY), _turn(2, voice=SECRETARY)]) == SECRETARY


def test_the_llm_tier_stamps_the_voice_it_was_given():
    """`_sole_voice` decides; `_parse_memories` must actually carry the decision through."""
    out = _parse_memories('{"preference": ["likes mornings"]}', SCOPE, 0.8, SECRETARY)
    assert [m.first_heard_by for m in out] == [SECRETARY]


# ── the two stores must answer the same question ─────────────────────────────
psycopg = pytest.importorskip("psycopg")

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path

from cogno_engram.adapters.postgres import PostgresStore, ensure_schema  # noqa: E402

DSN = resolve_test_dsn()


async def _pg_store():
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    await ensure_schema(conn, embedding_dim=8)
    return conn


@pytest.mark.asyncio
async def test_postgres_keeps_the_first_voice_TOO():
    """The parity assertion, and it is not cosmetic.

    The in-memory upsert and the Postgres `ON CONFLICT` are two hand-written copies of one rule.
    This repo has already measured the two adapters DISAGREEING about the same function, in the
    direction that leaks. If the double kept the second voice while the real store kept the
    first, every test above would pass and say nothing about production.
    """
    conn = await _pg_store()
    store = PostgresStore(dsn=DSN)
    scope = f"{SCOPE}/pg"
    async with conn:
        await conn.execute("DELETE FROM memories WHERE scope = %s", (scope,))
    await store.save_memory(MemoryRecord(scope, "preference", "likes mornings",
                                         first_heard_by=SECRETARY))
    await store.save_memory(MemoryRecord(scope, "preference", "likes mornings",
                                         confidence=0.9, first_heard_by=BOOKKEEPER))
    rows = await store.load_memories(scope)
    assert len(rows) == 1
    assert rows[0].first_heard_by == SECRETARY
    assert rows[0].confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_ensure_schema_ADDS_the_columns_to_a_database_that_predates_them():
    """`CREATE TABLE IF NOT EXISTS` is a NO-OP against a live table.

    A deployment that already has turns and memories would otherwise get the new code and none
    of the columns — the exact hazard the edge-curation migration above it documents. Dropping
    the columns here reproduces that database.
    """
    conn = await _pg_store()
    async with conn:
        for table, column in (("memories", "first_heard_by"), ("turns", "voiced_by")):
            await conn.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS {column}")
        await ensure_schema(conn, embedding_dim=8)
        for table, column in (("memories", "first_heard_by"), ("turns", "voiced_by")):
            cur = await conn.execute(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = %s AND column_name = %s", (table, column))
            assert await cur.fetchone() is not None, f"{table}.{column} was not migrated in"
