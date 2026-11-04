"""A ``dsn=`` store holds AT MOST ``pool_max_size`` server connections — counted by the SERVER.

``test_pg_own_pool.py`` pins the plumbing with doubles; this file asks Postgres itself, through
``pg_stat_activity`` and ``pg_backend_pid()``, what a store really opened:

* the twin — N statements (sequential and concurrent) on a pooled store run on at most
  ``pool_max_size`` backends, and the server never shows more than that for the store;
* the control — the same N statements with ``pool_max_size=0`` (the old path) run on N
  backends, one per statement: the count discriminates, it is not a constant;
* ``close()`` — after it, the server shows NO backend of the store; before it, at least
  ``pool_min_size``;
* the session lock is NOT one of the pool's: held, it is a backend of its own, and the pool's
  statements still run beside it.

Each store marks its connections with an ``application_name`` of its own (in the DSN), so the
count is the store's and nobody else's on the server. Gated like the rest of the Postgres
suite: it aims at ``engram_test`` by itself and skips when nothing is listening. Every name
here is invented.
"""

from __future__ import annotations

import asyncio
import uuid
from urllib.parse import urlsplit, urlunsplit

import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("psycopg_pool")

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path

from cogno_engram.adapters.postgres import PostgresStore  # noqa: E402

DSN = resolve_test_dsn()
N = 40
MAX = 3



@pytest.fixture(autouse=True)
def _needs_a_database():
    # read at RUN time: the conftest blanks this module's `DSN` when nothing listens there
    if not DSN:
        pytest.skip("set ENGRAM_TEST_DSN to a reachable Postgres to run")


def _named(dsn: str, app: str) -> str:
    """``dsn`` with an ``application_name`` that marks THIS store's connections."""
    parts = urlsplit(dsn)
    query = "&".join(q for q in (parts.query, f"application_name={app}") if q)
    return urlunsplit(parts._replace(query=query))


async def _backends(app: str) -> int:
    """How many server backends carry ``app`` right now — asked on a connection of its own."""
    async with await psycopg.AsyncConnection.connect(DSN, autocommit=True) as conn:
        cur = await conn.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE application_name = %s", (app,))
        row = await cur.fetchone()
        return int(row[0])


async def _settled(app: str, want: int) -> int:
    """The backend count once the server has caught up (a closed connection leaves
    ``pg_stat_activity`` a moment after the client let go)."""
    seen = await _backends(app)
    for _ in range(50):
        if seen == want:
            break
        await asyncio.sleep(0.1)
        seen = await _backends(app)
    return seen


async def _pid(store: PostgresStore) -> "tuple[int, str]":
    """One statement on the store: the backend that ran it (pid + its start, so a reused pid of
    a later backend is still another backend)."""
    async with store._conn() as conn:
        cur = await conn.execute(
            "SELECT pid, backend_start::text AS started FROM pg_stat_activity "
            "WHERE pid = pg_backend_pid()")
        row = await cur.fetchone()
        return int(row["pid"]), str(row["started"])


async def _run(store: PostgresStore, app: str) -> "tuple[set, int]":
    """N statements, half one after another and half at once; the distinct backends that ran
    them, and the most the server showed for ``app`` while they ran."""
    seen: set = set()
    peak = 0
    for _ in range(N // 2):
        seen.add(await _pid(store))
        peak = max(peak, await _backends(app))
    got = await asyncio.gather(*(_pid(store) for _ in range(N // 2)))
    seen.update(got)
    peak = max(peak, await _backends(app))
    return seen, peak


async def test_n_statements_run_on_at_most_max_backends_and_close_frees_them():
    app = f"engram_pool_{uuid.uuid4().hex[:10]}"
    store = PostgresStore(dsn=_named(DSN, app), pool_min_size=1, pool_max_size=MAX)
    try:
        seen, peak = await _run(store, app)
        assert 1 <= len(seen) <= MAX, seen
        assert 1 <= peak <= MAX, peak
        assert await _backends(app) >= 1          # min_size: the pool keeps one open
        # held here so only close() can free its connections: an unreferenced pool is also
        # reclaimed by the garbage collector, which would let a close() that closes nothing pass
        held = store._own_pool
        assert held is not None
    finally:
        await store.close()
    assert await _settled(app, 0) == 0            # close() gave every backend back
    del held


async def test_CONTROL_without_the_pool_every_statement_is_a_backend_of_its_own():
    app = f"engram_nopool_{uuid.uuid4().hex[:10]}"
    store = PostgresStore(dsn=_named(DSN, app), pool_min_size=0, pool_max_size=0)
    seen, _peak = await _run(store, app)
    assert len(seen) == N, len(seen)              # one backend per statement: the old path
    assert await _settled(app, 0) == 0            # and none of them stays open
    await store.close()


async def test_the_session_lock_is_a_backend_beside_the_pool():
    app = f"engram_lock_{uuid.uuid4().hex[:10]}"
    store = PostgresStore(dsn=_named(DSN, app), pool_min_size=1, pool_max_size=1)
    try:
        await _pid(store)                          # the pool's one connection exists
        async with store.session_lock("t-inventado/x", "s-inventada"):
            assert await _settled(app, 2) == 2    # the pool's + the lock's own
            await _pid(store)                      # a max=1 pool still serves under the lock
        assert await _settled(app, 1) == 1        # the lock's connection is gone with it
    finally:
        await store.close()
    assert await _settled(app, 0) == 0
