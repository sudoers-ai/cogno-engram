"""A ``dsn=`` store keeps ONE pool of its own instead of a connection per statement.

No database: ``psycopg_pool.AsyncConnectionPool`` and ``psycopg.AsyncConnection.connect`` are
replaced by counting doubles, so what is pinned is the store's PLUMBING — how many pools, built
how, closed when — and the server-side proof (N statements → at most ``max`` backends) is
``test_pool_postgres.py``.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import pytest

psycopg = pytest.importorskip("psycopg")
psycopg_pool = pytest.importorskip("psycopg_pool")
from psycopg.rows import dict_row  # noqa: E402

from cogno_engram.adapters import postgres as pg  # noqa: E402
from cogno_engram.adapters.postgres import (  # noqa: E402
    DEFAULT_POOL_MAX_SIZE,
    DEFAULT_POOL_MIN_SIZE,
    PostgresDocumentStore,
    PostgresKnowledgeGraph,
    PostgresStore,
)

DSN = "postgresql://u:p@127.0.0.1:1/never_test"


class FakeConn:
    def __init__(self, origin: str) -> None:
        self.origin = origin
        self.row_factory = None
        self.executed: list[str] = []
        self.closed = False

    async def execute(self, sql, params=None):
        self.executed.append(sql)

    async def close(self):
        self.closed = True


class FakePool:
    """Counts what the store does with ``psycopg_pool.AsyncConnectionPool``."""
    built: list["FakePool"] = []
    fail_next_open = 0

    def __init__(self, conninfo="", **kw) -> None:
        self.conninfo, self.kw = conninfo, kw
        self.opened = self.closed = 0
        self.borrowed = 0
        FakePool.built.append(self)

    @staticmethod
    async def check_connection(conn) -> None:      # what the store passes as `check=`
        return None

    async def open(self, wait: bool = False) -> None:
        await asyncio.sleep(0)                      # a real yield: concurrent callers interleave
        if FakePool.fail_next_open:
            FakePool.fail_next_open -= 1
            raise psycopg.OperationalError("open failed")
        self.opened += 1

    async def close(self) -> None:
        self.closed += 1

    @asynccontextmanager
    async def connection(self):
        assert self.opened and not self.closed, "borrowed from a pool that is not open"
        self.borrowed += 1
        yield FakeConn("pool")


@pytest.fixture
def doubles(monkeypatch):
    FakePool.built = []
    FakePool.fail_next_open = 0
    monkeypatch.setattr(psycopg_pool, "AsyncConnectionPool", FakePool)
    connects: list[FakeConn] = []

    async def connect(*a, **kw):
        c = FakeConn("fresh")
        connects.append(c)
        return c
    monkeypatch.setattr(psycopg.AsyncConnection, "connect", connect)
    return connects


async def _statements(store, n: int) -> None:
    for _ in range(n):
        async with store._conn() as conn:
            await conn.execute("SELECT 1")


# ── one pool, built like a fresh connection ──────────────────────────────────────────────

@pytest.mark.parametrize("cls", [PostgresStore, PostgresKnowledgeGraph, PostgresDocumentStore])
async def test_n_statements_one_pool_and_no_connection_per_statement(doubles, cls):
    store = cls(dsn=DSN)
    await _statements(store, 25)
    assert len(FakePool.built) == 1
    pool = FakePool.built[0]
    assert (pool.opened, pool.borrowed, pool.closed) == (1, 25, 0)
    assert doubles == []                            # no connection of its own, not one


async def test_the_pool_is_configured_like_the_fresh_connection_it_replaces(doubles):
    await _statements(PostgresStore(dsn=DSN), 1)
    pool = FakePool.built[0]
    assert pool.conninfo == DSN
    assert pool.kw["kwargs"] == {"autocommit": True, "row_factory": dict_row,
                                 "prepare_threshold": None}
    assert pool.kw["check"] is FakePool.check_connection
    assert pool.kw["open"] is False
    assert (pool.kw["min_size"], pool.kw["max_size"]) == (DEFAULT_POOL_MIN_SIZE,
                                                          DEFAULT_POOL_MAX_SIZE)


async def test_sizes_and_timeout_are_the_callers(doubles):
    await _statements(PostgresStore(dsn=DSN, pool_min_size=2, pool_max_size=5,
                                    pool_timeout=7.5), 1)
    kw = FakePool.built[0].kw
    assert (kw["min_size"], kw["max_size"], kw["timeout"]) == (2, 5, 7.5)


async def test_concurrent_first_statements_share_one_opening(doubles):
    store = PostgresStore(dsn=DSN)
    await asyncio.gather(*(_statements(store, 3) for _ in range(20)))
    assert len(FakePool.built) == 1
    assert FakePool.built[0].opened == 1 and FakePool.built[0].borrowed == 60


async def test_a_failed_open_is_retried_by_the_next_statement(doubles):
    store = PostgresStore(dsn=DSN)
    FakePool.fail_next_open = 1
    with pytest.raises(psycopg.OperationalError):
        await _statements(store, 1)
    await _statements(store, 1)
    assert len(FakePool.built) == 2 and FakePool.built[1].borrowed == 1


# ── the off switch: the old path, byte for byte ──────────────────────────────────────────

async def test_pool_max_size_0_is_a_connection_per_statement(doubles):
    store = PostgresStore(dsn=DSN, pool_max_size=0, pool_min_size=0)
    await _statements(store, 7)
    assert FakePool.built == []
    assert len(doubles) == 7 and all(c.closed for c in doubles)
    await store.close()                             # nothing to close, never raises


# ── close ────────────────────────────────────────────────────────────────────────────────

async def test_close_closes_the_pool_once_and_is_idempotent(doubles):
    store = PostgresStore(dsn=DSN)
    await _statements(store, 3)
    await store.close()
    await store.close()
    assert FakePool.built[0].closed == 1


async def test_close_before_any_statement_builds_nothing(doubles):
    await PostgresStore(dsn=DSN).close()
    assert FakePool.built == []


async def test_a_store_used_after_close_opens_a_new_pool(doubles):
    store = PostgresStore(dsn=DSN)
    await _statements(store, 1)
    await store.close()
    await _statements(store, 1)
    assert len(FakePool.built) == 2
    assert FakePool.built[0].closed == 1 and FakePool.built[1].closed == 0


async def test_close_never_touches_the_callers_pool(doubles):
    callers = FakePool()
    await callers.open()
    store = PostgresStore(pool=callers)
    await _statements(store, 4)
    await store.close()
    assert callers.closed == 0 and callers.borrowed == 4
    assert len(FakePool.built) == 1                 # only the caller's: none of the store's own


# ── the session lock keeps a connection of its own ───────────────────────────────────────

async def test_session_lock_on_a_dsn_store_takes_its_own_connection(doubles):
    store = PostgresStore(dsn=DSN)
    async with store.session_lock("acme", "s1"):
        pass
    assert FakePool.built == []                     # the lock never borrowed from the pool
    assert len(doubles) == 1 and doubles[0].closed
    assert [s.split("(")[0] for s in doubles[0].executed] == ["SELECT pg_advisory_lock",
                                                              "SELECT pg_advisory_unlock"]


async def test_session_lock_on_a_callers_pool_stays_on_that_pool(doubles):
    callers = FakePool()
    await callers.open()
    async with PostgresStore(pool=callers).session_lock("acme", "s1"):
        pass
    assert callers.borrowed == 1 and doubles == []


# ── a new event loop gets a new pool ─────────────────────────────────────────────────────

def test_a_store_moved_to_another_loop_opens_a_pool_there(doubles):
    store = PostgresStore(dsn=DSN)
    asyncio.run(_statements(store, 2))
    asyncio.run(_statements(store, 2))
    assert len(FakePool.built) == 2
    assert [p.borrowed for p in FakePool.built] == [2, 2]


# ── refused configurations ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw", [dict(pool_max_size=-1), dict(pool_min_size=-1),
                                dict(pool_min_size=3, pool_max_size=2),
                                dict(pool_timeout=0), dict(pool_timeout=-1.0)])
def test_refused_pool_configurations(kw):
    with pytest.raises(ValueError):
        PostgresStore(dsn=DSN, **kw)


def test_the_module_names_the_defaults():
    assert pg.DEFAULT_POOL_MIN_SIZE <= pg.DEFAULT_POOL_MAX_SIZE
    assert pg.DEFAULT_POOL_MAX_SIZE > 0 and pg.DEFAULT_POOL_TIMEOUT_S > 0
