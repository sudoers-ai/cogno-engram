from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from functools import lru_cache
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import pytest

from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore

# ── safety gate on the database the destructive suites target ────────────────────────────
#
# `test_postgres_integration.py` DROPs and recreates the engram tables, and recreates them
# with a TINY embedding dim so runs are deterministic. Against a throwaway database that is
# exactly right. Against a live one it is worse than data loss: the tables come back with a
# `vector(8)` embedding column facing a 768-dimension embedder, so the memory layer stops
# accepting writes entirely.
#
# That happened on 2026-08-04 — this suite ran with ENGRAM_TEST_DSN pointing at the demo
# box's live database and took out `memories`, `knowledge_nodes` and `knowledge_edges`.
# `cogno-host` already had this guard and was the only repo in the batch that refused.
#
# A database whose name says "test" is one someone is willing to lose. Anything else is
# assumed real until proven otherwise — a deployment is never worth a green test run.
_TEST_DB_MARKER = "test"
_DSN_ENV = "ENGRAM_TEST_DSN"


def names_a_test_database(dsn: str) -> bool:
    """Whether ``dsn`` targets a database safe to DROP TABLE in.

    Matches on the database NAME only. Matching the whole DSN would accept
    ``postgres:test@host/cogno`` — "test" in the password, production in the path.
    """
    return _TEST_DB_MARKER in urlsplit(dsn).path.lstrip("/").lower()


# ── the disposable database is the DEFAULT, not something you have to remember ───────────
#
# The guard above turns the 2026-08-04 mistake into a refusal, but it still leaves the
# person to TYPE a DSN — and the shape that did the damage is the one a dev shell hands
# you: `COGNO_PG_DSN` is already exported, and copying it into ENGRAM_TEST_DSN is one
# keystroke. So stop asking. With ENGRAM_TEST_DSN unset the suite now aims at
# `engram_test` on whatever LOCAL server the shell already points at.
#
# The database name is the part that is never carried across: `_for_test_database`
# OVERWRITES it. Handing this function the exact DSN that caused the outage returns the
# disposable one — which is the property `tests/test_db_guard.py` pins, in both directions.
_TEST_DATABASE = "engram_test"

# Only a LOCAL server is adopted implicitly. A `COGNO_PG_DSN` pointing at a managed cloud
# instance is somebody's production server, and `engram_test` is not ours to create there.
_LOCAL_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1", "0.0.0.0"})

_PROBE_TIMEOUT_S = 0.5


def _for_test_database(dsn: str) -> str:
    """``dsn`` with its database name REPLACED by this repo's disposable one."""
    return urlunsplit(urlsplit(dsn)._replace(path=f"/{_TEST_DATABASE}"))


def default_test_dsn(env: Mapping[str, str] | None = None) -> str:
    """Where the DSN-using tests go when nobody names a database.

    Server and credentials come from the ambient `COGNO_PG_DSN` when it is local, else from
    libpq's own `PG*` variables (whose defaults are what CI's postgres service container
    serves). The database is always `_TEST_DATABASE` — there is no input that changes it.
    """
    env = os.environ if env is None else env
    ambient = (env.get("COGNO_PG_DSN") or "").strip()
    if ambient and (urlsplit(ambient).hostname or "") in _LOCAL_HOSTS:
        return _for_test_database(ambient)
    host = env.get("PGHOST") or "localhost"
    return (f"postgresql://{env.get('PGUSER') or 'postgres'}:"
            f"{env.get('PGPASSWORD') or 'postgres'}@"
            f"{quote(host, safe='') if host.startswith('/') else host}:"
            f"{env.get('PGPORT') or '5432'}/{_TEST_DATABASE}")


@lru_cache(maxsize=None)
def _server_is_listening(dsn: str) -> bool:
    """Whether something answers at ``dsn``'s address — a TCP probe, no query, no auth.

    Keeps the old ergonomics: with no server around, the modules skip exactly as they did
    when the variable was simply unset. A box with no Postgres must not go red.
    """
    parts = urlsplit(dsn)
    host = unquote(parts.hostname or "localhost")
    if host.startswith("/"):
        return os.path.exists(host)
    try:
        with socket.create_connection((host, parts.port or 5432), _PROBE_TIMEOUT_S):
            return True
    except OSError:
        return False


def resolve_test_dsn(env: Mapping[str, str] | None = None) -> str:
    """The DSN the DSN-using tests run against; ``""`` means skip.

    An explicit ENGRAM_TEST_DSN wins — including a dangerous one, which is the whole point:
    it must reach the collection guard by name rather than be silently corrected.
    """
    env = os.environ if env is None else env
    explicit = (env.get(_DSN_ENV) or "").strip()
    if explicit:
        return explicit
    fallback = default_test_dsn(env)
    return fallback if _server_is_listening(fallback) else ""


def pytest_collection_modifyitems(items) -> None:
    """Abort when a DSN-using test is about to run against a non-test database.

    Checks the RESOLVED DSN, not the raw variable: the guard has to inspect the same string
    the fixtures will open, or the two can drift apart and only one of them is checked.

    Fires on COLLECTION, not on every session: a stale ``ENGRAM_TEST_DSN`` in someone's
    shell must not stop ``pytest tests/test_in_memory_store.py``, which never opens a
    connection. The trigger is a collected test whose module reads the DSN — those are the
    ones that ``DROP TABLE``.
    """
    dsn = resolve_test_dsn()
    if not dsn or names_a_test_database(dsn):
        return
    if not any(getattr(getattr(i, "module", None), "DSN", None) for i in items):
        return                                   # nothing collected would touch that database
    database = urlsplit(dsn).path.lstrip("/")
    pytest.exit(
        f"refusing to run: {_DSN_ENV} points at database {database!r}, which is not a "
        f"test database (its name must contain {_TEST_DB_MARKER!r}). These tests DROP TABLE "
        f"and recreate the engram tables with a tiny embedding dim — running them here would "
        f"destroy real data AND leave the schema unusable by a real embedder. Unset "
        f"{_DSN_ENV} and the suite aims at {_TEST_DATABASE!r} on the same server by itself.",
        returncode=pytest.ExitCode.USAGE_ERROR,
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def buffer() -> InMemoryBuffer:
    return InMemoryBuffer()


@pytest.fixture
def graph() -> InMemoryGraph:
    return InMemoryGraph()
