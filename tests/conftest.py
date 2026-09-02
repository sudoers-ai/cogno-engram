from __future__ import annotations

import os
import socket
import warnings
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


class UnreachableTestDatabase(UserWarning):
    """The DSN is set but nothing answers there — named once, with the password masked."""


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


def _without_password(dsn: str) -> str:
    """``dsn`` with its password replaced by ``***``, so a skip message can name it.

    Splits the netloc on the LAST ``@`` exactly as ``urlsplit`` itself does, so a password
    that contains one is removed whole rather than half-printed. A DSN carrying no password
    comes back untouched.
    """
    parts = urlsplit(dsn)
    userinfo, at, hostport = parts.netloc.rpartition("@")
    if not at or ":" not in userinfo:
        return dsn
    return urlunsplit(parts._replace(netloc=f"{userinfo.split(':', 1)[0]}:***@{hostport}"))


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


# ── a server that is not there is the ENVIRONMENT, not a failing test ────────────────────
#
# With no ENGRAM_TEST_DSN the DSN-using modules skip: `resolve_test_dsn` probes the fallback
# server and returns "" when nothing answers. An EXPLICIT DSN took no such probe, so a
# variable left over in a shell — from an earlier session, from a container that has since
# stopped — turned the suite red instead. Measured on 432f867 with a DSN aimed at a dead
# port: 22 failed + 8 errors, every one of them `psycopg.OperationalError: connection
# failed`, and identical on `main` and on any branch. On a box running several worktrees at
# once that red reads as a defect in the code under test, and costs whoever sees it a round
# of hunting themselves.
#
# The distinction is narrow ON PURPOSE, and the TCP probe is what keeps it narrow: it opens
# a connection and asks nothing, so it can only ever answer "nothing is listening there".
# A wrong password, an absent database, a missing `vector` extension, a broken query — each
# of those happens on a socket that CONNECTED, so none of them can reach this branch. They
# stay red, which is the point: a guard wide enough to swallow them would be worse than the
# noise it removes.
#
# WHAT it does is blank the resolved DSN on the modules that read it, rather than skip their
# tests wholesale. The two are not the same, and the difference is measurable: marking every
# item of every module that exposes `DSN` also silenced the five parametrisations of
# `test_the_net_catches_legacy_shapes_and_NOTHING_else`, which live in
# `test_postgres_integration.py` but drive a FAKE connection object and pass with no server
# anywhere. Module-level `DSN` is the right unit for the abort above (over-refusing costs a
# run; under-refusing costs a database) and the wrong one here, where over-skipping quietly
# retires passing tests. Each module already knows which of ITS tests need the database —
# that knowledge is its own `if not DSN` gate — so hand it the same "" it would have got
# from an unset variable and let it decide. The suite then reports exactly what it reports
# with no DSN at all, which is what `test_dsn_skip.py` pins in both directions.
#
# The decision is taken during COLLECTION, before any fixture runs — the same reason the
# abort above is: a check that fires once a test is RUNNING has already let pytest reach the
# fixture whose next statement is `DROP TABLE`. `cogno-host` carried that shape until
# 2026-08-26.
def _blank_the_dsn_when_nothing_is_listening(dsn: str, targets: list) -> None:
    """Hand the DSN-using modules an empty DSN when ``dsn``'s server does not answer."""
    if _server_is_listening(dsn):
        return                       # something answers: whatever the tests find is real
    for module in {i.module for i in targets if getattr(i, "module", None) is not None}:
        module.DSN = ""
    warnings.warn(
        f"nothing is listening at {_without_password(dsn)} (from {_DSN_ENV}), so the tests "
        f"that need a database are skipping. This is the ENVIRONMENT, not a defect in the "
        f"code under test: start that server, or unset {_DSN_ENV} and the suite aims at "
        f"{_TEST_DATABASE!r} on whatever local server the shell already points at.",
        UnreachableTestDatabase, stacklevel=1,
    )


def pytest_collection_modifyitems(items) -> None:
    """Decide, per DSN-using test, between running it, refusing it, and skipping it.

    Three outcomes, and none of them collapses into another:

    * the database name does not say "test" → ABORT the whole session, loudly and by name;
    * the name is safe but nothing answers there → SKIP those tests (the environment, not
      the code — see ``_blank_the_dsn_when_nothing_is_listening``);
    * the name is safe and the server answers → do nothing, and whatever the tests find is
      reported as it happens. A failure against a reachable server is a real failure.

    Checks the RESOLVED DSN, not the raw variable: the guard has to inspect the same string
    the fixtures will open, or the two can drift apart and only one of them is checked. The
    unreachable case is decided AFTER the name, so a dangerous DSN aimed at a dead server is
    still refused by name rather than quietly skipped — otherwise the person keeps a live
    DSN in their shell and only finds out once the server is back up.

    Fires on COLLECTION, not on every session: a stale ``ENGRAM_TEST_DSN`` in someone's
    shell must not stop ``pytest tests/test_in_memory_store.py``, which never opens a
    connection. The trigger is a collected test whose module reads the DSN — those are the
    ones that ``DROP TABLE``.
    """
    dsn = resolve_test_dsn()
    if not dsn:
        return                       # nobody named a database; the modules' own skips apply
    targets = [i for i in items if getattr(getattr(i, "module", None), "DSN", None)]
    if not targets:
        return                                   # nothing collected would touch that database
    if names_a_test_database(dsn):
        _blank_the_dsn_when_nothing_is_listening(dsn, targets)
        return
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
