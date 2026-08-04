import os
from urllib.parse import urlsplit

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


def names_a_test_database(dsn: str) -> bool:
    """Whether ``dsn`` targets a database safe to DROP TABLE in.

    Matches on the database NAME only. Matching the whole DSN would accept
    ``postgres:test@host/cogno`` — "test" in the password, production in the path.
    """
    return _TEST_DB_MARKER in urlsplit(dsn).path.lstrip("/").lower()


def pytest_collection_modifyitems(items) -> None:
    """Abort when a DSN-using test is about to run against a non-test database.

    Fires on COLLECTION, not on every session: a stale ``ENGRAM_TEST_DSN`` in someone's
    shell must not stop ``pytest tests/test_in_memory_store.py``, which never opens a
    connection. The trigger is a collected test whose module reads the DSN — those are the
    ones that ``DROP TABLE``.

    Unset DSN is the normal case: those modules skip on their own and nothing runs.
    """
    dsn = os.environ.get("ENGRAM_TEST_DSN", "").strip()
    if not dsn or names_a_test_database(dsn):
        return
    if not any(getattr(getattr(i, "module", None), "DSN", None) for i in items):
        return                                   # nothing collected would touch that database
    database = urlsplit(dsn).path.lstrip("/")
    pytest.exit(
        f"refusing to run: ENGRAM_TEST_DSN points at database {database!r}, which is not a "
        f"test database (its name must contain {_TEST_DB_MARKER!r}). These tests DROP TABLE "
        f"and recreate the engram tables with a tiny embedding dim — running them here would "
        f"destroy real data AND leave the schema unusable by a real embedder. Create a "
        f"throwaway database (e.g. engram_test) and point ENGRAM_TEST_DSN at that instead.",
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
