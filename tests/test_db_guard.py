"""The destructive-suite guard: what it refuses, and what it must NOT get in the way of.

``tests/conftest.py`` refuses to let a DSN-using test run against a database whose name does
not say "test". The abort itself ends the session, so it is exercised here in a subprocess;
the predicate it delegates to is checked directly.

The case that motivated it: on 2026-08-04 the Postgres suite ran with ``ENGRAM_TEST_DSN``
pointing at the live demo database ``.../cogno``. It dropped ``memories``,
``knowledge_nodes`` and ``knowledge_edges`` and recreated them with a ``vector(8)``
embedding column against a 768-dimension embedder — the memory layer stopped accepting
writes at all, which is worse than the row loss.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from conftest import names_a_test_database  # the sibling conftest, on pytest's rootdir path

_ROOT = Path(__file__).resolve().parent.parent
_LIVE_DSN = "postgresql://x:y@localhost:5432/cogno"


# ── which databases the predicate accepts ────────────────────────────────────────────────

def test_refuses_the_live_database_name():
    # the exact DSN shape that did the damage
    assert not names_a_test_database("postgresql://postgres:pw@localhost:55435/cogno")


def test_refuses_when_only_the_credentials_say_test():
    # "test" is in the DSN but the DATABASE is live — matching the whole string would
    # have let exactly this through.
    assert not names_a_test_database("postgresql://postgres:test@localhost:55432/cogno")


def test_refuses_the_default_postgres_database():
    # `postgres` exists on every server, production ones included. The old CI DSN and the
    # module docstring both used it; both now say engram_test.
    assert not names_a_test_database("postgresql://postgres:postgres@localhost:5432/postgres")


def test_accepts_a_throwaway_database():
    assert names_a_test_database("postgresql://postgres:postgres@localhost:5432/engram_test")
    assert names_a_test_database("postgresql://u:p@h:5432/TEST_db")


# ── blast radius: it aborts the dangerous run, not every run ─────────────────────────────
#
# A stale ENGRAM_TEST_DSN in someone's shell must not stop
# ``pytest tests/test_in_memory_store.py``, which never opens a connection. A guard annoying
# enough to be worked around is a guard that stops guarding — so the trigger is a COLLECTED
# test whose module reads the DSN, not the mere presence of the variable.

def _run(target: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider"],
        cwd=_ROOT, capture_output=True, text=True,
        env={**os.environ, "ENGRAM_TEST_DSN": _LIVE_DSN},
    )


def test_a_live_dsn_does_not_block_the_in_memory_tests():
    r = _run("tests/test_in_memory_store.py")
    assert r.returncode == 0, r.stdout[-2000:]
    assert "refusing to run" not in r.stdout


def test_a_live_dsn_does_block_the_destructive_suite():
    r = _run("tests/test_postgres_integration.py")
    assert r.returncode != 0
    assert "refusing to run" in r.stdout
