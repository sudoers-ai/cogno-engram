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
import re
import subprocess
import sys
from pathlib import Path

from conftest import (  # the sibling conftest, on pytest's rootdir path
    _TEST_DATABASE,
    default_test_dsn,
    names_a_test_database,
    resolve_test_dsn,
)

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

def _run(target: str, *extra: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "pytest", target, "-q", "-p", "no:cacheprovider", *extra],
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
    assert "cogno" in r.stdout, "the abort must NAME the database it refused"


def test_it_refuses_during_COLLECTION_and_not_once_a_test_is_running():
    """``--collect-only`` opens no connection — so if the abort still fires, it fired first.

    The distinction is the whole guard: a check that runs inside a fixture has already let
    ``pytest`` reach the point where the next statement is ``DROP TABLE``. ``cogno-host``
    carried exactly that shape (a session-scoped autouse fixture) until 2026-08-26, and it
    is invisible to any assertion that only looks at the exit code of a full run.
    """
    r = _run("tests/test_postgres_integration.py", "--collect-only")
    assert "refusing to run" in r.stdout, r.stdout[-1500:]
    assert r.returncode != 0


# ── the convention the guard rests on ────────────────────────────────────────────────────
#
# The trigger is a collected test whose module exposes a module-level ``DSN``. That makes the
# guard fail OPEN for a future module that reads ENGRAM_TEST_DSN some other way — it would
# connect, DROP, and never trip the abort. Assert the convention instead of trusting it.

def test_every_module_reading_the_dsn_exposes_it_as_a_module_attribute():
    offenders, scanned = [], []
    for f in sorted((_ROOT / "tests").glob("test_*.py")):
        src = f.read_text()
        if f.name == Path(__file__).name:
            continue
        if "ENGRAM_TEST_DSN" not in src and "resolve_test_dsn" not in src:
            continue
        scanned.append(f.name)
        if not re.search(r"^DSN\s*=", src, re.M):
            offenders.append(f.name)
    # A scan that matches nothing passes for free. Since the modules stopped naming the
    # variable directly (they call `resolve_test_dsn` now), the match term is the thing
    # that can silently go stale — so assert it still finds them.
    assert len(scanned) >= 5, f"the scan matched only {scanned} — it has stopped seeing the suites"
    assert not offenders, (
        f"{offenders} read ENGRAM_TEST_DSN but expose no module-level `DSN`, so "
        f"pytest_collection_modifyitems in conftest.py cannot see them and would let a "
        f"DROP TABLE run against a live database."
    )


# ── the disposable database is the DEFAULT ───────────────────────────────────────────────
#
# Owner, 2026-08-26: "Já temos um test só para os testes de integração, isso deveria ser
# padrão." The guard turned the 2026-08-04 mistake into a refusal; this turns it into
# something nobody has to remember. What makes it safe is not a check but a construction:
# the database name is never carried in from anywhere, it is written.

def test_the_default_is_always_the_disposable_database():
    # the very DSN that caused the outage, handed in as the ambient one
    got = default_test_dsn({"COGNO_PG_DSN": "postgresql://postgres:pw@localhost:55435/cogno"})
    assert got == "postgresql://postgres:pw@localhost:55435/engram_test"
    assert names_a_test_database(got)


def test_the_default_keeps_the_server_and_credentials_it_was_given():
    # host, port, user and password ride across — only the database is replaced. Otherwise
    # "the default" would mean "a server nobody is running", i.e. a permanent skip.
    got = default_test_dsn({"COGNO_PG_DSN": "postgresql://u:p@localhost:6000/cogno?sslmode=require"})
    assert got.startswith("postgresql://u:p@localhost:6000/engram_test")


def test_no_ambient_dsn_falls_back_to_libqp_defaults_which_is_what_ci_serves():
    assert default_test_dsn({}) == "postgresql://postgres:postgres@localhost:5432/engram_test"
    assert default_test_dsn({"PGHOST": "db", "PGPORT": "6543", "PGUSER": "u", "PGPASSWORD": "s"}) \
        == "postgresql://u:s@db:6543/engram_test"


def test_a_REMOTE_ambient_dsn_is_not_adopted():
    """`engram_test` on someone's managed instance is not ours to create, let alone DROP."""
    got = default_test_dsn({"COGNO_PG_DSN": "postgresql://u:p@db.prod.example.com:5432/cogno"})
    assert "prod.example.com" not in got
    assert got == "postgresql://postgres:postgres@localhost:5432/engram_test"


def test_every_default_names_a_test_database_whatever_the_environment_says():
    for env in ({}, {"COGNO_PG_DSN": _LIVE_DSN}, {"COGNO_PG_DSN": "postgresql:///cogno"},
                {"PGHOST": "h", "COGNO_PG_DSN": "postgresql://a:b@127.0.0.1/postgres"}):
        assert names_a_test_database(default_test_dsn(env)), env


def test_the_marker_lives_in_the_name_the_default_uses():
    # `_TEST_DATABASE` and `names_a_test_database` are two halves of one rule; if the
    # constant were ever renamed to something without "test", the default would build a
    # DSN its own guard refuses.
    assert names_a_test_database(f"postgresql://h/{_TEST_DATABASE}")


def test_an_explicit_dsn_still_wins_including_a_dangerous_one():
    # It must NOT be quietly corrected: the collection guard has to see it and say the name
    # out loud, or the person keeps a live DSN in their shell and never learns.
    assert resolve_test_dsn({"ENGRAM_TEST_DSN": _LIVE_DSN}) == _LIVE_DSN


def test_nothing_listening_means_skip_exactly_as_before():
    # port 1 answers nowhere; the modules' own `skipif(not DSN)` then does what it always did
    assert resolve_test_dsn({"COGNO_PG_DSN": "postgresql://u:p@127.0.0.1:1/cogno"}) == ""
