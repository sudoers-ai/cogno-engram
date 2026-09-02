"""A Postgres that is not there must SKIP the suite, and nothing else may.

With ``ENGRAM_TEST_DSN`` unset the DSN-using modules have always skipped. With it SET but
the server gone — a container that has since stopped, a variable left in a shell from an
earlier session — they went red instead: measured on 432f867, ``22 failed, 355 passed, 82
skipped, 8 errors``, every one of the 30 a ``psycopg.OperationalError: connection failed``,
and identical on ``main`` and on any branch. On a box running several worktrees at once,
a red with no cause is what makes somebody spend a round hunting themselves.

The fix must be narrow in BOTH directions, so each half is pinned here:

* it must not go red — ``test_a_dead_port_skips_instead_of_failing``;
* it must not go quiet — a fix that skipped everything would pass that test on its own, so
  ``test_a_dead_port_reports_exactly_what_no_dsn_reports`` compares it against the control,
  and ``test_a_reachable_server_is_left_alone`` keeps a failure that is NOT a connection
  failure red. That last one is the one that matters: a guard wide enough to swallow a bad
  password, an absent database or a broken query would hide the defects the suite exists to
  find.
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from conftest import (  # the sibling conftest, on pytest's rootdir path
    _server_is_listening,
    _without_password,
    pytest_collection_modifyitems,
)

_ROOT = Path(__file__).resolve().parent.parent

# The modules that carry a module-level DSN: the three that went red, plus
# `test_postgres_integration.py`, which holds tests that need NO server at all and so is
# what makes the "did it over-skip?" half of the comparison bite.
_DSN_MODULES = [
    "tests/test_fold_regressions.py",
    "tests/test_fold_migration.py",
    "tests/test_folding_parity.py",
    "tests/test_postgres_integration.py",
]
_DEAD = "postgresql://postgres:hunter2@127.0.0.1:1/engram_test"   # port 1 answers nowhere
_SUMMARY = re.compile(r"\b(\d+) (passed|failed|skipped|error|errors)\b")


def _env(**overrides: str) -> dict[str, str]:
    """A hermetic environment: no DSN is inherited, and the fallback points nowhere.

    ``PGHOST``/``PGPORT`` are pinned at a dead port because the fallback DSN is otherwise
    built from libpq's defaults — which, in the ``integration`` CI job, name a Postgres that
    IS running. Without this the control would quietly run the destructive suite for real
    and compare two different things.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("ENGRAM_TEST_DSN", "COGNO_PG_DSN")}
    return {**env, "PGHOST": "127.0.0.1", "PGPORT": "1", **overrides}


def _run(*targets: str, env: dict[str, str]) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets],
        cwd=_ROOT, capture_output=True, text=True, env=env,
    )


def _counts(out: str) -> dict[str, int]:
    """``{"passed": 5, "skipped": 96}`` from pytest's own summary line."""
    lines = [ln for ln in out.splitlines() if _SUMMARY.search(ln)]
    assert lines, f"no pytest summary line in:\n{out[-3000:]}"
    counts = {kind: int(n) for n, kind in _SUMMARY.findall(lines[-1])}
    counts["failed"] = counts.get("failed", 0) + counts.get("error", 0) + counts.get("errors", 0)
    return {"passed": counts.get("passed", 0), "skipped": counts.get("skipped", 0),
            "failed": counts["failed"]}


# ── the defect: a DSN whose server is gone ───────────────────────────────────────────────

def test_a_dead_port_skips_instead_of_failing():
    r = _run(*_DSN_MODULES, env=_env(ENGRAM_TEST_DSN=_DEAD))
    got = _counts(r.stdout)
    assert got["failed"] == 0, f"a server that is not there must not go red:\n{r.stdout[-3000:]}"
    assert got["skipped"] > 0, "…and it must actually reach the tests to skip them"
    assert r.returncode == 0, r.stdout[-3000:]


def test_the_message_names_the_dsn_that_failed_and_blames_the_environment():
    r = _run("tests/test_fold_regressions.py", env=_env(ENGRAM_TEST_DSN=_DEAD))
    assert "127.0.0.1:1/engram_test" in r.stdout, r.stdout[-2000:]
    assert "ENGRAM_TEST_DSN" in r.stdout
    assert "ENVIRONMENT, not a defect in the code" in r.stdout, r.stdout[-2000:]


def test_the_message_never_prints_the_password():
    r = _run("tests/test_fold_regressions.py", env=_env(ENGRAM_TEST_DSN=_DEAD))
    assert "hunter2" not in r.stdout, "the skip message printed the password"
    assert "hunter2" not in r.stderr
    assert "postgres:***@" in r.stdout, r.stdout[-2000:]


# ── the control: without it, a fix that skips EVERYTHING passes the test above ───────────

def test_no_dsn_at_all_still_skips():
    r = _run(*_DSN_MODULES, env=_env())
    got = _counts(r.stdout)
    assert got["failed"] == 0, r.stdout[-3000:]
    assert got["skipped"] > 0
    assert r.returncode == 0


def test_a_dead_port_reports_exactly_what_no_dsn_reports():
    """The two situations must be indistinguishable in the RESULT, and only there.

    This is the assertion that catches over-skipping. The first shape of this fix marked
    every item of every module exposing ``DSN`` — and so also retired the five
    parametrisations of ``test_the_net_catches_legacy_shapes_and_NOTHING_else``, which drive
    a fake connection object and pass with no server anywhere: 350 passed / 109 skipped
    against the control's 355 / 104. Comparing counts is what turned that from an invisible
    regression into a red line.
    """
    dead = _counts(_run(*_DSN_MODULES, env=_env(ENGRAM_TEST_DSN=_DEAD)).stdout)
    none = _counts(_run(*_DSN_MODULES, env=_env()).stdout)
    assert dead == none, f"dead-port run {dead} differs from the no-DSN control {none}"
    assert none["passed"] > 0, "the control passes nothing — it would accept any fix at all"


def test_the_two_situations_are_still_told_apart():
    """Same result, different diagnosis: only the set DSN is worth naming back."""
    dead = _run("tests/test_fold_regressions.py", env=_env(ENGRAM_TEST_DSN=_DEAD))
    none = _run("tests/test_fold_regressions.py", env=_env())
    assert "nothing is listening at" in dead.stdout
    assert "nothing is listening at" not in none.stdout, none.stdout[-2000:]


# ── the line that must NOT move: a reachable server keeps its failures ───────────────────

@contextmanager
def _a_socket_that_answers_but_is_not_postgres():
    """Accepts a TCP connection and closes it — reachable, and useless to a client.

    Stands in for every failure that is not "nothing is listening": a wrong password, an
    absent database, a missing ``vector`` extension, a broken query. All of them happen on a
    socket that CONNECTED, which is precisely why the guard's TCP probe cannot reach them.
    """
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    srv.settimeout(0.2)
    stop = threading.Event()

    def serve() -> None:
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        yield srv.getsockname()[1]
    finally:
        stop.set()
        thread.join(timeout=3)
        srv.close()


def test_a_reachable_server_is_left_alone():
    """A failure that is not a connection failure stays RED. The whole point of the fix.

    Run against a socket that answers, ``test_fold_regressions.py`` fails on the protocol —
    not on ``connection refused``. If this ever goes green, the guard has grown wide enough
    to swallow real defects and the suite has stopped being able to fail.
    """
    with _a_socket_that_answers_but_is_not_postgres() as port:
        dsn = f"postgresql://postgres:hunter2@127.0.0.1:{port}/engram_test"
        assert _server_is_listening(dsn), "precondition: the guard's own probe sees it"
        r = _run("tests/test_fold_regressions.py", env=_env(ENGRAM_TEST_DSN=dsn))
    assert _counts(r.stdout)["failed"] > 0, f"a real failure was swallowed:\n{r.stdout[-3000:]}"
    assert r.returncode != 0
    assert "nothing is listening at" not in r.stdout, "it was skipped as unreachable instead"


class _FakeModule:
    def __init__(self, dsn: str) -> None:
        self.DSN = dsn


class _FakeItem:
    """What the collection hook sees: something with a ``.module`` that exposes a ``DSN``."""

    def __init__(self, module: _FakeModule) -> None:
        self.module = module

    def add_marker(self, marker) -> None:       # pragma: no cover — nothing should mark us
        raise AssertionError(f"the guard marked an item instead of blanking its DSN: {marker}")


def test_the_guard_reads_the_server_and_not_the_name(monkeypatch):
    """The same rule at the seam, without a subprocess: reachable ⇒ the DSN is left intact.

    Both DSNs below name ``engram_test`` on ``127.0.0.1``, so nothing but whether something
    ANSWERS separates them — which is the property the fix rests on.
    """
    with _a_socket_that_answers_but_is_not_postgres() as port:
        dsn = f"postgresql://postgres:pw@127.0.0.1:{port}/engram_test"
        monkeypatch.setenv("ENGRAM_TEST_DSN", dsn)
        module = _FakeModule(dsn)
        pytest_collection_modifyitems([_FakeItem(module)])
        assert module.DSN == dsn, "a reachable server had its tests silenced"

    monkeypatch.setenv("ENGRAM_TEST_DSN", _DEAD)
    module = _FakeModule(_DEAD)
    with pytest.warns(UserWarning, match="nothing is listening"):
        pytest_collection_modifyitems([_FakeItem(module)])
    assert module.DSN == "", "an unreachable server did not get blanked"


# ── it did not weaken the guard it sits next to ──────────────────────────────────────────

def test_a_dangerous_dsn_is_still_refused_even_when_its_server_is_dead():
    """Order matters: the NAME is judged before reachability.

    Nothing can be dropped on a server that is down, so skipping would be safe for this run
    — and wrong for the next one. The person still has a live DSN in their shell, and the
    only moment they can learn that is now, while the abort can still say the name out loud.
    """
    r = _run("tests/test_postgres_integration.py",
             env=_env(ENGRAM_TEST_DSN="postgresql://x:y@127.0.0.1:1/cogno"))
    assert "refusing to run" in r.stdout, r.stdout[-2000:]
    assert "'cogno'" in r.stdout, "the abort must NAME the database it refused"
    assert r.returncode != 0


# ── masking ──────────────────────────────────────────────────────────────────────────────

def test_the_password_is_removed_whole_not_by_prefix():
    assert _without_password("postgresql://u:pw@h:5432/db") == "postgresql://u:***@h:5432/db"
    # a password containing '@': urlsplit splits on the LAST one, and so must the mask
    assert _without_password("postgresql://u:p@ss@h:5432/db") == "postgresql://u:***@h:5432/db"
    assert "s3cr3t" not in _without_password("postgresql://u:s3cr3t@h/db")


def test_a_dsn_with_nothing_to_mask_survives_unchanged():
    for dsn in ("postgresql://h:5432/db", "postgresql://user@h/db", "postgresql:///db"):
        assert _without_password(dsn) == dsn
