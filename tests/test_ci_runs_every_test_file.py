"""Every versioned test file must sit UNDER a path that some CI job hands to pytest.

A test file no workflow reaches is not a weaker test — it is not a test. It was written,
reviewed, merged, and it has never executed anywhere automated; it is green only in the
reviewer's memory. A sweep across this ecosystem on 2026-09-02 counted six such files and 56
node ids in ``cogno-anima``; ``cogno-praxis`` had carried a bookkeeper suite the same way until
its #90.

The mechanism is never exotic: CI ENUMERATES what to run, by name, and the enumeration goes
stale in silence. Usually the names are the children of ``tests/`` (``pytest tests/unit``,
``pytest tests/integration``), so the day a third child appears the workflow keeps passing and
the new directory runs nowhere.

**This repo has already been bitten by it, one level finer.** Until #37 the integration job
read ``pytest tests/test_postgres_integration.py tests/test_redis_integration.py -q`` — a list
of FILES — and that list had gone stale: ``test_folding_parity.py`` and
``test_fold_migration.py`` existed, passed on the machine of whoever wrote them, and had never
run in CI. #37 replaced the list with a bare ``pytest -q``, and that is why all three of this
repo's pytest steps name no path at all today: the ``test`` job's
``pytest -q --cov=cogno_engram …``, the integration job's ``pytest -q``, and the collation-C
step's ``ENGRAM_TEST_DSN=… pytest -q``. Each is scoped by ``testpaths = ["tests"]`` in
``pyproject.toml``, and ``tests`` is an ancestor of every test module in the tree.

So ``cogno-engram`` is the ecosystem's REFERENCE form: immune by construction, because there
is no enumeration left to go stale. What this file adds is that the immunity stops being a
property of the current text of ``ci.yml`` — a rule someone has to remember — and becomes a
rule that RUNS. The day somebody "optimises" these steps back into
``pytest tests/unit && pytest tests/integration``, something says so. It is also the control
for the guard itself: a check that is red everywhere proves nothing, and this is the repo where
it must be green.

**The invariant: the path CI invokes is an ANCESTOR of every test file it should collect.**

Four things this had to get right, each of them a way the check could have been born useless:

* it enumerates with ``git ls-files``, not a walk of the disk. An uncommitted file is not a
  contract, and a scratch copy under ``tests/`` is not something CI ever promised to run;
* a CONDITIONAL job still counts as an invocation. A suite gated on ``schedule`` — assertions
  against a live model must not decide whether main is green — is a real answer, not a
  violation. WHICH paths are reached only that way is a second, weaker, separate assertion at
  the bottom of this file. Here that set is EMPTY, and empty is itself the claim: every test
  file in this repo gates a pull request;
* the option table is the false-GREEN surface. ``pytest --rootdir tests`` collects nothing
  named ``tests``, but a parser that does not know ``--rootdir`` takes a value reads the next
  token as a path and pronounces the whole suite covered. Unknown flags are therefore assumed
  to take NO value (the ``-q``/``-x``/``-s`` case) and every flag that could swallow a
  path-shaped token is listed in ``_TAKES_VALUE`` and pinned one-by-one by a test below;
* ``testpaths`` is read from pytest itself, not re-parsed out of ``pyproject.toml``. A second
  parser is a second answer, and ``tomllib`` does not exist on the 3.10 leg of this matrix.
  Since every invocation here resolves through that fallback, a wrong reading of it would not
  weaken this guard — it would switch it off entirely, so it is asserted directly.

Parsed with a real YAML parser and a real shell lexer: a regex over ``ci.yml`` would be a third
transcription of a format two libraries already read correctly.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path, PurePosixPath

import yaml

# `tests/` is FLAT in this repo — no `unit`/`integration` children — so the repo root is two
# levels up from this module, not three. Getting this number wrong points the guard at a
# directory with no `.github` and no git index, where it reads nothing and passes; the first
# assertion in "guards on the guard" below is what makes that impossible.
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

# Paths whose ONLY invocation comes from a conditional job or step. EMPTY here, and that is the
# measured state, not an unfilled placeholder: `ci.yml` triggers on `pull_request`, neither of
# its two jobs carries an `if:`, and neither does any of their steps — so every test file in
# this repo gates a pull request. A non-empty entry would be a decision that has to keep being
# made; see the second assertion at the bottom.
NIGHTLY_ONLY: "frozenset[str]" = frozenset()

# Options that consume the NEXT token. A missing entry here is the one way this guard goes
# falsely green: the swallowed token reads as a collection path, and `--rootdir tests` would
# then claim the whole suite is invoked. The `--flag=value` spellings need no entry, which is
# how `--cov=cogno_engram` in this repo's ci.yml is read.
_TAKES_VALUE = frozenset({
    "-k", "-m", "-p", "-n", "-c", "-o", "-W", "-r", "--maxfail", "--rootdir", "--deselect",
    "--ignore", "--ignore-glob", "--confcutdir", "--import-mode", "--basetemp", "--junitxml",
    "--junit-xml", "--log-file", "--numprocesses", "--dist", "--cov", "--cov-report",
    "--cov-config", "--cov-fail-under", "--timeout", "--durations", "--tb", "--color",
    "--capture",
})
# Shell tokens that end one command and begin another.
_SEPARATORS = frozenset({"|", "||", "&", "&&", ";", "(", ")", "<", ">", ">>", "|&", "&>"})
# Wrappers that may sit in front of `pytest` on the same command line.
_WRAPPERS = frozenset({"sudo", "env", "time", "nice", "xvfb-run", "poetry", "uv", "hatch",
                       "pdm", "rye", "run", "coverage", "python", "python3", "-m"})
_PYTEST = frozenset({"pytest", "py.test"})


# ── reading the workflows ─────────────────────────────────────────────────────────────
def _strip_heredocs(text: str) -> str:
    """Drop ``<<EOF ... EOF`` bodies. No workflow here embeds one today; the helper exists so
    that the day one does — a python or psql probe, the natural next thing to add beside the
    collation-C step — its body is not lexed as shell, which is noise at best and an
    unbalanced quote at worst."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        if "<<" not in line:
            continue
        tail = line.split("<<", 1)[1].lstrip("-").strip()
        if not tail:
            continue
        marker = tail.split()[0].strip("\"'")
        while i < len(lines) and lines[i].strip() != marker:
            i += 1
        i += 1                                    # the terminator line itself
    return "\n".join(out)


def _commands(run_text: str) -> "list[list[str]]":
    """A ``run:`` block, split into argv lists. A newline separates commands as surely as a
    ``;`` does, so a ``pytest`` on one line cannot absorb the arguments of the
    ``ruff check ... tests`` on the next."""
    text = _strip_heredocs(run_text).replace("\\\n", " ")
    cmds: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        lex = shlex.shlex(line, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        try:
            tokens = list(lex)
        except ValueError:
            # Only silence a line that cannot be a pytest invocation at all — a guard that
            # quietly skips its own subject is the state it exists to end.
            assert "pytest" not in line, f"unparseable run: line that mentions pytest: {line!r}"
            continue
        current: list[str] = []
        for token in tokens:
            if token in _SEPARATORS:
                if current:
                    cmds.append(current)
                current = []
            else:
                current.append(token)
        if current:
            cmds.append(current)
    return cmds


def _pytest_argv(cmd: "list[str]") -> "list[str] | None":
    """``cmd``'s arguments if it invokes pytest, else None. Steps over ``VAR=value`` prefixes
    — the collation-C step's ``ENGRAM_TEST_DSN=… pytest -q`` is exactly that shape — and the
    ``python -m`` / ``poetry run`` / ``coverage run -m`` wrappers."""
    i = 0
    while i < len(cmd) and cmd[i] not in _PYTEST and (
            cmd[i] in _WRAPPERS or ("=" in cmd[i] and not cmd[i].startswith("-"))):
        i += 1
    if i >= len(cmd) or cmd[i] not in _PYTEST:
        return None
    return cmd[i + 1:]


def _paths(argv: "list[str]") -> "list[str]":
    """The positional collection arguments, normalised repo-relative. An empty result means
    pytest was handed no path and falls back to ``testpaths`` — which, in this repo, is what
    every one of the three invocations does."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token.startswith("-"):
            skip = token in _TAKES_VALUE
            continue
        if token.isdigit():
            continue                              # a file descriptor from `2>&1`, not a path
        out.append(_norm(token))
    return out


def _norm(raw: str) -> str:
    path = raw.split("::", 1)[0]                  # a node id points at its file
    path = path.removeprefix("./").rstrip("/")
    return str(PurePosixPath(path)) if path else "."


def _fallback_scope(pytestconfig) -> "list[str]":
    """What a bare ``pytest`` collects — asked of pytest, which owns the semantics, instead of
    re-parsed here. With no ``testpaths`` configured pytest collects from the rootdir, and a
    rootdir covers everything."""
    configured = [_norm(str(p)) for p in (pytestconfig.getini("testpaths") or [])]
    return configured or ["."]


def _triggers(doc: dict) -> "set[str]":
    """YAML 1.1 reads a bare ``on:`` as the boolean True, so both spellings must be tried or
    every workflow looks untriggered and every job looks conditional."""
    on = doc.get("on", doc.get(True))
    if isinstance(on, (dict, list)):
        return {str(k) for k in on}
    return {str(on)} if on else set()


def pytest_commands() -> "list[tuple[list[str], str, bool]]":
    """``(argv, "workflow:job", pr_gated)`` for every pytest command any workflow runs.

    ``pr_gated`` is deliberately conservative: True only when the workflow triggers on
    ``pull_request`` AND neither the job nor the step carries an ``if:``. A condition this
    cannot evaluate is read as conditional, so the answer can only ever understate the gate —
    never invent one."""
    found: list[tuple[list[str], str, bool]] = []
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        doc = yaml.safe_load(workflow.read_text(encoding="utf-8")) or {}
        on_pull_request = "pull_request" in _triggers(doc)
        for job_name, job in (doc.get("jobs") or {}).items():
            job_open = on_pull_request and not (job or {}).get("if")
            for step in ((job or {}).get("steps") or []):
                if not isinstance(step, dict) or not step.get("run"):
                    continue
                gated = job_open and not step.get("if")
                for cmd in _commands(step["run"]):
                    argv = _pytest_argv(cmd)
                    if argv is not None:
                        found.append((argv, f"{workflow.name}:{job_name}", gated))
    return found


def invocations(fallback: "list[str]") -> "list[tuple[str, str, bool]]":
    """``(path, "workflow:job", pr_gated)`` for every path any workflow hands to pytest, with
    ``fallback`` substituted wherever a command names none."""
    return [(path, where, gated)
            for argv, where, gated in pytest_commands()
            for path in (_paths(argv) or fallback)]


# ── reading the repo ──────────────────────────────────────────────────────────────────
def versioned_test_files() -> "list[str]":
    """Committed test modules. ``git ls-files``, never a walk of the working tree: a file that
    is not committed is not a contract, and a leftover copy under ``tests/`` is not something
    anybody promised CI would run."""
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True)
    return sorted(p for p in proc.stdout.split("\0")
                  if p and _is_test_module(PurePosixPath(p).name))


def _is_test_module(name: str) -> bool:
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def _covers(path: str, file: str) -> bool:
    return path in (".", "") or file == path or file.startswith(path + "/")


# ── guards on the guard ───────────────────────────────────────────────────────────────
def test_the_workflows_parse_and_invoke_pytest_at_all(pytestconfig):
    """Everything below is vacuous if ``REPO_ROOT`` counted the wrong number of parents, the
    directory moved, the YAML changed shape, or the lexer quietly stopped recognising a pytest
    line. ``tests/`` is flat here, so ``parents[1]`` is the repo root — and a repo root is the
    thing that holds both ``.github`` and the git index the enumeration below reads."""
    assert (REPO_ROOT / ".github").is_dir(), REPO_ROOT
    assert (REPO_ROOT / "pyproject.toml").is_file(), REPO_ROOT
    assert WORKFLOWS.is_dir(), WORKFLOWS
    found = invocations(_fallback_scope(pytestconfig))
    assert found, "no workflow invokes pytest — this guard would pass over an empty set"
    assert versioned_test_files(), "git ls-files found no test modules — the guard has no subject"


def test_a_bare_pytest_means_testpaths_and_not_nothing(pytestconfig):
    """The load-bearing test for THIS repo: every one of its pytest steps takes this path.
    Reading ``pytest -q`` as invoking nothing would make the only form that is immune by
    construction look exactly like the defect — and would leave the invariant below asserting
    over an empty set."""
    assert _paths(_pytest_argv(["pytest", "-q"])) == []
    fallback = _fallback_scope(pytestconfig)
    assert fallback == ["tests"], fallback

    # the env-prefixed spelling the collation-C step uses
    env_prefixed = _commands(
        "ENGRAM_TEST_DSN=postgresql://postgres:postgres@localhost:5432/engram_ctest pytest -q")
    assert len(env_prefixed) == 1, env_prefixed
    assert _pytest_argv(env_prefixed[0]) == ["-q"]

    # and the fallback is what actually covers this repo — not a technicality that happens to
    # be true of nothing. Every committed test module sits under `testpaths`.
    uncovered = [f for f in versioned_test_files()
                 if not any(_covers(p, f) for p in fallback)]
    assert not uncovered, uncovered


def test_this_repo_keeps_the_immune_form_a_bare_pytest_on_the_pull_request_gate(pytestconfig):
    """The property that makes ``cogno-engram`` the reference, asserted rather than described.

    At least one pytest command on the PULL-REQUEST gate must name NO path, so its scope comes
    from ``testpaths`` and no enumeration exists to go stale. This is what #37 bought when it
    replaced a two-file list that had already gone stale, and it is what goes red the day the
    last bare invocation is "optimised" into ``pytest tests/unit``. It does not forbid ADDING a
    targeted run beside it — a focused extra step is not the defect; losing the unscoped one
    is."""
    bare = [where for argv, where, gated in pytest_commands() if gated and not _paths(argv)]
    assert bare, (
        "no pull-request-gated job hands pytest the bare, path-free form any more. Whatever "
        "paths are named now, they are an enumeration, and an enumeration goes stale in "
        "silence — that is what this repo's #37 fixed.\n"
        f"  pytest commands found: {[(a, w, g) for a, w, g in pytest_commands()]}")
    # All THREE of them, measured — the test job, the integration job, and the collation-C
    # step. `>=` rather than `==` on purpose: a fourth unscoped run is not a regression, while
    # any of these three growing a path list is exactly the erosion this catches. The
    # invariant below would stay green through it, because one surviving bare run still covers
    # every file; this is the assertion that notices the loss.
    assert len(bare) >= 3, bare


def test_the_lexer_reads_the_forms_this_repo_actually_uses():
    cases = {
        "pytest -q --cov=cogno_engram --cov-report=term-missing --cov-fail-under=90": [],
        "pytest -q": [],
        ("ENGRAM_TEST_DSN=postgresql://postgres:postgres@localhost:5432/engram_ctest "
         "pytest -q"): [],
        "python -m pytest tests -q": ["tests"],
        "pytest tests/test_folding_parity.py tests/test_fold_migration.py -q":
            ["tests/test_folding_parity.py", "tests/test_fold_migration.py"],
        "pytest ./tests/ -q": ["tests"],
        "pytest tests/test_db_guard.py::test_the_guard_refuses": ["tests/test_db_guard.py"],
        "pytest -q || [ $? -eq 5 ]": [],
        "pytest tests -q  # the pull-request gate": ["tests"],
    }
    for line, want in cases.items():
        cmds = [c for c in _commands(line) if _pytest_argv(c) is not None]
        assert len(cmds) == 1, (line, cmds)
        assert _paths(_pytest_argv(cmds[0])) == want, line


def test_a_command_that_is_not_pytest_is_not_read_as_one():
    """``ci.yml`` runs three non-pytest commands that a sloppy matcher would claim: the bench
    gate, the linter, and the ``psql`` that the collation-C step's ``pytest`` shares a block
    with."""
    for line in ("python3 cognobench.py --min-score 100",
                 "ruff check cogno_engram cognobench tests examples",
                 "mypy cogno_engram",
                 "pip install -e '.[dev,postgres,redis]'",
                 'PGPASSWORD=postgres psql -h localhost -U postgres -c "CREATE DATABASE x"'):
        assert all(_pytest_argv(c) is None for c in _commands(line)), line


def test_an_option_value_is_never_mistaken_for_a_collection_path():
    """The false-GREEN surface, one line per flag that could swallow a path-shaped token.
    ``--rootdir tests`` collects nothing; a parser that thinks otherwise declares every file
    covered and this guard silently stops working."""
    for line in ("pytest --rootdir tests tests/test_folding.py",
                 "pytest -k tests tests/test_folding.py",
                 "pytest -m tests tests/test_folding.py",
                 "pytest -p no:cacheprovider tests/test_folding.py",
                 "pytest --ignore tests/test_postgres_integration.py tests/test_folding.py",
                 "pytest --deselect tests/test_postgres_integration.py tests/test_folding.py",
                 "pytest -c tests/pytest.ini tests/test_folding.py",
                 "pytest -o testpaths=tests tests/test_folding.py",
                 "pytest --cov cogno_engram tests/test_folding.py",
                 "pytest -n 4 tests/test_folding.py"):
        assert _paths(_pytest_argv(_commands(line)[0])) == ["tests/test_folding.py"], line


def test_a_neighbouring_command_does_not_donate_its_arguments():
    """``ruff check cogno_engram cognobench tests examples`` on the next line is not a pytest
    path, and ``[ $? -eq 5 ]`` after a ``||`` is not one either. Either would silently widen
    what looks invoked — and here, where every real invocation is path-free, a donated
    ``tests`` would make the fallback look like something a workflow actually typed."""
    block = ("pytest tests/test_folding.py -q\n"
             "ruff check cogno_engram cognobench tests examples\n"
             "mypy cogno_engram")
    paths = [p for c in _commands(block)
             if (argv := _pytest_argv(c)) is not None for p in _paths(argv)]
    assert paths == ["tests/test_folding.py"], paths


def test_a_heredoc_body_is_not_shell():
    """No workflow here holds one yet; the helper is what keeps the first one from breaking
    the lexer, so it is pinned before it is needed rather than after."""
    block = ("python - <<'EOF'\n"
             "import os  # pytest tests, but this isn't a command\n"
             "print(\"the installed adapter doesn't honour the timeout\")\n"
             "EOF\n"
             "pytest -q")
    cmds = [c for c in _commands(block) if _pytest_argv(c) is not None]
    assert len(cmds) == 1, cmds
    assert _paths(_pytest_argv(cmds[0])) == []


def test_the_invariant_would_notice_an_uninvoked_file(pytestconfig):
    """Mutation. Without this, a ``_covers`` that answered True unconditionally — or a
    ``versioned_test_files`` that returned nothing interesting — would satisfy every other
    assertion in the file. It is the reason a green run below means anything in a repo where
    the invariant is expected to hold."""
    invoked = {p for p, _, _ in invocations(_fallback_scope(pytestconfig))}
    orphan = "docs/a_folder_testpaths_does_not_reach/test_never_run.py"
    assert not any(_covers(p, orphan) for p in invoked), sorted(invoked)
    assert _covers("tests", "tests/test_folding.py")
    assert _covers(".", "tests/test_folding.py")
    assert not _covers("tests", "tests_helpers/test_x.py")
    assert _is_test_module("test_x.py") and _is_test_module("x_test.py")
    assert not _is_test_module("conftest.py") and not _is_test_module("test_data.json")


# ── the invariant ─────────────────────────────────────────────────────────────────────
def test_every_versioned_test_file_is_under_a_path_ci_invokes(pytestconfig):
    invoked = {p for p, _, _ in invocations(_fallback_scope(pytestconfig))}
    orphans = [f for f in versioned_test_files() if not any(_covers(p, f) for p in invoked)]
    assert not orphans, (
        "these test files are committed but no CI job ever collects them. They pass in review "
        "and have never run anywhere automated. Widen an invoked path — or hand pytest the "
        "root and let `testpaths` do the scoping, which is the form that cannot regress and "
        "the one this repo already uses — rather than deleting them.\n"
        f"  invoked: {sorted(invoked)}\n"
        f"  orphaned ({len(orphans)}): {orphans}")


def test_which_paths_run_only_on_a_conditional_job_is_a_decision_not_a_drift(pytestconfig):
    """The weaker, separate half, and the reason the assertion above does not simply demand a
    pull-request gate for everything. A path reached only by a nightly IS invoked, and that is
    right wherever the assertions depend on a live model or a paid API: sampling one must not
    decide whether main is green. But it is a DECISION, so it is written down in
    ``NIGHTLY_ONLY`` and changing it has to mean changing that line. Here it is empty — every
    test file gates a pull request — and the day something moves out of that gate, this says
    so."""
    found = invocations(_fallback_scope(pytestconfig))
    gated = {p for p, _, is_pr in found if is_pr}
    conditional = {p for p, _, _ in found} - gated
    assert conditional == NIGHTLY_ONLY, (
        "the set of paths only a conditional job invokes has changed. A path that moved OUT of "
        "the pull-request gate is a suite quietly leaving CI; one that moved IN needs "
        f"NIGHTLY_ONLY updated.\n  now: {sorted(conditional)}\n"
        f"  declared: {sorted(NIGHTLY_ONLY)}")
