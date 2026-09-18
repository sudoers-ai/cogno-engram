"""The ``py.typed`` marker reaches the BUILT WHEEL — the only place it does any work.

A marker that exists in the checkout and not in the wheel is worse than no marker: every reader of
the repo sees a typed package, and every INSTALL of it is untyped. PEP 561 is read from
site-packages, so the checkout's copy is never consulted by the consumer.

Measured on cogno-host 2026-09-18. With the siblings installed as wheels — what its CI does — a
probe importing a deliberately wrong name from this package came back GREEN: the host was
type-checking nothing of it, because `ignore_missing_imports = true` turns an unmarked dependency
into `Any` without a word of warning. Three of the sixteen pinned siblings were invisible that way,
and in one of them the blindness had been hiding a real contract error for 17 days.

**IT BUILDS A COPY, AND THAT IS THE TEST'S OWN SCAR.** Written in place first, it passed over the
deletion of the marker itself: ``setuptools`` stages into ``build/lib`` and never removes what
disappeared, so every run after the first unzipped a wheel assembled from a leftover.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import zipfile

import pytest

# The repo root is FOUND, not counted in `..`s: this file lives at a different depth in each
# repo, and a wrong root builds a directory with no `pyproject.toml` instead of failing the
# assertion — which reads as a broken test rather than as a missing marker.
_ROOT = next(d for d in pathlib.Path(__file__).resolve().parents
             if (d / "pyproject.toml").exists())
_SKIP = {".git", "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__",
          ".venv", "venv"}


def _pristine_copy(dest: pathlib.Path) -> pathlib.Path:
    """The repo without any build state — so the wheel is assembled from THESE files, not from
    whatever a previous build happened to stage."""
    def ignore(_dir, names):
        return [n for n in names if n in _SKIP or n.endswith(".egg-info")]
    shutil.copytree(_ROOT, dest, ignore=ignore)
    assert not (dest / "build").exists(), "the copy carried build state — the guard above failed"
    return dest


def test_py_typed_is_inside_the_built_wheel(tmp_path):
    src = _pristine_copy(tmp_path / "src")
    out = subprocess.run(
        # Build ISOLATION is left ON (pip's default): with `--no-build-isolation` the runner has to
        # already have the declared backend importable, and a CI virtualenv for 3.12 does not ship
        # `setuptools` — measured, `BackendUnavailable`, red on 3.12 and green on 3.11 in the same
        # matrix. An isolated build is also the faithful reproduction of how the wheel is really
        # made. `--no-cache-dir` stays: pip caches wheels built from a local directory, and with
        # the cache on, this test passed over a mutation that deleted the marker outright.
        [sys.executable, "-m", "pip", "wheel", "--no-deps",
         "--no-cache-dir", "-w", str(tmp_path / "wheel"), str(src)],
        capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        pytest.fail("could not build a wheel, so this test measured nothing:\n"
                    + out.stdout[-3000:] + out.stderr[-3000:])

    wheels = sorted((tmp_path / "wheel").glob("cogno_engram-*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {[w.name for w in wheels]}"
    names = zipfile.ZipFile(wheels[0]).namelist()
    assert "cogno_engram/py.typed" in names, (
        "the wheel carries no PEP 561 marker — a consumer that installs this package type-checks "
        f"NOTHING of it. Wheel contents: {sorted(names)}")
