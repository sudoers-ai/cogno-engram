"""``cogno_engram.lexical`` — what one RANKING may cost, bounded by :data:`MAX_CANDIDATES`.

A ranking is CPU, and on an event loop CPU cannot be interrupted: while :func:`rank` tokenizes,
nothing else in the process runs, and a wall-clock ceiling only starts counting when the task
next yields. So the WORK is bounded before it starts — by COUNT here (the size of each candidate
is the caller's to cut; see the constant's docstring).

**Two claims, and only one of them is timed.**

* **The bound is MECHANICAL, and so is its proof** — no clock: over ~3 MB of invented
  section-sized text (~90 characters per section, the shape a caller's section splitter
  produces), a builder asked for :data:`MAX_CANDIDATES` returns exactly that many, out of a pool
  more than ten times larger.
* **What the bound BUYS is a RATIO, not a number of milliseconds.** The capped ranking and the
  uncapped one are timed INTERCALATED — capped, uncapped, capped, uncapped, capped, uncapped —
  in the same process, at the same load, and the MINIMUM of each side is compared:
  ``uncapped >= 10 × capped``. The pool is ~16 times the cap, so the ratio sits near 16 on a
  quiet machine and stays there on a busy one, because contention inflates both legs alike.

**Why not an absolute bound any more.** The first form asserted the capped ranking under 50 ms,
and it read the machine rather than the code: ``time.process_time`` counts only this process's
CPU, but a process sharing its cores and caches with busy neighbours spends MORE of its own CPU
on the same work, so a loaded machine pushed the reading over the bound with no change in the
code — measured on the consumer that first carried this twin, 93.6 ms and 160.9 ms under load.
The number stays as a MEASUREMENT: on 2026-09-24, on a 20-thread box at load ~9, the capped
ranking read ~22 ms and the uncapped ~325–350 ms (min of three, GC paused, no tracer).

Three choices make each reading a measurement of this code and not of the neighbours, each
inherited from the consumer that first measured it:

* the clock is ``time.process_time`` — the process's CPU, so a neighbour that takes the core
  does not count as our work;
* the cyclic GC is paused around the timed part — a generation-2 collection walks the WHOLE
  test process's heap, which is not this code's allocation;
* no trace function runs while the clock does (:func:`_untraced`) — this repo's CI measures
  coverage, and a line tracer quadruples the reading.
"""

from __future__ import annotations

import gc
import sys
import time
from contextlib import contextmanager

import pytest

from cogno_engram import GraphEdge
from cogno_engram.lexical import (MAX_CANDIDATES, Candidate, graph_candidates, memory_candidates,
                                  rank, variants)

#: What the cap must buy, as a ratio of the MINIMUM readings of each side (the pool is ~16× the
#: cap, so the ratio sits near 16; ten leaves room for noise and none for a cap that stopped
#: capping, where the ratio is ~1).
_MIN_RATIO = 10.0
#: How many (capped, uncapped) pairs are timed, intercalated.
_PAIRS = 3

#: Invented sections — a clinic's price list, hours and policies, nothing anybody wrote for real.
_SECTIONS = (
    "Limpeza simples R$ 180,00; com raspagem R$ 260,00. Parcelamos em até três vezes.",
    "Atendemos de segunda a sexta das 8h às 19h e aos sábados das 8h às 12h.",
    "Remarcar a consulta com 24 horas de antecedência não tem custo; depois, taxa de R$ 50.",
    "Clareamento a laser em duas sessões, com avaliação prévia obrigatória e gratuita.",
    "Aceitamos os convênios Plano Alfa e Plano Beta; os outros mediante reembolso.",
    "Crianças até doze anos acompanhadas por um responsável durante toda a consulta.",
)
_QUESTION = ("cancellation fee policy", "quanto custa remarcar a consulta?")


class _Rec:
    """A memory record — what :func:`memory_candidates` builds from."""

    def __init__(self, rid: str, content: str) -> None:
        self.id, self.content = rid, content


def _records(target_bytes: int) -> "list[_Rec]":
    """Section-sized records totalling about ``target_bytes`` of text, each distinct."""
    out: "list[_Rec]" = []
    size, i = 0, 0
    while size < target_bytes:
        text = f"Tópico {i}\n{_SECTIONS[i % len(_SECTIONS)]}"
        out.append(_Rec(f"m{i}", text))
        size += len(text.encode("utf-8"))
        i += 1
    return out


@contextmanager
def _untraced():
    """No trace function while the clock runs — restored after, the same object.

    A coverage run (this repo's CI runs the suite under ``--cov``) installs a tracer that is
    called on every line: measured on the 3.10 leg, the capped ranking read 86.6 ms under it and
    ~22 ms without. That is the instrument's cost, not this code's — the same reason the GC is
    paused. Both legs of the ratio are timed under the same condition, so it cannot pass by the
    tracer being off in one half and on in the other."""
    tracer = sys.gettrace()
    sys.settrace(None)
    try:
        yield
    finally:
        sys.settrace(tracer)


def _time_rank(pool: "list[Candidate]", asked: "list[str]") -> float:
    """One ranking of ``pool``, on the process's CPU clock, GC paused, no tracer."""
    gc.disable()
    try:
        with _untraced():
            t0 = time.process_time()
            rank(pool, asked)
            return time.process_time() - t0
    finally:
        gc.enable()


def _intercalated(capped: "list[Candidate]", uncapped: "list[Candidate]",
                  pairs: int = _PAIRS) -> "tuple[float, float]":
    """``(min capped, min uncapped)`` over ``pairs`` readings each, taken c, u, c, u, … — so the
    two sides see the same load, and a burst on the machine lands on both or on neither."""
    asked = variants(*_QUESTION)
    rank(capped[:50], asked)                                # warm-up: regexes, unicodedata
    c, u = [], []
    for _ in range(pairs):
        c.append(_time_rank(capped, asked))
        u.append(_time_rank(uncapped, asked))
    return min(c), min(u)


@pytest.fixture(scope="module")
def three_mb() -> "list[_Rec]":
    recs = _records(3_000_000)
    assert sum(len(r.content.encode("utf-8")) for r in recs) >= 3_000_000
    return recs


def test_over_3MB_the_builder_stops_at_MAX_CANDIDATES_mechanically(three_mb):
    """The bound, proved without a clock: the capped build is exactly the cap, the uncapped one
    is more than ten times it — and the capped pool still answers (the policy section is among
    the first topics)."""
    capped = memory_candidates(three_mb, limit=MAX_CANDIDATES)
    uncapped = memory_candidates(three_mb)
    assert len(capped) == MAX_CANDIDATES
    assert len(uncapped) > 10 * MAX_CANDIDATES, "the cap was hit, or this measures nothing"
    top = rank(capped, variants(*_QUESTION))[0]
    assert top[0] >= 0.5 and "Remarcar a consulta" in top[1].text


def test_TWIN_the_capped_ranking_is_an_order_of_magnitude_cheaper_timed_intercalated(three_mb):
    """What the cap buys, as a ratio of minimums over intercalated readings (module docstring).
    Its PAIR is inside it: the uncapped leg is the same pool without the cap, timed under the same
    load — and with the cap removed the two legs are the same pool, and the ratio is ~1."""
    capped = memory_candidates(three_mb, limit=MAX_CANDIDATES)
    uncapped = memory_candidates(three_mb)
    c, u = _intercalated(capped, uncapped)
    assert u >= _MIN_RATIO * c, (f"capped {c * 1000:.1f} ms vs uncapped {u * 1000:.1f} ms — the "
                                 f"cap bought only {u / c:.1f}×, under the {_MIN_RATIO:.0f}× it must")


def test_the_candidate_bound_stops_the_BUILD_not_only_the_result():
    """The builders stop AT ``limit`` — so a caller asks for one more than it may keep, learns
    the bound was hit, and never builds the tail."""
    edges = [GraphEdge(scope="t", source=f"N{i}", target=f"M{i}", relation="R")
             for i in range(10_000)]
    built = graph_candidates([(0, 0, 1, edges)], limit=MAX_CANDIDATES + 1)
    assert len(built) == MAX_CANDIDATES + 1, "asked for one more than it may keep, no more"

    mems = memory_candidates((_Rec(f"m{i}", f"memória {i}") for i in range(10_000)),
                             limit=MAX_CANDIDATES + 1)
    assert len(mems) == MAX_CANDIDATES + 1
    # the limit stops the walk mid-list, including across start nodes
    two = graph_candidates([(0, 0, 1, edges[:3]), (0, 1, 2, edges[3:6])], limit=4)
    assert [c.id for c in two] == ["edge:1.0", "edge:1.1", "edge:1.2", "edge:2.0"]
