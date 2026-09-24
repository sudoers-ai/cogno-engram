"""``cogno_engram.lexical`` — what one RANKING may cost, bounded by :data:`MAX_CANDIDATES`.

A ranking is CPU, and on an event loop CPU cannot be interrupted: while :func:`rank` tokenizes,
nothing else in the process runs, and a wall-clock ceiling only starts counting when the task
next yields. So the WORK is bounded before it starts — by COUNT here (the size of each candidate
is the caller's to cut; see the constant's docstring).

**The twin and its pair.** ~3 MB of invented section-sized text (~90 characters per section,
the shape a caller's section splitter produces), capped at :data:`MAX_CANDIDATES`, must rank in
under 50 ms; the SAME pool uncapped must come out far above that, or the twin would be a
constant — a fast machine, not a bound. Three choices make it a measurement of this code and not
of the neighbours, each inherited from the consumer that first measured it:

* the clock is ``time.process_time`` — the process's CPU, so a neighbour that takes the core
  does not count as our work;
* the cyclic GC is paused around the timed part — a generation-2 collection walks the WHOLE
  test process's heap, which is not this code's allocation;
* the reading is the minimum of three after a warm-up.
"""

from __future__ import annotations

import gc
import time

import pytest

from cogno_engram import GraphEdge
from cogno_engram.lexical import (MAX_CANDIDATES, SOURCE_MATERIAL, Candidate, graph_candidates,
                                  memory_candidates, rank, variants)

#: The bound on one ranking over the capped pool.
_BOUND_S = 0.050

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


def _pool(target_bytes: int) -> "list[Candidate]":
    """Section-sized candidates totalling about ``target_bytes`` of text, each distinct."""
    out: "list[Candidate]" = []
    size, i = 0, 0
    while size < target_bytes:
        text = f"Tópico {i}\n{_SECTIONS[i % len(_SECTIONS)]}"
        out.append(Candidate(id=f"mat:{i}.0", source=SOURCE_MATERIAL, text=text, prior=i))
        size += len(text.encode("utf-8"))
        i += 1
    return out


def _rank_time(pool: "list[Candidate]", runs: int = 3) -> float:
    asked = variants(*_QUESTION)
    rank(pool[:50], asked)                                  # warm-up: regexes, unicodedata
    best = float("inf")
    for _ in range(runs):
        gc.disable()
        try:
            t0 = time.process_time()
            rank(pool, asked)
            best = min(best, time.process_time() - t0)
        finally:
            gc.enable()
    return best


@pytest.fixture(scope="module")
def three_mb() -> "list[Candidate]":
    pool = _pool(3_000_000)
    assert sum(len(c.text.encode("utf-8")) for c in pool) >= 3_000_000
    return pool


def test_TWIN_3MB_of_sections_capped_at_MAX_CANDIDATES_ranks_under_50ms(three_mb):
    capped = three_mb[:MAX_CANDIDATES]
    assert len(three_mb) > 10 * MAX_CANDIDATES, "the cap was hit, or this measures nothing"
    dt = _rank_time(capped)
    assert dt < _BOUND_S, f"the ranking held the loop {dt * 1000:.1f} ms over the capped pool"
    # ...and the capped pool still answers: the policy section is among the first topics
    top = rank(capped, variants(*_QUESTION))[0]
    assert top[0] >= 0.5 and "Remarcar a consulta" in top[1].text


def test_PAIR_the_same_3MB_uncapped_is_far_over_the_bound(three_mb):
    """The negative control, PRODUCED: with no cap the very same ranking blows the bound — so
    the twin above measures the cap, not a fast machine. (One run: it is the slow one.)"""
    dt = _rank_time(three_mb, runs=1)
    assert dt > 3 * _BOUND_S, f"uncapped took only {dt * 1000:.1f} ms — the pair measures nothing"


def test_the_candidate_bound_stops_the_BUILD_not_only_the_result():
    """The builders stop AT ``limit`` — so a caller asks for one more than it may keep, learns
    the bound was hit, and never builds the tail."""
    edges = [GraphEdge(scope="t", source=f"N{i}", target=f"M{i}", relation="R")
             for i in range(10_000)]
    built = graph_candidates([(0, 0, 1, edges)], limit=MAX_CANDIDATES + 1)
    assert len(built) == MAX_CANDIDATES + 1, "asked for one more than it may keep, no more"

    class _Rec:
        def __init__(self, i):
            self.id, self.content = f"m{i}", f"memória {i}"

    mems = memory_candidates((_Rec(i) for i in range(10_000)), limit=MAX_CANDIDATES + 1)
    assert len(mems) == MAX_CANDIDATES + 1
    # the limit stops the walk mid-list, including across start nodes
    two = graph_candidates([(0, 0, 1, edges[:3]), (0, 1, 2, edges[3:6])], limit=4)
    assert [c.id for c in two] == ["edge:1.0", "edge:1.1", "edge:1.2", "edge:2.0"]
