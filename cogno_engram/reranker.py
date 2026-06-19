"""
cogno_engram.reranker — two-pass reranking for retrieved memories.

Ported from the parent's ``reranker.py``. Composable: the host fetches
candidates via ``MemoryStore.load_memories`` (already relevance-ordered) and
passes them here to refine the ordering before injecting the top-k.

  Pass 1 (always, pure): ``score = sim·w_sim + recency·w_recency + category·w_cat``
      - sim       — base relevance (explicit ``base_scores`` if the caller has
                    them, else a position heuristic since the input is ranked);
      - recency   — exponential decay from ``created_at`` (half-life configurable);
      - category  — a static boost per memory category (preferences > behaviour).

  Pass 2 (optional): a cross-encoder. To keep cogno-engram dependency-free, the
      cross-encoder is a **host-injected callable** ``(query, [content]) -> [score]``
      (e.g. wrapping ``sentence-transformers``), not a bundled model.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

from cogno_engram.types import MemoryRecord

logger = logging.getLogger("cogno_engram.reranker")

DEFAULT_HALF_LIFE_DAYS = 7.0

# Static category boosts — higher surfaces more readily (preferences/facts first).
DEFAULT_CATEGORY_BOOST: dict[str, float] = {
    "preference": 1.00, "preferences": 1.00,
    "fact": 0.95, "facts": 0.95,
    "pii": 0.95,
    "goal": 0.90, "goals": 0.90,
    "personality": 0.80,
    "behavior": 0.70, "behaviour": 0.70,
    "sentiment": 0.60,
}
DEFAULT_BOOST = 0.50

# A cross-encoder: scores each (query, content) pair; higher = more relevant.
CrossEncoder = Callable[[str, list[str]], Sequence[float]]


@dataclass
class RerankConfig:
    w_sim: float = 0.60
    w_recency: float = 0.25
    w_category: float = 0.15
    half_life_days: float = DEFAULT_HALF_LIFE_DAYS
    category_boost: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CATEGORY_BOOST))
    default_boost: float = DEFAULT_BOOST


def recency_score(created_at: Optional[datetime], now: Optional[datetime] = None,
                  *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS) -> float:
    """Exponential decay: 1.0 brand-new, ~0.5 at half-life, →0 for old."""
    if created_at is None:
        return 0.5
    now = now or datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
    return math.exp(-math.log(2) / half_life_days * age_days)


def rerank(
    memories: list[MemoryRecord],
    *,
    query_text: Optional[str] = None,
    top_k: int = 5,
    now: Optional[datetime] = None,
    config: Optional[RerankConfig] = None,
    base_scores: Optional[Sequence[float]] = None,
    cross_encoder: Optional[CrossEncoder] = None,
) -> list[MemoryRecord]:
    """Two-pass rerank ``memories`` (already relevance-ordered) → top-k.

    ``base_scores`` (parallel to ``memories``) supplies explicit relevance; when
    absent, a position heuristic is used (1st=1.0, 2nd=0.95, …). ``cross_encoder``,
    if given with ``query_text``, runs Pass 2 over the Pass-1 shortlist.
    """
    if not memories:
        return []
    cfg = config or RerankConfig()

    scored: list[tuple[float, int, MemoryRecord]] = []
    for idx, mem in enumerate(memories):
        if base_scores is not None and idx < len(base_scores):
            sim = float(base_scores[idx])
        else:
            sim = max(0.0, 1.0 - idx * 0.05)
        rec = recency_score(mem.created_at, now, half_life_days=cfg.half_life_days)
        cat = cfg.category_boost.get(mem.category.lower(), cfg.default_boost)
        score = sim * cfg.w_sim + rec * cfg.w_recency + cat * cfg.w_category
        scored.append((score, idx, mem))

    # Descending by score; ties broken toward the earlier (more relevant) position.
    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    pass1 = [mem for _, _, mem in scored]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("stage=reranker pass=1 candidates=%d top_score=%.3f",
                     len(scored), scored[0][0] if scored else 0.0)

    if cross_encoder is not None and query_text:
        shortlist = pass1[: max(top_k * 2, 20)]
        ce_scores = cross_encoder(query_text, [m.content for m in shortlist])
        order = sorted(range(len(shortlist)), key=lambda i: ce_scores[i], reverse=True)
        logger.debug("stage=reranker pass=2 event=cross_encoder shortlist=%d top_k=%d",
                     len(shortlist), top_k)
        return [shortlist[i] for i in order][:top_k]

    return pass1[:top_k]
