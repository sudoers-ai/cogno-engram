from datetime import datetime, timedelta, timezone

from cogno_engram.reranker import RerankConfig, recency_score, rerank
from cogno_engram.types import MemoryRecord

NOW = datetime(2026, 6, 17, tzinfo=timezone.utc)


def _mem(content, category="fact", age_days=0.0):
    return MemoryRecord("s", category, content, created_at=NOW - timedelta(days=age_days))


def test_empty_returns_empty():
    assert rerank([]) == []


def test_recency_decay_curve():
    assert recency_score(NOW, NOW) == 1.0
    assert abs(recency_score(NOW - timedelta(days=7), NOW) - 0.5) < 0.01
    assert recency_score(None) == 0.5


def test_recency_breaks_a_sim_tie():
    # same base score + category → the newer memory wins on recency
    old = _mem("old", age_days=30)
    new = _mem("new", age_days=0)
    out = rerank([old, new], now=NOW, base_scores=[1.0, 1.0])
    assert out[0].content == "new"


def test_category_boost_orders_preference_over_behavior():
    pref = _mem("a preference", category="preference", age_days=0)
    beh = _mem("a behavior", category="behavior", age_days=0)
    # equal sim + recency → category boost decides
    out = rerank([beh, pref], now=NOW, base_scores=[1.0, 1.0])
    assert out[0].category == "preference"


def test_position_heuristic_keeps_strong_first():
    a = _mem("first", age_days=0)
    b = _mem("second", age_days=0)
    out = rerank([a, b], now=NOW)            # no base_scores → position heuristic
    assert out[0].content == "first"


def test_base_scores_override_position():
    a = _mem("weak", age_days=0)
    b = _mem("strong", age_days=0)
    out = rerank([a, b], now=NOW, base_scores=[0.1, 0.99])
    assert out[0].content == "strong"


def test_top_k_limits_results():
    mems = [_mem(f"m{i}") for i in range(10)]
    assert len(rerank(mems, top_k=3)) == 3


def test_weights_recency_only_makes_newest_win():
    old = _mem("old", category="preference", age_days=60)   # high category, old
    new = _mem("new", category="behavior", age_days=0)      # low category, new
    cfg = RerankConfig(w_sim=0.0, w_recency=1.0, w_category=0.0)
    out = rerank([old, new], now=NOW, config=cfg, base_scores=[1.0, 1.0])
    assert out[0].content == "new"


def test_cross_encoder_pass_reorders():
    a = _mem("about cats", age_days=0)
    b = _mem("about dogs", age_days=0)

    def fake_ce(query, contents):
        # score 'dogs' higher regardless of Pass-1 order
        return [1.0 if "dogs" in c else 0.0 for c in contents]

    out = rerank([a, b], query_text="puppies", now=NOW, cross_encoder=fake_ce)
    assert out[0].content == "about dogs"


def test_cross_encoder_skipped_without_query():
    a, b = _mem("x", age_days=0), _mem("y", age_days=0)
    calls = []

    def fake_ce(query, contents):
        calls.append(1)
        return [0.0] * len(contents)

    rerank([a, b], cross_encoder=fake_ce)    # no query_text → Pass 2 skipped
    assert not calls
