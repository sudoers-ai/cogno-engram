"""
Integration tests for the Redis ConversationBuffer adapter.

Gated: set ``ENGRAM_TEST_REDIS_URL`` to a reachable Redis; the suite auto-skips
otherwise. Uses unique scopes per test (no shared state) so it is safe against
any Redis.

    docker run -d --rm --name engram-redis -p 56379:6379 redis:7-alpine
    ENGRAM_TEST_REDIS_URL=redis://localhost:56379/0 \
        python3 -m pytest tests/test_redis_integration.py
"""

from __future__ import annotations

import os
from uuid import uuid4

import pytest

pytest.importorskip("redis")

from cogno_engram.adapters.redis_buffer import RedisConversationBuffer  # noqa: E402
from cogno_engram.types import TurnRecord  # noqa: E402

URL = os.getenv("ENGRAM_TEST_REDIS_URL", "")


async def _redis_available() -> bool:
    if not URL:
        return False
    try:
        from redis.asyncio import from_url
        client = from_url(URL)
        await client.ping()
        await client.aclose()
        return True
    except Exception:
        return False


@pytest.fixture
async def buffer():
    if not await _redis_available():
        pytest.skip("set ENGRAM_TEST_REDIS_URL to a reachable Redis to run")
    buf = RedisConversationBuffer(redis_url=URL, max_size=5)
    yield buf
    await buf.aclose()


def _turn(n, **kw):
    base = dict(session_id="sess", scope="bench", turn_n=n, user_input=f"msg {n}")
    base.update(kw)
    return TurnRecord(**base)


async def test_blank_scope_rejected(buffer):
    with pytest.raises(ValueError):
        await buffer.push("", "sess", _turn(1))


async def test_push_and_window_roundtrip(buffer):
    scope, sid = f"t/{uuid4()}", "s1"
    await buffer.push(scope, sid, _turn(1, goal="pagar conta", pii_types=["EMAIL"]))
    await buffer.push(scope, sid, _turn(2))
    window = await buffer.window(scope, sid, size=10)
    assert [t.turn_n for t in window] == [1, 2]
    assert window[0].goal == "pagar conta" and window[0].pii_types == ["EMAIL"]


async def test_window_slides_to_requested_size(buffer):
    scope, sid = f"t/{uuid4()}", "s1"
    for n in range(1, 5):
        await buffer.push(scope, sid, _turn(n))
    assert [t.turn_n for t in await buffer.window(scope, sid, size=2)] == [3, 4]


async def test_max_size_bounds_the_list(buffer):
    scope, sid = f"t/{uuid4()}", "s1"
    for n in range(1, 9):                 # max_size=5
        await buffer.push(scope, sid, _turn(n))
    window = await buffer.window(scope, sid, size=100)
    assert [t.turn_n for t in window] == [4, 5, 6, 7, 8]


async def test_clear(buffer):
    scope, sid = f"t/{uuid4()}", "s1"
    await buffer.push(scope, sid, _turn(1))
    await buffer.clear(scope, sid)
    assert await buffer.window(scope, sid) == []


async def test_scope_isolation(buffer):
    a, b = f"t/{uuid4()}", f"t/{uuid4()}"
    await buffer.push(a, "sess", _turn(1))
    await buffer.push(b, "sess", _turn(2))
    assert [t.turn_n for t in await buffer.window(a, "sess")] == [1]


async def test_ttl_is_set(buffer):
    scope, sid = f"t/{uuid4()}", "s1"
    await buffer.push(scope, sid, _turn(1))
    ttl = await buffer._redis.ttl(buffer._key(scope, sid))
    assert ttl > 0
