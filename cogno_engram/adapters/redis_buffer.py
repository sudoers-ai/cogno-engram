"""
cogno_engram.adapters.redis_buffer — the reference ``ConversationBuffer`` adapter.

A stateless, distributed sliding window over Redis lists (the KV-shaped port):
``RPUSH`` the turn, ``LTRIM`` to bound the window, ``EXPIRE`` for TTL — so any
worker replica serves the next turn of a session without losing episodic
context. Ported from the parent's ``redis_buffer.py`` with the business identity
collapsed into the opaque ``scope``.

Requires ``redis`` (``pip install "cogno-engram[redis]"``). Only the JSON-able
turn fields are buffered; the opaque cognition objects (``intent``/``id_result``/
``metrics``) are dropped — the short-term window doesn't need them.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from redis.asyncio import Redis, from_url

from cogno_engram.types import TurnRecord

DEFAULT_TTL_SECONDS = 24 * 3600
DEFAULT_MAX_SIZE = 50
DEFAULT_PREFIX = "engram:buf"

# The JSON-able subset of TurnRecord persisted in the window.
_FIELDS = ("session_id", "scope", "turn_n", "user_input", "response", "feedback",
           "goal", "goal_status", "sentiment", "domains", "pii_types")


def _require_scope(scope: str) -> str:
    if not scope or not scope.strip():
        raise ValueError("scope must be a non-empty string (engram isolates every row by scope)")
    return scope


def _dump(turn: TurnRecord) -> str:
    data = {f: getattr(turn, f) for f in _FIELDS}
    data["created_at"] = turn.created_at.isoformat() if turn.created_at else None
    return json.dumps(data, ensure_ascii=False)


def _load(raw: str) -> TurnRecord:
    data = json.loads(raw)
    created = data.pop("created_at", None)
    return TurnRecord(created_at=datetime.fromisoformat(created) if created else None, **data)


class RedisConversationBuffer:
    """Reference ``ConversationBuffer`` over Redis lists."""

    def __init__(self, *, redis_url: Optional[str] = None, client: Optional[Redis] = None,
                 ttl_seconds: int = DEFAULT_TTL_SECONDS, max_size: int = DEFAULT_MAX_SIZE,
                 prefix: str = DEFAULT_PREFIX) -> None:
        if client is not None:
            self._redis: Redis = client
        elif redis_url:
            self._redis = from_url(redis_url, decode_responses=True)
        else:
            raise ValueError("provide either redis_url= or client=")
        self._ttl = ttl_seconds
        self._max = max_size
        self._prefix = prefix

    def _key(self, scope: str, session_id: str) -> str:
        return f"{self._prefix}:{scope}:{session_id}"

    async def push(self, scope: str, session_id: str, turn: TurnRecord) -> None:
        _require_scope(scope)
        key = self._key(scope, session_id)
        pipe = self._redis.pipeline()
        pipe.rpush(key, _dump(turn))
        pipe.ltrim(key, -self._max, -1)     # bound the window
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def window(self, scope: str, session_id: str, *, size: int = 10) -> list[TurnRecord]:
        _require_scope(scope)
        items = await self._redis.lrange(self._key(scope, session_id), -size, -1)  # type: ignore[misc]
        return [_load(i) for i in items]

    async def clear(self, scope: str, session_id: str) -> None:
        _require_scope(scope)
        await self._redis.delete(self._key(scope, session_id))

    async def aclose(self) -> None:
        await self._redis.aclose()
