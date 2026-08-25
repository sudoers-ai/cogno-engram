"""
End-to-end memory lifecycle on REAL infrastructure (Postgres + pgvector + Redis).

Gated: needs both ``ENGRAM_TEST_DSN`` and ``ENGRAM_TEST_REDIS_URL``; auto-skips
otherwise. Drives the full substrate — Redis buffer + Postgres store/graph +
Tier-1/2/3 consolidation + hybrid retrieval + rerank — against live services,
proving the adapters compose end to end (not just in isolation).
"""

from __future__ import annotations

import json
import os
from uuid import uuid4

import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("redis")

from cogno_engram import hypnos, rerank  # noqa: E402
from cogno_engram.adapters.postgres import (  # noqa: E402
    PostgresKnowledgeGraph,
    PostgresStore,
    ensure_schema,
)
from cogno_engram.adapters.redis_buffer import RedisConversationBuffer  # noqa: E402
from cogno_engram.types import AUDIENCE_STAFF, RetrievalQuery, TurnRecord  # noqa: E402

DSN = os.getenv("ENGRAM_TEST_DSN", "")
REDIS_URL = os.getenv("ENGRAM_TEST_REDIS_URL", "")
EMB_DIM = 8


class ScriptedBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.model = "stub"

    async def generate(self, system, prompt):
        return (self.responses.pop(0) if self.responses else "{}"), 7, 5


class StubEmbedder:
    async def embed(self, text):
        v = [0.0] * EMB_DIM
        for ch in text.lower():
            v[ord(ch) % EMB_DIM] += 1.0
        return v


async def _services_up() -> bool:
    if not DSN or not REDIS_URL:
        return False
    try:
        conn = await psycopg.AsyncConnection.connect(DSN, connect_timeout=3)
        await conn.close()
        from redis.asyncio import from_url
        client = from_url(REDIS_URL)
        await client.ping()
        await client.aclose()
        return True
    except Exception:
        return False


@pytest.fixture
async def infra():
    if not await _services_up():
        pytest.skip("set ENGRAM_TEST_DSN and ENGRAM_TEST_REDIS_URL to run")
    conn = await psycopg.AsyncConnection.connect(DSN, autocommit=True)
    for tbl in ("knowledge_edges", "knowledge_nodes", "memories", "turns", "sessions"):
        await conn.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
    await ensure_schema(conn, embedding_dim=EMB_DIM)
    await conn.close()
    buf = RedisConversationBuffer(redis_url=REDIS_URL, max_size=10)
    yield PostgresStore(dsn=DSN), PostgresKnowledgeGraph(dsn=DSN), buf
    await buf.aclose()


async def test_full_lifecycle_on_real_infra(infra):
    store, kg, buffer = infra
    emb = StubEmbedder()
    scope = f"acme/{uuid4()}"

    session = await store.create_session(scope)
    turns = [
        TurnRecord(session.id, scope, 1, "bom dia", response="oi"),
        TurnRecord(session.id, scope, 2, "quero agendar corte", response="claro",
                   goal="agendar corte", goal_status="NEW", domains=["SCHEDULING"]),
        TurnRecord(session.id, scope, 3, "prefiro pix", response="ok",
                   goal="agendar corte", goal_status="COMPLETED"),
    ]
    prev = None
    for t in turns:
        await store.save_turn(t)
        await buffer.push(scope, session.id, t)
        for m in hypnos.micro_consolidate(t, prev):
            m.embedding = await emb.embed(m.content)
            await store.save_memory(m)
        prev = t

    # Redis short-term window
    assert [w.turn_n for w in await buffer.window(scope, session.id, size=2)] == [2, 3]

    # Tier-2 periodic → Postgres memories + KG
    backend = ScriptedBackend([
        json.dumps({"preference": ["prefere pagar no pix"]}),
        json.dumps({"nodes": [{"label": "João", "type": "PERSON"}, {"label": "Corte", "type": "EVENT"}],
                    "edges": [{"source": "João", "target": "Corte", "relation": "WANTS"}]}),
    ])
    await hypnos.periodic_consolidate(store, backend, scope=scope, session_id=session.id,
                                      embedder=emb, kg=kg)
    assert [e.relation for e in await kg.walk(scope, "João", max_depth=1, audience=AUDIENCE_STAFF)] == ["WANTS"]

    # hybrid retrieval + rerank
    hits = await store.load_memories(
        scope, query=RetrievalQuery(text="prefere pagar", embedding=await emb.embed("prefere pagar")),
        limit=10)
    assert any("pix" in m.content for m in rerank(hits, query_text="prefere pagar", top_k=3))

    # Tier-3 final → summary + close
    result = await hypnos.consolidate_session(
        store, ScriptedBackend([json.dumps({"fact": ["cliente recorrente"]})]),
        session=session, kg=kg, embedder=emb)
    assert "Domains: SCHEDULING" in result.summary
    assert (await store.get_session(session.id)).ended_at is not None
