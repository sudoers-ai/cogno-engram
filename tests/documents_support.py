"""Shared helpers for the document-store tests — invented data only.

Not a test module (no ``test_`` prefix): the modules that use a database import from here and
expose their own module-level ``DSN``, which is what the collection guard in ``conftest.py``
inspects before any of them may DROP a table.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pytest

from cogno_engram.adapters.in_memory import InMemoryDocumentStore
from cogno_engram.documents import COMMIT_READY, MEDIA_MARKDOWN, KbChunk, embed_model_label

EMB_DIM = 8
MODEL_A = embed_model_label("stub:alpha", EMB_DIM)
MODEL_B = embed_model_label("stub:beta", EMB_DIM)
KB_TABLES = ("kb_chunks", "kb_originals", "kb_drafts", "kb_versions", "kb_tombstones",
             "kb_documents")
DERIVED_TS_CONFIGS = ("cogno_portuguese_unaccent", "cogno_simple_unaccent")


def vec(*head: float) -> list[float]:
    return (list(head) + [0.0] * EMB_DIM)[:EMB_DIM]


async def fresh_postgres(dsn: str, *, ts_config: str = "portuguese", unaccent: bool = False,
                         **kwargs):
    """A ``PostgresDocumentStore`` over freshly created ``kb_*`` tables, or a SKIP. The tables and
    the store are built with the SAME ``ts_config`` — a store querying with one configuration
    over a ``tsv`` generated with another matches nothing and says nothing."""
    psycopg = pytest.importorskip("psycopg")
    if not dsn:
        pytest.skip("no test Postgres answers — the in-memory leg ran")
    from cogno_engram.adapters.postgres import PostgresDocumentStore, ensure_schema
    try:
        conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True, connect_timeout=3)
    except Exception as exc:                            # noqa: BLE001
        pytest.skip(f"test Postgres unreachable: {type(exc).__name__}")
    for table in KB_TABLES:
        await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    # The derived unaccent configurations too: they are created ONCE and then left alone by
    # design, so a stale one from an earlier run would be measured instead of the code.
    for derived in DERIVED_TS_CONFIGS:
        await conn.execute(f"DROP TEXT SEARCH CONFIGURATION IF EXISTS {derived}")
    await ensure_schema(conn, embedding_dim=EMB_DIM, ts_config=ts_config, unaccent=unaccent)
    await conn.close()
    return PostgresDocumentStore(dsn=dsn, embedding_dim=EMB_DIM, ts_config=ts_config,
                                 unaccent=unaccent, **kwargs)


def store_factory(kind: str, dsn: str):
    async def factory(**kwargs):
        if kind == "memory":
            return InMemoryDocumentStore(embedding_dim=EMB_DIM, **kwargs)
        return await fresh_postgres(dsn, **kwargs)
    return factory


async def publish(store, owner_key: str, *, title: str = "Manual", profiles=("EMPLOYEE",),
                  chunks=(("Horários › Sábado\n\nAbrimos das 8h às 12h.", vec(1.0)),),
                  model: str = MODEL_A, original: bytes = b"# Manual\n") -> str:
    doc = await store.create_document(owner_key, title=title, profiles=list(profiles),
                                      media_type=MEDIA_MARKDOWN)
    v = await store.begin_version(owner_key, doc.id, sha256=hashlib.sha256(original).hexdigest(),
                                  embed_model=model, size_bytes=len(original), original=original)
    ok = await store.add_chunks(owner_key, doc.id, v.version, [
        KbChunk(ordinal=i, content=text, heading_path=(title,), embedding=e)
        for i, (text, e) in enumerate(chunks)])
    assert ok
    assert await store.commit_version(owner_key, doc.id, v.version, pages=0) == COMMIT_READY
    return doc.id


class StubEmbedder:
    """Deterministic, zero-network. ``embed_with_usage`` reports a KNOWN token count per call
    (the number of whitespace words, plus ``bonus``) so a test can sum what was reported."""

    def __init__(self, dim: int = EMB_DIM, *, fail_after: "int | None" = None,
                 width: "int | None" = None, bonus: int = 3) -> None:
        self.dim = dim
        self.width = width or dim
        self.fail_after = fail_after
        self.bonus = bonus
        self.calls = 0
        self.reported: list[int] = []

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [(b - 128) / 128.0 for b in digest[: self.width]]

    async def embed_with_usage(self, text: str):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("embedder down")
        tokens = len(text.split()) + self.bonus
        self.reported.append(tokens)
        return self._vector(text), tokens


class PlainEmbedder:
    """An embedder WITHOUT usage reporting — only ``embed``."""

    def __init__(self, dim: int = EMB_DIM) -> None:
        self.dim = dim
        self.calls = 0

    async def embed(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0] + [0.0] * (self.dim - 1)


@dataclass
class Page:
    number: int
    text: str


@dataclass
class Bookmark:
    level: int
    title: str
    page: int


@dataclass
class Extracted:
    pages: list
    outline: list = field(default_factory=list)


class ExtractorFailure(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class FakePdfExtractor:
    """A ``TextExtractor`` double: records what it was handed and answers as told."""

    media_types = frozenset({"application/pdf"})

    def __init__(self, pages=None, *, outline=None, raises: "str | None" = None,
                 answer=None) -> None:
        self.pages = pages if pages is not None else [Page(1, "Página um.")]
        self.outline = outline or []
        self.raises = raises
        self.answer = answer
        self.calls: list[dict] = []

    async def extract(self, data: bytes, *, media_type: str, max_bytes: int, max_pages: int,
                      timeout_s: float):
        self.calls.append(dict(size=len(data), media_type=media_type, max_bytes=max_bytes,
                               max_pages=max_pages, timeout_s=timeout_s))
        if self.raises is not None:
            raise ExtractorFailure(self.raises)
        if self.answer is not None:
            return self.answer
        return Extracted(pages=list(self.pages), outline=list(self.outline))
