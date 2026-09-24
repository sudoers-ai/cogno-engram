"""The document store's Postgres-only guarantees — what an in-memory double cannot show.

Gated like every DSN suite here (``tests/conftest.py``): a database whose name says "test", and
a skip when nothing answers. These tests DROP the ``kb_*`` tables and, in one of them, create
and drop a short-lived ROLE on that server.

* the health probe EXERCISES the schema: drop ANY column of ANY document table — the list is
  read from the catalogue after a fresh migration, not written here — and the probe fails;
* the reader path never reads a stored original: a role that may not SELECT the bytes column
  can still search and list, and is refused the moment it asks for the bytes (the control);
* the vector score is the SAME number in both adapters for the same chunks;
* the document ceiling holds under concurrent creation.
"""

from __future__ import annotations

import asyncio
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest

from cogno_engram.adapters.in_memory import InMemoryDocumentStore
from cogno_engram.documents import MEDIA_MARKDOWN, DocumentLimitReached
from cogno_engram.ingest import documents_probe

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import EMB_DIM, KB_TABLES, MODEL_A, fresh_postgres, publish, vec

DSN = resolve_test_dsn()      # ENGRAM_TEST_DSN, else `engram_test` on the local server


async def _connect():
    psycopg = pytest.importorskip("psycopg")
    return await psycopg.AsyncConnection.connect(DSN, autocommit=True, connect_timeout=3)


@pytest.fixture
async def pg():
    return await fresh_postgres(DSN)


async def _kb_columns(conn) -> "list[tuple[str, str]]":
    cur = await conn.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = ANY(%s) "
        "ORDER BY table_name, ordinal_position", (list(KB_TABLES),))
    return [(r[0], r[1]) for r in await cur.fetchall()]


async def _recreate(conn) -> None:
    from cogno_engram.adapters.postgres import ensure_documents_schema
    for table in KB_TABLES:
        await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    await ensure_documents_schema(conn, embedding_dim=EMB_DIM)


# ── the probe fails when the schema is missing ANY piece ─────────────────────────────────

async def test_the_probe_passes_on_a_fresh_schema_and_fails_without_any_column(pg):
    conn = await _connect()
    try:
        columns = await _kb_columns(conn)
        # The enumeration is the catalogue's; prove it saw the tables before trusting it.
        assert {t for t, _ in columns} == set(KB_TABLES)
        assert len(columns) >= 35, columns
        await documents_probe(pg, embed_model=MODEL_A)                  # CONTROL: healthy passes

        survived = []
        for table, column in columns:
            await _recreate(conn)
            await conn.execute(f"ALTER TABLE {table} DROP COLUMN {column} CASCADE")
            try:
                await documents_probe(pg, embed_model=MODEL_A)
            except Exception:                                          # noqa: BLE001 — expected
                continue
            survived.append(f"{table}.{column}")
        assert not survived, f"the probe did not notice these columns missing: {survived}"

        for table in KB_TABLES:                                        # and a whole table
            await _recreate(conn)
            await conn.execute(f"DROP TABLE {table} CASCADE")
            with pytest.raises(Exception):
                await documents_probe(pg, embed_model=MODEL_A)
    finally:
        await _recreate(conn)
        await conn.close()


async def test_the_probe_reads_nothing_and_writes_nothing(pg):
    o = f"acme{uuid4().hex[:6]}/p"
    await publish(pg, o, profiles=("GUEST",))
    conn = await _connect()
    try:
        async def counts():
            out = {}
            for table in KB_TABLES:
                cur = await conn.execute(f"SELECT count(*) FROM {table}")
                out[table] = (await cur.fetchone())[0]
            return out
        before = await counts()
        await documents_probe(pg, embed_model=MODEL_A)
        assert await counts() == before
    finally:
        await conn.close()


# ── the reader path never reads the stored bytes ─────────────────────────────────────────

async def test_the_reader_path_never_reads_an_original(pg):
    """A role that may read every document column EXCEPT ``kb_originals.data`` runs the whole
    reader path; the one call that needs the bytes is refused — the control that proves the
    privilege really bites."""
    psycopg = pytest.importorskip("psycopg")
    from cogno_engram.adapters.postgres import PostgresDocumentStore

    o = f"acme{uuid4().hex[:6]}/p"
    doc_id = await publish(pg, o, profiles=("GUEST",), original=b"%PDF-1.7 bytes nobody reads")
    role = f"kb_reader_{uuid4().hex[:8]}"
    conn = await _connect()
    try:
        await conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'reader-pw'")
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        await conn.execute(f"GRANT SELECT ON kb_documents, kb_versions, kb_chunks, kb_tombstones "
                           f"TO {role}")
        await conn.execute(f"GRANT SELECT (document_id, version) ON kb_originals TO {role}")
        parts = urlsplit(DSN)
        host = parts.hostname or "localhost"
        netloc = f"{role}:reader-pw@{host}" + (f":{parts.port}" if parts.port else "")
        reader = PostgresDocumentStore(dsn=urlunsplit(parts._replace(netloc=netloc)),
                                       embedding_dim=EMB_DIM)

        res = await reader.search(o, profile="GUEST", text="sábado", vector=vec(1.0),
                                  embed_model=MODEL_A)
        assert [h.document_id for h in res.hits] == [doc_id]
        assert [d.id for d in await reader.readable_documents(o, profile="GUEST")] == [doc_id]
        [listed] = await reader.list_documents(o)
        assert listed.active.has_original                    # asked of the KEY, not the bytes
        with pytest.raises(psycopg.errors.InsufficientPrivilege):      # CONTROL
            await reader.get_original(o, doc_id)
    finally:
        await conn.execute(f"DROP OWNED BY {role}")
        await conn.execute(f"DROP ROLE IF EXISTS {role}")
        await conn.close()


# ── parity: the vector score is one number in both adapters ──────────────────────────────

async def test_the_vector_score_is_the_same_number_in_both_adapters(pg):
    memory = InMemoryDocumentStore(embedding_dim=EMB_DIM)
    chunks = (("sábado de manhã", vec(1.0, 0.2)), ("domingo fechado", vec(0.3, 1.0, 0.1)),
              ("feriado", vec(-0.5, 0.5)), ("horário de verão", vec(0.9, -0.4, 0.2)))
    o = f"acme{uuid4().hex[:6]}/p"
    for store in (memory, pg):
        await publish(store, o, profiles=("GUEST",), chunks=chunks)
    q = vec(0.8, 0.1, 0.3)
    got = {}
    for name, store in (("memory", memory), ("postgres", pg)):
        res = await store.search(o, profile="GUEST", text="zzz", vector=q, embed_model=MODEL_A,
                                 limit=10)
        got[name] = [(h.ordinal, h.vector_score) for h in res.hits]
    assert [x for x, _ in got["memory"]] == [x for x, _ in got["postgres"]]
    for (_, a), (_, b) in zip(got["memory"], got["postgres"]):
        assert a == pytest.approx(b, abs=1e-6)


# ── the ceiling under concurrency ────────────────────────────────────────────────────────

async def test_the_document_ceiling_holds_under_concurrent_creation(pg):
    o = f"acme{uuid4().hex[:6]}/p"

    async def create():
        try:
            await pg.create_document(o, title="M", profiles=["GUEST"], media_type=MEDIA_MARKDOWN,
                                     max_documents=3)
            return True
        except DocumentLimitReached:
            return False

    results = await asyncio.gather(*(create() for _ in range(8)))
    assert sum(results) == 3
    assert len(await pg.list_documents(o)) == 3


# ── the migration is additive and idempotent ─────────────────────────────────────────────

async def test_ensure_schema_twice_keeps_every_document(pg):
    from cogno_engram.adapters.postgres import ensure_schema
    o = f"acme{uuid4().hex[:6]}/p"
    doc_id = await publish(pg, o, profiles=("GUEST",))
    conn = await _connect()
    try:
        await ensure_schema(conn, embedding_dim=EMB_DIM)
        await ensure_schema(conn, embedding_dim=EMB_DIM)
    finally:
        await conn.close()
    assert [h.document_id for h in (await pg.search(o, profile="GUEST", text="sábado")).hits] \
        == [doc_id]
