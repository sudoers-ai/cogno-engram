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


# ── the lexical score is ts_rank_cd normalised 32, EXACTLY ──────────────────────────────

_OR_QUERY = "replace(plainto_tsquery('portuguese', %s)::text, ''' & ''', ''' | ''')::tsquery"


async def test_the_lexical_score_is_ts_rank_cd_with_normalisation_32(pg):
    """``rank / (rank + 1)`` computed by Postgres itself for the same text and query — pinned
    to the VALUE, because a clamp downstream would hide an unnormalised rank as a saturated 1.0
    and the range test alone could not tell."""
    o = f"acme{uuid4().hex[:6]}/p"
    long_text = " ".join(["sábado"] * 400 + ["horário"] * 200)
    await publish(pg, o, profiles=("GUEST",), chunks=((long_text, vec(1.0)),
                                                     ("sábado curto", vec(1.0))))
    conn = await _connect()
    try:
        for query in ("sábado", "sábado horário"):
            res = await pg.search(o, profile="GUEST", text=query, limit=10)
            assert len(res.hits) == 2
            for hit in res.hits:
                cur = await conn.execute(
                    f"SELECT ts_rank_cd(to_tsvector('portuguese', %s), {_OR_QUERY})",
                    (hit.content, query))
                raw = (await cur.fetchone())[0]
                assert hit.lexical_score == pytest.approx(raw / (raw + 1.0), rel=1e-5)
                assert 0.0 < hit.lexical_score < 1.0
        # CONTROL — the raw rank of the long chunk really is above 1, so the normalisation bites.
        cur = await conn.execute(
            f"SELECT ts_rank_cd(to_tsvector('portuguese', %s), {_OR_QUERY})",
            (long_text, "sábado"))
        assert (await cur.fetchone())[0] > 1.0
    finally:
        await conn.close()


# ── the in-memory lexical stand-in: same ORDER, same zeros — never the same number ───────

#: A DECLARED corpus: ASCII words only, each query word at most once per chunk. Under the
#: `simple` configuration (lowercase only: no stemming, no unaccent) that is where the two
#: measures agree in ORDER — `ts_rank_cd` counts occurrences and weighs proximity, the in-memory
#: stand-in counts distinct words (see `in_memory._doc_lexical`). Chunk `n` carries `n` of the
#: three query words, except the two that carry none.
_ORDER_QUERY = "alfa beta gama"
_ORDER_CORPUS = (
    "delta epsilon zeta",              # 0 of 3
    "alfa eta teta",                   # 1 of 3
    "iota beta kapa gama",             # 2 of 3
    "gama lambda alfa beta",           # 3 of 3
    "mi ni xi",                        # 0 of 3
    "beta omicron",                    # 1 of 3
)


async def _order(store, owner, **kwargs) -> "list[tuple[int, bool]]":
    res = await store.search(owner, profile="GUEST", text=_ORDER_QUERY, limit=20, **kwargs)
    return [(h.ordinal, h.lexical_score == 0.0) for h in res.hits]


async def test_the_lexical_order_is_the_same_in_both_adapters():
    """Same ORDER of hits and the 0 in the same place, under `simple`, on the declared corpus.
    NOT the same number: the values are asserted to DIFFER, so this can never be read as a claim
    that the double reproduces `ts_rank_cd`."""
    pg = await fresh_postgres(DSN, ts_config="simple")
    memory = InMemoryDocumentStore(embedding_dim=EMB_DIM)
    same = vec(1.0)                          # one vector for all: the vector ties, words decide
    o = f"acme{uuid4().hex[:6]}/p"
    for store in (memory, pg):
        await publish(store, o, profiles=("GUEST",), chunks=tuple((t, same) for t in _ORDER_CORPUS))

    lexical = {name: await _order(store, o) for name, store in (("memory", memory), ("pg", pg))}
    hybrid = {name: await _order(store, o, vector=same, embed_model=MODEL_A)
              for name, store in (("memory", memory), ("pg", pg))}

    # lexical search: the chunks with a query word, best first; the zeros absent on BOTH sides
    assert lexical["memory"] == lexical["pg"] == [(3, False), (2, False), (1, False), (5, False)]
    # hybrid with tied vectors: every chunk, and the two without a query word score 0 on BOTH
    assert hybrid["memory"] == hybrid["pg"] == [(3, False), (2, False), (1, False), (5, False),
                                                (0, True), (4, True)]
    # …and NOT the same number
    mem_values = [h.lexical_score for h in (await memory.search(
        o, profile="GUEST", text=_ORDER_QUERY, limit=20)).hits]
    pg_values = [h.lexical_score for h in (await pg.search(
        o, profile="GUEST", text=_ORDER_QUERY, limit=20)).hits]
    assert mem_values != pytest.approx(pg_values)
