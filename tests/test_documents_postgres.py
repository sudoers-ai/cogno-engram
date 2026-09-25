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
        await conn.execute(f"GRANT SELECT ON kb_documents, kb_versions, kb_chunks, kb_tombstones, "
                           f"kb_drafts TO {role}")
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
        # the management view of the TEXT does not touch the bytes either
        text = await reader.version_text(o, doc_id)
        assert text.has_original and [c.ordinal for c in text.chunks] == [0]
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


# ── parity: the removal TRAIL is one sequence in both adapters ───────────────────────────

async def _removal_trail(store, o: str) -> "list[tuple[str, str, tuple[int, ...], str]]":
    """Every way a version leaves the store, on ONE owner, then the trail it left — document ids
    replaced by the order they were created in, so two stores' uuids compare."""
    from cogno_engram.documents import KbChunk

    async def staged(doc_id: str, tag: str) -> int:
        v = await store.begin_version(o, doc_id, sha256=tag * 64, embed_model=MODEL_A,
                                      size_bytes=len(tag), original=tag.encode())
        await store.add_chunks(o, doc_id, v.version, [
            KbChunk(ordinal=0, content=f"sábado {tag}", embedding=vec(1.0))])
        return v.version

    a = await publish(store, o, profiles=("GUEST",))                    # A v1 served
    v2 = await staged(a, "2")
    assert await store.commit_version(o, a, v2, pages=0) == "ready"     # swap: (1,)
    v3 = await staged(a, "3")
    assert await store.fail_version(o, a, v3, reason="timeout")        # failed: (3,) timeout
    v4 = await staged(a, "4")
    assert await store.commit_version(o, a, v4, pages=0) == "ready"     # swap: (2, 3)
    b = (await store.create_document(o, title="B", profiles=["GUEST"],
                                     media_type=MEDIA_MARKDOWN)).id
    late, newer = await staged(b, "5"), await staged(b, "6")
    assert await store.commit_version(o, b, late, pages=0) == "superseded"   # late: (1,)
    assert await store.commit_version(o, b, newer, pages=0) == "ready"       # nothing older
    assert await store.delete_document(o, b, actor="admin-1")                # delete: (2,)
    label = {a: "A", b: "B"}
    return [(label[t.document_id], t.kind, t.versions, t.actor, t.reason)
            for t in await store.tombstones(o, limit=50)]


async def test_the_removal_trail_is_the_same_sequence_in_both_adapters(pg):
    """The in-memory double is what a host codes against; if its trail drifts from the one
    Postgres writes, the host's audit view is tested against a store it will never run on."""
    o = f"acme{uuid4().hex[:6]}/p"
    memory = await _removal_trail(InMemoryDocumentStore(embedding_dim=EMB_DIM), o)
    postgres = await _removal_trail(pg, o)
    assert memory == postgres
    # newest first — every version the commits removed named exactly once, and the failure's
    # reason on its own stone only
    assert memory == [("B", "deleted", (2,), "admin-1", ""), ("B", "superseded", (1,), "", ""),
                      ("A", "superseded", (2, 3), "", ""), ("A", "failed", (3,), "", "timeout"),
                      ("A", "superseded", (1,), "", "")]


# ── a database from before `failed` tombstones gets the column, and its rows read blank ──

async def test_a_tombstone_table_without_reason_is_migrated_and_its_old_rows_read_blank(pg):
    from cogno_engram.adapters.postgres import ensure_schema
    o = f"acme{uuid4().hex[:6]}/p"
    doc_id = await publish(pg, o)
    assert await pg.delete_document(o, doc_id, actor="admin-1")
    conn = await _connect()
    try:
        await conn.execute("ALTER TABLE kb_tombstones DROP COLUMN reason")     # the old shape
        with pytest.raises(Exception):                                          # CONTROL: bites
            await pg.tombstones(o)
        await ensure_schema(conn, embedding_dim=EMB_DIM)
    finally:
        await conn.close()
    [old] = await pg.tombstones(o)
    assert (old.kind, old.actor, old.reason) == ("deleted", "admin-1", "")


# ── parity: a version's text is the same slices in both adapters ────────────────────────

async def test_the_version_text_is_the_same_slices_in_both_adapters(pg):
    """The same upload, read through every window the host uses — whole, paged by the cursor,
    one PDF page, a draft — gives field-for-field the same answer from the double and from
    Postgres (the document id aside, which is each store's own uuid)."""
    from dataclasses import asdict, replace

    from cogno_engram.documents import MEDIA_PDF
    from cogno_engram.ingest import ingest, prepare
    from documents_support import FakePdfExtractor, Page, StubEmbedder

    pages = [Page(1, "Capa."), Page(2, "\n\n".join(" ".join(f"item{p}x{w:02d}" for w in range(40))
                                                    for p in range(10))), Page(3, "Fim.")]
    md = "# Guia\n\nIntro.\n\n## Horários\n\nSábado 8h-12h.\n\n## Preços\n\nR$ 450,00.\n"

    async def read(store) -> list:
        o = f"acme{uuid4().hex[:6]}/p"
        pdf = (await store.create_document(o, title="Regulamento", profiles=["GUEST"],
                                           media_type=MEDIA_PDF)).id
        await ingest(store, o, pdf, data=b"%PDF invented", embedder=StubEmbedder(),
                     embed_model=MODEL_A, extractor=FakePdfExtractor(pages=pages))
        mdoc = (await store.create_document(o, title="Guia", profiles=["GUEST"],
                                            media_type=MEDIA_MARKDOWN)).id
        await prepare(store, o, mdoc, data=md.encode(), embed_model=MODEL_A)
        out, after = [], None
        for _ in range(20):                    # the whole PDF, 2 at a time — BOUNDED, never hangs
            part = await store.version_text(o, pdf, after=after, limit=2)
            out.append(part)
            if not part.has_more:
                break
            after = part.next_after
        out += [await store.version_text(o, pdf, page=2), await store.version_text(o, pdf, page=2,
                                                                                    limit=1),
                await store.version_text(o, mdoc, version=1), await store.version_text(o, mdoc)]
        return [None if t is None else asdict(replace(t, document_id="-")) for t in out]

    memory = await read(InMemoryDocumentStore(embedding_dim=EMB_DIM))
    postgres = await read(pg)
    assert memory == postgres
    assert len(memory) >= 5 and memory[-1] is None             # the draft has no SERVED text
    assert memory[-2]["state"] == "awaiting_confirmation"


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


# ── unaccent: the document search folds accents on BOTH sides of the match ──────────────

_WEEKEND = (("Abrimos no Sábado de manhã.", vec(1.0)),       # ordinal 0 — accented
            ("No sabado seguinte fechamos.", vec(1.0)),      # ordinal 1 — typed without accent
            ("Horário do domingo: fechado.", vec(1.0)))      # ordinal 2 — no Saturday at all


async def _words(store, owner, text) -> "set[int]":
    res = await store.search(owner, profile="GUEST", text=text, limit=10)
    return {h.ordinal for h in res.hits}


async def test_unaccent_matches_sabado_and_sabado_both_ways_and_still_stems():
    pg = await fresh_postgres(DSN, ts_config="portuguese", unaccent=True)
    o = f"acme{uuid4().hex[:6]}/p"
    await publish(pg, o, profiles=("GUEST",), chunks=_WEEKEND)
    assert await _words(pg, o, "sabado") == {0, 1}          # «sabado» finds «Sábado»
    assert await _words(pg, o, "Sábado") == {0, 1}          # …and «Sábado» finds «sabado»
    assert await _words(pg, o, "Sábados") == {0, 1}         # the stem still applies
    assert await _words(pg, o, "orçamento") == set()        # an unrelated word finds nothing
    assert await _words(pg, o, "domingo") == {2}            # CONTROL: the search does select


async def test_without_unaccent_sabado_does_not_find_sabado():
    """The red, produced: the SAME corpus and question under the base configuration — the
    accented chunk is missed. This is the defect the flag exists to close."""
    pg = await fresh_postgres(DSN, ts_config="portuguese", unaccent=False)
    o = f"acme{uuid4().hex[:6]}/p"
    await publish(pg, o, profiles=("GUEST",), chunks=_WEEKEND)
    assert 0 not in await _words(pg, o, "sabado")
    assert await _words(pg, o, "sabado") == {1}


async def test_the_derived_configuration_folds_only_non_ascii_words_and_is_made_once():
    from cogno_engram.adapters.postgres import documents_ts_config, ensure_schema
    await fresh_postgres(DSN, ts_config="portuguese", unaccent=True)
    conn = await _connect()
    try:
        await ensure_schema(conn, embedding_dim=EMB_DIM, ts_config="portuguese", unaccent=True)
        name = documents_ts_config("portuguese", unaccent=True)
        assert name == "cogno_portuguese_unaccent"
        cur = await conn.execute(
            "SELECT t.alias, array_agg(d.dictname::text ORDER BY m.mapseqno) "
            "FROM pg_ts_config_map m JOIN pg_ts_config c ON c.oid = m.mapcfg "
            "JOIN pg_ts_dict d ON d.oid = m.mapdict "
            "JOIN LATERAL ts_token_type(c.cfgparser) t ON t.tokid = m.maptokentype "
            "WHERE c.cfgname = %s AND t.alias IN ('word', 'hword', 'hword_part', 'asciiword') "
            "GROUP BY t.alias ORDER BY t.alias", (name,))
        mapping = {alias: dicts for alias, dicts in await cur.fetchall()}
        assert mapping == {"asciiword": ["portuguese_stem"],
                           "hword": ["unaccent", "portuguese_stem"],
                           "hword_part": ["unaccent", "portuguese_stem"],
                           "word": ["unaccent", "portuguese_stem"]}
        cur = await conn.execute("SELECT count(*) FROM pg_ts_config WHERE cfgname = %s", (name,))
        assert (await cur.fetchone())[0] == 1
    finally:
        await conn.close()


async def test_unaccent_touches_the_document_tables_only():
    from cogno_engram.adapters.postgres import documents_tsv_config
    await fresh_postgres(DSN, ts_config="portuguese", unaccent=True)
    conn = await _connect()
    try:
        assert await documents_tsv_config(conn) == "cogno_portuguese_unaccent"
        cur = await conn.execute(
            "SELECT pg_get_expr(d.adbin, d.adrelid) FROM pg_attrdef d JOIN pg_attribute a "
            "ON a.attrelid = d.adrelid AND a.attnum = d.adnum "
            "WHERE d.adrelid = to_regclass('memories') AND a.attname = 'tsv'")
        memories_expr = (await cur.fetchone())[0]
        assert "unaccent" not in memories_expr
    finally:
        await conn.close()


async def test_a_changed_configuration_is_named_as_an_error_and_rebuilt_on_request(caplog):
    """Switching an EXISTING database to `unaccent` leaves `kb_chunks.tsv` as it was generated
    — `CREATE TABLE IF NOT EXISTS` never touches it — so the migration says so, by name, and
    `rebuild_documents_tsv` is the step that makes the switch real."""
    import logging

    from cogno_engram.adapters.postgres import (PostgresDocumentStore, documents_tsv_config,
                                                ensure_schema, rebuild_documents_tsv)
    plain = await fresh_postgres(DSN, ts_config="portuguese", unaccent=False)
    o = f"acme{uuid4().hex[:6]}/p"
    await publish(plain, o, profiles=("GUEST",), chunks=_WEEKEND)
    folded = PostgresDocumentStore(dsn=DSN, embedding_dim=EMB_DIM, ts_config="portuguese",
                                   unaccent=True)
    conn = await _connect()
    try:
        with caplog.at_level(logging.ERROR, logger="cogno_engram.postgres"):
            await ensure_schema(conn, embedding_dim=EMB_DIM, ts_config="portuguese", unaccent=True)
        assert "event=kb_ts_config_mismatch" in caplog.text
        assert await documents_tsv_config(conn) == "portuguese"        # nothing was altered
        assert 0 not in await _words(folded, o, "sabado")               # the silent mismatch
        await rebuild_documents_tsv(conn, ts_config="portuguese", unaccent=True)
        assert await documents_tsv_config(conn) == "cogno_portuguese_unaccent"
        assert await _words(folded, o, "sabado") == {0, 1}
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="cogno_engram.postgres"):
            await ensure_schema(conn, embedding_dim=EMB_DIM, ts_config="portuguese", unaccent=True)
        assert "kb_ts_config_mismatch" not in caplog.text                 # healthy: silent
    finally:
        await conn.close()


async def test_the_probe_runs_over_an_unaccent_store():
    pg = await fresh_postgres(DSN, ts_config="portuguese", unaccent=True)
    await documents_probe(pg, embed_model=MODEL_A)
