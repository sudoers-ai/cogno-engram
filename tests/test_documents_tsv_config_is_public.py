"""`documents_tsv_config` is PUBLIC API: a host reads what `kb_chunks.tsv` was built with.

Before it was public, the host imported the private `_documents_tsv_config` — a private import
across libraries, the shape that breaks silently on a rename. These tests pin the public name,
the compatibility alias, and the parsing of the catalogue expression without a database (a fake
connection hands back the exact strings `pg_get_expr` returns); the real catalogue read runs in
`test_documents_postgres.py`.
"""

from __future__ import annotations

import pytest

from cogno_engram.adapters import postgres as pg


class _Cursor:
    def __init__(self, row):
        self._row = row

    async def fetchone(self):
        return self._row


class _Conn:
    """Answers the catalogue query with ``row`` — a tuple, a dict row, or ``None``."""

    def __init__(self, row):
        self._row = row
        self.sql: "list[str]" = []

    async def execute(self, sql, *args):
        self.sql.append(sql)
        return _Cursor(self._row)


def test_the_reader_is_public_and_the_old_name_is_the_same_object():
    assert pg.documents_tsv_config is pg._documents_tsv_config
    assert not pg.documents_tsv_config.__name__.startswith("_")


@pytest.mark.parametrize("expr,want", [
    ("to_tsvector('cogno_portuguese_unaccent'::regconfig, content)", "cogno_portuguese_unaccent"),
    ("to_tsvector('portuguese'::regconfig, content)", "portuguese"),
    ("to_tsvector('simple'::regconfig, content)", "simple"),
    ("lower(content)", None),                       # a generated column of another shape
    ("", None),
])
async def test_the_generated_expression_is_read_back_to_its_configuration(expr, want):
    assert await pg.documents_tsv_config(_Conn((expr,))) == want
    assert await pg.documents_tsv_config(_Conn({"pg_get_expr": expr})) == want   # dict rows


async def test_no_column_is_None_and_the_query_names_kb_chunks_tsv():
    conn = _Conn(None)
    assert await pg.documents_tsv_config(conn) is None
    assert "to_regclass('kb_chunks')" in conn.sql[0] and "attname = 'tsv'" in conn.sql[0]


async def test_the_config_the_schema_BUILDS_is_the_one_the_reader_reads():
    """The pair that matters to a host: what `documents_ts_config` names is exactly what the
    reader gives back from the column that name generates."""
    for base, unaccent in (("portuguese", False), ("portuguese", True), ("simple", False)):
        name = pg.documents_ts_config(base, unaccent=unaccent)
        expr = f"to_tsvector('{name}'::regconfig, content)"
        assert await pg.documents_tsv_config(_Conn((expr,))) == name
