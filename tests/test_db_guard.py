"""The destructive-suite guard's predicate.

``tests/conftest.py`` refuses to let the ``DROP TABLE`` suites run against a database whose
name does not say "test". The fixture itself ends in ``pytest.exit`` and so cannot be
exercised from inside the same run — the predicate it delegates to can, and is, here.

The case that motivated it: on 2026-08-04 the Postgres suite ran with ``ENGRAM_TEST_DSN``
pointing at the live demo database ``.../cogno``. It dropped ``memories``,
``knowledge_nodes`` and ``knowledge_edges`` and recreated them with a ``vector(8)``
embedding column against a 768-dimension embedder.
"""

from __future__ import annotations

from conftest import names_a_test_database  # the sibling conftest, on pytest's rootdir path


def test_refuses_the_live_database_name():
    # the exact DSN shape that did the damage
    assert not names_a_test_database("postgresql://postgres:pw@localhost:55435/cogno")


def test_refuses_when_only_the_credentials_say_test():
    # "test" is in the DSN but the DATABASE is live — matching the whole string would
    # have let exactly this through.
    assert not names_a_test_database("postgresql://postgres:test@localhost:55432/cogno")


def test_refuses_the_default_postgres_database():
    # `postgres` exists on every server, production ones included. The old CI DSN and the
    # module docstring both used it; both now say engram_test.
    assert not names_a_test_database("postgresql://postgres:postgres@localhost:5432/postgres")


def test_accepts_a_throwaway_database():
    assert names_a_test_database("postgresql://postgres:postgres@localhost:5432/engram_test")
    assert names_a_test_database("postgresql://u:p@h:5432/TEST_db")
