"""ts_config is interpolated into SQL by name — it must be validated as an
identifier so it can never be an injection vector. Pure, no DB needed."""

import pytest

psycopg = pytest.importorskip("psycopg")  # noqa: F841 — adapter imports psycopg at module load

from cogno_engram.adapters.postgres import (  # noqa: E402
    PostgresStore,
    _validate_ts_config,
)


def test_accepts_plain_and_schema_qualified_identifiers():
    assert _validate_ts_config("portuguese") == "portuguese"
    assert _validate_ts_config("english") == "english"
    assert _validate_ts_config("my_schema.unaccent_pt") == "my_schema.unaccent_pt"


@pytest.mark.parametrize("bad", [
    "english; DROP TABLE memories",
    "english', content) --",
    "english OR 1=1",
    "two words",
    "",
    "'quoted'",
])
def test_rejects_injection_shaped_configs(bad):
    with pytest.raises(ValueError):
        _validate_ts_config(bad)


def test_store_constructor_rejects_bad_ts_config():
    # Raises before any connection is attempted (pure validation).
    with pytest.raises(ValueError):
        PostgresStore(dsn="postgresql://x", ts_config="bad; DROP")
