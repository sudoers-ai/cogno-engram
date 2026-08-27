"""The edge-status vocabulary is written in ONE place, and the SQL reads it from there.

Why this file exists, and why it is a SOURCE-TEXT test rather than a behavioural one.

``types.py`` declares ``accepted``/``proposed``/``rejected`` and the postgres adapter already
IMPORTED those constants — and then hand-wrote the same three words inside seven SQL fragments,
forty lines from a query that interpolates the constant properly. Both halves of one file
disagreed about where the vocabulary lives.

**A drift here does not raise; it stops matching.** ``AND e.status = 'acepted'`` is valid SQL
that returns zero rows, so ``walk`` reaches nothing and ``neighbors`` discloses nobody — a graph
that looks empty rather than broken. The DDL default is the same shape from the other side: a
value outside the vocabulary is mapped to ``proposed`` by ``sanitize_edge_status`` on the way
back out, so every new edge would quietly become unreviewed and leave the walk.

The behavioural suites (``test_edge_curation``, ``test_audience_parity``,
``test_postgres_integration``) do cover those paths, and a mutation in any of the seven sites
dies in them. **They cannot cover the site that has not been written yet** — a fourth status, or
an eighth query — which is what this test is for, in the mould of
``cogno-anima``'s ``test_code_domains_match_prompt_domains_exactly``: pin the LIST, not a count.
"""
from __future__ import annotations

import re
from pathlib import Path

from cogno_engram.types import (EDGE_ACCEPTED, EDGE_PROPOSED, EDGE_REJECTED,
                                VALID_EDGE_STATUS)

_ADAPTERS = Path(__file__).resolve().parents[1] / "cogno_engram" / "adapters"


def _code_lines(path: Path) -> list[tuple[int, str]]:
    """Lines with the comment tail removed, so PROSE may quote what code may not write.

    A comment explaining ``status = 'accepted'`` is documentation and must stay readable; only
    a string the interpreter ships to Postgres is a second copy of the vocabulary. The split is
    on the first ``#``, which is crude in general and exact here: no SQL fragment in this
    package contains one (asserted below, so the crudeness cannot rot into a blind spot).
    """
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        code = line.split("#", 1)[0]
        if code.strip():
            out.append((n, code))
    return out


def test_no_status_literal_is_typed_by_hand_in_sql() -> None:
    """No adapter writes a status value as a literal — every use goes through the constant."""
    offenders = []
    for path in sorted(_ADAPTERS.glob("*.py")):
        for n, code in _code_lines(path):
            for value in sorted(VALID_EDGE_STATUS):
                # the value in quotes: `'accepted'` / `"accepted"`. The NAME `EDGE_ACCEPTED`
                # never matches, which is the whole point of the distinction.
                if re.search(rf"""(['"]){re.escape(value)}\1""", code):
                    offenders.append(f"{path.name}:{n}: {code.strip()}")
    assert not offenders, (
        "a status value is typed by hand instead of read from cogno_engram.types:\n  "
        + "\n  ".join(offenders))


def test_the_comment_split_cannot_hide_a_real_offender() -> None:
    """The `#`-split above is only safe while no SQL fragment contains a `#`.

    Without this, someone adds a query carrying a `#` (a jsonb path operator, a colour, an
    anchor in a URL) and the check silently stops reading the rest of that line — the offender
    would be INSIDE the discarded half. A filter that quietly narrows its own denominator is
    the defect this package keeps catching elsewhere.
    """
    for path in sorted(_ADAPTERS.glob("*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            before = line.split("#", 1)[0]
            if "#" in line and ("'" in before or '"' in before):
                # a `#` after an opening quote on the same line would be inside a string
                assert before.count("'") % 2 == 0 and before.count('"') % 2 == 0, (
                    f"{path.name}:{n} has a `#` inside a string — the comment split in "
                    f"`_code_lines` would discard code: {line.strip()}")


def test_the_vocabulary_is_exactly_these_three_names() -> None:
    """Pin the LIST, not its size.

    A fourth status must be a DECISION: adding it to `VALID_EDGE_STATUS` alone turns this red,
    and whoever turns it green has to look at every query that filters on status — which is the
    review that a passing count would not have asked for.
    """
    assert VALID_EDGE_STATUS == {EDGE_ACCEPTED, EDGE_PROPOSED, EDGE_REJECTED}
    assert (EDGE_ACCEPTED, EDGE_PROPOSED, EDGE_REJECTED) == ("accepted", "proposed", "rejected")


def test_every_status_the_sql_filters_on_comes_from_the_vocabulary() -> None:
    """The other direction: the postgres adapter interpolates only known status names.

    The test above forbids the literal; this one forbids the plausible-looking constant. A
    query that read `{EDGE_ARCHIVED}` would be an f-string over a name that is not part of the
    vocabulary — it would render, and the filter would match nothing.
    """
    src = (_ADAPTERS / "postgres.py").read_text()
    interpolated = set(re.findall(r"\{(EDGE_[A-Z_]+)\}", src))
    known = {"EDGE_ACCEPTED", "EDGE_PROPOSED", "EDGE_REJECTED"}
    assert interpolated, "the adapter stopped interpolating the vocabulary — is it hand-typed again?"
    assert interpolated <= known, f"unknown status constant in SQL: {sorted(interpolated - known)}"
