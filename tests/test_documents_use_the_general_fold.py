"""The document code folds text with the GENERAL fold — never with ``fold_label``.

``fold_label`` is the definition of "these two graph LABELS are the same node" (``folding.py``):
a decision about identity, with a Postgres twin (``engram_fold``) and a parity test of its own.
The document store only needs a neutral text fold for its in-memory lexical stand-in — the
``textfold.fold`` that ``#64`` brings here — and borrowing the identity rule for it would tie
two unrelated contracts: a change to how LABELS are folded would silently move how documents
are MATCHED.

It was written as ``xfail(strict=True)`` while the stand-in still borrowed ``fold_label``
(``#64``, which brought ``textfold``, had not landed); ``#64`` landed first, the swap was made in
this PR, and the marker came off — the test now has to PASS.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ("cogno_engram/documents.py", "cogno_engram/ingest.py", "cogno_engram/chunking.py")
IN_MEMORY = "cogno_engram/adapters/in_memory.py"
SECTION = "# ── documents ──"
NAME = "fold_label"


def _documents_section() -> str:
    text = (ROOT / IN_MEMORY).read_text(encoding="utf-8")
    assert text.count(SECTION) == 1, "the in-memory documents section marker moved or doubled"
    return text[text.index(SECTION):]


def _offenders() -> "dict[str, int]":
    found = {m: (ROOT / m).read_text(encoding="utf-8").count(NAME) for m in MODULES}
    found[f"{IN_MEMORY} (documents section)"] = _documents_section().count(NAME)
    return {where: n for where, n in found.items() if n}


def test_the_scan_sees_what_it_scans():
    """CONTROL — a scan that finds nothing must be able to find something: the section is the
    real one (it holds the store), the modules exist, and the SAME count over the rest of the
    in-memory adapter (the graph, which legitimately uses the label fold) is not zero."""
    section = _documents_section()
    assert "class InMemoryDocumentStore" in section and "def _doc_terms" in section
    assert all((ROOT / m).is_file() for m in MODULES)
    graph_half = (ROOT / IN_MEMORY).read_text(encoding="utf-8").split(SECTION)[0]
    assert graph_half.count(NAME) > 0


def test_no_document_module_folds_with_the_label_rule():
    assert _offenders() == {}


def test_the_stand_in_folds_with_the_general_fold():
    """The positive half: what the stand-in DOES use. `ø` is where the two folds differ
    (`fold_label` transliterates it to `o`, the general fold leaves it), so it tells them apart."""
    from cogno_engram.adapters.in_memory import _doc_terms
    from cogno_engram.folding import fold_label
    from cogno_engram.textfold import fold

    assert fold("Søren") != fold_label("Søren")                  # CONTROL: the probe discriminates
    assert _doc_terms("Søren, SÁBADO!") == {fold("Søren"), "sabado"}
