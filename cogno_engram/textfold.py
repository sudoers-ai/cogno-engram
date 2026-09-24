"""cogno_engram.textfold — ONE accent and case fold for every lexicon a consumer matches against;
the differences between consumers as keyword arguments the caller has to SAY.

**Why one.** A fold is copied easily — "lower-case, NFKD, drop the combining marks" is three
lines — and every copy is a place the rule can drift. Measured where this function came from:
several private copies of that rule answered identically on almost every string, and differed by
exactly one step each on the rest, and those single steps were what their lexicons depended on (a
punctuation-to-space step so a marker written with a comma still matches; a typographic
apostrophe mapped to ASCII so phone traffic matches; a whitespace collapse so a label compares
whole). A copy that drifts moves a match result with no red anywhere, so the copies became this
function and the steps became parameters, off by default.

**The base, and why each part of it is what it is:**

* **NFKD, not NFD.** The K is compatibility decomposition: a ligature (``ﬁ`` → ``fi``), a roman
  numeral (``Ⅻ`` → ``XII``), a circled digit (``①`` → ``1``) and the mathematical alphabets
  people type display names in (``𝐀𝐧𝐚`` → ``Ana``) all become the letters a person types.
* **The case fold LAST.** With the case fold first, the characters whose compatibility
  decomposition is UPPER-case — the mathematical alphabets, the superscript capitals, ``℃``/``℉``,
  ``ϒ`` — come out upper-case, a second fold changes them again, and a plain-text search never
  reaches them. With it last the fold is idempotent over EVERY code point. The two orders differ
  on the code points that were not idempotent under the old one, plus one (``Ϲ``, GREEK CAPITAL
  LUNATE SIGMA, idempotent under both and still different) — a set of about 630, whose exact size
  is a fact about the Unicode database of the running Python, not about this function.
* **``casefold``, not ``lower``.** ``casefold`` is the fold made for comparison: «Straße» is
  «Strasse». The two differ on about 190 code points after NFKD and the marks (``ß``, the Greek
  final sigma, the Cherokee syllabary…) and on the final sigma IN CONTEXT (``lower`` turns a
  word-final ``Σ`` into ``ς``; ``casefold`` into ``σ``); on every one of them ``casefold`` only
  folds FURTHER than ``lower``, never in another direction.
* ``None`` folds to ``""``.

``tests/test_textfold.py`` pins each of those properties as a SET over all of Unicode, never as a
count.

**What this is NOT: a key fold.** A value derived from this function must not become a persisted
KEY, because the day the fold improves, every key re-derives. This library's key fold is
:func:`cogno_engram.folding.fold_label`, deliberately distinct: it must agree with Postgres
``unaccent`` (it transliterates, ``ø`` → ``o``, and uses NFD), and it does not follow this one.
"""

from __future__ import annotations

import unicodedata

# The two apostrophes phones insert for "'": U+2019 RIGHT SINGLE QUOTATION MARK (the default on
# common mobile keyboards) and U+02BC MODIFIER LETTER APOSTROPHE. NFKD maps neither — they are
# punctuation and a modifier letter, not decomposable letters — so they are folded by hand, on
# request.
_APOSTROPHES = ("’", "ʼ")


def fold(text: "str | None", *, punctuation: bool = False, apostrophes: bool = False,
         collapse_whitespace: bool = False, strip: bool = False) -> str:
    """Lower-case and strip accents, so ``nao`` and ``não`` compare equal.

    Keyword steps, applied in this order after the base fold:

    * ``apostrophes`` — ``’``/``ʼ`` become ``'``;
    * ``punctuation`` — every character that is neither alphanumeric nor whitespace becomes a
      space; runs are NOT collapsed;
    * ``collapse_whitespace`` — runs of whitespace become one space, ends trimmed;
    * ``strip`` — leading/trailing whitespace removed.
    """
    # NFKD FIRST, the case fold LAST — the order is the property (see the module docstring):
    # with the case fold first, a compatibility character whose decomposition is upper-case
    # comes out upper-case and a second fold changes it again. `casefold`, not `lower`: «Straße»
    # is «Strasse».
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch)).casefold()
    if apostrophes:
        for mark in _APOSTROPHES:
            folded = folded.replace(mark, "'")
    if punctuation:
        folded = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in folded)
    if collapse_whitespace:
        folded = " ".join(folded.split())
    if strip:
        folded = folded.strip()
    return folded
