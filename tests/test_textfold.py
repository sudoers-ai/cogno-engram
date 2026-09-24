"""``cogno_engram.textfold.fold`` — the ONE accent/case fold, pinned as PROPERTIES over all of Unicode.

The first seven tests MOVED here with the function (from the reference host, whose own module now
re-exports this one and keeps, on its side, the tests that pin each of ITS consumers against the
copy that consumer used to carry). Every property is asserted as a SET, never as a count: the
count of code points on which two folds disagree is a fact about the Unicode database of the
running Python (it grows between 3.10 and 3.12), and the property is what holds on every leg of
the matrix.
"""

from __future__ import annotations

import random
import string
import unicodedata

from cogno_engram.folding import fold_label
from cogno_engram.textfold import fold


def _every_code_point() -> "list[str]":
    return [chr(cp) for cp in range(0x110000) if not 0xD800 <= cp <= 0xDFFF]


# Three written-out bases, independent of `fold`, that the properties below are stated over.

def _order_fixed_base(s):
    """NFKD → combining marks removed → ``lower`` — the order fixed, before ``casefold``."""
    folded = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in folded if not unicodedata.combining(c)).lower()


def _new_base(s):
    """NFKD → combining marks removed → ``casefold`` — the base :func:`fold` promises."""
    folded = unicodedata.normalize("NFKD", s or "")
    return "".join(c for c in folded if not unicodedata.combining(c)).casefold()


def _old_order_base(s):
    """``lower`` → NFKD → combining marks removed — the order that was not idempotent."""
    folded = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(c for c in folded if not unicodedata.combining(c))


ORDER_CHANGED = [c for c in _every_code_point() if _order_fixed_base(c) != _old_order_base(c)]
CASEFOLD_CHANGED = [c for c in _every_code_point() if _order_fixed_base(c) != _new_base(c)]
FINAL_SIGMA_IN_CONTEXT = ["ΟΔΟΣ", "ΟΔΟΣ ΚΑΙ", "Straße ΟΔΟΣ", "ΣΟΦΙΑΣ"]

# Ϲ (U+03F9, GREEK CAPITAL LUNATE SIGMA) is idempotent under BOTH orders and still differs: the
# old order lowers it to ϲ, whose NFKD is the FINAL ς; the new order decomposes it to Σ first, and
# `lower` gives σ. It is the one member of the set that the non-idempotence does not explain.
_ORDER_CHANGED_WITHOUT_NON_IDEMPOTENCE = frozenset({"Ϲ"})


def _old_order_not_idempotent():
    return {c for c in _every_code_point() if _old_order_base(_old_order_base(c)) != _old_order_base(c)}


def _fold_twice_changes(text):
    return fold(fold(text)) != fold(text)


def _styled(text: str, upper: int, lower: int) -> str:
    """``text`` in a mathematical alphabet (A–Z from ``upper``, a–z from ``lower``) — the letters
    people type "fancy" display names in, generated rather than pasted."""
    return "".join(chr(upper + ord(c) - 65) if "A" <= c <= "Z"
                   else chr(lower + ord(c) - 97) if "a" <= c <= "z" else c for c in text)


_BOLD = (0x1D400, 0x1D41A)
_BOLD_SCRIPT = (0x1D4D0, 0x1D4EA)


# ── moved with the function ──────────────────────────────────────────────────────────


def test_the_three_differences_are_explicit_parameters_not_hidden_variants():
    """Punctuation → space, apostrophes, whitespace collapse, strip. Each is off by default and
    does exactly one thing."""
    assert fold("Não, está ótimo!") == "nao, esta otimo!"
    assert fold("Não, está ótimo!", punctuation=True) == "nao  esta otimo "      # runs kept
    assert fold("what’s ʼcause") == "what’s ʼcause"           # untouched by NFKD
    assert fold("what’s ʼcause", apostrophes=True) == "what's 'cause"
    assert fold("  a \t b  ", collapse_whitespace=True) == "a b"
    assert fold("  a \t b  ", strip=True) == "a \t b"
    assert fold("  a \t b  ") == "  a \t b  "


def test_what_the_base_fold_deliberately_leaves_alone():
    """A stand-alone combining mark is dropped like any other; compatibility characters
    (ligature, roman numeral, circled digit) expand — that is the K in NFKD. ``ß`` expands to
    ``ss`` because the base case-FOLDS (``lower`` would leave it), so «Straße» is «Strasse»."""
    assert fold("Straße") == fold("Strasse") == "strasse"
    assert fold("À") == "a"
    assert fold("ﬁnal Ⅻ ①") == "final xii 1"


def test_the_fold_is_idempotent_on_every_code_point():
    bad = [hex(ord(c)) for c in _every_code_point() if _fold_twice_changes(c)]
    assert bad == [], f"folding a folded string changed it on {len(bad)} code points: {bad[:5]}"


def test_the_fold_is_idempotent_on_100_000_strings():
    rng = random.Random(0)
    alphabet = "aeiouçãõáéíóúâêôàüñßøłİıﬁ ’-ÆæŒœ€–ᴬϒ℃𝐉𝐨𝓜" + string.ascii_letters
    corpus = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12)))
              for _ in range(100_000)]
    bad = [t for t in corpus if _fold_twice_changes(t)]
    assert bad == [], f"{len(bad)} strings change on a second fold, e.g. {bad[:3]!r}"


def test_a_display_name_in_fancy_letters_folds_to_the_plain_name():
    """The case that made the order a defect: the fold is what search compares, and a label in
    mathematical bold/script letters must fold to what a person types."""
    assert fold(_styled("Zuleide Brum", *_BOLD)) == "zuleide brum"
    assert fold(_styled("Odete da Farofa", *_BOLD_SCRIPT)) == "odete da farofa"
    assert fold("ᴬna") == "ana"
    assert fold("Straße") == "strasse"     # casefold (the ORDER never touched ß)


def test_the_order_changed_exactly_where_the_old_order_was_not_idempotent_plus_lunate_sigma():
    """The two families every sentence about the order quotes (non-idempotent under the old
    order, plus Ϲ) — asserted as the SET they make up, never as a count."""
    non_idempotent = _old_order_not_idempotent()
    assert non_idempotent, "the old order was idempotent everywhere — this set no longer reaches it"
    assert set(ORDER_CHANGED) == non_idempotent | _ORDER_CHANGED_WITHOUT_NON_IDEMPOTENCE, (
        f"added: {sorted(hex(ord(c)) for c in set(ORDER_CHANGED) - non_idempotent)[:5]}, "
        f"missing: {sorted(hex(ord(c)) for c in non_idempotent - set(ORDER_CHANGED))[:5]}")
    assert "\U0001d409" in ORDER_CHANGED and "ᴬ" in ORDER_CHANGED and "Ϲ" in ORDER_CHANGED
    assert "é" not in ORDER_CHANGED and "ß" not in ORDER_CHANGED and "İ" not in ORDER_CHANGED


def test_casefold_changed_only_by_folding_further_and_the_final_sigma_in_context():
    """On every code point where ``casefold`` and ``lower`` differ after NFKD and the marks,
    casefolding the lowered form lands on the same answer — the change is extra folding, never a
    different direction; and the final sigma differs IN CONTEXT, which no code-point list holds."""
    assert CASEFOLD_CHANGED
    further = [hex(ord(c)) for c in CASEFOLD_CHANGED if _new_base(_order_fixed_base(c)) != _new_base(c)]
    assert further == [], f"casefold did not merely refine lower on {len(further)}: {further[:5]}"
    assert "ß" in CASEFOLD_CHANGED and "ς" in CASEFOLD_CHANGED and "Ꭰ" in CASEFOLD_CHANGED
    assert "é" not in CASEFOLD_CHANGED and "Σ" not in CASEFOLD_CHANGED    # Σ alone: σ either way
    for text in FINAL_SIGMA_IN_CONTEXT:                                   # …but not in a word
        assert _order_fixed_base(text) != _new_base(text), text


# ── the function's own parity: it takes the new order AND casefold, where each one matters ──


def test_fold_answers_the_new_base_where_the_ORDER_changed_and_the_old_order_diverges_there():
    """The parity that pins THE FUNCTION to the order: on every code point the order changed,
    :func:`fold` answers NFKD → marks → casefold — and, the control, the old order DIVERGES from it
    there, so the set reaches the change instead of agreeing by accident."""
    wrong = [(hex(ord(c)), fold(c), _new_base(c)) for c in ORDER_CHANGED if fold(c) != _new_base(c)]
    assert not wrong, f"fold does not answer the new order on {len(wrong)}: {wrong[:3]}"
    assert sum(1 for c in ORDER_CHANGED if _old_order_base(c) != fold(c)) > 0


def test_fold_answers_casefold_where_casefold_changed_and_lower_diverges_there():
    reached = CASEFOLD_CHANGED + FINAL_SIGMA_IN_CONTEXT
    wrong = [(t, fold(t), _new_base(t)) for t in reached if fold(t) != _new_base(t)]
    assert not wrong, f"fold does not answer casefold on {len(wrong)}: {wrong[:3]}"
    assert all(_order_fixed_base(t) != fold(t) for t in reached), "the control: lower diverges"


def test_none_folds_to_empty():
    assert fold(None) == "" and fold("") == ""


# ── the OTHER fold of this library stays distinct, and says so ───────────────────────


def test_the_label_fold_is_NOT_this_fold_and_the_difference_is_the_transliteration():
    """``folding.fold_label`` derives a node KEY and must agree with Postgres ``unaccent``, so it
    transliterates and uses NFD; this fold does neither. If this ever went green on equality, one
    of the two docstrings would be lying — and aligning them "in passing" would re-key a graph."""
    assert fold_label("Øverby") == "overby" and fold("Øverby") == "øverby"
    assert fold_label("Æsir") == "aesir" and fold("Æsir") == "æsir"
    assert fold("ﬁ") == "fi", "NFKD expands the ligature"
    differ = sum(1 for c in _every_code_point() if fold_label(c) != fold(c))
    assert differ > 0, "the two folds agree everywhere — then there is no reason for two"
