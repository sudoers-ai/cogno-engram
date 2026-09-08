"""No LIVE personal datum may enter this tree — and the allowlists are ENUMERATED here.

This repository is PUBLIC and Apache-licensed. A personal datum committed here is not
untidiness, it is disclosure: it ships to every clone, every mirror and every model that
scrapes GitHub, and it does so under a licence that invites redistribution.

**Why this guard exists, and the honest thing to say first.** The sweep of 2026-09-08 found
in this repo exactly ONE live datum, and it is of the class this guard CANNOT catch: a real
third party's first name, quoted from the live knowledge graph in ``CHANGELOG.md``. It was
anonymised by hand in the same commit that added this file. So this guard did not find that,
and would not have.

What it is for is the class that DOES have a shape, and that this repo has so far been lucky
about: the same sweep found a real subscriber's mobile number pasted into fixtures of
``cogno-anima`` (2 files) and ``cogno-soma`` (1). Every one of them arrived the same way —
somebody reproduced a live turn and pasted it, and the paste outlived the debugging session.
This repo's tree currently holds no phone number of its own at all; the two entries in
``ALLOWED_SYNTHETIC`` below are values THIS FILE writes, in its own prose and its own twins.
That is a guard standing where nothing has fallen yet, which is the only useful place for one.

**Anonymise, never delete.** The value of a line like the one in ``CHANGELOG.md`` is the
MEASUREMENT it carries — *ten groups where the same label is typed both CONCEPT and PERSON* —
and a ``git rm`` destroys that while a consistent synthetic preserves it whole. The name was
incidental to the finding and is now the same synthetic ``cogno-host`` chose for that person,
so one entity has one fake across the ecosystem. **Changing the WORDS of a measured quotation
would turn it into a fabricated one**: the rule is anonymise the NAME and keep the SENTENCE,
never rewrite the sentence.

**The allowlists are literal sets in this file, and that is the whole design.** A guard that
answered "is this number one of our contacts?" by querying the customer base would be a SECOND
copy of the problem — the repo would hold, or reach for, the very data it exists to keep out.
So the question it answers is the narrow one it can answer offline: *is this one of the
synthetics this repo has already agreed to use?* Anything else fails, including a real value
that belongs to nobody in particular. Adding an entry is a deliberate act, with a comment.

**What counts as a phone**, and why it is not "ten to thirteen digits": that would fire on
timestamps, ids, hashes and hex. A hit must be a Brazilian number by CONSTRUCTION — an
assigned area code (:data:`_DDD`), then either a 9-digit mobile opening with ``9`` or an
8-digit landline opening with ``2``-``5`` — optionally carrying the ``55`` country code and
the usual separators. The country code is stripped before comparison, so the same fake written
another way is one entry and not two: a value cannot slip through by being reformatted.

**The regex finds a SHAPE; the construction rules DECIDE.** This is inherited as a lesson, not
as a style. ``cogno-host``'s first cut spelled the mobile ``9`` into the pattern itself
(``9?[0-9]{4}``), which made the subscriber check unreachable — a mutation deleting that check
stayed green, because no input could ever arrive at it. Dead code inside a guard is a claim
the guard does not check. The pattern here therefore matches any 4-5 digit block and
:func:`phone_numbers_in` alone decides what is a number; both halves of that decision have a
mutation twin below.

**Enumerated with ``git ls-files``, not a walk of the disk.** An uncommitted file is not a
contract, and a scratch copy under ``tests/`` is not something this repo ever published. The
walk would also have to learn about ``.venv``, ``htmlcov`` and ``__pycache__``; the index
already knows.

────────────────────────────────────────────────────────────────────────────────────────────
**WHAT THIS DOES NOT CATCH.** Stated here so nobody reads more into a green than is in it.

* **A person's NAME.** This is measured, not assumed. ``cogno-host`` ran its own 115-name
  lexicon over its tree: **732 firings across 90 of 422 files**, and of the **5** real names
  that had to be removed it would have caught **1**. A first name has no shape that separates
  it from a persona, a fixture or half the prose in a Portuguese repo, and the only way to
  raise the recall is to LIST THE REAL NAMES in the repo — which is re-leaking them. That
  class is a human review rule, and it is written where reviewers look (``README.md``): *if
  you quote a turn, anonymise the NAME and keep the SENTENCE.* Changing the words of a
  MEASURED quotation turns it into a FABRICATED one, which is why the rule is anonymise, not
  rewrite.
* **CPF/CNPJ.** A document number has exactly one structural property — its check digits — and
  every generator emits valid ones, so a real CPF and a synthetic CPF are indistinguishable by
  construction. A guard here could only be an allowlist with no filter in front of it, firing
  on every new legitimate fixture until somebody switched it off. The 2026-09-08 sweep found
  no document number of any kind in this repo, real or synthetic.
* **Addresses, postcodes, and tenant/session ids.** Same reason — no shape to key on that does
  not also fire on ordinary prose — and the sweep found none of them live here. Enumerating
  live ids in this file would import the very data it exists to keep out.
* **Graph labels generally.** This repo stores a knowledge graph, so a node label can be
  anything a contact ever said. Nothing in a fixture distinguishes ``Anselmo/PERSON`` from
  ``Cardiologia/CONCEPT`` by shape; that is the same NAME problem one bullet up, and it is why
  the one finding here was fixed by review and not by this file.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

# This repo keeps its tests directly under ``tests/`` (no ``unit`` level), so the root is ONE
# parent up, not two. ``test_the_sweep_actually_reads_the_tree`` is what catches this if the
# layout ever moves — a wrong root makes the sweep read the wrong tree, or no tree at all.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Area codes actually assigned in Brazil. A number whose DDD is not one of these is not a
# number, whatever its length.
_DDD = frozenset(
    str(d) for d in (
        *range(11, 20), 21, 22, 24, 27, 28, *range(31, 36), 37, 38,
        *range(41, 50), 51, 53, 54, 55, *range(61, 70),
        71, 73, 74, 75, 77, 79, *range(81, 90), *range(91, 100),
    )
)

# Shape only. Boundaries exclude an adjacent hyphen on both sides: separators are legal INSIDE
# a written number (``97000-1111``) and never at its edges, so the exclusion costs nothing and
# removes the one false positive worth removing — the digit groups of a UUID.
_RUN = re.compile(
    r"(?<![0-9A-Za-z-])"
    r"((?:\+?55)?[\s\-.()]*[1-9][0-9][\s\-.()]*[0-9]{4,5}[\s\-.()]*[0-9]{4})"
    r"(?![0-9A-Za-z-])"
)

# The synthetics this repo has agreed to use, in NATIONAL form (DDD + subscriber, no country
# code). Every entry is a number nobody answers. Adding one is a deliberate act — say which
# fixture needs it and why the value is safe.
# Both entries are values THIS FILE writes. The sweep enumerates the index and this file is in
# it, so a guard that quotes a number in its own prose has to allow that number — there is no
# third state. Nothing else in this repo is a phone.
ALLOWED_SYNTHETIC = frozenset({
    "11970001111",    # the ecosystem's example contact (cogno-host's choice for that
                      # subscriber) — quoted in the boundary note above, nowhere else
    "11987654321",    # canonical fake, used only by this file's own mutation twins below
})

# Mail domains this repo has agreed to use. RFC 2606 reserves example.com/.net/.org and the
# .test/.invalid/.example TLDs precisely so fixtures need not borrow somebody's real domain;
# the rest are enumerated one by one.
ALLOWED_EMAIL_DOMAINS = frozenset({
    "x.com",                         # one-letter stub, the node/edge fixtures
    "db.prod.example.com",           # a DSN host in a docstring, not a mailbox — the detector
                                     # reads `user@db.prod.example.com` out of a connection URI
    "example.com", "y.zz",           # RFC 2606 and the stubs this file's own twins use
})

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9.-]+\.[A-Za-z]{2,})\b")

# Binary/large blobs the index tracks but nobody pastes a phone number into.
_SKIP_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".woff",
                            ".woff2", ".ttf", ".zip", ".gz", ".whl", ".so", ".lock"})


def _national(digits: str) -> "str | None":
    """Country code stripped, so one number is one entry however it was written."""
    if digits.startswith("55") and len(digits) in (12, 13):
        return digits[2:]
    if len(digits) in (10, 11):
        return digits
    return None


def phone_numbers_in(text: str) -> "set[str]":
    """Every Brazilian phone number in ``text``, in national form.

    Pure, and the mutation twins call it directly — a detector that scanned nothing would
    otherwise pass the sweep by finding nothing."""
    out: set[str] = set()
    for match in _RUN.finditer(text):
        digits = re.sub(r"[^0-9]", "", match.group(1))
        national = _national(digits)
        if national is None:
            continue
        ddd, subscriber = national[:2], national[2:]
        if ddd not in _DDD:
            continue
        if len(subscriber) == 9 and not subscriber.startswith("9"):
            continue
        if len(subscriber) == 8 and subscriber[0] not in "2345":
            continue
        out.add(national)
    return out


def email_domains_in(text: str) -> "set[str]":
    """Every mail domain in ``text``, lowercased. Pure, for the same reason."""
    return {m.group(1).lower().rstrip(".") for m in _EMAIL.finditer(text)}


def _tracked_files() -> "list[Path]":
    """The INDEX, not the disk: an uncommitted file is not a contract."""
    out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
                         capture_output=True, text=True, check=True)
    return [REPO_ROOT / rel for rel in out.stdout.split("\0")
            if rel and Path(rel).suffix.lower() not in _SKIP_SUFFIXES]


def _readable() -> "list[tuple[Path, str]]":
    found: list[tuple[Path, str]] = []
    for path in _tracked_files():
        try:
            found.append((path, path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue          # binary or unreadable: nothing to read a datum in
    return found


def test_the_sweep_actually_reads_the_tree():
    """The denominator, pinned. A guard that swept an empty tree would be green forever."""
    files = _tracked_files()
    assert len(files) > 60, f"only {len(files)} files swept — the enumeration is broken"
    names = {p.name for p in files}
    assert {"CHANGELOG.md", "fold_migration.py"} <= names, (
        "the tree is not being read — CHANGELOG.md is where this repo's one live datum was")


def test_no_phone_number_outside_the_enumerated_synthetics():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for number in phone_numbers_in(text) - ALLOWED_SYNTHETIC:
            offenders.setdefault(number, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "phone number(s) outside ALLOWED_SYNTHETIC — anonymise the VALUE (do not delete the "
        "line: it carries a measurement), keep the form, then add the synthetic to the list "
        f"with a comment: { {k: v[:3] for k, v in offenders.items()} }")


def test_no_email_outside_the_enumerated_domains():
    offenders: dict[str, list[str]] = {}
    for path, text in _readable():
        for domain in email_domains_in(text) - ALLOWED_EMAIL_DOMAINS:
            offenders.setdefault(domain, []).append(str(path.relative_to(REPO_ROOT)))
    assert not offenders, (
        "mail domain(s) outside ALLOWED_EMAIL_DOMAINS — prefer an RFC 2606 reserved domain "
        "(example.com, or a .test/.invalid TLD) for any new fixture: "
        f"{ {k: v[:3] for k, v in offenders.items()} }")


def test_the_detector_finds_a_number_in_every_shape_it_is_written():
    """Mutation twin for the FORM half: the same number, five ways, is one entry."""
    for written in ("5511987654321", "+55 11 98765-4321", "(11) 98765-4321",
                    "11 98765-4321", "11987654321"):
        assert phone_numbers_in(f"liga no {written} depois") == {"11987654321"}, written


def test_the_detector_does_not_fire_on_things_that_are_not_numbers():
    """The other half of the bar: a detector that over-matches gets switched off."""
    assert phone_numbers_in("123e4567-e89b-42d3-a456-426614174000") == set()   # uuid
    assert phone_numbers_in("sha 5511970001111abcdef") == set()                # inside a hash
    assert phone_numbers_in("(01) 98765-4321") == set()                        # no such DDD
    # Both halves of the construction rule, each REACHABLE — this is the mutation that
    # survived in cogno-host, and it survived because the pattern decided instead of the rule.
    assert phone_numbers_in("(11) 68765-4321") == set()   # 9 digits not opening with 9
    assert phone_numbers_in("(11) 8888-7777") == set()    # 8 digits outside 2-5


def test_the_email_detector_reads_the_domain_and_not_the_mailbox():
    assert email_domains_in("escreve para joao.silva@example.com hoje") == {"example.com"}
    assert email_domains_in("a@x.com e b@y.zz") == {"x.com", "y.zz"}
    assert email_domains_in("sem correio nenhum aqui") == set()
