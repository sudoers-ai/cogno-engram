"""`KbDocument.sections` — a document's OUTLINE, for a reader who has only its title.

A document titled after its year or its owner says nothing about what it covers; its section
headings do. `readable_documents` now returns them, in document order, for the ACTIVE version
of every document the profile may read — in BOTH adapters, by ONE pure rule
(`section_headings`): the section depth is the FIRST depth of `heading_path` with at least two
distinct headings, so nothing is hard-coded to a Markdown level.

Invented data only (a made-up warehouse business).
"""

from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from cogno_engram.documents import (
    COMMIT_READY,
    MAX_SECTIONS_PER_DOCUMENT,
    MEDIA_MARKDOWN,
    KbChunk,
    section_headings,
)

from conftest import resolve_test_dsn  # noqa: E402 — the sibling conftest, on pytest's path
from documents_support import MODEL_A, store_factory, vec

# Module-level ON PURPOSE (see test_documents_store.py): the Postgres leg DROPs the kb_* tables.
DSN = resolve_test_dsn()

TITLE = "Relatório Anual 2026"
#: (heading_path, text) — the title, a UNIQUE H1, then the H2s the document is really about.
REPORT = (
    ((TITLE, "Relatório Anual", "Receita de Aluguel — Galpão Norte"), "Janeiro 4.200."),
    ((TITLE, "Relatório Anual", "Receita de Aluguel — Galpão Norte"), "Fevereiro 4.200."),
    ((TITLE, "Relatório Anual", "Folha de Pagamento"), "Três pessoas."),
    ((TITLE, "Relatório Anual", "Impostos", "Municipal"), "IPTU pago."),
    ((TITLE, "Relatório Anual", "Impostos", "Federal"), "Em dia."),
)
SECTIONS = ("Receita de Aluguel — Galpão Norte", "Folha de Pagamento", "Impostos")


def _rows(chunks) -> "list[tuple[int, int, str]]":
    return [(depth, ordinal, heading) for ordinal, (path, _) in enumerate(chunks)
            for depth, heading in enumerate(path)]


# ── the rule, pure ─────────────────────────────────────────────────────────────────────────────

def test_a_unique_top_heading_puts_the_sections_one_level_below_it():
    assert section_headings(_rows(REPORT)) == SECTIONS


def test_no_unique_top_heading_gives_level_one():
    chunks = [((TITLE, "Horários"), "x"), ((TITLE, "Preços"), "y"), ((TITLE, "Horários"), "z")]
    assert section_headings(_rows(chunks)) == ("Horários", "Preços")


def test_a_document_that_never_splits_has_no_sections():
    assert section_headings(_rows([((TITLE, "Único"), "a"), ((TITLE, "Único"), "b")])) == ()
    assert section_headings(_rows([((TITLE,), "a")])) == ()
    assert section_headings([]) == ()


def test_document_order_is_the_FIRST_appearance_and_blanks_and_repeats_go():
    rows = [(1, 5, "Depois"), (1, 1, "Antes"), (1, 9, "Antes"), (1, 3, "  "), (1, 2, ""),
            (1, 7, "Meio  com   espaços")]
    assert section_headings(rows) == ("Antes", "Depois", "Meio com espaços")


def test_the_store_ceiling_holds():
    rows = [(1, i, f"Secção {i:03d}") for i in range(MAX_SECTIONS_PER_DOCUMENT + 25)]
    got = section_headings(rows)
    assert len(got) == MAX_SECTIONS_PER_DOCUMENT == 50
    assert got[0] == "Secção 000" and got[-1] == "Secção 049"


# ── both adapters ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=["memory", "postgres"])
async def docs(request):
    return await store_factory(request.param, DSN)()


async def _publish(store, owner: str, chunks, *, title: str = TITLE, profiles=("ADMIN",),
                   commit: bool = True) -> str:
    doc = await store.create_document(owner, title=title, profiles=list(profiles),
                                      media_type=MEDIA_MARKDOWN)
    original = repr(chunks).encode()
    v = await store.begin_version(owner, doc.id, sha256=hashlib.sha256(original).hexdigest(),
                                  embed_model=MODEL_A, size_bytes=len(original), original=original)
    assert await store.add_chunks(owner, doc.id, v.version, [
        KbChunk(ordinal=i, content=" › ".join(path) + "\n\n" + text, heading_path=path,
                embedding=vec(1.0))
        for i, (path, text) in enumerate(chunks)])
    if commit:
        assert await store.commit_version(owner, doc.id, v.version, pages=0) == COMMIT_READY
    return doc.id


def _owner() -> str:
    return f"acme{uuid4().hex[:8]}/persona-a"


async def test_readable_documents_carries_the_outline_of_what_the_profile_may_read(docs):
    o = _owner()
    await _publish(docs, o, REPORT, profiles=("ADMIN",))
    await _publish(docs, o, [(("Manual", "Horários"), "x"), (("Manual", "Preços"), "y")],
                   title="Manual", profiles=("GUEST", "ADMIN"))
    admin = {d.title: d.sections for d in await docs.readable_documents(o, profile="ADMIN")}
    assert admin == {TITLE: SECTIONS, "Manual": ("Horários", "Preços")}
    # the RBAC is the store's: a profile never sees the outline of a document it may not read
    guest = {d.title: d.sections for d in await docs.readable_documents(o, profile="GUEST")}
    assert guest == {"Manual": ("Horários", "Preços")}


async def test_the_outline_is_the_ACTIVE_versions_and_other_reads_carry_none(docs):
    o = _owner()
    doc_id = await _publish(docs, o, REPORT)
    # a newer version still being built does not move the outline the readers see
    newer = [((TITLE, "Relatório Anual", "Outra Coisa"), "a"),
             ((TITLE, "Relatório Anual", "Mais Outra"), "b")]
    v = await docs.begin_version(o, doc_id, sha256="b" * 64, embed_model=MODEL_A, size_bytes=1,
                                 original=b"x")
    assert await docs.add_chunks(o, doc_id, v.version, [
        KbChunk(ordinal=i, content=t, heading_path=p, embedding=vec(1.0))
        for i, (p, t) in enumerate(newer)])
    (served,) = await docs.readable_documents(o, profile="ADMIN")
    assert served.sections == SECTIONS
    assert all(d.sections == () for d in await docs.list_documents(o))


async def test_an_owner_with_nothing_readable_is_an_empty_list(docs):
    assert await docs.readable_documents(_owner(), profile="ADMIN") == []
