"""The chunker — CHARACTERS, ~2000 with 15% overlap, every chunk under its heading path."""

from __future__ import annotations

import pytest

from cogno_engram.chunking import (
    DEFAULT_CHUNKING,
    ChunkingConfig,
    chunk_markdown,
    chunk_pages,
    chunk_text,
)
from cogno_engram.documents import ExtractedPage, ExtractionError, OutlineEntry

SMALL = ChunkingConfig(target_chars=200, overlap=0.15)


def bodies(chunks):
    return [c.content.split("\n\n", 1)[1] if "\n\n" in c.content else c.content for c in chunks]


def test_the_default_unit_is_characters_2000_with_15_percent_overlap():
    assert (DEFAULT_CHUNKING.target_chars, DEFAULT_CHUNKING.overlap) == (2000, 0.15)
    assert DEFAULT_CHUNKING.overlap_chars == 300


def test_every_chunk_carries_its_heading_path_and_a_skipped_level_still_nests():
    md = ("Intro antes de tudo.\n\n# Horários\n\nTexto geral.\n\n### Sábado\n\n8h às 12h.\n\n"
          "## Feriados\n\nFechado.\n\n# Preços\n\nR$ 450,00.\n")
    chunks = chunk_markdown("Guia", md)
    got = [(c.heading_path, b) for c, b in zip(chunks, bodies(chunks))]
    assert got == [
        (("Guia",), "Intro antes de tudo."),
        (("Guia", "Horários"), "Texto geral."),
        (("Guia", "Horários", "Sábado"), "8h às 12h."),
        (("Guia", "Horários", "Feriados"), "Fechado."),
        (("Guia", "Preços"), "R$ 450,00."),
    ]
    assert chunks[2].content == "Guia › Horários › Sábado\n\n8h às 12h."
    assert [c.ordinal for c in chunks] == list(range(5)) and {c.page for c in chunks} == {None}


def test_a_top_heading_equal_to_the_title_is_not_said_twice():
    [c] = chunk_markdown("Manual", "# manual\n\nconteúdo\n")
    assert c.heading_path == ("Manual",)


def test_a_hash_inside_a_code_fence_is_code_not_a_heading():
    md = "# Setup\n\n```bash\n# not a heading\necho ok\n\n# still code\n```\n\ndepois\n"
    [c] = chunk_markdown("T", md)
    assert c.heading_path == ("T", "Setup")
    assert "# not a heading" in c.content and "# still code" in c.content


def test_headings_with_closing_hashes_and_bare_markers():
    chunks = chunk_markdown("T", "## Horário ##\n\nx\n\n#\n\ny\n\n#hashtag\n")
    assert chunks[0].heading_path == ("T", "Horário")
    assert "#hashtag" in chunks[0].content                 # no space: text, not a heading


def test_a_heading_with_nothing_under_it_makes_no_chunk():
    assert chunk_markdown("T", "# A\n\n# B\n\n## C\n\ntexto\n")[0].heading_path == ("T", "B", "C")
    assert chunk_markdown("T", "# A\n\n# B\n") == []
    assert chunk_markdown("T", "") == []


def test_no_body_exceeds_the_target_and_consecutive_chunks_overlap():
    words = " ".join(f"palavra{i:03d}" for i in range(200))           # ~2000 characters
    chunks = chunk_markdown("T", f"# S\n\n{words}\n", SMALL)
    bs = bodies(chunks)
    assert len(bs) > 5
    assert all(len(b) <= SMALL.target_chars for b in bs)
    for prev, nxt in zip(bs, bs[1:]):
        head = nxt.split(" ", 1)[0]
        assert head in prev and prev.index(head) > len(prev) - SMALL.overlap_chars - 12
    # nothing lost: every word appears in some chunk
    assert all(f"palavra{i:03d}" in " ".join(bs) for i in range(200))


def test_the_overlap_never_crosses_a_section():
    a = " ".join(["alfa"] * 60)
    b = " ".join(["beta"] * 60)
    chunks = chunk_markdown("T", f"# A\n\n{a}\n\n# B\n\n{b}\n", SMALL)
    for c in chunks:
        if c.heading_path == ("T", "B"):
            assert "alfa" not in c.content


def test_paragraphs_are_packed_up_to_the_target():
    paras = "\n\n".join(f"parágrafo {i} curto." for i in range(20))
    chunks = chunk_markdown("T", paras, SMALL)
    assert 1 < len(chunks) < 20                                         # packed, not one each
    assert "parágrafo 0 curto.\n\nparágrafo 1 curto." in chunks[0].content


def test_a_run_without_whitespace_is_cut_hard():
    url = "https://exemplo.invalid/" + "a" * 1000
    chunks = chunk_markdown("T", url, SMALL)
    assert len(chunks) > 1 and all(len(b) <= SMALL.target_chars for b in bodies(chunks))


def test_the_chunker_is_deterministic():
    md = "# A\n\n" + "\n\n".join("texto " * 40 for _ in range(10))
    assert chunk_markdown("T", md, SMALL) == chunk_markdown("T", md, SMALL)


def test_the_chunk_ceiling_stops_the_work_with_over_limit():
    md = "\n\n".join(f"parágrafo {i} " * 20 for i in range(100))
    with pytest.raises(ExtractionError) as err:
        chunk_markdown("T", md, ChunkingConfig(target_chars=200, max_chunks=3))
    assert err.value.reason == "over_limit"


def test_pages_are_chunked_per_page_never_across_and_keep_their_number():
    pages = [ExtractedPage(1, "Primeira página.\r\nLinha dois."),
             ExtractedPage(2, "   "),
             ExtractedPage(3, " ".join(["terceira"] * 80))]
    chunks = chunk_pages("Doc", pages, config=SMALL)
    assert chunks[0].page == 1 and chunks[0].content == "Doc\n\nPrimeira página.\nLinha dois."
    assert {c.page for c in chunks[1:]} == {3} and len(chunks) > 2
    assert all(c.heading_path == ("Doc",) for c in chunks)


def test_the_bookmark_trail_in_force_on_each_page_is_its_path():
    pages = [ExtractedPage(n, f"p{n}") for n in range(1, 6)]
    outline = [OutlineEntry(1, "Parte I", 2), OutlineEntry(2, "Cap. 1", 2),
               OutlineEntry(2, "Cap. 2", 4), OutlineEntry(1, "Parte II", 5)]
    paths = {c.page: c.heading_path for c in chunk_pages("Doc", pages, outline)}
    assert paths == {1: ("Doc",), 2: ("Doc", "Parte I", "Cap. 1"), 3: ("Doc", "Parte I", "Cap. 1"),
                     4: ("Doc", "Parte I", "Cap. 2"), 5: ("Doc", "Parte II")}


def test_a_blank_title_leaves_only_the_headings():
    [c] = chunk_markdown("  ", "# A\n\nx\n")
    assert c.heading_path == ("A",) and c.content == "A\n\nx"


@pytest.mark.parametrize("kwargs", [dict(target_chars=10), dict(overlap=0.5), dict(overlap=-0.1),
                                    dict(max_chunks=0)])
def test_nonsense_configurations_are_refused(kwargs):
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)


# ── chunk_text: the inverse of the head the chunker writes, in the SAME module ──────────
#
# A reader of a version's text (``DocumentStore.version_text``) shows the passage beside its
# heading path. Taking the head off anywhere else would copy the chunker's format into a second
# place; this pins the two against each other over what the chunker actually emits.

ROUND_TRIP_MD = ("Antes de tudo.\n\n# Horários\n\nTexto geral.\n\n### Sábado\n\n8h às 12h.\n\n"
                 "Segundo parágrafo\n\ncom linha em branco.\n\n# Preços\n\n"
                 + " ".join(f"palavra{i:03d}" for i in range(120)) + "\n")


@pytest.mark.parametrize("separator", [" › ", " / ", "::", " — "])
def test_chunk_text_takes_off_exactly_the_head_the_chunker_wrote(separator):
    config = ChunkingConfig(target_chars=200, overlap=0.15, path_separator=separator)
    pages = [ExtractedPage(1, "Primeira página.\n\nOutro parágrafo."),
             ExtractedPage(2, " ".join(["segunda"] * 60))]
    outline = [OutlineEntry(1, "Parte I", 1), OutlineEntry(2, "Cap. 1", 2)]
    chunks = chunk_markdown("Guia", ROUND_TRIP_MD, config) + chunk_pages("Doc", pages, outline,
                                                                          config)
    assert len(chunks) > 6 and any(len(c.heading_path) >= 3 for c in chunks)   # the shape exists
    for c in chunks:
        text = chunk_text(c.content, c.heading_path)
        # the head is gone, whole — and putting it back gives the stored content byte for byte
        assert f"{separator.join(c.heading_path)}\n\n{text}" == c.content, c.ordinal
    # CONTROL — the body keeps its OWN blank lines: only the first one (the head's) is taken
    [two] = [c for c in chunks if "Segundo parágrafo" in c.content]
    assert chunk_text(two.content, two.heading_path).count("\n\n") >= 1


@pytest.mark.parametrize("content, path", [
    ("Horários › Sábado\n\nAbrimos.", ("Manual",)),       # a head that is not THIS path
    ("Manual › Outro\n\nAbrimos.", ("Manual", "Horários")),
    ("Manual\nAbrimos.", ("Manual",)),                    # no blank line after the head
    ("Abrimos às 8h.", ("Manual",)),                       # no head at all
    ("Manual\n\nAbrimos.", ()),                          # no path to take off
])
def test_chunk_text_leaves_a_chunk_whole_when_its_head_is_not_the_path(content, path):
    assert chunk_text(content, path) == content
