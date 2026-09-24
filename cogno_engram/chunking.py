"""
cogno_engram.chunking — cut a document into the pieces a search returns.

**The unit is the CHARACTER, deliberately.** A chunk is ~``target_chars`` characters (2000 by
default) with a ``overlap`` share (15%) of the previous chunk repeated at its head. No model
tokenizer: a tokenizer is a dependency, it belongs to ONE model, and the deployment's embedder
can be swapped — a chunk size measured in some model's tokens would silently change meaning with
it. Characters are the same number in a test and on the box. (For budgeting, roughly four
characters make a token in the Latin-script languages this has been measured on; that estimate
lives in :func:`cogno_engram.ingest.estimate_tokens`, not here.)

**Every chunk carries its heading path**, rendered at the head of its ``content``
(``"Manual › Horários › Sábado"`` then the text) and kept structured in ``heading_path``. A
passage that says "das 8h às 12h" is not an answer until the search can see it is under
*Sábado*; the path is what makes a short passage findable and citable.

* **Markdown** is cut by heading (ATX ``#``…``######``, fenced code blocks respected — a ``#``
  inside a fence is code, not a heading), then by paragraph within the section.
* **Pages** (what a PDF extractor returns) are cut by page, then by paragraph; a chunk never
  spans two pages, so ``page`` is always one number. When the extractor reports the PDF's
  bookmarks, the path of a page is the bookmark trail in force on it.

The overlap never crosses a section or a page: text under another heading is another context,
and repeating it would make a chunk about *Sábado* start with *Domingo*.

Deterministic — the same input gives the same chunks, byte for byte — because re-running an
ingestion must upsert the same ordinals rather than add new ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from cogno_engram.documents import (
    REASON_OVER_LIMIT,
    ExtractedPage,
    ExtractionError,
    KbChunk,
    OutlineEntry,
)


@dataclass(frozen=True)
class ChunkingConfig:
    """How big a chunk is, in CHARACTERS, and how many a document may make.

    ``max_chunks`` bounds the WORK of one document (every chunk is one embedding call): past it
    the chunker stops and raises ``over_limit`` instead of cutting a 10 MB file into ten
    thousand calls."""

    target_chars: int = 2000
    overlap: float = 0.15
    max_chunks: int = 5000
    path_separator: str = " › "

    def __post_init__(self) -> None:
        if self.target_chars < 50:
            raise ValueError("target_chars must be at least 50")
        if not 0.0 <= self.overlap < 0.5:
            raise ValueError("overlap must be in [0, 0.5)")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be at least 1")

    @property
    def overlap_chars(self) -> int:
        return int(self.target_chars * self.overlap)


DEFAULT_CHUNKING = ChunkingConfig()

_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t#]*$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")


def _normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")


def _paragraphs(lines: Sequence[str]) -> list[str]:
    """Blank-line separated paragraphs; a fenced block is ONE paragraph, blank lines included."""
    out: list[str] = []
    cur: list[str] = []
    fence = ""
    for line in lines:
        m = _FENCE.match(line)
        if fence:
            cur.append(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = ""
            continue
        if m:
            fence = m.group(1)
            cur.append(line)
            continue
        if line.strip():
            cur.append(line.rstrip())
        elif cur:
            out.append("\n".join(cur).strip("\n"))
            cur = []
    if cur:
        out.append("\n".join(cur).strip("\n"))
    return [p for p in out if p.strip()]


def _split_long(paragraph: str, size: int) -> list[str]:
    """A paragraph longer than ``size`` cut at whitespace into pieces of at most ``size`` (a run
    with no whitespace at all is cut hard — a 5000-character URL is still text)."""
    pieces: list[str] = []
    rest = paragraph.strip()
    while len(rest) > size:
        cut = rest.rfind(" ", 0, size + 1)
        nl = rest.rfind("\n", 0, size + 1)
        cut = max(cut, nl)
        if cut <= 0:
            cut = size
        pieces.append(rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    if rest:
        pieces.append(rest)
    return pieces


def _tail(body: str, chars: int) -> str:
    """The last ~``chars`` characters of ``body``, starting at a word boundary."""
    if chars <= 0 or len(body) <= chars:
        return "" if chars <= 0 else body
    start = len(body) - chars
    space = body.find(" ", start)
    newline = body.find("\n", start)
    marks = [i for i in (space, newline) if i != -1]
    if marks:
        start = min(marks) + 1
    return body[start:].strip()


def _pack(paragraphs: Sequence[str], config: ChunkingConfig) -> list[str]:
    """Paragraphs packed into bodies of at most ``target_chars``, each after the first opening
    with the tail of the one before it. A body is never ONLY overlap."""
    overlap = config.overlap_chars
    piece = max(1, config.target_chars - overlap)
    units: list[tuple[str, bool]] = []            # (text, continues the previous unit's paragraph)
    for para in paragraphs:
        parts = _split_long(para, piece) if len(para) > piece else [para]
        units.extend((part, i > 0) for i, part in enumerate(parts))

    bodies: list[str] = []
    cur = ""
    has_unit = False
    for text, continues in units:
        joiner = " " if continues else "\n\n"
        if has_unit and len(cur) + len(joiner) + len(text) > config.target_chars:
            bodies.append(cur)
            cur = _tail(cur, overlap)
            has_unit = False
        cur = f"{cur}{joiner}{text}" if cur else text
        has_unit = True
    if has_unit and cur.strip():
        bodies.append(cur)
    return bodies


def _content(path: Sequence[str], body: str, config: ChunkingConfig) -> str:
    head = config.path_separator.join(path)
    return f"{head}\n\n{body}" if head else body


class _Emitter:
    """Numbers chunks across the whole document and enforces ``max_chunks`` AS it goes."""

    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config
        self.chunks: list[KbChunk] = []

    def emit(self, path: tuple[str, ...], bodies: Sequence[str], page: "int | None") -> None:
        for body in bodies:
            if len(self.chunks) >= self.config.max_chunks:
                raise ExtractionError(REASON_OVER_LIMIT,
                                      f"more than {self.config.max_chunks} chunks")
            self.chunks.append(KbChunk(ordinal=len(self.chunks),
                                       content=_content(path, body, self.config),
                                       heading_path=path, page=page))


def _root(title: str) -> tuple[str, ...]:
    t = " ".join(str(title or "").split())
    return (t,) if t else ()


def chunk_markdown(title: str, text: str, config: ChunkingConfig = DEFAULT_CHUNKING) -> list[KbChunk]:
    """Markdown → chunks, by heading then paragraph. Text before the first heading sits under the
    document title alone. A heading with no text under it makes no chunk of its own (it is still
    in the path of what follows it)."""
    out = _Emitter(config)
    root = _root(title)
    stack: list[tuple[int, str]] = []
    section: list[str] = []
    fence = ""

    def flush() -> None:
        path = root + tuple(t for _, t in stack)
        out.emit(path, _pack(_paragraphs(section), config), None)

    for line in _normalise(text or "").split("\n"):
        m_fence = _FENCE.match(line)
        if fence:
            if m_fence and m_fence.group(1)[0] == fence[0] and len(m_fence.group(1)) >= len(fence):
                fence = ""
            section.append(line)
            continue
        if m_fence:
            fence = m_fence.group(1)
            section.append(line)
            continue
        m = _HEADING.match(line)
        heading = " ".join((m.group(2) or "").split()) if m else ""
        if m and heading:
            flush()
            section = []
            level = len(m.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            continue
        section.append(line)
    flush()
    return out.chunks


def _outline_paths(outline: Sequence[OutlineEntry], pages: Sequence[int]) -> dict[int, tuple[str, ...]]:
    """The bookmark trail in force on each page: every entry at or before the page, in order."""
    entries = sorted(enumerate(outline), key=lambda ie: (ie[1].page, ie[0]))
    paths: dict[int, tuple[str, ...]] = {}
    stack: list[tuple[int, str]] = []
    i = 0
    for number in sorted(set(pages)):
        while i < len(entries) and entries[i][1].page <= number:
            entry = entries[i][1]
            while stack and stack[-1][0] >= entry.level:
                stack.pop()
            stack.append((entry.level, " ".join(entry.title.split())))
            i += 1
        paths[number] = tuple(t for _, t in stack)
    return paths


def chunk_pages(title: str, pages: Sequence[ExtractedPage],
                outline: Sequence[OutlineEntry] = (),
                config: ChunkingConfig = DEFAULT_CHUNKING) -> list[KbChunk]:
    """Pages → chunks, by page then paragraph; never across a page. Blank pages make nothing."""
    out = _Emitter(config)
    root = _root(title)
    trail = _outline_paths(outline, [p.number for p in pages])
    for page in pages:
        lines = _normalise(page.text or "").split("\n")
        out.emit(root + trail.get(page.number, ()), _pack(_paragraphs(lines), config), page.number)
    return out.chunks
