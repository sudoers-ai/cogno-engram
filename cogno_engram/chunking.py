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
    KbTextChunk,
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
    # Room for the overlap AND the joiner ("\n\n", two characters) in front of a piece: a
    # piece sized to `target - overlap` overflowed by the joiner whenever the tail had no
    # word boundary to shrink at (a long URL) — measured by
    # `test_a_run_without_whitespace_is_cut_hard`.
    piece = max(1, config.target_chars - overlap - 2)
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


def _path(root: tuple[str, ...], trail: Sequence[str]) -> tuple[str, ...]:
    """The document title, then the heading trail — without repeating the title when the
    document's own top heading IS the title (``# Manual`` in a document titled *Manual*), which
    is the common case and would otherwise spend characters of every chunk saying it twice."""
    trail = tuple(trail)
    if root and trail and trail[0].casefold() == root[0].casefold():
        trail = trail[1:]
    return root + trail


def _content(path: Sequence[str], body: str, config: ChunkingConfig) -> str:
    """What a chunk's ``content`` is: the heading path, a blank line, the passage. Its inverse is
    :func:`chunk_text`, right below — the two are one rule, pinned by a round-trip test."""
    head = config.path_separator.join(path)
    return f"{head}\n\n{body}" if head else body


def chunk_text(content: str, heading_path: Sequence[str]) -> str:
    """The passage of a chunk WITHOUT the heading path :func:`_content` put at its head — what a
    person reads beside the path, which travels structured in ``heading_path``.

    The separator the chunker joined the path with is a ``ChunkingConfig`` choice the store does
    not record, so it is READ from the head itself rather than assumed: the head is stripped only
    when it is EXACTLY the path's titles joined by one separator, then a blank line. Anything else
    — a chunk written by another writer, a path that is not at its head — comes back whole: at
    worst a reader sees the path twice, never a passage with its first line cut off. Pure."""
    path = [str(t) for t in heading_path]
    head, blank, body = str(content).partition("\n\n")
    if not path or not blank or not head.startswith(path[0]):
        return content
    if len(path) == 1:
        return body if head == path[0] else content
    gap = head.find(path[1], len(path[0]))
    separator = head[len(path[0]):gap] if gap > len(path[0]) else ""
    return body if separator and head == separator.join(path) else content


def _overlap_length(prev: str, nxt: str, config: ChunkingConfig) -> int:
    """How many leading characters of ``nxt`` repeat the tail of ``prev`` — the overlap
    :func:`_pack` put there — or ``0`` when there is none that the chunker could have written.

    Two chunks with the same heading path can be two SECTIONS with the same heading, and between
    those there is no overlap — so only what :func:`_pack` could have written is accepted, and
    each condition below is a NECESSARY one for a real cut inside a section:

    * **the previous chunk is longer than ``overlap_chars``.** ``_pack`` flushes a body only when
      the next unit does not fit (``len(body) + joiner + unit > target``), and a unit is at most
      ``target − overlap − 2`` characters, so a body cut mid-section is always longer than the
      overlap. A shorter one is the END of a section (its whole text would otherwise have been
      "repeated", which ``_tail`` does for a short body — but never mid-section);
    * **the repeated head is EXACTLY ``_tail(prev, overlap_chars)``** — the chunker's own function,
      which is deterministic — followed by a joiner (a blank line between paragraphs, one space
      inside a split paragraph). Not "the longest suffix of ``prev`` that ``nxt`` opens with": a
      section that happens to open with the words the last one ended on matches THAT, and not
      this;
    * **the two did not fit together**: ``len(prev) + len(rest) > target_chars``, where ``rest``
      is what follows the repeated head (joiner included) — the flush condition again.

    Anything else is ``0``: the caller then keeps both texts whole, which at worst shows a
    sentence twice. What it cannot tell apart, said plainly: a section whose heading repeats the
    one before AND that opens by repeating, verbatim, the previous section's last ~``overlap_chars``
    characters from the same word — that head is taken as overlap and shown once."""
    most = config.overlap_chars
    if most <= 0 or len(prev) <= most or not nxt:
        return 0
    head = _tail(prev, most)
    rest = nxt[len(head):]
    if head and nxt.startswith(head) and (rest.startswith("\n\n") or rest.startswith(" ")) \
            and len(prev) + len(rest) > config.target_chars:
        return len(head)
    return 0


def join_passages(chunks: Sequence[KbTextChunk], *, previous: "KbTextChunk | None" = None,
                  config: ChunkingConfig = DEFAULT_CHUNKING) -> list[KbTextChunk]:
    """Chunks read back in ``ordinal`` order → ONE passage per run of the same section, with the
    overlap the chunker repeated at the head of each chunk REMOVED — what a reader needs to show
    a document as one text rather than as pieces that each restate the end of the one before.

    The inverse of :func:`_pack`'s overlap, as :func:`chunk_text` is the inverse of
    :func:`_content`: the two rules live together so a reader never re-derives the chunker's. Pure.

    * A run is consecutive ordinals with the SAME ``heading_path`` and the SAME ``page`` — the
      chunker never overlaps across a section or a page, so neither does this.
    * Inside a run, each chunk's repeated head (:func:`_overlap_length`) is dropped and the rest
      appended with the joiner the chunker wrote. A chunk whose head is NOT a plausible overlap is
      appended whole after a blank line: at worst a reader sees a sentence twice, never a passage
      with words missing.
    * ``previous`` is the chunk just BEFORE ``chunks[0]`` (a continuation that read from the
      middle of a document): it is not returned, but when ``chunks[0]`` continues its run, the
      head they share is removed from ``chunks[0]`` too.
    * Each returned passage carries the ``ordinal`` of the FIRST chunk of its run.
    """
    out: list[KbTextChunk] = []
    last = previous
    for chunk in chunks:
        same_run = (last is not None and chunk.ordinal == last.ordinal + 1
                    and tuple(chunk.heading_path) == tuple(last.heading_path)
                    and chunk.page == last.page)
        text = chunk.text
        if same_run and last is not None:
            cut = _overlap_length(last.text, text, config)
            if out and out[-1].ordinal <= last.ordinal:
                joined = out[-1].text + (text[cut:] if cut else "\n\n" + text)
                out[-1] = KbTextChunk(ordinal=out[-1].ordinal, page=out[-1].page,
                                      heading_path=out[-1].heading_path, text=joined)
                last = chunk
                continue
            text = text[cut:].lstrip() if cut else text
        out.append(KbTextChunk(ordinal=chunk.ordinal, page=chunk.page,
                               heading_path=tuple(chunk.heading_path), text=text))
        last = chunk
    return out


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
        out.emit(_path(root, [t for _, t in stack]), _pack(_paragraphs(section), config), None)

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
        if m and not heading:
            continue                                   # a bare `#` marks nothing and says nothing
        if m:
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
        out.emit(_path(root, trail.get(page.number, ())), _pack(_paragraphs(lines), config),
                 page.number)
    return out.chunks
