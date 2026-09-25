"""cogno_engram.lexical — does a candidate carry the words of the QUESTION? One tokenizer, one
score, one floor.

**The defect this exists for.** A retrieval that fetches by PROXIMITY — the nodes nearest the
query's embedding, walked a couple of hops; the memories nearest the query — always has a
nearest, so it always returns something. Handed to a model as ``status=success``, the nearest
noise is either invented from or read as proof that nothing better exists. What turns it into an
honest *nothing relevant* is a FLOOR, and a floor needs a score that means the same thing
wherever it is taken.

**What this module computes, and only this:**

* a TOKENIZER (:func:`tokens`) and the function words it discards (:data:`STOPWORDS`) — the ONE
  definition of what a word is, shared by every consumer that ranks text and by every floor over
  that ranking;
* one RELEVANCE score per candidate (:func:`relevance`): the share of the question's
  meaning-carrying words the candidate contains, taken against every VARIANT of the question
  (the text a model rewrote into canonical English and the contact's own words), the better one
  counting;
* a ranking (:func:`rank`) with a ONE-HOP inheritance along a graph walk and a deterministic
  tie-break, and the decision over it (:func:`decide`): relevant, *nothing relevant*, or
  *error* — a source that broke is never reported as a source that holds nothing — and, when the
  caller names who the question is ABOUT (:func:`anchor`, :func:`speaks_of_self`), *partial*:
  nothing answers the question directly, but the graph holds edges about who it names
  (:func:`partial`), which is said instead of *nothing relevant*;
* candidates built from this library's OWN types (:func:`graph_candidates` over the
  ``(variant, rank, node_id, edges)`` a walk produces, :func:`memory_candidates` over memory
  records), each under a CONTENT-FREE id a reply can cite.

**What it does NOT do — the caller's, every time.** It fetches nothing, and it decides nothing
about who may read what: the candidates arrive already fetched with the caller's audience and
the caller's scopes. It bounds no wall-clock time (a ranking is CPU, and the caller owns the
event loop it runs on). It bounds the WORK it is handed only by count
(:data:`MAX_CANDIDATES`); the SIZE of each candidate is the caller's to cut.

**Why the floor is LEXICAL and not a vector distance.** A vector score depends on the embedder,
and a curve that picks a floor is measured offline — over invented content, with a model-free
stub embedder in a unit test. A floor over the stub's cosine would be a number about the stub.
The share of folded words is the same number in a test and in production, so the curve
transfers; vectors still decide which candidates are FETCHED. What it costs is stated, not
assumed: a relevant candidate that shares no word with the question — a PARAPHRASE («quanto
recebo» against «remuneração») — scores zero.

Portuguese data in an English module (the stopword list, the plural fold) is domain data that
must stay Portuguese to work: it is what the tokenizer is FOR.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Optional, Sequence

from cogno_engram.textfold import fold

# ── words ────────────────────────────────────────────────────────────────────────────
#
# The fold is :func:`cogno_engram.textfold.fold` — the ONE accent/case fold, the same object a
# consumer's own lexicons use, never a second spelling of it. Not
# :func:`cogno_engram.folding.fold_label`: that one derives a KEY and must agree with Postgres
# ``unaccent`` (it transliterates), and measured against this fold it differs on 3,757 code
# points — a word it found this tokenizer would not, and the other way round.

_WORD = re.compile(r"[a-z0-9]{2,}")


def _singular(token: str) -> str:
    """``ementas`` → ``ementa``. Portuguese plurals, by the one rule that covers almost all of
    them, and deliberately no more.

    **It decides the case it exists for.** A question such as *«ementa de <discipline>»* names a
    KIND (ementa) and a TOPIC (the discipline). When the topic appears in every document of a
    small corpus — the timetable, the syllabus and the reading list each have a section on it —
    the only word that can say WHICH document answers is the kind, and the kind lives in that
    document's title. The title says *Ementas*, the contact says *ementa*, and to a tokenizer
    that reads bytes those are two unrelated words.

    ``ss`` is excluded so English ``class``/``process`` survive intact, and four characters is
    the floor so ``as``/``dos``/``nos`` are untouched. What it does NOT do: ``-ões``/``-ais``
    (``reposições`` stays apart from ``reposição``, ``materiais`` from ``material``). Those are a
    stemmer, and a stemmer is a thing to measure on its own corpus rather than to smuggle in
    behind a language fix.
    """
    return token[:-1] if len(token) >= 4 and token.endswith("s") and not token.endswith(
        "ss") else token


def tokens(text: str) -> "list[str]":
    """What a word is — the ONE definition a ranking and every floor over it share.

    :func:`~cogno_engram.textfold.fold` (no keyword step), then every run of two or more
    ``[a-z0-9]``, each through :func:`_singular`.
    A consumer that ranks with one tokenizer and filters with another has two chances to
    disagree about the same sentence; pass THIS callable to both (it is one object, so a test
    can assert identity rather than agreement).
    """
    return [_singular(t) for t in _WORD.findall(fold(text))]


#: Function words — the floor's blind-spot list. NOT a ranking table: it decides only whether a
#: candidate shares any MEANING with the question, and a preposition shares none.
#:
#: **It exists because a ranking always has a best.** With no stopword list a question about
#: something a corpus does not hold (*«preço do estacionamento»*) comes back with whatever
#: document carries the word ``"do"`` — and a lookup that answers every question with its best
#: documents has no way to say "there is nothing written about that", which is exactly the
#: answer that stops a model inventing one.
#:
#: Document frequency was the obvious alternative and is worse on a SMALL corpus: with three
#: documents, ``"de"`` and a topical word can both appear in all three, so IDF drops the topic
#: too. IDF needs a denominator a three-document corpus does not have.
#:
#: Portuguese and English, deliberately SHORT, and it errs towards recall: a false positive
#: hands the model true-but-unasked-for material, which it can still answer around; a false
#: negative says *nothing recorded* about material that is right there. Words that look
#: structural in one corpus and topical in another (``antes``, ``durante``, ``depois``) are left
#: OUT for that reason.
#:
#: Stored through :func:`tokens` rather than ``.split()``, so the list is held in the alphabet the
#: tokenizer actually produces: ``quais`` folds to ``quai`` on both sides and keeps working,
#: where a raw ``"quais"`` in the set would silently stop matching anything. The single-letter
#: entries drop out, which changes nothing — :func:`tokens` never emits a token shorter than two.
STOPWORDS = frozenset(tokens("""
a as ao aos da das de do dos na nas no nos em um uma uns umas os se ser sao ou que qual quais
quando onde como por pra para pelo pela com sem sobre mais menos muito ja nao sim meu minha
seu sua ele ela eles elas eu tu voce voces isso isto aquilo esse essa este esta aquele aquela
tem ter tenho temos foi era estao ha ate tambem so bem aqui ali la entao qualquer cada
the of and or to for in on at is are was be it this that with from by an my your we you they
does what when where how which who
"""))


# ── the numbers ──────────────────────────────────────────────────────────────────────

#: The relevance floor. A candidate below it is not about the question; a call whose best
#: candidate is below it answers *nothing relevant*.
#:
#: **Chosen on a curve, not by a guess — and the curve is the CONSUMER's.** The floor is a
#: point on the scale :func:`relevance` defines, so it moves whenever the tokenizer, the stopword
#: list or the scoring does; it was picked by the consumer that owns a labelled set (~40
#: invented questions over three sources and four readers: the maximum of the mean of
#: success@3 and *nothing-relevant*-on-negatives, the widest plateau, its midpoint). That
#: consumer's calibration test asserts THIS constant is what its rule picks, so a change here
#: that moves the optimum turns it red at the pin bump instead of leaving a stale number. A
#: caller with another corpus passes its own ``floor``.
RELEVANCE_FLOOR = 0.3

#: Words are compared by their first ``STEM_PREFIX`` characters (0 = whole words). A crude
#: stemmer, and a CROSS-LANGUAGE one by accident of Latin: *implant*/*implante*,
#: *cancelar*/*cancelamento*, *allergy*/*allergic*, *specialty*/*specializes* meet at six.
#:
#: **OFF, and that is a measurement, not a preference.** Over the labelled set above, every
#: prefix moves the objective by at most one question either way, while every prefix that
#: recovers a paraphrase also lets a negative through — a difference such a set cannot resolve,
#: on a knob whose collisions in Portuguese (*mensa*lidade/*mensa*gem) are exactly the kind a
#: small invented set would not contain. It stays a parameter so a replay over real calls can
#: decide it.
STEM_PREFIX = 0

#: One hop along the graph: an edge whose SOURCE is the TARGET of a matched edge inherits this
#: share of that edge's score («a Zuleide tem cachorro?» matches *Zuleide HAS_PET Farofa*, and
#: *Farofa IS_A beagle* is the rest of the answer while sharing no word with it). Forward only,
#: one pass, never iterated: the walk a proximity retrieval makes, turned into a score. ON
#: because the ablation over the labelled set says it costs nothing and keeps one more of the
#: walk's own answers.
HOP_DECAY = 0.5

#: How many results the answer carries.
TOP_K = 5

#: The most candidates one call should SCORE — the ranking's cost bound, by COUNT.
#:
#: **Why it exists.** A ranking is CPU, and on an event loop CPU cannot be interrupted: a
#: ``rank`` over ~3 MB of section-sized candidates held the loop ~370–400 ms, during which
#: nothing else in the process ran. At this many section-sized candidates it is ~16 times
#: cheaper (~22 ms against ~350 ms on a quiet 20-thread box, 2026-09-24) — and it is the RATIO
#: that ``tests/test_lexical_cost.py`` asserts, timed intercalated with the uncapped pool, because
#: an absolute number of milliseconds is a fact about the machine's load, not about this code.
#:
#: **What it does not bound: the SIZE of a candidate.** Cost is linear in the total text scored,
#: so 2 000 candidates of 1.5 KB each are the same 3 MB again. A caller that builds candidates
#: from a large document cuts the document before it builds them; the builders here stop AT a
#: ``limit`` (so a caller asks for one more than it may keep and learns the bound was hit
#: without building the tail).
MAX_CANDIDATES = 2000

SOURCE_GRAPH = "graph"
SOURCE_MEMORY = "memory"
SOURCE_MATERIAL = "material"      # a caller's own written material, one candidate per section
#: Tie-break order between sources, and the order a record counts them in.
SOURCES = (SOURCE_GRAPH, SOURCE_MEMORY, SOURCE_MATERIAL)

#: The closed alphabet of :func:`decide`.
DECISION_RELEVANT = "relevant"
DECISION_NOTHING = "nothing_relevant"
DECISION_ERROR = "error"          # a source that should have answered could not, and nothing
                                  # relevant came from the others — never "nothing relevant"
DECISION_PARTIAL = "partial"      # nothing clears the floor for the QUESTION, but the graph holds
                                  # edges about who or what the question NAMES — see :func:`partial`

_CLIP = 600


# ── the score ────────────────────────────────────────────────────────────────────────

def terms(text: str, prefix: Optional[int] = None) -> frozenset:
    """The meaning-carrying words of ``text``: :func:`tokens` minus :data:`STOPWORDS`, each cut to
    its first ``prefix`` characters (:data:`STEM_PREFIX` when not given; ``0`` keeps whole
    words). The prefix cut is applied AFTER the stopword test, so it never turns a function word
    into a content one."""
    n = STEM_PREFIX if prefix is None else prefix
    words = (t for t in tokens(text or "") if t not in STOPWORDS)
    return frozenset(t[:n] if n and len(t) > n else t for t in words)


def relevance(candidate: frozenset, questions: Sequence[frozenset]) -> float:
    """The share of the question's words the candidate carries — the better of the variants.

    Coverage of the QUESTION, not of the candidate: a long section that mentions the question's
    two words is about it; a one-line edge that mentions one of four is a quarter about it. A
    greeting dilutes the raw turn («boa noite, quanto custa a mensalidade?» carries five words
    and a timetable matches one of them, *noite*: 0.2) — which is why the raw turn is never the
    only variant: a canonical rewrite is scored too, and the better one counts.
    """
    best = 0.0
    for q in questions:
        if not q:
            continue
        best = max(best, len(q & candidate) / len(q))
    return best


# ── candidates ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """One thing a retrieval could answer with.

    ``id`` is CONTENT-FREE by construction (a store id, a node id, an ordinal) and is the only
    identifier meant to leave the caller's process; ``text`` is what is scored and rendered.
    ``prior`` is the position in its source's own recall order, the last tie-break. ``old``
    marks a candidate a BASELINE retrieval also returns (see ``baseline_nodes`` in
    :func:`graph_candidates`) — what lets a caller measure how much of an old path a new one
    keeps.
    """

    id: str
    source: str
    text: str
    prior: int = 0
    old: bool = False
    #: The edge's two ends, folded — graph candidates only, for the one-hop inheritance.
    head: str = ""
    tail: str = ""

    def words(self, prefix: Optional[int] = None) -> frozenset:
        return terms(self.text, prefix)


def edge_end(label: Any) -> str:
    """An edge end as the one-hop inheritance compares it: its :func:`tokens`, space-joined."""
    return " ".join(tokens(str(label or "")))


def _detail(attributes: Any) -> str:
    if not isinstance(attributes, dict) or not attributes:
        return ""
    parts = [f"{k}: {' '.join(str(attributes[k]).split())}" for k in sorted(attributes)
             if str(attributes[k]).strip()]
    detail = "; ".join(parts)[:160]
    return f" ({detail})" if detail else ""


def edge_text(edge: Any) -> str:
    """An edge as one line, in the shape :func:`~cogno_engram.graph_context.format_graph_context`
    renders (``source --[RELATION]--> target (key: value)``), so a result read from here reads
    like the line a graph block carries for the same edge.

    The same SHAPE, not the same bytes: the detail here is cut at 160 characters with no
    ellipsis, where the graph block cuts at 120 and marks the cut. This is the text that is
    SCORED, and a cut that moved would move scores.
    """
    return (f"{getattr(edge, 'source', '')} --[{getattr(edge, 'relation', '')}]--> "
            f"{getattr(edge, 'target', '')}{_detail(getattr(edge, 'attributes', None))}")


def graph_candidates(walks: Iterable[tuple], limit: Optional[int] = None, *,
                     baseline_nodes: int = 0) -> "list[Candidate]":
    """Edges from ``walks`` — ``(variant, rank, node_id, edges)`` per start node, in fetch order.

    Deduplicated by ``(source, relation, target)``: two start nodes one hop apart walk the same
    edge, and it is one fact. The id is ``edge:<start node id>.<n>`` — an integer the store
    assigned and an ordinal, nothing a person wrote. ``limit`` STOPS the build at that many —
    the work is bounded, not only the result (:data:`MAX_CANDIDATES`).

    ``baseline_nodes``: the first that-many start nodes of variant 0 are the ones a BASELINE
    retrieval walks (a proximity search that takes the nearest N nodes of the first query), so
    their edges are marked ``old``. ``0`` marks none.

    A walk that did not start from a query variant — an ANCHOR's walk (see :func:`partial`), seeded
    on a label the question names rather than on the nodes nearest its embedding — passes any
    ``variant`` other than ``0`` (``-1`` by convention) and a content-free ``node_id`` of the
    caller's (an ordinal such as ``a0``): it is never marked ``old``, and an edge it shares with a
    query walk stays ONE candidate under the id it got first.
    """
    seen: "set[tuple]" = set()
    out: "list[Candidate]" = []
    for variant, rank, node_id, edges in walks:
        n = 0
        for edge in edges:
            if limit is not None and len(out) >= limit:
                return out
            key = (str(getattr(edge, "source", "")).lower(), str(getattr(edge, "relation", "")),
                   str(getattr(edge, "target", "")).lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(Candidate(id=f"edge:{node_id}.{n}", source=SOURCE_GRAPH,
                                 text=edge_text(edge), prior=len(out),
                                 old=(variant == 0 and rank < baseline_nodes),
                                 head=edge_end(getattr(edge, "source", "")),
                                 tail=edge_end(getattr(edge, "target", ""))))
            n += 1
    return out


def memory_candidates(records: Iterable[Any],
                      limit: Optional[int] = None) -> "list[Candidate]":
    """Memories, deduplicated by store id. The id is ``mem:<store id>`` — opaque by construction.
    A record with no id or no content is skipped (there is nothing to cite, or nothing to say)."""
    seen: "set[str]" = set()
    out: "list[Candidate]" = []
    for r in records:
        if limit is not None and len(out) >= limit:
            break
        rid = str(getattr(r, "id", "") or "")
        content = str(getattr(r, "content", "") or "")
        if not rid or rid in seen or not content.strip():
            continue
        seen.add(rid)
        out.append(Candidate(id=f"mem:{rid}", source=SOURCE_MEMORY, text=content,
                             prior=len(out)))
    return out


# ── ranking and the decision ─────────────────────────────────────────────────────────

def rank(candidates: Sequence[Candidate], asked: Sequence[str], *,
         prefix: Optional[int] = None,
         hop: Optional[float] = None) -> "list[tuple[float, Candidate]]":
    """Every candidate with its relevance to the query variants ``asked``, best first.

    Deterministic: ties go to the source order (:data:`SOURCES`), then to each source's own
    recall order, then to the id. ``prefix`` and ``hop`` default to :data:`STEM_PREFIX` /
    :data:`HOP_DECAY`; a bench passes them explicitly to measure each one alone."""
    decay = HOP_DECAY if hop is None else hop
    questions = [terms(q, prefix) for q in asked]
    own = [relevance(c.words(prefix), questions) for c in candidates]
    score = list(own)
    if decay:
        best_into: "dict[str, float]" = {}
        for c, s in zip(candidates, own):
            if c.source == SOURCE_GRAPH and c.tail and s > 0:
                best_into[c.tail] = max(best_into.get(c.tail, 0.0), s)
        for i, c in enumerate(candidates):
            if c.source == SOURCE_GRAPH and c.head in best_into:
                score[i] = max(score[i], decay * best_into[c.head])
    order = {s: i for i, s in enumerate(SOURCES)}
    scored = list(zip(score, candidates))
    scored.sort(key=lambda sc: (-sc[0], order.get(sc[1].source, len(order)), sc[1].prior,
                                sc[1].id))
    return scored


def chosen(ranked: Sequence[tuple], floor: float = RELEVANCE_FLOOR,
           k: int = TOP_K) -> "list[tuple[float, Candidate]]":
    """What the answer would carry: the ranked candidates AT or above ``floor``, top ``k``.
    A score of zero never passes, whatever the floor: a candidate sharing no word with the
    question is not "barely relevant", it is unrelated."""
    return [(s, c) for s, c in ranked if s >= floor and s > 0][:k]


def variants(query: str, original: str = "") -> "list[str]":
    """The question as the executor wrote it (canonical English) and as the contact wrote it —
    the second only when it carries words the first does not. Order matters: variant 0 is the
    one a baseline retrieval embeds, and :func:`graph_candidates` marks its first nodes as the
    baseline's answer."""
    out = [query]
    if (original or "").strip() and terms(original) - terms(query):
        out.append(original)
    return out


# ── anchors: who and what the question is ABOUT ──────────────────────────────────────
#
# **The defect this is for.** The score is the share of the QUESTION's words a candidate
# carries, and two shapes of question defeat it while the answer sits among the candidates:
#
# * the question speaks of the one asking («meus horários de outubro») and the edge that
#   answers it names that person and none of the question's words (``<the asker> --[TEACHES]-->
#   <a class>``): the SUBJECT of the question is a word the score cannot see;
# * the question names an entity and asks several things about it, and the edge names the entity
#   and nothing else that was asked: one word of four is a quarter, under the floor.
#
# Measured on a consumer's replay of real calls: wherever the proximity walk held an edge a human
# labelled relevant and this module answered *nothing relevant*, that edge WAS a candidate and
# scored under the floor. It was not a fetch that missed it — so fetching more of the person's
# edges does not move that shape; a rule about what an edge TOUCHING the named entity means does.
#
# **The rule, and why it is a TIER and not a boost.** An ANCHOR is a label the question is about:
# an entity it names, or the asker when it speaks in the first person. The CALLER decides which —
# who the asker is, and what named what, are not this module's to know. A graph candidate is ABOUT
# an anchor when one of its two ENDS carries the anchor's words, whole and in order. When nothing
# clears the floor, :func:`decide` answers :data:`DECISION_PARTIAL` with those candidates instead
# of *nothing relevant* — not "this answers the question" but "nothing answers it directly, and
# this is what is recorded about who it names" (:func:`render` says exactly that). Anything that
# clears the floor is untouched: the tier never competes with a relevant result, moves no score and
# no floor, and a caller that passes no anchors gets the decision it always got.
#
# What it costs is stated, not assumed: a question about an attribute nobody recorded («does <the
# clinic> have parking?») that names an entity WITH edges now gets that entity's edges as a
# partial answer where it used to get *nothing relevant*. That is the price of never saying
# "nothing" about a person the graph does know, and the render is what keeps it honest.

#: First-person SINGULAR words, Portuguese and English, in the tokenizer's alphabet: a question
#: carrying one speaks of the one ASKING. Some are also :data:`STOPWORDS` (``eu``, ``meu``,
#: ``minha``, ``my``) — they carry no TOPIC, which is exactly why the score never sees them — but
#: they carry a REFERENCE, and a reference is what an anchor is. The PLURAL (``nós``, ``nosso``,
#: ``we``, ``our``) is left out on purpose: said by a business's own staff it means the business,
#: not the person. English ``I`` is one character, never a token; :func:`speaks_of_self` reads it
#: on its own.
FIRST_PERSON = frozenset(tokens("eu me mim comigo meu minha meus minhas my me mine myself"))

# `I` over the folded text, as a whole word (so «iPhone» and «Turma II» do not count).
_I = re.compile(r"(?<![a-z0-9])i(?![a-z0-9])")


def speaks_of_self(text: str) -> bool:
    """Does ``text`` speak in the first person singular — :data:`FIRST_PERSON`, or English ``I``?
    The caller turns a ``True`` into an anchor on the one asking; this module never knows who
    that is."""
    return (bool(FIRST_PERSON.intersection(tokens(text or "")))
            or bool(_I.search(fold(text or ""))))


def anchor(label: Any) -> "tuple[str, ...]":
    """An anchor's words, in order: the label's :func:`tokens` with function words trimmed from
    both ENDS («o Orientador» → ``("orientador",)``) and KEPT inside («Rua das Acácias» keeps its
    ``das``, so it still matches the edge end it names). ``()`` when no content word is left — a
    pronoun or an article alone is not an anchor. Memoised: a caller asks :func:`about` once per
    candidate, and a call holds a handful of anchors against up to :data:`MAX_CANDIDATES`."""
    return _anchor(str(label or ""))


@lru_cache(maxsize=4096)
def _anchor(label: str) -> "tuple[str, ...]":
    words = tokens(label)
    while words and words[0] in STOPWORDS:
        words.pop(0)
    while words and words[-1] in STOPWORDS:
        words.pop()
    return tuple(words)


def _anchor_words(labels: Iterable[Any]) -> "tuple[tuple[str, ...], ...]":
    out: "list[tuple[str, ...]]" = []
    for label in labels:
        words = anchor(label)
        if words and words not in out:
            out.append(words)
    return tuple(out)


def _carries(end: str, words: "tuple[str, ...]") -> bool:
    ends = end.split()
    n = len(words)
    return any(tuple(ends[i:i + n]) == words for i in range(len(ends) - n + 1))


def _about(candidate: Candidate, words: "Sequence[tuple[str, ...]]") -> bool:
    if candidate.source != SOURCE_GRAPH:
        return False
    return any(_carries(end, w) for end in (candidate.head, candidate.tail) if end for w in words)


def about(candidate: Candidate, anchors: Iterable[Any]) -> bool:
    """Is ``candidate`` an edge whose SOURCE or TARGET carries one of the ``anchors``' words,
    whole and in order? «Zulmira» is carried by the end «Zulmira Pervinca» and not by «Zulmirinha»; the
    fold is the tokenizer's (accents and case), so «Otávio» is carried by «OTAVIO BRANDÃO».

    Graph candidates only: an edge RELATES two things, so "is it about this one" has an answer. A
    memory or a section of material has no ends — it is about whatever its words say, and that is
    what the score already measures."""
    return _about(candidate, _anchor_words(anchors))


def partial(ranked: Sequence[tuple], anchors: Iterable[Any],
            k: int = TOP_K) -> "list[tuple[float, Candidate]]":
    """The :data:`DECISION_PARTIAL` answer over an ALREADY-RANKED list: the candidates
    :func:`about` an anchor, in the ranking's own order (their own score first — a word of the
    question still counts — then each source's recall order), then the edges ONE hop on from them
    (an edge whose source is the target of one of those: the rule :data:`HOP_DECAY` applies to
    scores, applied to the tier), top ``k``. Forward only, one hop, never iterated — two hops
    leave the entity, and the tier would stop being about it."""
    words = _anchor_words(anchors)
    if not words:
        return []
    direct = [(s, c) for s, c in ranked if _about(c, words)]
    ids = {c.id for _, c in direct}
    tails = {c.tail for _, c in direct if c.tail}
    hop = [(s, c) for s, c in ranked
           if c.source == SOURCE_GRAPH and c.id not in ids and c.head in tails]
    return (direct + hop)[:k]


def render(query: str, picked: Sequence[tuple], *,
           decision: str = DECISION_RELEVANT) -> str:
    """The payload an executor would read — each result under its id, so a reply can cite
    ``[edge:12.0]``. A :data:`DECISION_PARTIAL` answer SAYS it is one: nothing recorded answers
    the question directly, and what follows is only what is recorded about who or what it names —
    a reader that took it for the answer would be the defect this tier must not create."""
    if not picked:
        return f"Nothing relevant recorded about {query!r}."
    if decision == DECISION_PARTIAL:
        lines = [f"Nothing recorded answers {query!r} directly. Recorded about who or what it "
                 f"names (it may not answer the question; cite by id):"]
    else:
        lines = [f"What we have recorded about {query!r} (most relevant first; cite by id):"]
    for _, c in picked:
        text = " ".join(c.text.split())
        lines.append(f"[{c.id}] {text[:_CLIP]}")
    return "\n".join(lines)


def decide(ranked: Sequence[tuple], *, failed: Sequence[str] = (),
           floor: float = RELEVANCE_FLOOR,
           anchors: Sequence[Any] = ()) -> "tuple[str, list]":
    """``(decision, picked)`` over an ALREADY-RANKED list (ranking is the CPU; it runs once).
    *Nothing relevant* only when every source that should have answered DID — a source that
    broke and a source that holds nothing are different facts.

    ``anchors`` — the labels the question is ABOUT (see :func:`partial`) — are read only when
    nothing clears the floor and no source broke: then the candidates about them are the answer,
    as :data:`DECISION_PARTIAL`, and *nothing relevant* is said only when there are none. Empty
    (the default), the decision is exactly the one this function made before anchors existed. A
    broken source still wins over the tier: "nothing answers it directly" is a claim about every
    source, and one of them could not be asked."""
    picked = chosen(ranked, floor)
    if picked:
        return DECISION_RELEVANT, picked
    if failed:
        return DECISION_ERROR, picked
    tier = partial(ranked, anchors) if anchors else []
    if tier:
        return DECISION_PARTIAL, tier
    return DECISION_NOTHING, picked
