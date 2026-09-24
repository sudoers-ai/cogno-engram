"""``cogno_engram.lexical`` — ANCHORS: who and what the question is about, and the *partial* tier.

The shape these tests reproduce was measured on a consumer's replay of real calls: every call where
the proximity walk held an edge a person had labelled relevant and the lexical decision answered
*nothing relevant* had that edge AMONG ITS CANDIDATES, scoring under the floor. Three forms of it:

* the question speaks of the asker («meus horários de outubro») and the edge names the asker and
  none of the question's words;
* the question names a thing the asker's colleague made, and the edge that answers it is one hop
  on from that thing;
* the question names a person and asks several things about her, and the edge names her and
  nothing else that was asked — one word of four.

Every name, class and relation here is INVENTED; none comes from a real conversation. Each twin is
run in BOTH worlds — without anchors (the decision before this change: *nothing relevant*) and with
them (*partial*) — so a test that passes in only one of them is visibly not a constant.
"""

from __future__ import annotations

from cogno_engram import GraphEdge
from cogno_engram.lexical import (DECISION_ERROR, DECISION_NOTHING, DECISION_PARTIAL,
                                  DECISION_RELEVANT, FIRST_PERSON, RELEVANCE_FLOOR, SOURCE_GRAPH,
                                  SOURCE_MEMORY, STOPWORDS, TOP_K, about, anchor,
                                  decide, graph_candidates, memory_candidates, partial, rank,
                                  render, speaks_of_self, tokens, variants)


def _e(source: str, relation: str, target: str) -> GraphEdge:
    return GraphEdge(scope="t", source=source, target=target, relation=relation)


def _walk(*edges: GraphEdge, variant: int = 0, node: object = 1) -> tuple:
    return (variant, 0, node, list(edges))


class _Record:
    def __init__(self, rid, content):
        self.id, self.content = rid, content


# ── the words ────────────────────────────────────────────────────────────────────────


def test_speaks_of_self_reads_the_first_person_SINGULAR_and_not_the_plural():
    for text in ("meus horários de outubro", "qual o nome do meu gato?", "eu tenho aula hoje?",
                 "my October schedule", "which classes do I teach", "I'm late for class",
                 "pode me dizer o horário?"):
        assert speaks_of_self(text), text
    # the plural is the BUSINESS when staff say it; and `I` is a whole word only
    for text in ("quais são os nossos horários?", "our opening hours", "vocês abrem sábado?",
                 "iPhone repair price", "a Turma II começa quando?", "", "   "):
        assert not speaks_of_self(text), text


def test_the_first_person_list_is_held_in_the_tokenizers_own_alphabet():
    """A word the tokenizer can never produce is a dead entry that reads as a live one — the rule
    the stopword list already follows (``meus`` folds to ``meu`` on both sides)."""
    assert not [w for w in FIRST_PERSON if tokens(w) != [w]]
    assert {"meu", "minha", "eu", "my", "me"} <= FIRST_PERSON
    assert not {"nos", "nosso", "we", "our"} & FIRST_PERSON


def test_an_anchor_trims_function_words_at_its_ENDS_and_keeps_them_inside():
    assert anchor("o Orientador") == ("orientador",)
    assert anchor("Rua das Acácias") == ("rua", "das", "acacia"), "the inner «das» stays"
    assert anchor("  Marisa Lobo ") == ("marisa", "lobo")
    assert anchor("ela") == () and anchor("") == () and anchor(None) == ()


def test_about_is_whole_words_in_order_on_either_END_and_graph_only():
    edge = graph_candidates([_walk(_e("Dulce Amaral", "TRANSFERS_TO", "Marisa Lobo"))])[0]
    assert about(edge, ["Marisa"]) and about(edge, ["marisa lobo"]) and about(edge, ["DULCE"])
    assert about(edge, ["Dúlce"]), "the tokenizer's fold: accents and case"
    assert not about(edge, ["Mari"]), "whole words — «Mari» is not «Marisa»"
    assert not about(edge, ["Lobo Marisa"]), "in order"
    assert not about(edge, ["transfers"]), "the RELATION is not an end"
    assert not about(edge, []), "no anchor, nothing is about it"
    mem = memory_candidates([_Record("m1", "Marisa Lobo prefers email")])[0]
    assert mem.source == SOURCE_MEMORY and not about(mem, ["Marisa"]), "a memory has no ends"


# ── the three shapes, each in BOTH worlds ────────────────────────────────────────────

_ASKER = "Gisele Arantes"

#: A walk whose first edges are about other things (the shape: the answer sat DEEP in the block),
#: then the asker's own relations, which share no word with the question.
_SHAPE_ASKER = [_walk(
    _e("Escola Horizonte", "LOCATED_AT", "Avenida Brasil, 900"),
    _e("Escola Horizonte", "OFFERS", "curso de inglês"),
    _e("curso de inglês", "HAS_LEVEL", "intermediário"),
    _e("Escola Horizonte", "OPENS_AT", "7h"),
    _e("Rogério Pinto", "COORDINATES", "curso de inglês"),
    _e(_ASKER, "TEACHES", "Turma 7B"),
    _e(_ASKER, "TEACHES", "Turma 8A"),
    _e("Turma 7B", "MEETS_ON", "quarta-feira"),
)]


def test_SHAPE_the_asker_nothing_relevant_without_the_anchor_partial_with_it():
    cands = graph_candidates(_SHAPE_ASKER)
    asked = variants("my October schedule", "meus horários de outubro")
    ranked = rank(cands, asked)
    own = {c.text: s for s, c in ranked}
    teaches = [c for c in cands if "TEACHES" in c.text]
    # the measured shape, produced: the answer IS a candidate, deep in the walk, under the floor
    assert len(teaches) == 2 and all(c.prior >= 5 for c in teaches)
    assert all(own[c.text] < RELEVANCE_FLOOR for c in teaches)
    # the world BEFORE: nothing relevant, over candidates that held the answer
    assert decide(ranked) == (DECISION_NOTHING, [])
    # the world AFTER: the asker is what the question is about
    decision, picked = decide(ranked, anchors=[_ASKER])
    assert decision == DECISION_PARTIAL
    ids = [c.id for _, c in picked]
    assert ids[:2] == [c.id for c in teaches], "the asker's own edges first, in recall order"
    # ...then the rest of the answer, ONE hop on (the class meets on Wednesdays)
    assert any("MEETS_ON" in c.text for _, c in picked)
    assert not any("Escola Horizonte" in c.text for _, c in picked), "nothing that is not hers"


def test_SHAPE_one_hop_on_from_what_the_question_names_and_never_two():
    walk = [_walk(
        _e("Secretaria", "PUBLISHES", "calendário"),
        _e("Orientador", "AUTHORED", "Roteiro Trimestral"),
        _e("Roteiro Trimestral", "FOCUSES_ON", "conversação"),
        _e("conversação", "PRACTICED_IN", "laboratório 3"),
    )]
    ranked = rank(graph_candidates(walk),
                  variants("save in the calendar what the Orientador created yesterday",
                           "grava na agenda o que o Orientador criou ontem"))
    # one word of five («orientador») — under the floor, so the world BEFORE says nothing
    assert max(s for s, _ in ranked) < RELEVANCE_FLOOR
    assert decide(ranked)[0] == DECISION_NOTHING
    decision, picked = decide(ranked, anchors=["o Orientador"])
    assert decision == DECISION_PARTIAL
    tier = [c.text for _, c in picked]
    assert tier[0].startswith("Orientador --[AUTHORED]"), "what it names, first"
    assert tier[1].startswith("Roteiro Trimestral --[FOCUSES_ON]"), "one hop on: its focus"
    assert not any("PRACTICED_IN" in t for t in tier), "two hops leave the entity"
    assert not any("Secretaria" in t for t in tier)


def test_SHAPE_a_named_person_diluted_by_a_long_question():
    walk = [_walk(
        _e("Escola Horizonte", "OFFERS", "curso de espanhol"),
        _e("Dulce Amaral", "TRANSFERS_TO", "Marisa Lobo"),
        _e("Marisa Lobo", "WORKS_IN", "financeiro"),
    )]
    asked = variants("can Marisa approve refunds above the limit",
                     "a Marisa aprova reembolso acima do limite?")
    ranked = rank(graph_candidates(walk), asked)
    best = max(s for s, c in ranked if "TRANSFERS_TO" in c.text)
    assert 0 < best < RELEVANCE_FLOOR, "one word of the question — a quarter, a fifth"
    assert decide(ranked)[0] == DECISION_NOTHING                          # BEFORE
    decision, picked = decide(ranked, anchors=["Marisa"])                 # AFTER
    assert decision == DECISION_PARTIAL
    assert {c.text.split(" --")[0] for _, c in picked} == {"Dulce Amaral", "Marisa Lobo"}


# ── what the tier must NOT do ────────────────────────────────────────────────────────


def test_CONTROL_the_tier_never_competes_with_a_relevant_result():
    """When anything clears the floor, anchors change NOTHING — the same decision and the same
    picked list, object for object. Produced: the anchor here has edges, so the tier WOULD fire."""
    walk = [_walk(_e("Gisele Arantes", "TEACHES", "Turma 7B"),
                  _e("Escola Horizonte", "OPENS_AT", "7h"))]
    ranked = rank(graph_candidates(walk), variants("school opening time",
                                                   "que horas a escola horizonte abre?"))
    assert partial(ranked, [_ASKER]), "the control: the tier has something to offer"
    assert decide(ranked, anchors=[_ASKER]) == decide(ranked)
    assert decide(ranked)[0] == DECISION_RELEVANT


def test_a_BROKEN_source_still_wins_over_the_tier():
    ranked = rank(graph_candidates(_SHAPE_ASKER), ["my October schedule"])
    assert decide(ranked, failed=["memory"], anchors=[_ASKER])[0] == DECISION_ERROR
    assert decide(ranked, anchors=[_ASKER])[0] == DECISION_PARTIAL, "the pair: without the break"


def test_an_anchor_nobody_recorded_is_still_nothing_relevant():
    ranked = rank(graph_candidates(_SHAPE_ASKER), ["my October schedule"])
    assert decide(ranked, anchors=["Heloísa Prado", "ela", ""]) == (DECISION_NOTHING, [])


def test_the_tier_is_top_k_and_graph_only():
    edges = [_e(_ASKER, "TEACHES", f"Turma {n}A") for n in range(TOP_K + 3)]
    cands = graph_candidates([_walk(*edges)]) + memory_candidates(
        [_Record("m1", "Gisele Arantes prefers the morning shift")])
    tier = partial(rank(cands, ["my October timetable"]), [_ASKER])
    assert len(tier) == TOP_K and all(c.source == SOURCE_GRAPH for _, c in tier)


def test_an_anchor_walk_is_never_old_and_a_shared_edge_is_one_candidate():
    shared = _e(_ASKER, "TEACHES", "Turma 7B")
    cands = graph_candidates([_walk(shared, node=4),
                              (-1, 0, "a0", [shared, _e(_ASKER, "LIVES_IN", "Niterói")])],
                             baseline_nodes=2)
    assert [c.id for c in cands] == ["edge:4.0", "edge:a0.0"], "first id wins, ids content-free"
    assert [c.old for c in cands] == [True, False]


def test_render_SAYS_a_partial_answer_is_partial():
    ranked = rank(graph_candidates(_SHAPE_ASKER), ["my October schedule"])
    decision, picked = decide(ranked, anchors=[_ASKER])
    text = render("my October schedule", picked, decision=decision)
    assert text.startswith("Nothing recorded answers 'my October schedule' directly.")
    assert "it may not answer the question" in text and "[edge:1.5]" in text
    # the pair: a relevant answer keeps its old header, byte for byte
    assert render("x", picked).startswith("What we have recorded about 'x'")
    assert render("x", []) == "Nothing relevant recorded about 'x'."


def test_first_person_words_carry_a_reference_the_SCORE_never_sees():
    """Why the asker has to be an anchor rather than a word: the possessives are stopwords, so
    the question's subject is invisible to the score — the assumption the tier stands on, pinned
    so a stopword edit that changes it is red here."""
    assert {"meu", "minha", "eu", "my"} <= STOPWORDS
