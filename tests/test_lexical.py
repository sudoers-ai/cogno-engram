"""``cogno_engram.lexical`` — the tokenizer, the score, the ranking and the decision.

The first seven tests MOVED here with the engine they test (from the reference host's hybrid
retrieval suite), their fixtures re-invented; the rest are this library's own. All content is
INVENTED — names, pets, clinics — and every name here is a fixture, not a person.
"""

from __future__ import annotations

import pytest

from cogno_engram import GraphEdge
from cogno_engram.folding import fold_label
from cogno_engram.lexical import (DECISION_ERROR, DECISION_NOTHING, DECISION_RELEVANT,
                                  RELEVANCE_FLOOR, SOURCE_GRAPH, SOURCE_MATERIAL,
                                  SOURCE_MEMORY, SOURCES, STOPWORDS, TOP_K, Candidate, chosen,
                                  decide, edge_end, edge_text, fold, graph_candidates,
                                  memory_candidates, rank, relevance, render, terms, tokens,
                                  variants)

_EDGES = [
    GraphEdge(scope="t", source="Dra. Quitéria Brum", target="ortodontia",
              relation="SPECIALIZES_IN"),
    GraphEdge(scope="t", source="Anselmo Pires", target="Rosália Pires", relation="MARRIED_TO"),
]


class _Record:
    def __init__(self, rid, content):
        self.id, self.content = rid, content


# ── moved with the engine ────────────────────────────────────────────────────────────


def test_terms_fold_accents_and_plurals_and_drop_function_words():
    assert terms("Qual a ESPECIALIDADE da Dra. Quitéria?") == {"especialidade", "dra", "quiteria"}
    assert terms("clínicas") == terms("clinica")


def test_relevance_is_the_better_of_the_two_languages():
    edge = terms("Dra. Quitéria Brum --[SPECIALIZES_IN]--> ortodontia")
    canonical, original = terms("orthodontics"), terms("quem faz ortodontia?")
    assert relevance(edge, [canonical]) == 0.0, "the English query alone misses the PT label"
    # one of the contact's three content words (quem, faz, ortodontia) — a third
    assert relevance(edge, [canonical, original]) == pytest.approx(1 / 3), \
        "the contact's own words find it"


def test_a_score_of_zero_never_passes_whatever_the_floor():
    c = Candidate(id="x", source=SOURCE_GRAPH, text="nothing in common")
    assert chosen([(0.0, c)], floor=0.0) == []


def test_the_original_is_a_variant_only_when_it_adds_words():
    assert variants("Quitéria", "quiteria") == ["Quitéria"]
    assert variants("Quitéria", "") == ["Quitéria"]
    assert variants("orthodontics", "quem faz ortodontia?") == ["orthodontics",
                                                                "quem faz ortodontia?"]


def test_one_hop_inherits_along_the_walk_and_only_forward():
    walk = [(0, 0, 1, [GraphEdge(scope="t", source="Zuleide", target="Farofa", relation="HAS_PET"),
                       GraphEdge(scope="t", source="Farofa", target="beagle", relation="IS_A"),
                       GraphEdge(scope="t", source="Odete", target="Zuleide", relation="KNOWS")])]
    cands = graph_candidates(walk)
    by = {c.text.split(" --")[0] + ">" + c.text.split("--> ")[1]: s
          for s, c in rank(cands, ["Zuleide pet"])}
    assert by["Zuleide>Farofa"] == 1.0
    assert by["Farofa>beagle"] == 0.5, "the rest of the answer, one hop on"
    assert by["Odete>Zuleide"] == 0.5, "matched on its own word (Zuleide), not inherited"
    # the inheritance is FORWARD only: nothing points INTO Odete, so a question about the pet
    # does not drag in who knows its owner
    by_pet = {c.id: s for s, c in rank(cands, ["Farofa"])}
    assert by_pet["edge:1.2"] == 0.0


def test_decide_is_deterministic_on_ties():
    a = Candidate(id="b", source=SOURCE_MATERIAL, text="Quitéria", prior=0)
    b = Candidate(id="a", source=SOURCE_GRAPH, text="Quitéria", prior=5)
    picked = decide(rank([a, b], ["Quitéria"]))[1]
    assert [c.id for _, c in picked] == ["a", "b"], "graph before material on a tie"


def test_the_stopword_list_is_stored_in_the_tokenizers_own_alphabet():
    """A stopword the tokenizer can never produce is a dead entry that reads as a live one.

    ``quais`` folds to ``quai`` and ``mais`` to ``mai``; held as raw text they would silently
    stop matching the day the fold arrived, and leak back into the floor as meaning-carrying
    words.
    """
    assert {"quai", "mai", "meno"} <= STOPWORDS
    assert not [w for w in STOPWORDS if tokens(w) != [w]]


# ── the fold and the tokenizer ───────────────────────────────────────────────────────


def test_the_fold_is_nfkd_then_marks_then_casefold():
    assert fold("Não, está ÓTIMO") == "nao, esta otimo"
    assert fold("Straße") == "strasse", "casefold, not lower"
    assert fold("𝐉𝐨𝐚𝐨") == "joao", "NFKD first: compatibility capitals come out lower-case"
    assert fold(None) == ""
    for s in ("ᴬ", "ϒ", "℃", "ẞ", "ΟΔΟΣ", "İstanbul", "ﬁnal"):
        assert fold(fold(s)) == fold(s), s


def test_the_word_fold_is_NOT_the_label_fold_and_says_so():
    """The two folds answer different questions, and the difference is observable: the label
    fold transliterates (it must agree with Postgres ``unaccent``), the word fold does not. If
    this ever went green on equality, one of the two docstrings would be lying."""
    assert fold_label("Øverby") == "overby"
    assert fold("Øverby") == "øverby"
    assert tokens("Øverby") == ["verby"], "the ø is not a word character to this tokenizer"


def test_tokens_fold_the_portuguese_plural_and_nothing_more():
    assert tokens("Ementas de Redes") == ["ementa", "de", "rede"]
    assert tokens("class process") == ["class", "process"], "-ss survives"
    assert tokens("dos nos as") == ["dos", "nos", "as"], "under four characters survives"
    assert tokens("reposições materiais") == ["reposicoe", "materiai"], "not a stemmer"
    assert tokens("a é x 1") == [], "single characters are never tokens"
    assert tokens("R$ 180,00 às 14h") == ["180", "00", "as", "14h"]


def test_the_floor_is_the_stopword_list_and_nothing_else():
    """Function words never count; one content word does. Without the second half this would
    pass over a ``terms`` that returned nothing at all."""
    assert terms("de do da para com the of") == frozenset()
    assert terms("de do da modelagem") == {"modelagem"}
    assert "de" in STOPWORDS and "modelagem" not in STOPWORDS


def test_the_prefix_is_applied_after_the_stopword_test():
    assert terms("implante implant", prefix=6) == {"implan"}
    assert terms("implante implant") == {"implante", "implant"}
    assert terms("quando", prefix=3) == frozenset(), "a function word stays a function word"


# ── what the default floor MEANS on this scale ───────────────────────────────────────


def test_the_default_floor_keeps_a_third_and_drops_a_quarter():
    """Not a calibration — the calibration is the consumer's, over its own labelled set. This is
    what 0.3 DOES here: of a four-content-word question, one shared word is not enough and two
    are; of a three-word question, one is."""
    q4 = ["prazo entrega lente progressiva"]
    q3 = ["prazo entrega lente"]
    one = Candidate(id="c", source=SOURCE_MEMORY, text="o prazo varia")
    two = Candidate(id="d", source=SOURCE_MEMORY, text="prazo de entrega: dez dias")
    assert [c.id for _, c in chosen(rank([one, two], q4))] == ["d"]
    assert [c.id for _, c in chosen(rank([one, two], q3))] == ["d", "c"]
    assert RELEVANCE_FLOOR == 0.3


# ── candidates ───────────────────────────────────────────────────────────────────────


def test_every_id_is_content_free():
    edges = graph_candidates([(0, 0, 42, _EDGES)])
    mems = memory_candidates([_Record("9b1d-uuid", "Prefers mornings")])
    ids = [c.id for c in edges + mems]
    assert ids == ["edge:42.0", "edge:42.1", "mem:9b1d-uuid"]
    for word in ("Quitéria", "Brum", "Anselmo", "Rosália", "Prefers", "ortodontia"):
        assert all(word.lower() not in i.lower() for i in ids), word


def test_an_edge_reached_from_two_start_nodes_is_one_candidate():
    same = GraphEdge(scope="t", source="ANSELMO PIRES", target="rosália pires",
                     relation="MARRIED_TO")
    cands = graph_candidates([(0, 0, 1, _EDGES), (0, 1, 2, [same])])
    assert [c.id for c in cands] == ["edge:1.0", "edge:1.1"], "deduplicated case-insensitively"
    assert [c.prior for c in cands] == [0, 1]


def test_baseline_nodes_marks_the_first_nodes_of_the_FIRST_variant_only():
    e = lambda s: [GraphEdge(scope="t", source=s, target="x", relation="R")]  # noqa: E731
    walks = [(0, 0, 1, e("a")), (0, 1, 2, e("b")), (0, 2, 3, e("c")), (1, 0, 4, e("d"))]
    marked = {c.text.split(" ")[0]: c.old for c in graph_candidates(walks, baseline_nodes=2)}
    assert marked == {"a": True, "b": True, "c": False, "d": False}
    assert not any(c.old for c in graph_candidates(walks)), "0 marks none"


def test_the_ends_are_folded_for_the_hop():
    c = graph_candidates([(0, 0, 1, [GraphEdge(scope="t", source="Dra. Quitéria",
                                                target="Clínicas Unidas", relation="R")])])[0]
    assert (c.head, c.tail) == ("dra quiteria", "clinica unida")
    assert edge_end(None) == "" and edge_end(7) == ""


def test_edge_text_is_the_graph_block_shape_with_its_own_detail_cut():
    plain = GraphEdge(scope="t", source="A", target="B", relation="R")
    assert edge_text(plain) == "A --[R]--> B"
    rich = GraphEdge(scope="t", source="A", target="B", relation="R",
                     attributes={"z": "last", "a": "  first\nline  ", "m": "   "})
    assert edge_text(rich) == "A --[R]--> B (a: first line; z: last)", "sorted, flattened"
    long = GraphEdge(scope="t", source="A", target="B", relation="R",
                     attributes={"note": "x" * 500})
    detail = edge_text(long).split(" (", 1)[1]
    assert len(detail) == 160 + 1 and detail.endswith("x)"), "cut at 160, no ellipsis"


def test_memory_candidates_skip_what_cannot_be_cited_or_said():
    recs = [_Record("m1", "Prefers mornings"), _Record("m1", "duplicate id"),
            _Record("", "no id"), _Record("m2", "   "), _Record("m3", "Allergic to latex")]
    assert [c.id for c in memory_candidates(recs)] == ["mem:m1", "mem:m3"]
    assert [c.prior for c in memory_candidates(recs)] == [0, 1]
    assert [c.id for c in memory_candidates(recs, limit=1)] == ["mem:m1"]


# ── the decision and the payload ─────────────────────────────────────────────────────


def test_a_broken_source_is_error_never_nothing_relevant():
    unrelated = rank([Candidate(id="x", source=SOURCE_GRAPH, text="estacionamento")], ["lente"])
    assert decide(unrelated)[0] == DECISION_NOTHING
    assert decide(unrelated, failed=[SOURCE_MEMORY])[0] == DECISION_ERROR
    # PAIR: a relevant hit wins over a broken source — the others did answer
    hit = rank([Candidate(id="y", source=SOURCE_GRAPH, text="lente")], ["lente"])
    assert decide(hit, failed=[SOURCE_MEMORY])[0] == DECISION_RELEVANT


def test_chosen_is_top_k_at_or_above_the_floor():
    cands = [Candidate(id=f"c{i}", source=SOURCE_MEMORY, text="lente", prior=i) for i in range(9)]
    picked = chosen(rank(cands, ["lente"]))
    assert [c.id for _, c in picked] == [f"c{i}" for i in range(TOP_K)]
    assert chosen([(0.3, cands[0])], floor=0.3) == [(0.3, cands[0])], "AT the floor passes"


def test_ties_follow_the_source_order_then_recall_order_then_id():
    pool = [Candidate(id="z", source=SOURCE_MATERIAL, text="lente", prior=0),
            Candidate(id="y", source=SOURCE_MEMORY, text="lente", prior=0),
            Candidate(id="b", source=SOURCE_GRAPH, text="lente", prior=1),
            Candidate(id="a", source=SOURCE_GRAPH, text="lente", prior=1),
            Candidate(id="q", source="elsewhere", text="lente", prior=0)]
    assert [c.id for _, c in rank(pool, ["lente"])] == ["a", "b", "y", "z", "q"]
    assert SOURCES == (SOURCE_GRAPH, SOURCE_MEMORY, SOURCE_MATERIAL)


def test_render_cites_by_id_and_clips_each_result():
    long = Candidate(id="mem:1", source=SOURCE_MEMORY, text="palavra\n\n" + "y" * 900)
    out = render("lens", [(1.0, long)])
    head, line = out.split("\n")
    assert head == "What we have recorded about 'lens' (most relevant first; cite by id):"
    assert line.startswith("[mem:1] palavra y") and len(line) == len("[mem:1] ") + 600
    assert render("lens", []) == "Nothing relevant recorded about 'lens'."


def test_a_hop_needs_a_scored_parent_and_can_be_switched_off():
    walk = [(0, 0, 1, [GraphEdge(scope="t", source="Zuleide", target="Farofa", relation="HAS_PET"),
                       GraphEdge(scope="t", source="Farofa", target="beagle", relation="IS_A")])]
    cands = graph_candidates(walk)
    assert [s for s, _ in rank(cands, ["Zuleide"], hop=0.0)] == [1.0, 0.0]
    assert [s for s, _ in rank(cands, ["estacionamento"])] == [0.0, 0.0], "nothing to inherit"
