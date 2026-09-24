"""The pure rules of ``cogno_engram.documents`` — one definition, both adapters."""

from __future__ import annotations

import pytest

from cogno_engram import documents as kb
from cogno_engram.documents import (
    EXTRACTOR_REASONS,
    VALID_KB_REASONS,
    ExtractionError,
    KbDocument,
    KbHit,
    KbVersion,
    embed_model_label,
    hit_order_key,
    hybrid_score,
    model_dimensions,
    owner_in_subtree,
    profile_can_read,
    read_extracted,
    require_model,
    require_owner,
    require_profile,
    require_vector,
    sanitize_profiles,
    sanitize_reason,
)

from documents_support import Bookmark, Extracted, Page


def test_the_extractor_alphabet_is_the_five_the_contract_names():
    assert EXTRACTOR_REASONS == {"no_text", "over_limit", "encrypted", "invalid", "timeout"}
    assert EXTRACTOR_REASONS < VALID_KB_REASONS


@pytest.mark.parametrize("raw,want", [
    ("GUEST", ("GUEST",)), (["b", " a ", "b", "", 3, None], ("a", "b")), ({"X"}, ("X",)),
    (None, ()), (42, ()), ("", ()), ([], ()),
])
def test_profiles_are_sanitised_and_a_bare_string_is_one_profile(raw, want):
    assert sanitize_profiles(raw) == want


def test_profile_membership_is_exact_after_stripping():
    assert profile_can_read(" GUEST ", ("GUEST",))
    assert not profile_can_read("guest", ("GUEST",))
    assert not profile_can_read("", ("",))
    assert not profile_can_read("GUEST", ())
    assert not profile_can_read(None, ("GUEST",))                      # type: ignore[arg-type]


@pytest.mark.parametrize("owner,prefix,inside", [
    ("t1", "t1", True), ("t1/p", "t1", True), ("t1/p/q", "t1", True),
    ("t10", "t1", False), ("t10/p", "t1", False), ("t1x", "t1", False), ("t", "t1", False),
])
def test_the_subtree_rule(owner, prefix, inside):
    assert owner_in_subtree(owner, prefix) is inside


@pytest.mark.parametrize("bad", ["", "  ", None, 3])
def test_blank_owners_and_profiles_are_refused(bad):
    with pytest.raises(ValueError):
        require_owner(bad)
    with pytest.raises(ValueError):
        require_profile(bad)


def test_the_model_label_carries_its_width_and_the_store_checks_it():
    label = embed_model_label("ollama:nomic-embed-text:latest", 768)
    assert label == "ollama:nomic-embed-text:latest@768" and model_dimensions(label) == 768
    assert require_model(label, 768) == label
    with pytest.raises(ValueError):
        require_model(label, 1536)
    for bad in ("nomic", "nomic@", "a b@768", "x@0", "@768", None):
        with pytest.raises(ValueError):
            model_dimensions(bad)
    with pytest.raises(ValueError):
        embed_model_label("with space", 768)


def test_vectors_are_checked_for_width_and_finiteness():
    assert require_vector((1, 2), 2) == [1.0, 2.0]
    for bad in ([1.0], [1.0, float("nan")], [1.0, float("inf")], "ab", None):
        with pytest.raises(ValueError):
            require_vector(bad, 2)


def test_reasons_are_a_closed_alphabet():
    assert sanitize_reason(" Timeout ") == "timeout"
    assert sanitize_reason("/tmp/x.pdf: boom") == "internal"
    assert sanitize_reason(None) == "internal"

    class Hostile:
        def __str__(self):
            raise RuntimeError

    assert sanitize_reason(Hostile()) == "internal"
    err = ExtractionError("rm -rf", "detail")
    assert err.reason == "internal" and err.detail == "detail"


def test_the_extractor_answer_is_read_structurally():
    got = read_extracted(Extracted(pages=[Page(1, "a"), Page(2, "b")],
                                   outline=[Bookmark(1, " Parte ", 2), Bookmark(2, "", 1)]))
    assert [(p.number, p.text) for p in got.pages] == [(1, "a"), (2, "b")]
    assert [(e.level, e.title, e.page) for e in got.outline] == [(1, "Parte", 2)]
    for bad in (object(), Extracted(pages=[Page(0, "a")]), Extracted(pages=[Page(1, b"a")]),
                Extracted(pages=None)):
        with pytest.raises(ExtractionError) as err:
            read_extracted(bad)
        assert err.value.reason == "invalid"


def test_the_score_is_renormalised_over_the_components_present():
    assert hybrid_score(None, 0.37, vector_weight=0.6, lexical_weight=0.4) == 0.37
    assert hybrid_score(0.5, 0.25, vector_weight=0.6, lexical_weight=0.4) == pytest.approx(0.4)
    assert hybrid_score(0.5, 0.25, vector_weight=6, lexical_weight=4) == pytest.approx(0.4)
    assert hybrid_score(2.0, -1.0, vector_weight=1, lexical_weight=1) == 0.5      # clamped inputs
    assert hybrid_score(None, float("nan"), vector_weight=1, lexical_weight=1) == 0.0
    for wv, wl in ((0, 0), (-1, 2)):
        with pytest.raises(ValueError):
            hybrid_score(0.1, 0.1, vector_weight=wv, lexical_weight=wl)


def _hit(doc, version, ordinal, score):
    return KbHit(id="", document_id=doc, version=version, ordinal=ordinal, title="", heading_path=(),
                 page=None, content="", embed_model="m@1", vector_score=None, lexical_score=score,
                 score=score)


def test_hits_order_best_first_then_document_version_ordinal():
    hits = [_hit("b", 1, 0, 0.5), _hit("a", 2, 1, 0.5), _hit("a", 2, 0, 0.5), _hit("a", 1, 9, 0.5),
            _hit("z", 1, 0, 0.9)]
    assert [(h.document_id, h.version, h.ordinal) for h in sorted(hits, key=hit_order_key)] == \
        [("z", 1, 0), ("a", 1, 9), ("a", 2, 0), ("a", 2, 1), ("b", 1, 0)]


def test_a_document_reads_the_state_of_its_latest_attempt():
    v = KbVersion(document_id="d", version=1, state="error", sha256="", embed_model="m@1")
    assert KbDocument(owner_key="o", id="d", title="", profiles=(), media_type=kb.MEDIA_PDF,
                      latest=v).status == "error"
    assert KbDocument(owner_key="o", id="d", title="", profiles=(),
                      media_type=kb.MEDIA_PDF).status == "processing"
    assert kb.chunk_id("d", 2, 3) == "kb:d.2.3"
