"""Condition (b): a vector is NEVER compared across embedding models — watched at the seam.

The contract suite (``test_documents_store.py``) proves it by OUTCOME on both adapters: the
same numbers labelled with another model find nothing. This one proves it by MECHANISM on the
in-memory adapter, where the one function that compares two vectors can be watched: a search
whose query model differs from a chunk's must not reach it AT ALL. "The cosine was computed and
then discarded" would pass an outcome test and fail this one.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from cogno_engram.adapters import in_memory
from cogno_engram.adapters.in_memory import InMemoryDocumentStore
from cogno_engram.documents import KB_EMBED_SPACE_UNAVAILABLE

from documents_support import EMB_DIM, MODEL_A, MODEL_B, publish, vec


@pytest.fixture
def cosines(monkeypatch):
    calls: list[tuple] = []
    real = in_memory._doc_cosine

    def spy(a, b):
        calls.append((tuple(a), tuple(b)))
        return real(a, b)

    monkeypatch.setattr(in_memory, "_doc_cosine", spy)
    return calls


async def _store_with(*models: str):
    store = InMemoryDocumentStore(embedding_dim=EMB_DIM)
    o = f"acme{uuid4().hex[:6]}/p"
    for model in models:
        await publish(store, o, profiles=("GUEST",), model=model,
                      chunks=(("abrimos no sábado", vec(1.0)), ("fechamos no domingo", vec(0.0, 1.0))))
    return store, o


async def test_a_query_of_another_model_never_reaches_the_cosine(cosines):
    store, o = await _store_with(MODEL_A)
    res = await store.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_B)
    assert cosines == []
    assert res.degradations == (KB_EMBED_SPACE_UNAVAILABLE,)
    assert [h.vector_score for h in res.hits] == [None]
    # CONTROL — the same query under the chunks' own model DOES compute, once per chunk.
    await store.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_A)
    assert len(cosines) == 2


async def test_one_foreign_model_among_the_readable_chunks_turns_the_whole_search_lexical(cosines):
    store, o = await _store_with(MODEL_A, MODEL_B)
    res = await store.search(o, profile="GUEST", text="sábado", vector=vec(1.0), embed_model=MODEL_B,
                             limit=10)
    assert cosines == []                                   # not even the MODEL_B chunks
    assert res.models_unavailable == (MODEL_A,)
    assert {h.vector_score for h in res.hits} == {None}


async def test_without_a_vector_nothing_is_compared(cosines):
    store, o = await _store_with(MODEL_A)
    res = await store.search(o, profile="GUEST", text="sábado", embed_model=MODEL_A)
    assert cosines == [] and res.models_unavailable == (MODEL_A,)
    res = await store.search(o, profile="GUEST", text="sábado")          # no label either
    assert cosines == [] and res.models_unavailable == (MODEL_A,)


async def test_a_reader_with_nothing_readable_is_not_degraded(cosines):
    store, o = await _store_with(MODEL_A)
    res = await store.search(o, profile="EMPLOYEE", text="sábado", vector=vec(1.0),
                             embed_model=MODEL_B)
    assert res.hits == () and res.degradations == () and cosines == []
