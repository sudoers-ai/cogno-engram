"""The three bench dimensions, driven directly against the in-memory adapter."""

from __future__ import annotations

from cogno_engram import hypnos
from cogno_engram.adapters.in_memory import InMemoryGraph, InMemoryStore
from cogno_engram.types import GraphEdge, GraphNode, MemoryRecord, RetrievalQuery, TurnRecord

from cognobench.consolidation_cases import CASES as CONSOLIDATION_CASES
from cognobench.embedder import HashingEmbedder
from cognobench.graph_cases import CASES as GRAPH_CASES
from cognobench.retrieval_cases import CASES as RETRIEVAL_CASES
from cognobench.types import CheckResult, DimensionResult


async def run_retrieval(limit: int | None = None) -> DimensionResult:
    dim = DimensionResult("retrieval")
    emb = HashingEmbedder()
    cases = RETRIEVAL_CASES[:limit] if limit else RETRIEVAL_CASES
    for case in cases:
        store = InMemoryStore()
        for category, content in case.memories:
            await store.save_memory(MemoryRecord("bench", category, content, embedding=emb.embed(content)))
        out = await store.load_memories(
            "bench", query=RetrievalQuery(text=case.query, embedding=emb.embed(case.query)), limit=3)
        top = out[0].content if out else ""
        dim.checks.append(CheckResult(
            case.id, "hit@1", case.expect_top, top, case.expect_top.lower() in top.lower()))
    return dim


async def run_consolidation(limit: int | None = None) -> DimensionResult:
    dim = DimensionResult("consolidation")
    cases = CONSOLIDATION_CASES[:limit] if limit else CONSOLIDATION_CASES
    for case in cases:
        turn = TurnRecord(**case.turn)
        prev = TurnRecord(**case.prev) if case.prev else None
        mems = hypnos.micro_consolidate(turn, prev)
        if case.expect_empty:
            dim.checks.append(CheckResult(case.id, "no_memory", "[]",
                                          str([m.content for m in mems]), not mems))
            continue
        for cat, substr in case.expect:
            hit = any(m.category == cat and substr.lower() in m.content.lower() for m in mems)
            dim.checks.append(CheckResult(case.id, f"{cat}:{substr}", "present",
                                          str([m.content for m in mems]), hit))
    return dim


async def run_graph(limit: int | None = None) -> DimensionResult:
    dim = DimensionResult("graph")
    cases = GRAPH_CASES[:limit] if limit else GRAPH_CASES
    for case in cases:
        graph = InMemoryGraph()
        for label, ntype in case.nodes:
            await graph.upsert_node(GraphNode("bench", label, ntype))
        for src, tgt, rel in case.edges:
            await graph.upsert_edge(GraphEdge("bench", src, tgt, rel, source_session="s1"))
        edges = await graph.walk("bench", case.start, max_depth=case.max_depth)
        got = {e.relation for e in edges}
        dim.checks.append(CheckResult(
            case.id, "reachable_relations", str(sorted(case.expect_relations)),
            str(sorted(got)), got == case.expect_relations))
    return dim


DIMENSIONS = {
    "retrieval": run_retrieval,
    "consolidation": run_consolidation,
    "graph": run_graph,
}
