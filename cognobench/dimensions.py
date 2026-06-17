"""The three bench dimensions, driven directly against the in-memory adapter."""

from __future__ import annotations

from cogno_engram import hypnos, rerank
from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore
from cogno_engram.types import GraphEdge, GraphNode, MemoryRecord, RetrievalQuery, TurnRecord

from cognobench.buffer_cases import CASES as BUFFER_CASES
from cognobench.consolidation_cases import CASES as CONSOLIDATION_CASES
from cognobench.embedder import HashingEmbedder
from cognobench.graph_cases import CASES as GRAPH_CASES
from cognobench.lifecycle_cases import CASES as LIFECYCLE_CASES
from cognobench.llm_consolidation_cases import CASES as LLM_CONSOLIDATION_CASES
from cognobench.retrieval_cases import CASES as RETRIEVAL_CASES
from cognobench.types import CheckResult, DimensionResult


async def run_retrieval(limit: int | None = None) -> DimensionResult:
    dim = DimensionResult("retrieval")
    emb = HashingEmbedder()
    cases = RETRIEVAL_CASES[:limit] if limit else RETRIEVAL_CASES
    for case in cases:
        store = InMemoryStore()
        for category, content in case.memories:
            embedding = emb.embed(content) if case.use_embedding else None
            await store.save_memory(MemoryRecord("bench", category, content, embedding=embedding))
        q_emb = emb.embed(case.query) if case.use_embedding else None
        out = await store.load_memories(
            "bench", query=RetrievalQuery(text=case.query, embedding=q_emb), limit=3)
        top = out[0].content if out else ""
        name = "hit@1" if case.use_embedding else "hit@1(bm25)"
        dim.checks.append(CheckResult(
            case.id, name, case.expect_top, top, case.expect_top.lower() in top.lower()))
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


async def run_lifecycle(limit: int | None = None) -> DimensionResult:
    """End-to-end: turns → Tier-1 micro → retrieval + rerank (deterministic, no LLM)."""
    dim = DimensionResult("lifecycle")
    emb = HashingEmbedder()
    cases = LIFECYCLE_CASES[:limit] if limit else LIFECYCLE_CASES
    for case in cases:
        store = InMemoryStore()
        session = await store.create_session("bench")
        prev = None
        for i, td in enumerate(case.turns, 1):
            turn = TurnRecord(session.id, "bench", i, td.get("user_input", "x"),
                              goal=td.get("goal", ""), goal_status=td.get("goal_status", ""),
                              sentiment=td.get("sentiment", ""), domains=td.get("domains", []),
                              pii_types=td.get("pii_types", []))
            await store.save_turn(turn)
            for m in hypnos.micro_consolidate(turn, prev):
                m.embedding = emb.embed(m.content)
                await store.save_memory(m)
            prev = turn
        hits = await store.load_memories(
            "bench", query=RetrievalQuery(text=case.query, embedding=emb.embed(case.query)), limit=10)
        top = rerank(hits, query_text=case.query, top_k=3)
        ok = any(case.expect_retrieved.lower() in m.content.lower() for m in top)
        dim.checks.append(CheckResult(case.id, "retrieved@3", case.expect_retrieved,
                                      str([m.content for m in top]), ok))
    return dim


async def run_buffer(limit: int | None = None) -> DimensionResult:
    """Short-term buffer retention — which terms survive the sliding window."""
    dim = DimensionResult("buffer")
    cases = BUFFER_CASES[:limit] if limit else BUFFER_CASES
    for case in cases:
        buf = InMemoryBuffer()
        for i, text in enumerate(case.turns, 1):
            await buf.push("bench", "sess", TurnRecord("sess", "bench", i, text))
        window = await buf.window("bench", "sess", size=case.window_size)
        text = " | ".join(t.user_input for t in window)
        for term in case.expect_present:
            dim.checks.append(CheckResult(case.id, f"present:{term}", "present", text,
                                          term.lower() in text.lower()))
        for term in case.expect_absent:
            dim.checks.append(CheckResult(case.id, f"absent:{term}", "absent", text,
                                          term.lower() not in text.lower()))
    return dim


async def run_llm_consolidation(limit: int | None = None, *,
                                model: str = "mistral:latest",
                                base_url: str = "http://localhost:11434") -> DimensionResult:
    """Gated: hypnos Tier-2 extraction QUALITY against a real Ollama model.

    Soft, model-dependent. Auto-skips (0 checks) if Ollama is unreachable, so the
    default deterministic run is unaffected.
    """
    from cognobench.ollama import OllamaBackend, is_available
    dim = DimensionResult("llm_consolidation")
    if not await is_available(base_url):
        return dim     # 0 checks → reported as skipped
    backend = OllamaBackend(model=model, base_url=base_url)
    cases = LLM_CONSOLIDATION_CASES[:limit] if limit else LLM_CONSOLIDATION_CASES
    for case in cases:
        store = InMemoryStore()
        session = await store.create_session("bench")
        for i, (user, resp) in enumerate(case.turns, 1):
            await store.save_turn(TurnRecord(session.id, "bench", i, user, response=resp))
        mems = await hypnos.periodic_consolidate(
            store, backend, scope="bench", session_id=session.id, extract_relations=False)
        blob = " | ".join(m.content for m in mems).lower()
        for term in case.expect_contains:
            dim.checks.append(CheckResult(case.id, f"contains:{term}(soft)", term, blob,
                                          term.lower() in blob))
    return dim


# Deterministic dimensions run by default; the LLM one is opt-in (`--only`).
DETERMINISTIC = ["retrieval", "buffer", "consolidation", "graph", "lifecycle"]

DIMENSIONS = {
    "retrieval": run_retrieval,
    "buffer": run_buffer,
    "consolidation": run_consolidation,
    "graph": run_graph,
    "lifecycle": run_lifecycle,
    "llm_consolidation": run_llm_consolidation,
}
