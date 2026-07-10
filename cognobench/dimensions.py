"""The three bench dimensions, driven directly against the in-memory adapter."""

from __future__ import annotations

from cogno_engram import hypnos, rerank
from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore
from cogno_engram.types import GraphEdge, GraphNode, MemoryRecord, RetrievalQuery, TurnRecord

from cognobench.buffer_cases import CASES as BUFFER_CASES
from cognobench.consolidation_cases import CASES as CONSOLIDATION_CASES
from cognobench.embedder import HashingEmbedder
from cognobench.graph_capture_cases import CASES as GRAPH_CAPTURE_CASES
from cognobench.graph_cases import CASES as GRAPH_CASES
from cognobench.graph_html import GraphSnapshot
from cognobench.lifecycle_cases import CASES as LIFECYCLE_CASES
from cognobench.llm_consolidation_cases import CASES as LLM_CONSOLIDATION_CASES
from cognobench.retrieval_cases import CASES as RETRIEVAL_CASES
from cognobench.types import CheckResult, DimensionResult


async def _snapshot(graph: InMemoryGraph, scope: str, title: str, note: str = "") -> GraphSnapshot:
    """Collect what a case left in the graph (for --graph-html)."""
    nodes = await graph.list_nodes(scope)
    seen: set[tuple[str, str, str]] = set()
    edges: list[tuple[str, str, str, float]] = []
    for node in nodes:
        for e in await graph.walk(scope, node.label, max_depth=3):
            key = (e.source.lower(), e.target.lower(), e.relation)
            if key not in seen:
                seen.add(key)
                edges.append((e.source, e.target, e.relation, e.confidence))
    return GraphSnapshot(title=title, nodes=[(n.label, n.node_type) for n in nodes],
                         edges=edges, note=note)


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


async def run_graph(limit: int | None = None, *,
                    viz: list[GraphSnapshot] | None = None) -> DimensionResult:
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
        if viz is not None:
            viz.append(await _snapshot(graph, "bench", f"graph · {case.id}",
                                       case.description or "synthetic walk case"))
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


async def run_graph_capture(limit: int | None = None, *,
                            model: str = "mistral:latest",
                            base_url: str = "http://localhost:11434",
                            viz: list[GraphSnapshot] | None = None) -> DimensionResult:
    """Gated: the FULL edge-capture path — hypnos Tier-2 ``extract_relations`` against a real
    model, writing into a real graph adapter at ``kg_scope``.

    Hard, model-independent invariants per case: every edge endpoint exists as a node, the
    confidence is in (0,1], edges carry the session tag (feedback pruning key), and graph rows
    land ONLY at ``kg_scope`` — never at the narrower memory scope (regression guard for the
    host's tenant-vs-identity scope split). Soft (model-dependent): the expected entity pairs
    are connected within a 2-hop walk. Auto-skips (0 checks) if Ollama is unreachable.
    """
    from cognobench.ollama import OllamaBackend, is_available
    dim = DimensionResult("graph_capture")
    if not await is_available(base_url):
        return dim     # 0 checks → reported as skipped
    backend = OllamaBackend(model=model, base_url=base_url)
    mem_scope, kg_scope = "bench/u1", "bench"
    cases = GRAPH_CAPTURE_CASES[:limit] if limit else GRAPH_CAPTURE_CASES
    for case in cases:
        store, graph = InMemoryStore(), InMemoryGraph()
        session = await store.create_session(mem_scope)
        for i, (user, resp) in enumerate(case.turns, 1):
            await store.save_turn(TurnRecord(session.id, mem_scope, i, user, response=resp))
        await hypnos.periodic_consolidate(
            store, backend, scope=mem_scope, session_id=session.id,
            kg=graph, kg_scope=kg_scope, extract_relations=True)

        nodes = await graph.list_nodes(kg_scope)
        labels = {n.label.lower() for n in nodes}
        edges = [e for e in graph._edges if e.scope == kg_scope]
        leaked = ([n for n in await graph.list_nodes(mem_scope)]
                  + [e for e in graph._edges if e.scope == mem_scope])
        # hard invariants
        dim.checks.append(CheckResult(case.id, "kg_scope_only", "no rows at memory scope",
                                      f"{len(leaked)} leaked", not leaked))
        dangling = [e for e in edges
                    if e.source.lower() not in labels or e.target.lower() not in labels]
        dim.checks.append(CheckResult(case.id, "no_dangling_edges", "[]",
                                      str([(e.source, e.target) for e in dangling]),
                                      not dangling))
        dim.checks.append(CheckResult(case.id, "confidence_valid", "(0,1]",
                                      str([e.confidence for e in edges]),
                                      all(0 < e.confidence <= 1.0 for e in edges)))
        dim.checks.append(CheckResult(case.id, "session_tagged", session.id,
                                      str({e.source_session for e in edges} or "{}"),
                                      all(e.source_session == session.id for e in edges)))
        # soft: expected connections reachable (labels vary less than relation names)
        for start, target in case.expect_connected:
            start_node = next((n.label for n in nodes if start in n.label.lower()), None)
            reached: set[str] = set()
            if start_node:
                for e in await graph.walk(kg_scope, start_node, max_depth=2):
                    reached.update((e.source.lower(), e.target.lower()))
            hit = any(target in r for r in reached)
            dim.checks.append(CheckResult(case.id, f"connected:{start}->{target}(soft)",
                                          "reachable", str(sorted(reached)) or "—", hit))
        if viz is not None:
            viz.append(await _snapshot(
                graph, kg_scope, f"graph_capture · {case.id} ({model})",
                case.description or "LLM-captured relations"))
    return dim


# Deterministic dimensions run by default; the LLM ones are opt-in (`--only`).
DETERMINISTIC = ["retrieval", "buffer", "consolidation", "graph", "lifecycle"]

DIMENSIONS = {
    "retrieval": run_retrieval,
    "buffer": run_buffer,
    "consolidation": run_consolidation,
    "graph": run_graph,
    "lifecycle": run_lifecycle,
    "llm_consolidation": run_llm_consolidation,
    "graph_capture": run_graph_capture,
}
