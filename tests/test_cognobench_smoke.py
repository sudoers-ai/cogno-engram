"""Smoke test guarding the EngramBench plumbing in CI (no DB, no model)."""

import pytest

from cognobench.dimensions import (
    run_buffer,
    run_consolidation,
    run_graph,
    run_lifecycle,
    run_llm_consolidation,
    run_retrieval,
)


async def test_retrieval_dimension_runs_and_scores_well():
    dim = await run_retrieval()
    assert dim.total >= 6
    assert dim.pct >= 75.0, [f"{c.case_id}:{c.actual}" for c in dim.failures]


async def test_consolidation_dimension_passes():
    dim = await run_consolidation()
    assert dim.total >= 8
    assert dim.pct == 100.0, [c.case_id for c in dim.failures]


async def test_graph_dimension_passes():
    dim = await run_graph()
    assert dim.total >= 5
    assert dim.pct == 100.0, [c.case_id for c in dim.failures]


async def test_lifecycle_dimension_passes():
    dim = await run_lifecycle()
    assert dim.total >= 5
    assert dim.pct == 100.0, [f"{c.case_id}:{c.actual}" for c in dim.failures]


async def test_buffer_dimension_passes():
    dim = await run_buffer()
    assert dim.total >= 6
    assert dim.pct == 100.0, [c.case_id for c in dim.failures]


async def test_llm_consolidation_dimension_gated():
    # Opt-in / model-dependent: auto-skips (0 checks) when Ollama is unreachable.
    dim = await run_llm_consolidation()
    if dim.total == 0:
        pytest.skip("Ollama unavailable")
    assert dim.pct >= 50.0, [c.case_id for c in dim.failures]


def test_runner_main_returns_zero():
    from cognobench.runner import main
    assert main(["--only", "consolidation", "graph"]) == 0


def test_graph_capture_cases_well_formed():
    from cognobench.graph_capture_cases import CASES
    assert CASES
    for c in CASES:
        assert c.turns and c.expect_connected
        for start, target in c.expect_connected:
            # reachability is matched case-insensitively against node labels
            assert start == start.lower() and target == target.lower(), c.id


async def test_graph_capture_dimension_gated():
    # Opt-in / model-dependent: auto-skips (0 checks) when Ollama is unreachable.
    from cognobench.dimensions import run_graph_capture
    dim = await run_graph_capture()
    if dim.total == 0:
        pytest.skip("Ollama unavailable")
    hard = [c for c in dim.checks if "(soft)" not in c.name]
    assert all(c.passed for c in hard), [f"{c.case_id}:{c.name}" for c in hard if not c.passed]


def test_graph_html_renders_nodes_and_edges(tmp_path):
    from cognobench.graph_html import GraphSnapshot, render_graphs_html
    html = render_graphs_html([GraphSnapshot(
        title="t", nodes=[("José", "PERSON"), ("Rex", "ANIMAL")],
        edges=[("José", "Rex", "OWNS", 0.9)])])
    assert "José" in html and "OWNS" in html and "<svg" in html
    assert "http" not in html.split("xmlns")[0]      # self-contained — no external refs


def test_runner_graph_html_writes_file(tmp_path):
    from cognobench.runner import main
    out = tmp_path / "graphs.html"
    assert main(["--only", "graph", "--graph-html", str(out)]) == 0
    text = out.read_text()
    assert "<svg" in text and "graph · vet_owner_pet_breed" in text
