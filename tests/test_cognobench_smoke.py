"""Smoke test guarding the EngramBench plumbing in CI (no DB, no model)."""


from cognobench.dimensions import (
    run_buffer,
    run_consolidation,
    run_graph,
    run_lifecycle,
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


def test_runner_main_returns_zero():
    from cognobench.runner import main
    assert main(["--only", "consolidation", "graph"]) == 0
