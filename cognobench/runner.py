from __future__ import annotations

import argparse
import asyncio
import inspect
import sys

from cognobench.dimensions import DETERMINISTIC, DIMENSIONS
from cognobench.graph_html import GraphSnapshot, render_graphs_html
from cognobench.report import render
from cognobench.types import DimensionResult

ALL = list(DIMENSIONS)


async def _run(only: list[str], limit: int | None, *, model: str | None = None,
               viz: list[GraphSnapshot] | None = None) -> list[DimensionResult]:
    names = only or DETERMINISTIC   # default: deterministic dims (LLM ones are opt-in)
    out = []
    for n in names:
        fn = DIMENSIONS[n]
        params = inspect.signature(fn).parameters
        kwargs: dict = {}
        if viz is not None and "viz" in params:
            kwargs["viz"] = viz
        if model and "model" in params:
            kwargs["model"] = model
        out.append(await fn(limit, **kwargs))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EngramBench — quality harness for cogno-engram")
    p.add_argument("--only", nargs="+", choices=ALL, default=[],
                   help="run only these dimensions (incl. opt-in 'llm_consolidation'/'graph_capture')")
    p.add_argument("--limit", type=int, default=None, help="cap cases per dimension")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="exit non-zero if overall %% is below this")
    p.add_argument("--model", default=None,
                   help="Ollama model for the LLM dimensions (default mistral:latest)")
    p.add_argument("--graph-html", default=None, metavar="PATH",
                   help="write a self-contained HTML visualization of every graph the "
                        "bench built/captured (graph + graph_capture dimensions)")
    args = p.parse_args(argv)

    viz: list[GraphSnapshot] | None = [] if args.graph_html else None
    dims = asyncio.run(_run(args.only, args.limit, model=args.model, viz=viz))
    print(render(dims))

    if args.graph_html and viz is not None:
        with open(args.graph_html, "w", encoding="utf-8") as f:
            f.write(render_graphs_html(viz))
        print(f"  graphs → {args.graph_html} ({len(viz)} graphs)")

    total_p = sum(d.passed for d in dims)
    total_t = sum(d.total for d in dims)
    overall = 100.0 * total_p / total_t if total_t else 0.0
    return 1 if overall < args.min_score else 0


if __name__ == "__main__":
    sys.exit(main())
