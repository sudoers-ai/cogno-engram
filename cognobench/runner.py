from __future__ import annotations

import argparse
import asyncio
import sys

from cognobench.dimensions import DIMENSIONS
from cognobench.report import render
from cognobench.types import DimensionResult

ALL = list(DIMENSIONS)


async def _run(only: list[str], limit: int | None) -> list[DimensionResult]:
    names = only or ALL
    return [await DIMENSIONS[n](limit) for n in names]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="EngramBench — quality harness for cogno-engram")
    p.add_argument("--only", nargs="+", choices=ALL, default=[],
                   help="run only these dimensions")
    p.add_argument("--limit", type=int, default=None, help="cap cases per dimension")
    p.add_argument("--min-score", type=float, default=0.0,
                   help="exit non-zero if overall %% is below this")
    args = p.parse_args(argv)

    dims = asyncio.run(_run(args.only, args.limit))
    print(render(dims))

    total_p = sum(d.passed for d in dims)
    total_t = sum(d.total for d in dims)
    overall = 100.0 * total_p / total_t if total_t else 0.0
    return 1 if overall < args.min_score else 0


if __name__ == "__main__":
    sys.exit(main())
