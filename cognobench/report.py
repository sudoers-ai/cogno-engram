from __future__ import annotations

from cognobench.types import DimensionResult

_BAR = 24


def _bar(pct: float) -> str:
    fill = round(_BAR * pct / 100)
    return "█" * fill + "·" * (_BAR - fill)


def render(dims: list[DimensionResult], *, show_failures: bool = True) -> str:
    lines = ["", "═" * 60, "  EngramBench (cogno-engram)", "═" * 60]
    total_p = sum(d.passed for d in dims)
    total_t = sum(d.total for d in dims)
    for d in dims:
        lines.append(f"  {d.name:<14} {_bar(d.pct)} {d.pct:5.1f}%  ({d.passed}/{d.total})")
    lines.append("-" * 60)
    overall = 100.0 * total_p / total_t if total_t else 0.0
    lines.append(f"  {'OVERALL':<14} {_bar(overall)} {overall:5.1f}%  ({total_p}/{total_t})")
    lines.append("═" * 60)
    if show_failures:
        for d in dims:
            if d.failures:
                lines.append(f"\n  ▼ {d.name} — {len(d.failures)} failed")
                for c in d.failures:
                    lines.append(f"    ✗ {c.case_id:<28} {c.name}  want={c.expected!r} got={c.actual!r}")
    return "\n".join(lines)
