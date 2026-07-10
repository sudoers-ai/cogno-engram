"""Self-contained HTML rendering of the graphs the bench builds/captures (``--graph-html``).

One static file, no external assets: each graph is an inline SVG (circular layout — the bench
graphs are small), nodes coloured by type, edges labelled with relation + confidence. Lets a
human VERIFY what the capture process actually produced instead of trusting the pass/fail line.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field


@dataclass
class GraphSnapshot:
    """What one bench case put in the graph — collected by the dims when --graph-html is on."""
    title: str
    nodes: list[tuple[str, str]] = field(default_factory=list)       # (label, node_type)
    edges: list[tuple[str, str, str, float]] = field(default_factory=list)
    #        (source, target, relation, confidence)
    note: str = ""

_TYPE_COLOURS = {
    "PERSON": "#4c9ee8", "ANIMAL": "#e8a04c", "PLACE": "#5cb85c", "ORG": "#b06ae0",
    "CONCEPT": "#8a8f98", "OBJECT": "#d9b23c", "EVENT": "#e05c7a",
}


def _svg(snap: GraphSnapshot, *, size: int = 420) -> str:
    n = max(len(snap.nodes), 1)
    cx = cy = size / 2
    radius = size / 2 - 70
    pos: dict[str, tuple[float, float]] = {}
    for i, (label, _) in enumerate(snap.nodes):
        ang = 2 * math.pi * i / n - math.pi / 2
        pos[label.lower()] = (cx + radius * math.cos(ang), cy + radius * math.sin(ang))

    parts = [f'<svg viewBox="0 0 {size} {size}" width="{size}" height="{size}" '
             f'xmlns="http://www.w3.org/2000/svg">']
    for src, tgt, rel, conf in snap.edges:
        a, b = pos.get(src.lower()), pos.get(tgt.lower())
        if not a or not b:
            continue
        mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        parts.append(f'<line x1="{a[0]:.0f}" y1="{a[1]:.0f}" x2="{b[0]:.0f}" y2="{b[1]:.0f}" '
                     f'stroke="#999" stroke-width="1.5" marker-end="url(#arr)"/>')
        parts.append(f'<text x="{mx:.0f}" y="{my - 4:.0f}" font-size="10" fill="#666" '
                     f'text-anchor="middle">{html.escape(rel)} ({conf:.1f})</text>')
    for label, ntype in snap.nodes:
        x, y = pos[label.lower()]
        colour = _TYPE_COLOURS.get(ntype.upper(), "#8a8f98")
        parts.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="18" fill="{colour}" opacity="0.85"/>')
        parts.append(f'<text x="{x:.0f}" y="{y + 32:.0f}" font-size="11" text-anchor="middle" '
                     f'fill="currentColor">{html.escape(label)}</text>')
    parts.append('<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="24" refY="4" '
                 'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="#999"/></marker></defs>')
    parts.append("</svg>")
    return "".join(parts)


def render_graphs_html(snaps: list[GraphSnapshot]) -> str:
    """The whole report: one card per graph, a legend, zero external requests."""
    legend = " ".join(
        f'<span style="display:inline-block;margin-right:1em">'
        f'<span style="display:inline-block;width:10px;height:10px;border-radius:50%;'
        f'background:{c};margin-right:4px"></span>{t}</span>'
        for t, c in _TYPE_COLOURS.items())
    cards = []
    for s in snaps:
        note = f'<p style="color:#888;font-size:0.85em">{html.escape(s.note)}</p>' if s.note else ""
        empty = ('<p style="color:#c66">no edges captured</p>' if not s.edges else "")
        cards.append(
            f'<div style="border:1px solid #ccc3;border-radius:8px;padding:1em;margin:1em 0">'
            f"<h3>{html.escape(s.title)}</h3>{note}{empty}"
            f'<div style="overflow-x:auto">{_svg(s)}</div>'
            f"<p style='font-size:0.85em'>{len(s.nodes)} nodes · {len(s.edges)} edges</p></div>")
    return (
        "<title>EngramBench — graphs</title>"
        "<h1>EngramBench — captured/built graphs</h1>"
        f"<p>{legend}</p>" + "\n".join(cards))
