"""
cogno_engram.graph_context — knowledge-graph ingestion + prompt formatting helpers.

Two host-facing conveniences ported from the parent:

  * ``ingest_entities`` — the **turn-level (channel 1)** ingestion: the host feeds
    the NER entities of a turn and they become graph nodes in real time (the
    LLM batch relation extraction in ``hypnos`` is channel 2).
  * ``format_graph_context`` — the parent's ``KnowledgeRetriever``: turn a walk
    into a compact ``[Knowledge Graph]`` prompt block (budgeted), ready to inject
    into a cognition prompt.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Union

from cogno_engram.ports import KnowledgeGraph
from cogno_engram.types import EDGE_ACCEPTED, VALID_NODE_TYPES, GraphEdge, GraphNode

# An entity is either a bare label or a (label, node_type) pair.
Entity = Union[str, tuple[str, str]]


async def ingest_entities(
    kg: KnowledgeGraph,
    scope: str,
    entities: Iterable[Entity],
    *,
    embedder: Optional[Any] = None,
    attributes: Optional[dict] = None,
) -> int:
    """Channel-1 ingestion: upsert each NER entity as a graph node. Returns the count.

    ``embedder`` (duck-typed ``cogno-anima`` Embedder) optionally embeds the node
    label for later semantic node search. ``attributes`` is merged into every
    ingested node (upsert semantics: keys overwrite) — e.g. the host stamps
    ``{"identity_id": ...}`` so a shared-scope graph keeps per-node provenance.
    """
    count = 0
    for ent in entities:
        if isinstance(ent, tuple):
            label, ntype = ent[0], (ent[1] or "CONCEPT").upper()
        else:
            label, ntype = ent, "CONCEPT"
        if not label or not str(label).strip():
            continue
        if ntype not in VALID_NODE_TYPES:
            ntype = "CONCEPT"
        node = GraphNode(scope, str(label).strip(), ntype, attributes=dict(attributes or {}))
        if embedder is not None:
            node.embedding = (await embedder.embed(node.label)) or None
        await kg.upsert_node(node)
        count += 1
    return count


_MAX_DETAIL_CHARS = 120


def _detail(edge: GraphEdge) -> str:
    """The edge's ``attributes`` as a short parenthetical, or ``""``.

    This is what the attribute exists FOR, and it was missing: the value was stored, merged,
    migrated and round-tripped through both stores while `format_graph_context` emitted only
    ``source --[relation]--> target``. The type's docstring claimed the opposite effect and the
    README advertised it — prose asserting what the code did not do, found by a review grepping
    for a consumer and finding none.

    Sorted for a stable render, bounded per edge so one verbose note cannot eat the whole block,
    and newlines are flattened: a value arrives from a person typing into an admin field, and a
    line break inside a bullet turns one fact into what reads as two.
    """
    if not isinstance(edge.attributes, dict) or not edge.attributes:
        return ""
    parts = []
    for key in sorted(edge.attributes):
        value = " ".join(str(edge.attributes[key]).split())
        if value:
            parts.append(f"{key}: {value}")
    if not parts:
        return ""
    detail = "; ".join(parts)
    if len(detail) > _MAX_DETAIL_CHARS:
        detail = detail[:_MAX_DETAIL_CHARS - 1].rstrip() + "…"
    return f" ({detail})"


def format_graph_context(edges: list[GraphEdge], *, max_chars: int = 2000,
                         header: str = "[Knowledge Graph]") -> str:
    """Render edges as a compact prompt block, e.g.::

        [Knowledge Graph]
        - José --[OWNS]--> Rex
        - Rex --[BREED]--> Pastor Alemão

    Stops once ``max_chars`` is reached (a budget guard for prompt injection).
    Returns an empty string when there are no edges.
    """
    if not edges:
        return ""
    # Defence in depth. ``KnowledgeGraph.walk`` already returns accepted edges only, and this
    # repeats the check at the LAST point before the text becomes a prompt — because the cost
    # of the two disagreeing is asymmetric: a dropped edge is a missed kindness, a leaked one
    # is the agent stating an unreviewed claim about a person as fact. Callers that build an
    # edge list by hand (a test, an admin view, a future store) get the same guarantee without
    # having to know it exists.
    edges = [e for e in edges if getattr(e, "status", EDGE_ACCEPTED) == EDGE_ACCEPTED]
    if not edges:
        return ""
    lines = [header]
    used = len(header)
    for e in edges:
        line = f"- {e.source} --[{e.relation}]--> {e.target}{_detail(e)}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
