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
from cogno_engram.types import VALID_NODE_TYPES, GraphEdge, GraphNode

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
    lines = [header]
    used = len(header)
    for e in edges:
        line = f"- {e.source} --[{e.relation}]--> {e.target}"
        if used + len(line) + 1 > max_chars:
            break
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines)
