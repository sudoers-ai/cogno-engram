"""
cogno_engram.types — the data carriers persisted/retrieved by the engram ports.

Every record is isolated by an opaque ``scope`` string. cogno-engram NEVER
interprets the scope — the host composes it (e.g. ``"tenant/phone"``) and owns
its meaning and any cross-scope aggregation. This is what decouples the
substrate from the host's business identity (no ``tenant_id``/``phone_id`` here).

The rich cognition objects (``IntentResult``/``IdResult``/``StageMetrics``) come
from the sibling ``cogno-anima`` library and are stored verbatim — engram treats
them as opaque payloads, so they are imported only under TYPE_CHECKING and the
core stays dependency-free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from cogno_anima.types import IntentResult, IdResult, StageMetrics


# Closed (but host-extensible) vocabulary of knowledge-graph node types.
VALID_NODE_TYPES: set[str] = {
    "PERSON", "ANIMAL", "PLACE", "ORG", "CONCEPT", "OBJECT", "EVENT",
}

# Confidence defaults per memory category — ported from the parent's Tier-1
# heuristics (a fact/PII is absolute; an inferred preference is soft).
DEFAULT_CONFIDENCE: dict[str, float] = {
    "fact": 1.0,
    "pii": 1.0,
    "goal_completed": 1.0,
    "goal_started": 0.9,
    "goal_changed": 0.9,
    "goal_abandoned": 0.85,
    "llm": 0.8,
    "preference": 0.7,
    "sentiment": 0.7,
}


@dataclass
class Session:
    id: str
    scope: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    summary: str = ""


@dataclass
class TurnRecord:
    session_id: str
    scope: str
    turn_n: int
    user_input: str
    response: str = ""
    # Host-captured feedback signal: -1 dislike / 0 neutral / +1 like.
    feedback: int = 0
    created_at: Optional[datetime] = None

    # Flat cognition signals the host maps from cogno-anima output. These let
    # the (LLM-free) Tier-1 micro consolidation run without engram importing
    # anima's internal types.
    goal: str = ""
    goal_status: str = ""          # NEW | ONGOING | COMPLETED | ABANDONED
    sentiment: str = ""
    domains: list[str] = field(default_factory=list)
    pii_types: list[str] = field(default_factory=list)

    # Full opaque cognition objects, persisted verbatim (engram never inspects).
    intent: Optional["IntentResult"] = None
    id_result: Optional["IdResult"] = None
    metrics: list["StageMetrics"] = field(default_factory=list)


@dataclass
class MemoryRecord:
    scope: str
    category: str
    content: str
    confidence: float = 1.0
    # Adjusted by host feedback (likes/dislikes); saturates at +/-10.
    feedback_score: float = 0.0
    embedding: Optional[list[float]] = None
    created_at: Optional[datetime] = None
    id: Optional[str] = None


@dataclass
class GraphNode:
    scope: str
    label: str
    node_type: str = "CONCEPT"
    attributes: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    id: Optional[int] = None


@dataclass
class GraphEdge:
    scope: str
    source: str                   # source node label
    target: str                   # target node label
    relation: str
    confidence: float = 1.0
    # The session that asserted this edge — the key for feedback-driven pruning.
    source_session: str = ""


@dataclass
class RetrievalQuery:
    text: str = ""                          # → lexical / BM25
    embedding: Optional[list[float]] = None  # → vector cosine (if backend supports)
    categories: Optional[list[str]] = None


@dataclass
class HybridWeights:
    """Linear fusion weights for hybrid retrieval (injectable, multi-tenant safe).

    Defaults mirror the parent: 0.60 vector + 0.40 lexical + 0.05 feedback.
    """
    vector: float = 0.60
    lexical: float = 0.40
    feedback: float = 0.05


@dataclass
class SessionSummary:
    session_id: str
    summary: str
    memories: list[MemoryRecord] = field(default_factory=list)


@dataclass
class NodeContext:
    """A node with its incident edges and neighbour nodes (one-hop context)."""
    node: GraphNode
    edges: list[GraphEdge] = field(default_factory=list)
    neighbors: list[GraphNode] = field(default_factory=list)
