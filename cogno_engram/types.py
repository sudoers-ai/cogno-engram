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
class TurnTrace:
    """Per-turn pipeline trace, persisted in its OWN table (``turn_traces``), keyed to a
    turn by ``(scope, session_id, turn_n)``. ``trace`` is an opaque dict the host composes
    (e.g. cogno-host's ``build_turn_trace``: the NOUMENO/NER/ID/EGO/Drift signals + the
    Aristotelian decomposition) — engram never interprets it. Feeds the audit/inspector
    views without bloating the flat ``turns`` record."""
    session_id: str
    scope: str
    turn_n: int
    trace: dict = field(default_factory=dict)
    created_at: Optional[datetime] = None


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
    created_at: Optional[datetime] = None      # set by the adapter, not the caller
    updated_at: Optional[datetime] = None


# ── Edge curation ──────────────────────────────────────────────────────────
#
# Who ASSERTED an edge decides whether it may be spoken. A host (or a staff member typing into
# an admin page) asserts; an LLM extraction PROPOSES. The difference matters because a graph
# edge becomes a sentence the agent says about a person as if it knew — "your son Pedro" is
# either a kindness or an invention, and nothing downstream can tell which.
EDGE_ACCEPTED = "accepted"
EDGE_PROPOSED = "proposed"
EDGE_REJECTED = "rejected"
VALID_EDGE_STATUS: frozenset[str] = frozenset({EDGE_ACCEPTED, EDGE_PROPOSED, EDGE_REJECTED})

# Relations that describe a PERSON's close world, as opposed to the open-vocabulary relations
# an extraction invents about a domain. Closed so a curation UI can label them, so a walk can
# ask for them by name, and so "has a kid" / "is father of" / "PARENT" do not become three
# facts about one child. Not enforced on write — an open relation is still a valid edge; this
# is the set the proximity feature knows how to render and review.
VALID_PROXIMITY_RELATIONS: frozenset[str] = frozenset({
    "PARENT_OF", "CHILD_OF", "SPOUSE_OF", "SIBLING_OF", "FRIEND_OF",
    "PET_OF", "OWNS_PET", "NICKNAME_OF", "SUPPORTS", "WORKS_AT",
    "STUDIES_AT", "LIVES_IN", "BORN_IN", "CELEBRATES", "PREFERS",
})


def sanitize_edge_status(raw: object) -> str:
    """Any input → a valid status. PURE, never raises. **Absent and garbled are not the same.**

    * absent (``None``/``""``) → ``accepted``. That is the back-compatible reading: an edge
      written before this field existed, or by a caller that says nothing, keeps reaching the
      prompt exactly as it did.
    * a valid status → itself.
    * anything else → ``proposed``, and this is the half a review had to point out. Folding a
      typo into ``accepted`` **inverts the caller's intent in the one direction the feature
      exists to prevent**: ``status="propsed"`` means someone meant to hold the edge for review,
      and it walked straight into the prompt as an assertion instead. An unreadable intent is
      held, not asserted — the cost of guessing wrong that way is a queue entry.
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return EDGE_ACCEPTED
    try:
        text = str(raw).strip().lower()
    except Exception:                       # noqa: BLE001 — a hostile __str__ is data
        return EDGE_PROPOSED
    return text if text in VALID_EDGE_STATUS else EDGE_PROPOSED


@dataclass
class GraphEdge:
    scope: str
    source: str                   # source node label
    target: str                   # target node label
    relation: str
    confidence: float = 1.0
    # The session that asserted this edge — the key for feedback-driven pruning.
    source_session: str = ""
    # Free-form detail the relation alone cannot carry: "Pedro" is a CHILD_OF edge, but
    # ``{"age": 8, "note": "joga futebol no sábado"}`` is what makes the agent sound like it
    # remembers rather than like it queried a database.
    attributes: dict = field(default_factory=dict)
    # Curation state. ``accepted`` edges are the only ones a walk returns — see the port.
    status: str = EDGE_ACCEPTED

    def __post_init__(self) -> None:
        """One normalisation, both stores.

        It used to live on the Postgres write path only, so the two adapters disagreed on any
        non-canonical value: ``status="Accepted"`` walked fine out of Postgres and, in memory,
        was invisible to ``walk()`` **and** absent from ``pending_edges()`` — an edge in a black
        hole, un-renderable and un-reviewable, with no error anywhere. Normalising in the type
        means neither store can be the one that gets it right.

        ``attributes`` is coerced here for the same reason: ``None`` is accepted in memory and
        becomes ``'null'::jsonb`` in Postgres, where the NEXT upsert's ``||`` merge fails on
        concatenating an object with a scalar — a divergence that only surfaces in production.
        """
        self.status = sanitize_edge_status(self.status)
        if not isinstance(self.attributes, dict):
            self.attributes = {}


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


def require_edge_status(raw: object) -> str:
    """A VERDICT, validated — raises on anything that is not one. Not the same question as
    :func:`sanitize_edge_status`, and a review had to point out that reusing it was wrong.

    ``sanitize_edge_status`` answers a STORAGE question: an edge arriving with no status was
    written before the field existed, so ``accepted`` is the back-compatible reading. Here there
    is no legacy caller: ``set_edge_status`` exists only to record a decision a human made, and
    "absent" means **no decision was submitted**.

    Borrowing the storage default inverted the safety: a TYPO'd verdict (``"acepted"``) was
    safely held as ``proposed``, while a MISSING one published the unreviewed edge — and the
    call returned ``True``, so the curation UI showed success. Measured in both adapters.
    """
    text = str(raw).strip().lower() if raw is not None else ""
    if text not in VALID_EDGE_STATUS:
        raise ValueError(
            f"not a verdict: {raw!r}. Expected one of {sorted(VALID_EDGE_STATUS)}. "
            f"A missing or unreadable verdict is a caller bug, and defaulting it would publish "
            f"an unreviewed edge — the one thing this field exists to prevent.")
    return text
