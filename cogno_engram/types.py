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
    """A typed node. ``node_type`` is CASE-NORMALISED here, at the boundary.

    **Why here and not in the helpers.** ``graph_context.ingest_entities`` and ``hypnos`` both
    upper-case the type before building a node, and both do it correctly — but they are two
    convenience paths, not the door. The DOCUMENTED way in is to build a ``GraphNode`` and call
    ``upsert_node``, which takes the value raw; and this is a PUBLIC library, so its callers are
    not only our own host. Normalising in a third helper would be a third copy of the rule.

    **Why it matters more than a style nit.** The Postgres unique index is
    ``(scope, engram_fold(label), node_type)``: the LABEL half is folded and the TYPE half is
    not. ``Rex/PERSON`` and ``Rex/person`` would be two rows for one thing — the same shape as
    ``José``/``Jose``, with half the work already done. **A half-folded identity is worse than a
    raw one, because it looks solved.**

    **PREVENTION, not repair, and that distinction is measured.** On 2026-08-27 the live box had
    ZERO rows outside upper case and ZERO pairs differing only in the type's case — verified with
    a POSITIVE CONTROL (the same query shape finds 10 groups of same-label/different-type, so it
    knows how to find things). There is nothing to migrate, which is exactly why this is cheap
    today: with one lower-case row already stored the answer would invert — normalise on WRITE
    only, and migrate first — because ``__post_init__`` also runs when the ADAPTERS build a node
    from a row, and an object that disagrees with its row makes read-modify-write create a
    SECOND row instead of updating the first.

    **Case only.** An unknown type is left alone: coercing it to ``CONCEPT`` here would rewrite a
    value on the way OUT of the database, which is a different (and lossy) decision from folding
    case. The write helpers already coerce against :data:`VALID_NODE_TYPES`; that is their job.
    """

    scope: str
    label: str
    node_type: str = "CONCEPT"
    attributes: dict = field(default_factory=dict)
    embedding: Optional[list[float]] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None      # set by the adapter, not the caller
    updated_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        # Total and never raising: a node whose type is unusable must still be a node — the
        # graph losing a row is worse than the row carrying an odd label.
        try:
            self.node_type = str(self.node_type or "CONCEPT").strip().upper() or "CONCEPT"
        except Exception:                                   # noqa: BLE001 — see above
            self.node_type = "CONCEPT"


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


# ── Audience: who may READ an edge ───────────────────────────────────────────
#
# "Tenant sees everything in the graph; an identity sees only its own life. They do not mix."
# — the product decision, 2026-08-25.
#
# **This lives on the EDGE, and that is forced, not preferred.** `knowledge_nodes` is unique on
# `(scope, lower(label), node_type)`, so the node "Maria" is ONE row for the whole tenant: two
# contacts who each mention a Maria share it. There is no "José's Maria" to mark. The EDGE
# `José --SPOUSE_OF--> Maria` is his; the node is just a label. Node visibility is DERIVED from
# the edges a reader may see.
AUDIENCE_UNCLASSIFIED = ""      # a writer that did not declare: staff sees it, no contact does
AUDIENCE_TENANT = "tenant"      # a business fact: staff and EVERY identity may read it
_AUDIENCE_IDENTITY_PREFIX = "identity:"

# Not a stored value — the argument a STAFF read passes. Distinct from anything `audience_for`
# can produce, so a stored row can never be mistaken for a staff request.
AUDIENCE_STAFF = "__staff__"


def audience_for(identity_id: object) -> str:
    """The audience value for one contact's own life, or ``""`` when there is no identity.

    The only producer of identity audiences. Callers never hand-write the string, so the prefix
    cannot drift and a raw identity id cannot be stored by accident.
    """
    text = str(identity_id or "").strip()
    return f"{_AUDIENCE_IDENTITY_PREFIX}{text}" if text else AUDIENCE_UNCLASSIFIED


def sanitize_audience(raw: object) -> str:
    """Any input → a storable audience. Pure, total, never raises.

    Anything unrecognisable becomes ``AUDIENCE_UNCLASSIFIED``, and that direction is the whole
    point: an unclassified edge is visible to STAFF and to NO contact. A writer that forgets, or
    garbles, costs a missing block — visible, annoying, safe — and never a leak. Two
    discriminators in this codebase defaulted the permissive way (`status` to `accepted`,
    `source_session` to empty) and both had to be undone after they had already spoken.
    """
    text = str(raw or "").strip()
    if text == AUDIENCE_TENANT:
        return AUDIENCE_TENANT
    if text.startswith(_AUDIENCE_IDENTITY_PREFIX) and text[len(_AUDIENCE_IDENTITY_PREFIX):].strip():
        return text
    return AUDIENCE_UNCLASSIFIED


def audience_can_read(audience: str, row_audience: str) -> bool:
    """May a reader with ``audience`` see a row stamped ``row_audience``? PURE — one rule, one
    place, so the two adapters cannot drift and a third one inherits it.
    """
    if audience == AUDIENCE_STAFF:
        return True                                   # tenant/staff read: everything
    row = sanitize_audience(row_audience)
    if row == AUDIENCE_TENANT:
        return True                                   # a business fact, for everyone
    return bool(row) and row == sanitize_audience(audience)


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
    # WHO MAY READ IT — see the block above. Default is unclassified: staff yes, contact no.
    audience: str = AUDIENCE_UNCLASSIFIED
    # WHEN the store first recorded it. `None` on an edge that has not been read back from a
    # store — a caller building one to WRITE cannot know it, and inventing a value here would
    # make "when did we learn this?" answerable with the moment somebody constructed an object.
    # The column has existed on `knowledge_edges` since the table did (`created_at timestamptz
    # NOT NULL DEFAULT now()`); it was written on every edge and then dropped on the way out,
    # because the dataclass had nowhere to put it. Surfacing it costs nothing and answers a
    # question the contact-graph view is built to ask.
    created_at: "Optional[datetime]" = None

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
        # Same reason as `status`: normalise in the TYPE, so the two adapters cannot disagree
        # about a non-canonical value and produce an edge that is invisible in one and visible
        # in the other. A garbled audience lands unclassified — staff-only — never public.
        self.audience = sanitize_audience(self.audience)
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


@dataclass
class GraphStats:
    """The dashboard's whole graph summary, from ONE aggregated read per shape.

    It exists because the caller that needed it was rebuilding it a node at a time: the host's
    ``knowledge_stats`` listed every node and then asked ``get_node_context`` for each — and that
    helper is itself ``find_node`` + ``walk`` + ``neighbors``, so the true cost was ``1 + 3N``
    (1165 queries for the 388 nodes of the live box, on every page open) and it grows with the
    graph. Nothing in the port could answer "how connected is each node" in bulk, so there was no
    other way to write it.

    The counts follow the SAME visibility rules the one-at-a-time version did, and that is the
    part worth stating: ``total_nodes``/``by_type`` count nodes the audience may see (for a
    non-staff reader that is DERIVED — a node is visible when some visible edge touches it), and
    degree counts DISTINCT ``(source, target, relation)`` accepted edges the audience may see,
    because an unreviewed edge is not walkable and the old code counted what ``walk`` returned.
    """
    total_nodes: int = 0
    total_edges: int = 0
    by_type: dict = field(default_factory=dict)
    top_connected: list = field(default_factory=list)   # [(GraphNode, degree)], most connected first


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
