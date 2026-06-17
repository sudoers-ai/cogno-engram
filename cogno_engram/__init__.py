"""cogno-engram — persistence substrate for the Cogno cognitive pipeline."""

from cogno_engram import hypnos
from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore
from cogno_engram.ports import (
    ConversationBuffer,
    KnowledgeGraph,
    MemoryStore,
    SupportsVectorSearch,
)
from cogno_engram.types import (
    DEFAULT_CONFIDENCE,
    VALID_NODE_TYPES,
    GraphEdge,
    GraphNode,
    HybridWeights,
    MemoryRecord,
    RetrievalQuery,
    Session,
    SessionSummary,
    TurnRecord,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # ports
    "MemoryStore", "SupportsVectorSearch", "ConversationBuffer", "KnowledgeGraph",
    # types
    "Session", "TurnRecord", "MemoryRecord", "GraphNode", "GraphEdge",
    "RetrievalQuery", "HybridWeights", "SessionSummary",
    "DEFAULT_CONFIDENCE", "VALID_NODE_TYPES",
    # reference adapters
    "InMemoryStore", "InMemoryBuffer", "InMemoryGraph",
    # consolidation
    "hypnos",
]
