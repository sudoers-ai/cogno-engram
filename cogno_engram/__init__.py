"""cogno-engram — persistence substrate for the Cogno cognitive pipeline."""

from cogno_engram import hypnos, maintenance
from cogno_engram.adapters.in_memory import InMemoryBuffer, InMemoryGraph, InMemoryStore
from cogno_engram.graph_context import format_graph_context, ingest_entities
from cogno_engram.reranker import RerankConfig, recency_score, rerank
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
    NodeContext,
    RetrievalQuery,
    Session,
    SessionSummary,
    TurnRecord,
    TurnTrace,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # ports
    "MemoryStore", "SupportsVectorSearch", "ConversationBuffer", "KnowledgeGraph",
    # types
    "Session", "TurnRecord", "TurnTrace", "MemoryRecord", "GraphNode", "GraphEdge", "NodeContext",
    "RetrievalQuery", "HybridWeights", "SessionSummary",
    "DEFAULT_CONFIDENCE", "VALID_NODE_TYPES",
    # reference adapters
    "InMemoryStore", "InMemoryBuffer", "InMemoryGraph",
    # consolidation + maintenance + reranking + graph helpers
    "hypnos", "maintenance", "rerank", "RerankConfig", "recency_score",
    "ingest_entities", "format_graph_context",
]
