"""cogno-engram — persistence substrate for the Cogno cognitive pipeline."""

from cogno_engram import chunking, documents, hypnos, ingest, maintenance, write_loss
from cogno_engram.adapters.in_memory import (
    InMemoryBuffer,
    InMemoryDocumentStore,
    InMemoryGraph,
    InMemoryStore,
)
from cogno_engram.graph_context import format_graph_context, ingest_entities
from cogno_engram.reranker import RerankConfig, recency_score, rerank
from cogno_engram.ports import (
    ConversationBuffer,
    DocumentStore,
    KnowledgeGraph,
    MemoryStore,
    SupportsVectorSearch,
)
from cogno_engram.documents import (
    KB_EMBED_SPACE_UNAVAILABLE,
    KbChunk,
    KbDocument,
    KbHit,
    KbSearchResult,
    KbTombstone,
    KbVersion,
    TextExtractor,
    embed_model_label,
)
from cogno_engram.ingest import IngestOutcome, documents_probe
from cogno_engram.ingest import ingest as ingest_document
from cogno_engram.types import (
    DEFAULT_CONFIDENCE,
    AUDIENCE_STAFF,
    AUDIENCE_TENANT,
    AUDIENCE_UNCLASSIFIED,
    EDGE_ACCEPTED,
    EDGE_PROPOSED,
    EDGE_REJECTED,
    GraphEdge,
    GraphNode,
    HybridWeights,
    MemoryRecord,
    NodeContext,
    GraphStats,
    RetrievalQuery,
    Session,
    SessionSummary,
    TurnRecord,
    TurnTrace,
    VALID_EDGE_STATUS,
    VALID_NODE_TYPES,
    VALID_PROXIMITY_RELATIONS,
    require_edge_status,
    audience_can_read,
    audience_for,
    sanitize_audience,
    sanitize_edge_status,
)

__version__ = "0.1.0"

__all__ = [
    # write losses this library counts instead of swallowing (see write_loss.py)
    "write_loss",
    "AUDIENCE_STAFF",
    "AUDIENCE_TENANT",
    "AUDIENCE_UNCLASSIFIED",
    "audience_can_read",
    "audience_for",
    "sanitize_audience",
    # edge curation (see types.VALID_EDGE_STATUS)
    "EDGE_ACCEPTED", "EDGE_PROPOSED", "EDGE_REJECTED", "VALID_EDGE_STATUS",
    "VALID_PROXIMITY_RELATIONS", "sanitize_edge_status", "require_edge_status",
    "__version__",
    # ports
    "MemoryStore", "SupportsVectorSearch", "ConversationBuffer", "KnowledgeGraph",
    "DocumentStore",
    # documents (see cogno_engram.documents / .chunking / .ingest)
    "documents", "chunking", "ingest", "ingest_document", "documents_probe", "IngestOutcome",
    "KbDocument", "KbVersion", "KbChunk", "KbHit", "KbSearchResult", "KbTombstone",
    "TextExtractor", "embed_model_label", "KB_EMBED_SPACE_UNAVAILABLE",
    # types
    "Session", "TurnRecord", "TurnTrace", "MemoryRecord", "GraphNode", "GraphEdge", "NodeContext", "GraphStats",
    "RetrievalQuery", "HybridWeights", "SessionSummary",
    "DEFAULT_CONFIDENCE", "VALID_NODE_TYPES",
    # reference adapters
    "InMemoryStore", "InMemoryBuffer", "InMemoryGraph", "InMemoryDocumentStore",
    # consolidation + maintenance + reranking + graph helpers
    "hypnos", "maintenance", "rerank", "RerankConfig", "recency_score",
    "ingest_entities", "format_graph_context",
]
