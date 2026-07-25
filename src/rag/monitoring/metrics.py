from prometheus_client import Counter, Histogram, Gauge

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "End-to-end request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Qdrant hybrid search latency",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

EMBED_LATENCY = Histogram(
    "rag_embed_duration_seconds",
    "Embedding latency",
    buckets=[0.01, 0.05, 0.1, 0.3],
)

DOCS_INGESTED = Counter(
    "rag_docs_ingested_total",
    "Total documents ingested",
)

QUERY_ERRORS = Counter(
    "rag_query_errors_total",
    "Total query errors",
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Currently active query requests",
)
