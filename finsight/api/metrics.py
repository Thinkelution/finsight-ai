"""Prometheus metrics for FinSight API and pipeline monitoring."""

from prometheus_client import Counter, Gauge, Histogram, Info

app_info = Info("finsight", "FinSight AI application info")
app_info.info({"version": "1.0.0", "model": "qwen2.5-14b"})

# Query metrics
query_total = Counter(
    "finsight_queries_total",
    "Total number of queries processed",
    ["asset_class", "status"],
)
query_latency = Histogram(
    "finsight_query_latency_seconds",
    "Query processing latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
)
query_chunks_used = Histogram(
    "finsight_query_chunks_used",
    "Number of chunks used per query",
    buckets=[0, 2, 4, 6, 8, 10, 15, 20],
)

# Ingestion metrics
articles_ingested = Counter(
    "finsight_articles_ingested_total",
    "Total articles ingested",
    ["source_type"],  # rss, web_scrape, social
)
articles_deduplicated = Counter(
    "finsight_articles_deduplicated_total",
    "Articles skipped due to deduplication",
)
ingestion_errors = Counter(
    "finsight_ingestion_errors_total",
    "Ingestion errors by source type",
    ["source_type"],
)

# Processing metrics
chunks_processed = Counter(
    "finsight_chunks_processed_total",
    "Total chunks processed and embedded",
)
embedding_latency = Histogram(
    "finsight_embedding_latency_seconds",
    "Embedding generation latency per batch",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)
sentiment_distribution = Counter(
    "finsight_sentiment_total",
    "Sentiment distribution of processed articles",
    ["label"],  # positive, negative, neutral
)

# Storage metrics
qdrant_points = Gauge(
    "finsight_qdrant_points",
    "Number of points in Qdrant collection",
)
qdrant_query_latency = Histogram(
    "finsight_qdrant_query_latency_seconds",
    "Qdrant vector search latency",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Alert metrics
alerts_fired = Counter(
    "finsight_alerts_fired_total",
    "Total alerts triggered",
    ["alert_type"],  # price_spike, breaking_news, sentiment_shift, correlation
)

# Market data metrics
market_data_fetches = Counter(
    "finsight_market_data_fetches_total",
    "Total market data fetch operations",
    ["status"],  # success, error
)
market_data_latency = Histogram(
    "finsight_market_data_latency_seconds",
    "Market data fetch latency",
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0],
)

# System metrics
llm_inference_latency = Histogram(
    "finsight_llm_inference_latency_seconds",
    "LLM inference latency",
    buckets=[1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0],
)
summary_refresh_latency = Histogram(
    "finsight_summary_refresh_latency_seconds",
    "Market summary refresh latency",
    buckets=[5.0, 10.0, 30.0, 60.0, 120.0],
)
