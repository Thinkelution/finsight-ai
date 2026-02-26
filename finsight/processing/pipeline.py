"""Orchestrates the full processing flow for articles."""

from finsight.config.logging import get_logger
from finsight.config.settings import settings
from finsight.processing.chunker import chunk_text
from finsight.processing.cleaner import clean_text
from finsight.processing.embedder import embed_chunks
from finsight.processing.ner import extract_entities
from finsight.processing.sentiment import score_sentiment

logger = get_logger(__name__)


class ProcessingPipeline:
    def process_article(self, article: dict) -> list[dict]:
        """Process a single article through the full pipeline.

        Returns a list of chunk payloads ready for indexing in Qdrant.
        """
        article_id = article.get("id", "unknown")
        logger.info("processing_article", article_id=article_id, source=article.get("source"))

        text = clean_text(article.get("text", ""))
        if not text or len(text) < 50:
            logger.warning("article_too_short", article_id=article_id)
            return []

        entities = extract_entities(text)
        sentiment = score_sentiment(text)

        chunks = chunk_text(
            text=text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

        if not chunks:
            return []

        chunk_texts = [c["text"] for c in chunks]
        try:
            embeddings = embed_chunks(chunk_texts)
        except Exception as e:
            logger.error("embedding_pipeline_failed", article_id=article_id, error=str(e))
            return []

        flat_entities = (
            entities.get("tickers", [])
            + entities.get("fx_pairs", [])
            + entities.get("companies", [])[:5]
        )
        geo_tags = entities.get("geopolitical", [])

        payloads = []
        for chunk, embedding in zip(chunks, embeddings):
            payloads.append(
                {
                    "text": chunk["text"],
                    "embedding": embedding,
                    "metadata": {
                        "article_id": article_id,
                        "source": article.get("source", ""),
                        "source_type": article.get("source_type", ""),
                        "url": article.get("url", ""),
                        "title": article.get("title", ""),
                        "published_at": article.get("published_at", ""),
                        "entities": flat_entities,
                        "geopolitical_tags": geo_tags,
                        "sentiment_score": sentiment.get("score", 0),
                        "sentiment_label": sentiment.get("label", "neutral"),
                        "asset_classes": article.get("asset_classes", []),
                        "regions": article.get("regions", []),
                        "chunk_index": chunk["chunk_index"],
                    },
                }
            )

        logger.info(
            "article_processed",
            article_id=article_id,
            chunks=len(payloads),
            sentiment=sentiment.get("label"),
        )
        return payloads

    def process_batch(self, articles: list[dict]) -> list[dict]:
        """Process a batch of articles."""
        all_payloads = []
        for article in articles:
            try:
                payloads = self.process_article(article)
                all_payloads.extend(payloads)
            except Exception as e:
                logger.error(
                    "article_processing_failed",
                    article_id=article.get("id", "unknown"),
                    error=str(e),
                )
        logger.info("batch_processed", articles=len(articles), chunks=len(all_payloads))
        return all_payloads
