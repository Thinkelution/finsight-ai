"""Background ingestion and maintenance tasks."""

from finsight.config.logging import get_logger
from finsight.workers.celery_app import app

logger = get_logger(__name__)


@app.task(bind=True, max_retries=3, default_retry_delay=30)
def fetch_rss_feeds(self):
    """Poll all configured RSS feeds and process new articles."""
    try:
        from finsight.ingestion.rss_fetcher import RSSFetcher
        from finsight.processing.pipeline import ProcessingPipeline
        from finsight.storage.indexer import index_chunks

        fetcher = RSSFetcher()
        pipeline = ProcessingPipeline()

        articles = fetcher.fetch_all()
        logger.info("rss_task_fetched", articles=len(articles))

        if articles:
            payloads = pipeline.process_batch(articles)
            if payloads:
                indexed = index_chunks(payloads)
                logger.info("rss_task_indexed", chunks=indexed)

        fetcher.close()
        return {"articles": len(articles)}

    except Exception as e:
        logger.error("rss_task_failed", error=str(e))
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def scrape_web_news(self):
    """Scrape configured financial news websites."""
    try:
        from finsight.ingestion.web_scraper import WebScraper
        from finsight.processing.pipeline import ProcessingPipeline
        from finsight.storage.indexer import index_chunks

        scraper = WebScraper()
        pipeline = ProcessingPipeline()

        articles = scraper.scrape_all()
        logger.info("scrape_task_fetched", articles=len(articles))

        if articles:
            payloads = pipeline.process_batch(articles)
            if payloads:
                indexed = index_chunks(payloads)
                logger.info("scrape_task_indexed", chunks=indexed)

        scraper.close()
        return {"articles": len(articles)}

    except Exception as e:
        logger.error("scrape_task_failed", error=str(e))
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=2, default_retry_delay=60)
def fetch_social_feeds(self):
    """Fetch posts from Reddit and StockTwits."""
    try:
        from finsight.ingestion.social_fetcher import SocialFetcher
        from finsight.processing.pipeline import ProcessingPipeline
        from finsight.storage.indexer import index_chunks

        fetcher = SocialFetcher()
        pipeline = ProcessingPipeline()

        articles = fetcher.fetch_all()
        logger.info("social_task_fetched", articles=len(articles))

        if articles:
            payloads = pipeline.process_batch(articles)
            if payloads:
                indexed = index_chunks(payloads)
                logger.info("social_task_indexed", chunks=indexed)

        fetcher.close()
        return {"articles": len(articles)}

    except Exception as e:
        logger.error("social_task_failed", error=str(e))
        raise self.retry(exc=e)


@app.task(bind=True, max_retries=1)
def refresh_market_summary(self):
    """Regenerate the rolling 24h market summary."""
    try:
        from finsight.ingestion.market_data import MarketDataFetcher
        from finsight.processing.embedder import embed_text
        from finsight.storage.retriever import TimeWeightedRetriever
        from finsight.storage.summariser import MarketSummariser

        retriever = TimeWeightedRetriever()
        summariser = MarketSummariser()
        market = MarketDataFetcher()

        query_emb = embed_text("global market summary today major moves")
        recent = retriever.retrieve(query_embedding=query_emb, k=15, hours_back=24)

        recent_dicts = []
        for r in recent:
            recent_dicts.append({
                "text": r.payload.get("text", ""),
                "metadata": r.payload.get("metadata", {}),
            })

        live_prices = market.get_live_prices()
        summary = summariser.generate_summary(recent_dicts, live_prices)
        logger.info("summary_task_complete", summary_len=len(summary))
        return {"summary_length": len(summary)}

    except Exception as e:
        logger.error("summary_task_failed", error=str(e))
        raise self.retry(exc=e)


@app.task
def cleanup_expired_chunks():
    """Remove chunks older than NEWS_EXPIRY_DAYS from Qdrant."""
    try:
        from finsight.storage.indexer import delete_expired_chunks
        delete_expired_chunks()
        logger.info("cleanup_task_complete")
        return {"status": "cleaned"}
    except Exception as e:
        logger.error("cleanup_task_failed", error=str(e))
        return {"status": "error", "error": str(e)}


@app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_single_article(self, article: dict):
    """Process and index a single article (called from ingestion on-demand)."""
    try:
        from finsight.processing.pipeline import ProcessingPipeline
        from finsight.storage.indexer import index_chunks

        pipeline = ProcessingPipeline()
        payloads = pipeline.process_article(article)
        if payloads:
            index_chunks(payloads)
        return {"chunks": len(payloads)}

    except Exception as e:
        logger.error("process_article_task_failed", error=str(e))
        raise self.retry(exc=e)


@app.task
def check_price_alerts():
    """Check current prices against previous prices for alert-worthy moves."""
    try:
        from finsight.ingestion.market_data import MarketDataFetcher
        from finsight.inference.alerter import MarketAlerter

        market = MarketDataFetcher()
        alerter = MarketAlerter()

        prices = market.get_live_prices()
        changes = prices.get("changes", {})
        rates = prices.get("rates", {})

        alerts_fired = 0
        for symbol, pct in changes.items():
            if abs(pct) > alerter.threshold * 100:
                current = rates.get(symbol, 0)
                previous = current / (1 + pct / 100) if pct != -100 else 0
                alert = alerter.check_price_move(symbol, current, previous)
                if alert:
                    alerts_fired += 1

        if changes:
            alerter.check_cross_asset_correlation(changes)

        logger.info("price_alert_check_complete", alerts=alerts_fired)
        return {"alerts_fired": alerts_fired}

    except Exception as e:
        logger.error("price_alert_check_failed", error=str(e))
        return {"error": str(e)}
