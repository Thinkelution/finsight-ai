import hashlib
from datetime import datetime
from pathlib import Path

import feedparser
import httpx
import trafilatura
import yaml

from finsight.config.logging import get_logger
from finsight.ingestion.deduplicator import Deduplicator

logger = get_logger(__name__)

SOURCES_PATH = Path(__file__).parent / "sources.yaml"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class RSSFetcher:
    def __init__(self, deduplicator: Deduplicator | None = None):
        self.dedup = deduplicator or Deduplicator()
        self._http = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        self.feeds = self._load_feeds()

    def _load_feeds(self) -> list[dict]:
        with open(SOURCES_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("rss_feeds", [])

    def fetch_feed(self, feed_config: dict) -> list[dict]:
        url = feed_config["url"]
        logger.info("fetching_rss", feed=feed_config["name"], url=url)

        try:
            resp = self._http.get(url)
            parsed = feedparser.parse(resp.text)
        except Exception as e:
            logger.error("rss_fetch_failed", feed=feed_config["name"], error=str(e))
            return []

        articles = []
        for entry in parsed.entries:
            article = self._process_entry(entry, feed_config)
            if article:
                articles.append(article)

        logger.info("rss_fetched", feed=feed_config["name"], count=len(articles))
        return articles

    def _process_entry(self, entry, feed_config: dict) -> dict | None:
        url = entry.get("link", "")
        if not url:
            return None

        title = entry.get("title", "")
        summary = entry.get("summary", "")

        text = self._extract_full_text(url)
        if not text:
            text = summary
        if not text or len(text) < 100:
            return None

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if self.dedup.is_duplicate(content_hash):
            return None
        self.dedup.mark_seen(content_hash)

        published = entry.get("published_parsed")
        pub_dt = (
            datetime(*published[:6]).isoformat()
            if published
            else datetime.utcnow().isoformat()
        )

        return {
            "id": content_hash,
            "url": url,
            "title": title,
            "text": text,
            "source": feed_config["name"],
            "source_type": "rss",
            "asset_classes": feed_config.get("asset_classes", []),
            "regions": feed_config.get("regions", []),
            "published_at": pub_dt,
        }

    def _extract_full_text(self, url: str) -> str | None:
        try:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                return trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    favor_precision=True,
                )
        except Exception as e:
            logger.warning("article_extract_failed", url=url, error=str(e))
        return None

    def fetch_all(self) -> list[dict]:
        all_articles = []
        for feed in self.feeds:
            articles = self.fetch_feed(feed)
            all_articles.extend(articles)
        return all_articles

    def close(self):
        self._http.close()
