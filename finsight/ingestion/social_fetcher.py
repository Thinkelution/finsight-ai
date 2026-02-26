"""Fetch financial discussions from Reddit and StockTwits."""

import hashlib
import json
from datetime import datetime
from pathlib import Path

import httpx
import yaml

from finsight.config.logging import get_logger
from finsight.ingestion.deduplicator import Deduplicator

logger = get_logger(__name__)

SOURCES_PATH = Path(__file__).parent / "sources.yaml"

REDDIT_BASE = "https://www.reddit.com"
STOCKTWITS_BASE = "https://api.stocktwits.com/api/2"
USER_AGENT = "FinSight/1.0 (financial research bot)"


class SocialFetcher:
    def __init__(self, deduplicator: Deduplicator | None = None):
        self.dedup = deduplicator or Deduplicator()
        self._http = httpx.Client(
            timeout=30.0,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        self.config = self._load_config()

    def _load_config(self) -> dict:
        with open(SOURCES_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("social_feeds", {})

    def fetch_reddit(self) -> list[dict]:
        subreddits = self.config.get("reddit", {}).get("subreddits", [])
        articles = []

        for sub_cfg in subreddits:
            name = sub_cfg["name"]
            sort = sub_cfg.get("sort", "hot")
            limit = sub_cfg.get("limit", 25)

            url = f"{REDDIT_BASE}/r/{name}/{sort}.json?limit={limit}"
            logger.info("fetching_reddit", subreddit=name)

            try:
                resp = self._http.get(url)
                resp.raise_for_status()
                data = resp.json()

                for post in data.get("data", {}).get("children", []):
                    article = self._process_reddit_post(post["data"], name)
                    if article:
                        articles.append(article)
            except Exception as e:
                logger.error("reddit_fetch_failed", subreddit=name, error=str(e))

        logger.info("reddit_fetched", total=len(articles))
        return articles

    def _process_reddit_post(self, post: dict, subreddit: str) -> dict | None:
        title = post.get("title", "")
        selftext = post.get("selftext", "")
        text = f"{title}\n\n{selftext}".strip()

        if len(text) < 50:
            return None
        if post.get("score", 0) < 10:
            return None

        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if self.dedup.is_duplicate(content_hash):
            return None
        self.dedup.mark_seen(content_hash)

        created_utc = post.get("created_utc", 0)
        pub_dt = datetime.utcfromtimestamp(created_utc).isoformat() if created_utc else datetime.utcnow().isoformat()

        return {
            "id": content_hash,
            "url": f"{REDDIT_BASE}{post.get('permalink', '')}",
            "title": title[:200],
            "text": text[:5000],
            "source": f"reddit_r/{subreddit}",
            "source_type": "social",
            "asset_classes": self._infer_asset_classes(subreddit, text),
            "regions": ["us"],
            "published_at": pub_dt,
            "social_score": post.get("score", 0),
            "num_comments": post.get("num_comments", 0),
        }

    def fetch_stocktwits(self) -> list[dict]:
        st_cfg = self.config.get("stocktwits", {})
        symbols = st_cfg.get("symbols", [])
        articles = []

        for symbol in symbols:
            url = f"{STOCKTWITS_BASE}/streams/symbol/{symbol}.json"
            logger.info("fetching_stocktwits", symbol=symbol)

            try:
                resp = self._http.get(url)
                resp.raise_for_status()
                data = resp.json()

                for message in data.get("messages", []):
                    article = self._process_stocktwits_message(message, symbol)
                    if article:
                        articles.append(article)
            except Exception as e:
                logger.error("stocktwits_fetch_failed", symbol=symbol, error=str(e))

        if st_cfg.get("trending"):
            articles.extend(self._fetch_stocktwits_trending())

        logger.info("stocktwits_fetched", total=len(articles))
        return articles

    def _process_stocktwits_message(self, msg: dict, symbol: str) -> dict | None:
        body = msg.get("body", "")
        if len(body) < 20:
            return None

        content_hash = hashlib.sha256(body.encode()).hexdigest()
        if self.dedup.is_duplicate(content_hash):
            return None
        self.dedup.mark_seen(content_hash)

        return {
            "id": content_hash,
            "url": f"https://stocktwits.com/symbol/{symbol}",
            "title": f"StockTwits: ${symbol}",
            "text": body,
            "source": f"stocktwits_{symbol}",
            "source_type": "social",
            "asset_classes": ["equities"],
            "regions": ["us"],
            "published_at": msg.get("created_at", datetime.utcnow().isoformat()),
            "sentiment_label": msg.get("entities", {}).get("sentiment", {}).get("basic"),
        }

    def _fetch_stocktwits_trending(self) -> list[dict]:
        url = f"{STOCKTWITS_BASE}/trending/symbols.json"
        try:
            resp = self._http.get(url)
            resp.raise_for_status()
            data = resp.json()
            trending_symbols = [s["symbol"] for s in data.get("symbols", [])[:10]]
            logger.info("stocktwits_trending", symbols=trending_symbols)
            return []  # trending symbols used for awareness, not article content
        except Exception as e:
            logger.warning("stocktwits_trending_failed", error=str(e))
            return []

    @staticmethod
    def _infer_asset_classes(subreddit: str, text: str) -> list[str]:
        sub_map = {
            "wallstreetbets": ["equities"],
            "stocks": ["equities"],
            "investing": ["equities", "macro"],
            "forex": ["forex"],
            "economics": ["macro"],
        }
        return sub_map.get(subreddit, ["equities"])

    def fetch_all(self) -> list[dict]:
        articles = []
        articles.extend(self.fetch_reddit())
        articles.extend(self.fetch_stocktwits())
        return articles

    def close(self):
        self._http.close()
