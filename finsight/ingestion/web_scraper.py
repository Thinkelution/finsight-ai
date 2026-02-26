"""Web scraper for financial news sites, press releases, and trending articles.

Uses httpx + trafilatura for standard sites, with playwright fallback
for JS-heavy pages (Bloomberg, etc.).
"""

import hashlib
import random
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

import httpx
import trafilatura
import yaml

from finsight.config.logging import get_logger
from finsight.ingestion.deduplicator import Deduplicator

logger = get_logger(__name__)

SOURCES_PATH = Path(__file__).parent / "sources.yaml"

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
]


class WebScraper:
    def __init__(self, deduplicator: Deduplicator | None = None):
        self.dedup = deduplicator or Deduplicator()
        self._http = httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": random.choice(USER_AGENTS)},
        )
        self._playwright_browser = None
        self.targets = self._load_targets()

    def _load_targets(self) -> list[dict]:
        with open(SOURCES_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("web_scrape_targets", [])

    def _get_browser(self):
        if self._playwright_browser is None:
            try:
                from playwright.sync_api import sync_playwright

                self._pw = sync_playwright().start()
                self._playwright_browser = self._pw.chromium.launch(headless=True)
            except Exception as e:
                logger.error("playwright_init_failed", error=str(e))
                return None
        return self._playwright_browser

    def scrape_target(self, target: dict) -> list[dict]:
        name = target["name"]
        logger.info("scraping_target", target=name)

        requires_browser = target.get("requires_browser", False)

        try:
            if requires_browser:
                links = self._extract_links_browser(target)
            else:
                links = self._extract_links_http(target)
        except Exception as e:
            logger.error("link_extraction_failed", target=name, error=str(e))
            return []

        articles = []
        for link in links[:15]:
            article = self._fetch_article(link, target)
            if article:
                articles.append(article)

        logger.info("scrape_complete", target=name, articles=len(articles))
        return articles

    def _extract_links_http(self, target: dict) -> list[str]:
        url = target["url"].replace("{today}", datetime.utcnow().strftime("%Y-%m-%d"))
        resp = self._http.get(url)
        resp.raise_for_status()

        from trafilatura import extract

        extracted = extract(resp.text, include_links=True, output_format="txt")
        if not extracted:
            return self._extract_links_from_html(resp.text, url)
        return self._extract_links_from_html(resp.text, url)

    def _extract_links_from_html(self, html: str, base_url: str) -> list[str]:
        """Parse anchor tags from raw HTML to find article links."""
        links = []
        import re

        for match in re.finditer(r'href=["\']([^"\']+)["\']', html):
            href = match.group(1)
            full_url = urljoin(base_url, href)
            if self._is_article_url(full_url):
                links.append(full_url)
        return list(dict.fromkeys(links))

    def _extract_links_browser(self, target: dict) -> list[str]:
        browser = self._get_browser()
        if not browser:
            return self._extract_links_http(target)

        page = browser.new_page()
        try:
            page.goto(target["url"], wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)

            selector = target.get("section_selector", "article a")
            elements = page.query_selector_all(selector)
            links = []
            for el in elements:
                href = el.get_attribute("href")
                if href:
                    full_url = urljoin(target["url"], href)
                    if self._is_article_url(full_url):
                        links.append(full_url)
            return list(dict.fromkeys(links))
        finally:
            page.close()

    @staticmethod
    def _is_article_url(url: str) -> bool:
        skip_patterns = [
            "/video/", "/podcast/", "/live/", "#", "javascript:",
            ".pdf", ".jpg", ".png", "/login", "/subscribe", "/signup",
            "/author/", "/tag/", "/category/",
        ]
        return (
            url.startswith("http")
            and len(url) > 30
            and not any(p in url.lower() for p in skip_patterns)
        )

    def _fetch_article(self, url: str, target: dict) -> dict | None:
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )
            if not text or len(text) < 100:
                return None

            content_hash = hashlib.sha256(text.encode()).hexdigest()
            if self.dedup.is_duplicate(content_hash):
                return None
            self.dedup.mark_seen(content_hash)

            title = trafilatura.extract(downloaded, output_format="xml")
            title_text = ""
            if title:
                import re
                m = re.search(r"<title>(.*?)</title>", title)
                if m:
                    title_text = m.group(1)

            return {
                "id": content_hash,
                "url": url,
                "title": title_text or url.split("/")[-1][:80],
                "text": text,
                "source": target["name"],
                "source_type": "web_scrape",
                "asset_classes": target.get("asset_classes", []),
                "regions": target.get("regions", []),
                "published_at": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.warning("article_fetch_failed", url=url, error=str(e))
            return None

    def scrape_all(self) -> list[dict]:
        all_articles = []
        for target in self.targets:
            articles = self.scrape_target(target)
            all_articles.extend(articles)
        return all_articles

    def close(self):
        self._http.close()
        if self._playwright_browser:
            self._playwright_browser.close()
            self._pw.stop()
