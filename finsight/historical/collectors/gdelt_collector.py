"""Collect historical news from GDELT 2.0 API.

GDELT (Global Database of Events, Language, and Tone) provides
comprehensive global news coverage. Free, no API key required.
Coverage from 2015 onwards.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

QUERY_THEMES = [
    "economy OR market OR stocks OR inflation OR recession",
    "trade OR tariff OR sanctions OR embargo",
    "federal reserve OR central bank OR interest rate",
    "oil OR energy OR commodities OR gold",
    "technology OR AI OR semiconductor",
    "war OR conflict OR military OR geopolitical",
    "election OR policy OR regulation OR government",
    "pandemic OR health OR pharma",
    "climate OR environment OR renewable",
    "crypto OR bitcoin OR blockchain",
]

DATA_DIR = Path("data/historical/news/gdelt")


def fetch_articles(
    query: str,
    start_dt: str,
    end_dt: str,
    max_records: int = 250,
) -> list[dict]:
    """Fetch articles from GDELT for a date range.

    Args:
        query: Search query string
        start_dt: Start datetime as YYYYMMDDHHMMSS
        end_dt: End datetime as YYYYMMDDHHMMSS
        max_records: Maximum articles to return (max 250)
    """
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": min(max_records, 250),
        "startdatetime": start_dt,
        "enddatetime": end_dt,
        "format": "json",
        "sort": "hybridrel",
    }

    try:
        with httpx.Client(timeout=30) as client:
            resp = client.get(GDELT_DOC_API, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            logger.warning("GDELT rate limited, sleeping 60s")
            time.sleep(60)
            return fetch_articles(query, start_dt, end_dt, max_records)
        logger.error(f"GDELT HTTP error: {e}")
        return []
    except Exception as e:
        logger.error(f"GDELT fetch error: {e}")
        return []


def collect_week(
    week_start: datetime,
    output_dir: Path | None = None,
) -> list[dict]:
    """Collect all relevant news for a specific week."""
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    week_end = week_start + timedelta(days=7)
    start_str = week_start.strftime("%Y%m%d000000")
    end_str = week_end.strftime("%Y%m%d000000")

    all_articles = []
    seen_urls = set()

    for query in QUERY_THEMES:
        articles = fetch_articles(query, start_str, end_str)
        for art in articles:
            url = art.get("url", "")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            all_articles.append({
                "title": art.get("title", ""),
                "url": url,
                "source": art.get("domain", art.get("source", "")),
                "date": art.get("seendate", ""),
                "language": art.get("language", ""),
                "tone": art.get("tone", 0),
                "theme": query.split(" OR ")[0],
            })

        time.sleep(1)  # rate limiting

    file_name = f"{week_start.strftime('%Y_%m_%d')}.jsonl"
    file_path = out / file_name
    with open(file_path, "w") as f:
        for art in all_articles:
            f.write(json.dumps(art) + "\n")

    logger.info(
        f"GDELT week {week_start.date()}: {len(all_articles)} articles → {file_path}"
    )
    return all_articles


def collect_range(
    start_date: str = "2016-01-01",
    end_date: str = "2026-02-26",
    output_dir: Path | None = None,
) -> int:
    """Collect GDELT data for an entire date range, week by week."""
    out = output_dir or DATA_DIR
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total = 0
    current = start

    while current < end:
        file_name = f"{current.strftime('%Y_%m_%d')}.jsonl"
        file_path = out / file_name

        if file_path.exists() and file_path.stat().st_size > 0:
            logger.info(f"Skipping {current.date()} (already collected)")
            count = sum(1 for _ in open(file_path))
            total += count
            current += timedelta(days=7)
            continue

        articles = collect_week(current, out)
        total += len(articles)
        current += timedelta(days=7)

        time.sleep(2)  # be respectful to GDELT

    logger.info(f"GDELT collection complete: {total} total articles")
    return total


def load_week(week_start: datetime, data_dir: Path | None = None) -> list[dict]:
    """Load previously collected articles for a specific week."""
    d = data_dir or DATA_DIR
    file_name = f"{week_start.strftime('%Y_%m_%d')}.jsonl"
    file_path = d / file_name

    if not file_path.exists():
        return []

    articles = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    total = collect_range("2024-01-01", "2024-02-01")
    print(f"Collected {total} articles")
