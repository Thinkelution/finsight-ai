"""Collect historical events from Wikipedia's Current Events portal.

Wikipedia's Portal:Current_events provides curated, high-quality summaries
of world events organized by date. Coverage from late 1990s to present.
Free to access, no API key required.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/historical/news/wikipedia")

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "FinSightBot/1.0 (https://finsight.ai; vivek@finsight.ai) python-httpx/0.27",
}


def fetch_month_events(year: int, month: int) -> list[dict]:
    """Fetch current events for a specific month by fetching daily pages."""
    import calendar
    month_name = MONTH_NAMES[month - 1]
    _, days_in_month = calendar.monthrange(year, month)

    all_events = []
    with httpx.Client(timeout=30, headers=WIKI_HEADERS) as client:
        for day in range(1, days_in_month + 1):
            title = f"Portal:Current_events/{year}_{month_name}_{day}"
            params = {
                "action": "parse",
                "page": title,
                "prop": "wikitext",
                "format": "json",
            }
            try:
                resp = client.get(WIKI_API, params=params)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                if "error" in data:
                    continue
                wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
                date_str = f"{year}-{month:02d}-{day:02d}"
                events = _parse_daily_wikitext(wikitext, date_str)
                all_events.extend(events)
            except Exception:
                continue
            time.sleep(0.3)  # polite rate limit

    if not all_events:
        # Fallback: try the monthly overview page
        title = f"Portal:Current_events/{month_name}_{year}"
        params = {"action": "parse", "page": title, "prop": "wikitext", "format": "json"}
        try:
            with httpx.Client(timeout=30, headers=WIKI_HEADERS) as client:
                resp = client.get(WIKI_API, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" not in data:
                        wikitext = data.get("parse", {}).get("wikitext", {}).get("*", "")
                        all_events = _parse_wikitext_events(wikitext, year, month)
        except Exception as e:
            logger.error(f"Failed to fetch {month_name} {year}: {e}")

    logger.info(f"Wikipedia {month_name} {year}: {len(all_events)} events")
    return all_events


def _parse_daily_wikitext(wikitext: str, date_str: str) -> list[dict]:
    """Parse a single day's wikitext into events."""
    events = []
    current_categories = []

    for line in wikitext.split("\n"):
        line = line.strip()

        cat_match = re.match(r"^[;*]\s*'''(.+?)'''", line)
        if cat_match:
            current_categories = [cat_match.group(1).strip("[]")]
            continue

        if line.startswith("*") and not line.startswith("**"):
            text = _clean_wikitext(line.lstrip("* "))
            if len(text) > 20:
                categories = _categorize_event(text, current_categories)
                events.append({
                    "date": date_str,
                    "text": text,
                    "categories": categories,
                    "source": "wikipedia_current_events",
                })
        elif line.startswith("**"):
            text = _clean_wikitext(line.lstrip("* "))
            if len(text) > 20:
                categories = _categorize_event(text, current_categories)
                events.append({
                    "date": date_str,
                    "text": text,
                    "categories": categories,
                    "source": "wikipedia_current_events",
                })

    return events


def _parse_wikitext_events(wikitext: str, year: int, month: int) -> list[dict]:
    """Parse Wikipedia wikitext to extract events by date."""
    events = []
    current_date = None
    current_categories = []

    for line in wikitext.split("\n"):
        line = line.strip()

        date_match = re.match(
            r"^[=]{2,3}\s*\[\[(\w+ \d+)\]\]\s*[=]{2,3}",
            line,
        )
        if not date_match:
            date_match = re.match(
                r"^[=]{2,3}\s*(\w+ \d+)\s*[=]{2,3}",
                line,
            )
        if date_match:
            date_str = date_match.group(1)
            try:
                current_date = datetime.strptime(
                    f"{date_str} {year}", "%B %d %Y"
                ).strftime("%Y-%m-%d")
            except ValueError:
                pass
            continue

        cat_match = re.match(r"^[;*]\s*'''(.+?)'''", line)
        if cat_match:
            current_categories = [cat_match.group(1).strip("[]")]
            continue

        if line.startswith("*") and current_date:
            text = _clean_wikitext(line.lstrip("* "))
            if len(text) > 20:
                categories = _categorize_event(text, current_categories)
                events.append({
                    "date": current_date,
                    "text": text,
                    "categories": categories,
                    "source": "wikipedia_current_events",
                })

    return events


def _clean_wikitext(text: str) -> str:
    """Remove wiki markup from text."""
    text = re.sub(r"\[\[([^|\]]+\|)?([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"'''?(.+?)'''?", r"\1", text)
    text = re.sub(r"\{\{[^}]+\}\}", "", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", "", text)
    text = re.sub(r"<ref[^>]*/>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


FINANCE_KEYWORDS = {
    "market", "stock", "economy", "trade", "gdp", "inflation",
    "recession", "bank", "fed", "interest rate", "bond", "treasury",
    "currency", "dollar", "euro", "oil", "gold", "commodity",
    "earnings", "profit", "revenue", "ipo", "merger", "acquisition",
    "bankruptcy", "debt", "deficit", "fiscal", "monetary", "tariff",
    "sanction", "export", "import", "unemployment", "jobs",
}

GEO_KEYWORDS = {
    "war", "conflict", "military", "invasion", "bomb", "attack",
    "election", "president", "minister", "parliament", "government",
    "treaty", "summit", "united nations", "nato", "eu",
    "protest", "coup", "crisis", "refugee", "nuclear",
    "pandemic", "earthquake", "hurricane", "flood", "disaster",
}


def _categorize_event(text: str, wiki_categories: list[str]) -> list[str]:
    """Assign categories based on content and wiki section headers."""
    cats = set()
    text_lower = text.lower()

    if any(kw in text_lower for kw in FINANCE_KEYWORDS):
        cats.add("finance")
    if any(kw in text_lower for kw in GEO_KEYWORDS):
        cats.add("geopolitical")

    for wc in wiki_categories:
        wc_lower = wc.lower()
        if any(t in wc_lower for t in ["business", "econom", "financ"]):
            cats.add("finance")
        elif any(t in wc_lower for t in ["armed", "conflict", "politic", "law"]):
            cats.add("geopolitical")
        elif any(t in wc_lower for t in ["science", "technol"]):
            cats.add("technology")
        elif any(t in wc_lower for t in ["disaster", "environment"]):
            cats.add("environment")

    if not cats:
        cats.add("general")

    return sorted(cats)


def collect_range(
    start_date: str = "2016-01-01",
    end_date: str = "2026-02-26",
    output_dir: Path | None = None,
) -> int:
    """Collect Wikipedia events for a date range, month by month."""
    out = output_dir or DATA_DIR
    out.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    total = 0
    current_year = start.year
    current_month = start.month

    while datetime(current_year, current_month, 1) <= end:
        file_name = f"{current_year}_{current_month:02d}.jsonl"
        file_path = out / file_name

        if file_path.exists() and file_path.stat().st_size > 0:
            count = sum(1 for _ in open(file_path))
            logger.info(f"Skipping {MONTH_NAMES[current_month-1]} {current_year} ({count} events)")
            total += count
        else:
            events = fetch_month_events(current_year, current_month)
            with open(file_path, "w") as f:
                for ev in events:
                    f.write(json.dumps(ev) + "\n")
            logger.info(
                f"Wikipedia {MONTH_NAMES[current_month-1]} {current_year}: "
                f"{len(events)} events → {file_path}"
            )
            total += len(events)
            time.sleep(1)

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    logger.info(f"Wikipedia collection complete: {total} total events")
    return total


def load_month(year: int, month: int, data_dir: Path | None = None) -> list[dict]:
    """Load previously collected events for a specific month."""
    d = data_dir or DATA_DIR
    file_path = d / f"{year}_{month:02d}.jsonl"

    if not file_path.exists():
        return []

    events = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def load_date_range(
    start_date: str, end_date: str, data_dir: Path | None = None
) -> list[dict]:
    """Load events within a specific date range."""
    d = data_dir or DATA_DIR
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_events = []
    current_year = start.year
    current_month = start.month

    while datetime(current_year, current_month, 1) <= end:
        events = load_month(current_year, current_month, d)
        for ev in events:
            ev_date = datetime.strptime(ev["date"], "%Y-%m-%d")
            if start <= ev_date <= end:
                all_events.append(ev)

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1

    return sorted(all_events, key=lambda x: x["date"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    total = collect_range("2024-01-01", "2024-03-01")
    print(f"Collected {total} events")
