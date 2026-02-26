"""Named entity extraction for financial text using spaCy + custom patterns."""

import re
from functools import lru_cache

from finsight.config.logging import get_logger

logger = get_logger(__name__)

TICKER_PATTERN = re.compile(r"\b[A-Z]{1,5}\b")
FX_PAIR_PATTERN = re.compile(
    r"\b(EUR|USD|GBP|JPY|AUD|CAD|CHF|NZD|CNY|HKD|SGD|NOK|SEK|DKK|ZAR|TRY|MXN|BRL|INR)"
    r"[/]?"
    r"(EUR|USD|GBP|JPY|AUD|CAD|CHF|NZD|CNY|HKD|SGD|NOK|SEK|DKK|ZAR|TRY|MXN|BRL|INR)\b"
)

COMMON_WORDS = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "ANY", "CAN",
    "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW", "ITS",
    "MAY", "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "DID", "GET", "HIM",
    "LET", "SAY", "SHE", "TOO", "USE", "DAY", "BIG", "FED", "GDP", "CPI",
    "IMF", "ECB", "BOJ", "BOE", "RBA", "SNB", "PMI", "ISM", "NFP", "CEO",
    "CFO", "IPO", "ETF", "NYSE", "SEC", "PER", "EPS", "YOY", "QOQ", "MOM",
    "ATH", "ATL", "WITH", "THAT", "THIS", "WILL", "YOUR", "FROM", "THEY",
    "BEEN", "HAVE", "MUCH", "SOME", "THAN", "THEM", "THEN", "WHAT", "WHEN",
    "ALSO", "BACK", "BEEN", "COME", "EACH", "EVEN", "FIND", "FIRST", "HERE",
    "INTO", "JUST", "KNOW", "LAST", "LIKE", "LONG", "LOOK", "MADE", "MAKE",
    "MANY", "MORE", "MOST", "MUST", "NEXT", "ONLY", "OVER", "SAID", "SAME",
    "TAKE", "TELL", "WELL", "VERY",
}

KNOWN_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "NVDA", "META", "TSLA",
    "BRK", "JPM", "JNJ", "V", "UNH", "MA", "PG", "HD", "DIS", "BAC",
    "XOM", "PFE", "CSCO", "ADBE", "NFLX", "CRM", "INTC", "AMD", "QCOM",
    "GS", "MS", "WFC", "C", "AXP", "BLK", "SCHW", "SPY", "QQQ", "IWM",
    "GLD", "SLV", "USO", "VIX",
}

GEOPOLITICAL_KEYWORDS = {
    "tariff", "sanctions", "trade war", "embargo", "nato", "opec",
    "fed", "ecb", "boj", "interest rate", "inflation", "recession",
    "gdp", "stimulus", "fiscal", "monetary policy", "debt ceiling",
    "government shutdown", "election", "regulation", "antitrust",
    "nuclear", "war", "conflict", "ceasefire", "treaty", "summit",
    "supreme court", "executive order", "infrastructure bill",
}


@lru_cache(maxsize=1)
def _load_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        logger.warning("spacy_unavailable_using_regex_ner")
        return None


def extract_entities(text: str) -> dict:
    """Extract financial entities: tickers, FX pairs, companies, people, orgs."""
    result = {
        "tickers": [],
        "fx_pairs": [],
        "companies": [],
        "people": [],
        "organizations": [],
    }

    fx_matches = FX_PAIR_PATTERN.findall(text)
    result["fx_pairs"] = list({f"{a}/{b}" for a, b in fx_matches})

    ticker_matches = TICKER_PATTERN.findall(text)
    valid_tickers = [
        t for t in ticker_matches
        if t in KNOWN_TICKERS or (len(t) >= 2 and t not in COMMON_WORDS)
    ]
    result["tickers"] = list(set(valid_tickers))[:20]

    text_lower = text.lower()
    geo_tags = [kw for kw in GEOPOLITICAL_KEYWORDS if kw in text_lower]
    result["geopolitical"] = geo_tags[:10]

    nlp = _load_spacy()
    if nlp:
        doc = nlp(text[:10000])
        for ent in doc.ents:
            if ent.label_ == "ORG":
                result["organizations"].append(ent.text)
            elif ent.label_ == "PERSON":
                result["people"].append(ent.text)

        result["companies"] = list(set(result["organizations"]))[:20]
        result["people"] = list(set(result["people"]))[:10]
        result["organizations"] = list(set(result["organizations"]))[:20]

    return result
