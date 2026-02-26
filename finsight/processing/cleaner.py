"""Clean and normalise raw article text."""

import re
import html


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)

    text = re.sub(r"<[^>]+>", " ", text)

    text = re.sub(r"https?://\S+", "", text)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)

    boilerplate = [
        r"(?i)subscribe to.*newsletter",
        r"(?i)sign up for.*alerts",
        r"(?i)click here to",
        r"(?i)read more at",
        r"(?i)follow us on",
        r"(?i)share this article",
        r"(?i)copyright \d{4}",
        r"(?i)all rights reserved",
        r"(?i)terms of (use|service)",
        r"(?i)privacy policy",
        r"(?i)cookie (policy|settings)",
    ]
    for pattern in boilerplate:
        text = re.sub(pattern + r"[^\n]*", "", text)

    text = text.strip()
    return text


def extract_headline(text: str) -> str:
    """Extract first sentence as headline if no title provided."""
    if not text:
        return ""
    first_line = text.split("\n")[0].strip()
    if len(first_line) > 200:
        first_line = first_line[:200] + "..."
    return first_line
