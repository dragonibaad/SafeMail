from __future__ import annotations

import re

URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
URGENCY_TERMS = {
    "urgent",
    "immediately",
    "verify",
    "account",
    "password",
    "click",
    "suspended",
    "limited",
    "security",
    "confirm",
}


def extract_suspicious_indicators(email_text: str) -> dict[str, list[str]]:
    urls = URL_REGEX.findall(email_text)
    words = re.findall(r"[A-Za-z']+", email_text.lower())
    matched_terms = sorted({w for w in words if w in URGENCY_TERMS})
    return {
        "urls": urls,
        "urgency_terms": matched_terms,
    }


def highlight_text(email_text: str) -> str:
    highlighted = URL_REGEX.sub(lambda m: f"<mark>{m.group(0)}</mark>", email_text)
    for term in sorted(URGENCY_TERMS, key=len, reverse=True):
        highlighted = re.sub(
            rf"\b({re.escape(term)})\b",
            r"<mark>\1</mark>",
            highlighted,
            flags=re.IGNORECASE,
        )
    return highlighted
