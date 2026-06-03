from __future__ import annotations

import re
from typing import Iterable

from harm_detection.config import INVALID_CATEGORY_LABELS, VALID_CATEGORY_CODES


WHITESPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def normalize_text(value: object, *, zero_is_empty: bool = True) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ").strip()
    if not text:
        return ""
    compact = WHITESPACE_RE.sub(" ", text)
    if zero_is_empty and compact.upper() in {"0", "NAN", "NONE", "NULL"}:
        return ""
    return compact


def text_present(*values: object) -> bool:
    return any(normalize_text(value) for value in values)


def parse_harm_labels(raw_value: object) -> list[str]:
    raw = normalize_text(raw_value, zero_is_empty=False).upper()
    if raw in INVALID_CATEGORY_LABELS:
        return []
    if not raw:
        return []
    tokens = [token.strip() for token in re.split(r"[|,;/]+", raw) if token.strip()]
    cleaned: list[str] = []
    for token in tokens:
        if token in VALID_CATEGORY_CODES and token not in cleaned:
            cleaned.append(token)
    return cleaned


def select_canonical_text(values: Iterable[object]) -> str:
    cleaned = [normalize_text(value) for value in values]
    cleaned = [value for value in cleaned if value]
    if not cleaned:
        return ""
    cleaned.sort(key=lambda item: (len(item), item), reverse=True)
    return cleaned[0]


def build_model_text(title: object, description: object, transcript: object) -> str:
    fields = [
        normalize_text(title),
        normalize_text(description),
        normalize_text(transcript),
    ]
    return "\n".join(field for field in fields if field)


def token_count(text: object) -> int:
    return len(TOKEN_RE.findall(normalize_text(text, zero_is_empty=False)))


def risk_band(probability: float) -> str:
    if probability < 0.35:
        return "low"
    if probability <= 0.65:
        return "medium"
    return "high"
