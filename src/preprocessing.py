"""Deterministic statement cleaning shared by training and inference."""
from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?%?")


def clean_text(value: Any) -> str:
    """Normalize markup/URLs and punctuation while retaining stopwords and numbers."""
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value)).lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    return " ".join(TOKEN_PATTERN.findall(text))


def normalize_liar_label(value: Any) -> str | None:
    """Normalize a valid LIAR six-class label; return None for invalid values."""
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    return label if label in LIAR_LABELS else None


LIAR_LABELS = ("pants-fire", "false", "barely-true", "half-true", "mostly-true", "true")
BINARY_INTERPRETATION = {
    "pants-fire": "FAKE / LOW TRUTHFULNESS",
    "false": "FAKE / LOW TRUTHFULNESS",
    "barely-true": "FAKE / LOW TRUTHFULNESS",
    "half-true": "REAL / HIGHER TRUTHFULNESS",
    "mostly-true": "REAL / HIGHER TRUTHFULNESS",
    "true": "REAL / HIGHER TRUTHFULNESS",
}


def binary_label(liar_label: str) -> str:
    """Project-level grouping, not a claim that half-true statements are factual."""
    return "FAKE / LOW TRUTHFULNESS" if liar_label in LIAR_LABELS[:3] else "REAL / HIGHER TRUTHFULNESS"
