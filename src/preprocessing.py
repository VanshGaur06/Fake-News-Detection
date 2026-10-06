"""Reusable text and label preprocessing for TruthLens."""
from __future__ import annotations

import html
import re
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML = re.compile(r"<[^>]+>")
_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")
_STOPWORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "against", "before", "after", "over", "under"}


def clean_text(value: Any, remove_stopwords: bool = True) -> str:
    """Normalize one text value consistently for training and prediction."""
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value)).lower()
    text = _URL.sub(" ", text)
    text = _HTML.sub(" ", text)
    words = _TOKEN.findall(text)
    if remove_stopwords:
        words = [word for word in words if word not in _STOPWORDS]
    return " ".join(words)


def normalize_label(value: Any) -> str | None:
    """Map common binary label spellings to FAKE or REAL."""
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().upper()
    mapping = {"FAKE": "FAKE", "FALSE": "FAKE", "F": "FAKE", "0": "FAKE",
               "REAL": "REAL", "TRUE": "REAL", "R": "REAL", "1": "REAL"}
    return mapping.get(label)


def combine_article(title: Any, text: Any) -> str:
    """Combine title and body while safely handling missing values."""
    return " ".join(part for part in (str(title).strip() if pd.notna(title) else "",
                                       str(text).strip() if pd.notna(text) else "") if part)
