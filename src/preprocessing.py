"""Text normalization shared by WELFake training and article inference."""
from __future__ import annotations

import html
import re
from typing import Any
import pandas as pd

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]*>")
SPACE_PATTERN = re.compile(r"\s+")
BINARY_LABELS = ("FAKE", "REAL")


def clean_text(value: Any) -> str:
    """Lowercase article text and strip markup, URLs, control noise, and excess space."""
    if value is None or pd.isna(value):
        return ""
    text = html.unescape(str(value)).lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = text.replace("\x00", " ")
    return SPACE_PATTERN.sub(" ", text).strip()
