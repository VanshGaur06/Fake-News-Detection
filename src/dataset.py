"""WELFake binary article loading, deduplication, and stratified splitting."""
from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WELFake_Dataset.csv"
RANDOM_STATE = 42
LABEL_MAP = {0: "FAKE", 1: "REAL"}  # WELFake's published encoding.
LABELS = ("FAKE", "REAL")


def _normalize_duplicate_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def load_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict]:
    """Load WELFake title/body text and binary labels; remove empty/duplicate articles."""
    if not path.exists():
        raise FileNotFoundError(f"WELFake dataset missing: {path}")
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise ValueError(f"Could not read WELFake CSV: {exc}") from exc
    source_rows = len(frame)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    required = {"title", "text", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"WELFake CSV must contain title, text, and label columns; found {list(frame.columns)}")
    frame["label"] = pd.to_numeric(frame["label"], errors="coerce")
    invalid = frame["label"].isna() | ~frame["label"].isin(LABEL_MAP)
    if invalid.any():
        raise ValueError(f"WELFake CSV has {int(invalid.sum())} invalid labels; expected 0=FAKE and 1=REAL.")
    frame["label"] = frame["label"].astype(int).map(LABEL_MAP)
    frame["title"] = frame["title"].fillna("").astype(str).str.strip()
    frame["text"] = frame["text"].fillna("").astype(str).str.strip()
    frame["article"] = (frame["title"] + " " + frame["text"]).str.replace(r"\s+", " ", regex=True).str.strip()
    before_empty = len(frame)
    frame = frame[frame["article"].ne("")].copy()
    empty_removed = before_empty - len(frame)
    frame["duplicate_key"] = frame["article"].map(_normalize_duplicate_key)

    # Drop duplicate-content groups with conflicting labels rather than allowing
    # the same article to leak across splits with contradictory targets.
    label_counts = frame.groupby("duplicate_key")["label"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    conflicting_rows = int(frame["duplicate_key"].isin(conflicting_keys).sum())
    frame = frame[~frame["duplicate_key"].isin(conflicting_keys)].copy()
    before_dedupe = len(frame)
    frame = frame.drop_duplicates("duplicate_key", keep="first").copy()
    duplicate_rows_removed = before_dedupe - len(frame)
    if frame.empty or set(frame["label"].unique()) != set(LABELS):
        raise ValueError("WELFake must contain usable articles for both FAKE and REAL after cleaning.")
    frame = frame.reset_index(drop=True)
    summary = {
        "dataset": "WELFake", "source_rows": int(source_rows),
        "total_samples": int(len(frame)), "empty_articles_removed": int(empty_removed),
        "duplicate_articles_removed": int(duplicate_rows_removed),
        "conflicting_duplicate_rows_removed": int(conflicting_rows),
        "class_distribution": {label: int(count) for label, count in frame["label"].value_counts().reindex(LABELS, fill_value=0).items()},
    }
    return frame[["title", "text", "article", "label"]], summary


def stratified_splits(frame: pd.DataFrame, random_state: int = RANDOM_STATE) -> dict[str, pd.DataFrame]:
    """Create fixed 80/10/10 splits with class ratios preserved in each split."""
    train, remainder = train_test_split(frame, test_size=0.2, random_state=random_state,
        shuffle=True, stratify=frame["label"])
    validation, test = train_test_split(remainder, test_size=0.5, random_state=random_state,
        shuffle=True, stratify=remainder["label"])
    splits = {"train": train.reset_index(drop=True), "validation": validation.reset_index(drop=True),
              "test": test.reset_index(drop=True)}
    keys = {name: set(part["article"].map(_normalize_duplicate_key)) for name, part in splits.items()}
    if (keys["train"] & keys["validation"] or keys["train"] & keys["test"] or keys["validation"] & keys["test"]):
        raise RuntimeError("Article text leakage detected between dataset splits.")
    return splits
