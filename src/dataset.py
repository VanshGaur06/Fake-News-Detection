"""LIAR dataset split loading and leakage-safe duplicate handling."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.preprocessing import LIAR_LABELS, clean_text, normalize_liar_label

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "liar"
LIAR_COLUMNS = ["id", "label", "statement", "subjects", "speaker", "speaker_job",
                "state_info", "party_affiliation", "barely_true_count", "false_count",
                "half_true_count", "mostly_true_count", "pants_fire_count", "context"]
SPLIT_FILES = {"train": "train.tsv", "validation": "valid.tsv", "test": "test.tsv"}


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"LIAR {split_name} split missing: {path}. Run python src/download_dataset.py")
    try:
        frame = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS, dtype={"id": str},
                            keep_default_na=False, quoting=3, on_bad_lines="error")
    except Exception as exc:
        raise ValueError(f"Could not parse {split_name} TSV: {exc}") from exc
    if frame.shape[1] != 14 or frame.empty:
        raise ValueError(f"{path} must contain non-empty rows with exactly 14 tab-separated LIAR columns.")
    frame["label"] = frame["label"].map(normalize_liar_label)
    invalid = int(frame["label"].isna().sum())
    if invalid:
        raise ValueError(f"{path} contains {invalid} missing or unknown labels; expected the six original LIAR labels.")
    frame["cleaned_statement"] = frame["statement"].map(clean_text)
    frame = frame[frame["cleaned_statement"].str.strip().ne("")].copy()
    frame["word_count"] = frame["cleaned_statement"].str.split().str.len()
    return frame.reset_index(drop=True)


def load_splits(data_dir: Path = DATA_DIR) -> tuple[dict[str, pd.DataFrame], dict[str, int]]:
    splits = {name: load_split(data_dir / filename, name) for name, filename in SPLIT_FILES.items()}
    # Reserve evaluation statements: remove any matching claim from training, and any
    # test-overlapping claim from validation so the test set cannot influence tuning.
    test_texts = set(splits["test"]["cleaned_statement"])
    validation_texts = set(splits["validation"]["cleaned_statement"])
    before_train = len(splits["train"])
    splits["train"] = splits["train"].drop_duplicates("cleaned_statement", keep="first")
    splits["train"] = splits["train"][~splits["train"]["cleaned_statement"].isin(test_texts | validation_texts)]
    before_validation = len(splits["validation"])
    splits["validation"] = splits["validation"].drop_duplicates("cleaned_statement", keep="first")
    splits["validation"] = splits["validation"][~splits["validation"]["cleaned_statement"].isin(test_texts)]
    before_test = len(splits["test"])
    splits["test"] = splits["test"].drop_duplicates("cleaned_statement", keep="first")
    if not splits["train"].shape[0] or not splits["validation"].shape[0] or not splits["test"].shape[0]:
        raise ValueError("LIAR splits must remain non-empty after exact duplicate statements are removed.")
    for name, frame in splits.items():
        missing_classes = sorted(set(LIAR_LABELS) - set(frame["label"]))
        if missing_classes:
            raise ValueError(f"The {name} split is missing classes: {', '.join(missing_classes)}")
    duplicate_summary = {"training_duplicates_removed": before_train - len(splits["train"]),
                         "validation_duplicates_removed": before_validation - len(splits["validation"]),
                         "test_duplicates_removed": before_test - len(splits["test"])}
    return splits, duplicate_summary
