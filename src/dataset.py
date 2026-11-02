"""Dataset loading and preprocessing for the LIAR benchmark."""
from pathlib import Path
import re
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

LIAR_COLUMNS = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state_info",
    "party",
    "barely_true_count",
    "false_count",
    "half_true_count",
    "mostly_true_count",
    "pants_on_fire_count",
    "context",
]

# Standard binary mapping for LIAR dataset
# Untruthful claims -> FAKE, Truthful/Nuanced claims -> REAL
LABEL_MAP = {
    "pants-fire": "FAKE",
    "false": "FAKE",
    "barely-true": "FAKE",
    "half-true": "REAL",
    "mostly-true": "REAL",
    "true": "REAL",
}

CLASSES = ["FAKE", "REAL"]


def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip URLs, HTML tags, and extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_split(split_name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load a single split (train, valid, or test) from TSV."""
    filename_map = {
        "train": "train.tsv",
        "validation": "valid.tsv",
        "test": "test.tsv",
    }
    file_path = data_dir / filename_map[split_name]
    if not file_path.exists():
        raise FileNotFoundError(f"Missing LIAR file: {file_path}")

    df = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=LIAR_COLUMNS,
        quoting=3,
        on_bad_lines="skip",
    )

    df["raw_label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["raw_label"].map(LABEL_MAP)
    df = df.dropna(subset=["label"]).copy()

    df["statement"] = df["statement"].fillna("").astype(str)
    df["cleaned_text"] = df["statement"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    return df


def load_all_splits(data_dir: Path = DATA_DIR) -> tuple[dict[str, pd.DataFrame], dict]:
    """Load train, validation, and test splits and return summary statistics."""
    splits = {
        "train": load_split("train", data_dir),
        "validation": load_split("validation", data_dir),
        "test": load_split("test", data_dir),
    }

    summary = {
        "dataset_name": "LIAR Benchmark",
        "split_counts": {k: len(v) for k, v in splits.items()},
        "train_class_counts": splits["train"]["label"].value_counts().to_dict(),
        "train_6way_counts": splits["train"]["raw_label"].value_counts().to_dict(),
    }
    return splits, summary
