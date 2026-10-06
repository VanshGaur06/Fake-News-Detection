"""Download only the three official LIAR benchmark TSV splits."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import urllib.error
import urllib.request
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DATA_DIR, LIAR_COLUMNS, SPLIT_FILES
from src.preprocessing import LIAR_LABELS

REPOSITORY_URL = "https://github.com/tfs4/liar_dataset"
RAW_BASE = "https://raw.githubusercontent.com/tfs4/liar_dataset/master"


def validate_tsv(content: bytes, filename: str) -> None:
    try:
        rows = csv.reader(io.StringIO(content.decode("utf-8")), delimiter="\t")
        first = next(rows)
    except (UnicodeDecodeError, StopIteration, csv.Error) as exc:
        raise ValueError(f"Downloaded {filename} is empty or not valid UTF-8 TSV.") from exc
    if len(first) != len(LIAR_COLUMNS) or first[1].strip().lower() not in LIAR_LABELS:
        raise ValueError(f"Downloaded {filename} does not match the expected 14-column LIAR format.")


def download_dataset(destination: Path = DATA_DIR) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for filename in SPLIT_FILES.values():
        url = f"{RAW_BASE}/{filename}"
        request = urllib.request.Request(url, headers={"User-Agent": "TruthLens-LIAR-downloader/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not download {filename} from {REPOSITORY_URL}: {exc}. "
                               f"Download train.tsv, valid.tsv and test.tsv and place them in {destination}.") from exc
        validate_tsv(content, filename)
        target = destination / filename
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(target)
        output_paths.append(target)
        print(f"Saved {target} ({len(content):,} bytes)")
    return output_paths


if __name__ == "__main__":
    try:
        download_dataset()
    except (RuntimeError, ValueError, OSError) as error:
        raise SystemExit(f"Dataset download failed: {error}")
