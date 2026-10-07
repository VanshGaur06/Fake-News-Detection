"""Print model predictions for deterministic, stratified held-out WELFake articles."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DATA_PATH, LABELS, load_dataset, stratified_splits
from src.predict import load_artifacts, predict_article


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=2, help="Number of held-out examples from each label")
    parser.add_argument("--max-input-chars", type=int, default=900, help="Maximum article preview printed per example")
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    frame, _ = load_dataset(DATA_PATH)
    heldout = stratified_splits(frame)["test"]
    artifacts = load_artifacts()
    if list(artifacts["model"].classes_) != list(LABELS):
        raise RuntimeError(f"Unexpected model classes: {artifacts['model'].classes_.tolist()}")

    samples = []
    for label in LABELS:
        samples.extend(heldout[heldout.label.eq(label)].sample(
            n=args.per_class, random_state=42, replace=False).to_dict("records"))
    for index, sample in enumerate(samples, start=1):
        result = predict_article(sample["title"], sample["text"], artifacts)
        full_input = " ".join(part for part in (sample["title"], sample["text"]) if part)
        preview = full_input[:args.max_input_chars]
        if len(full_input) > len(preview):
            preview += " … [truncated; full held-out article was classified]"
        print(f"\nEXAMPLE {index}")
        print(f"TRUE LABEL: {sample['label']}")
        print(f"INPUT: {preview}")
        print(f"PREDICTED LABEL: {result['label']}")
        print(f"FAKE PROBABILITY: {result['fake_probability']:.1%}")
        print(f"REAL PROBABILITY: {result['real_probability']:.1%}")
        print(f"MODEL: {result['model_name']}")


if __name__ == "__main__":
    main()
