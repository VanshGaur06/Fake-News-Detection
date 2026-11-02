"""Script to test sample predictions on held-out test claims."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import load_split
from src.predict import load_artifacts, predict_claim


def test_samples():
    print("Loading test split and model artifacts...")
    test_df = load_split("test")
    artifacts = load_artifacts()

    # Pick 2 FAKE and 2 REAL examples from the test set
    fake_samples = test_df[test_df["label"] == "FAKE"].head(2)
    real_samples = test_df[test_df["label"] == "REAL"].head(2)

    sample_batch = [
        ("FAKE", fake_samples),
        ("REAL", real_samples),
    ]

    print("\n" + "=" * 65)
    print("SAMPLE EVALUATION ON UNSEEN TEST CLAIMS")
    print("=" * 65)

    for ground_truth, subset in sample_batch:
        for idx, row in subset.iterrows():
            statement = row["statement"]
            speaker = row["speaker"] if str(row["speaker"]).strip() else "Unknown"
            result = predict_claim(statement, artifacts)

            print(f"\nGround Truth : {ground_truth} (Original label: {row['raw_label']})")
            print(f"Speaker      : {speaker}")
            print(f"Claim        : \"{statement}\"")
            print(f"Prediction   : {result['prediction']} (Confidence: {result['confidence']:.2%})")
            print(f"Probabilities: FAKE={result['fake_probability']:.1%}, REAL={result['real_probability']:.1%}")

            if result["explanation"]["words"]:
                top_words = [f"{w['word']} ({w['impact']})" for w in result["explanation"]["words"][:3]]
                print(f"Top Signals  : {', '.join(top_words)}")
            print("-" * 65)


if __name__ == "__main__":
    test_samples()
