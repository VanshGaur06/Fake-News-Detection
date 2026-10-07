# TruthLens

TruthLens is a Streamlit fake-news classifier for full news articles. Its production model is a binary Logistic Regression model trained on the WELFake dataset using article title and body text. It does not use the historical LIAR model or infer facts from external sources.

## Dataset

Place `WELFake_Dataset.csv` in `data/`. The project copy has columns `title`, `text`, and `label`; the serial/index column is ignored. The dataset record defines `0 → FAKE` and `1 → REAL` ([WELFake on Zenodo](https://zenodo.org/records/4561253)). The record describes WELFake as combining Kaggle, McIntire, Reuters, and BuzzFeed Political news data.

The source file in this workspace contains 72,134 rows. The source record's published class totals and the file's numeric label counts are inconsistent, so training follows the source's explicit numeric label mapping and reports the actual counts derived from the file. No article labels are inferred or rewritten from article wording.

Title and body are joined as the model input. `subject`, `date`, the serial/index column, and other metadata are excluded. The loader removes empty articles, removes duplicate article text, and drops duplicate-text groups with conflicting labels before splitting.

## Train and verify

```powershell
python src/train.py
python -m pytest tests -q
python scripts/test_model.py
streamlit run app.py
```

Training uses a fixed-seed stratified 80/10/10 train/validation/test split. TF-IDF is fitted on the training partition only, and Logistic Regression is fitted on that same training partition only. Validation and test partitions are transform-only. The training command prints dataset size, class distribution, split sizes, validation and test accuracy, macro precision/recall/F1, confusion matrices, and classification reports.

`scripts/test_model.py` selects deterministic examples from both labels in the held-out test split and prints each article's input preview, source label, predicted label, FAKE probability, REAL probability, and model name.

## Inference and explainability

The application loads only these WELFake artifacts:

- `models/fake_news_model.joblib`
- `models/fake_news_vectorizer.joblib`
- `models/fake_news_metadata.json`

The model's classes are `['FAKE', 'REAL']`. Displayed probabilities are returned directly by Logistic Regression `predict_proba()` and represent model confidence, **not independent fact verification**. Feature contributions use the saved TF-IDF vectorizer and the same binary Logistic Regression coefficients.

The previous LIAR artifacts (`final_model.joblib`, `tfidf_vectorizer.joblib`, `explanation_model.joblib`, and `metadata.joblib`) have been removed. Legacy LIAR TSV files under `data/` are not used by training or the app.

## Project layout

```text
app.py                         # TruthLens Streamlit UI
data/WELFake_Dataset.csv       # binary full-article source data (not committed)
models/                        # saved WELFake model and vectorizer
outputs/metrics/evaluation.json
src/dataset.py                 # WELFake loading, cleaning, and stratified split
src/preprocessing.py           # shared text cleaning
src/features.py                # TF-IDF configuration
src/train.py                   # training, metrics, and serialization
src/predict.py                 # inference through the WELFake artifacts
src/explain.py                 # binary Logistic Regression contributions
tests/test_project.py          # split, artifact, and held-out inference tests
scripts/test_model.py          # print held-out example predictions
```

WELFake is a benchmark, not a live fact-checking service. Its article labels and source composition limit how far its measured performance generalizes.
