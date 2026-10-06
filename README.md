# TruthLens

**See the signal behind the story.** TruthLens is an educational AI-based truthfulness classification demo using classical NLP and machine learning.

> **Benchmark scope:** LIAR contains short, fact-checked political statements. It is not a dataset of complete news articles. TruthLens predicts a LIAR truthfulness label; it does not independently verify facts.

## Aim and objectives

Demonstrate a complete, explainable NLP workflow on an established six-class benchmark: prepare the official splits, clean statement text, represent claims with TF-IDF, compare classical classifiers, select using validation macro F1, evaluate once on held-out test data, and inspect model confidence and influential features.

## Features

- Streamlit interface with Home, Analyze, Model Lab, Explainability, Dataset, and Methodology views.
- Six-class output: `pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, and `true`.
- Actual model probability distribution and predicted-class confidence.
- Clearly separate project-level interpretation: the first three LIAR labels map to **FAKE / LOW TRUTHFULNESS** and the latter three to **REAL / HIGHER TRUTHFULNESS**. In this grouping, `half-true` does not mean factually real.
- Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest.
- Validation model comparison, training-only five-fold CV, held-out test metrics, six-class and binary confusion matrices, and per-class scores.
- TF-IDF contribution explanations for linear estimators.
- Saved artifacts; Streamlit does not train models at startup.

## Dataset

Source: [tfs4/liar_dataset](https://github.com/tfs4/liar_dataset). The repository provides three official splits and 14 TSV columns: statement ID, truth label, statement text, subjects, speaker information, credit-history counts, and context. **Only statement text is used for model input**; metadata is excluded to avoid speaker/source shortcuts.

Download the three required files with:

```bash
python src/download_dataset.py
```

This creates:

```text
data/liar/train.tsv
data/liar/valid.tsv
data/liar/test.tsv
```

Or download those three files from the dataset repository and place them in `data/liar/`. The downloader obtains no notebook or derived CSV files. Raw TSVs are excluded from Git by `.gitignore`; do not commit the dataset unless there is a clear need and repository size/licensing have been checked.

## Label interpretation

TruthLens preserves all six original labels for the main ML task. The optional binary presentation groups `pants-fire`, `false`, and `barely-true` as **FAKE / LOW TRUTHFULNESS**, and `half-true`, `mostly-true`, and `true` as **REAL / HIGHER TRUTHFULNESS**. It is only a project-level simplification for the UI; it is not the benchmark's original task and is not a factual verdict.

## Method and leakage controls

1. Download and validate the original train, validation, and test TSVs. No random split is made.
2. Normalize statement punctuation, markup, URLs, and whitespace. Stopwords, negation, and numeric values are retained because claims are short and those tokens may matter.
3. Remove exact cleaned-claim duplicates from training, and remove training/validation claims that overlap later official splits. Validation claims that overlap test are also excluded. This avoids duplicates influencing selection or the held-out score.
4. Tune Logistic Regression and Linear SVM on the official validation split with a compact C/class-weight grid. Each candidate is fit on training statements only; the TF-IDF step is inside the pipeline.
5. Compare validation results using macro F1 first, then accuracy and weighted F1. Five-fold cross-validation measures the chosen configurations on training only; every fold fits its own TF-IDF vocabulary.
6. Refit the selected model using train plus validation, then evaluate it once on the untouched official test split. Six-class and binary-interpretation metrics are reported separately.
7. Linear SVM confidence uses sigmoid calibration fitted with train and validation only. Other classifiers use their `predict_proba` outputs. Probabilities express model estimates, not factual certainty.

## Models and metrics

- Multinomial Naive Bayes is a simple probabilistic text baseline.
- Logistic Regression is a sparse linear classifier with class probabilities.
- Linear SVM is suitable for high-dimensional sparse text; calibration provides probabilities.
- Random Forest provides a nonlinear tree-ensemble comparison.

The main metrics are six-class accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrix, and per-class precision/recall/F1. The binary interpretation has its own metrics and confusion matrix. Macro F1 weights the six categories equally, which makes it useful when class frequencies differ.

### Measured run on the downloaded LIAR splits

This workspace run selected **Logistic Regression** on validation macro F1 (26.31%). After exact-claim deduplication, the split sizes were 10,237 train / 1,283 validation / 1,283 test. On the untouched six-class test split: **accuracy 25.18%, macro precision 24.96%, macro recall 25.63%, macro F1 25.13%, weighted F1 25.13%**. Five-fold training CV for the selected configuration gave accuracy 24.56% ± 0.87% and macro F1 24.36% ± 0.90% (mean ± standard deviation).

For the separate two-group project interpretation, test accuracy was **63.13%**, macro precision 62.37%, macro recall 62.24%, macro F1 **62.29%**, and weighted F1 63.04%. These are dataset-specific measured results, not hardcoded app values or a guarantee of performance on new domains. The six-class challenge is substantially harder than the collapsed binary grouping.

## Explainability and limitations

For linear models, feature contributions use TF-IDF values multiplied by class-coefficient contrasts between the predicted class and its strongest competitor. They show which text patterns influenced this model; they do not prove a statement true or false. Random Forest does not receive a fabricated local linear explanation.

LIAR consists of political claims labeled by the original dataset annotators. Its labels, time period, topics, and source population have limits. Text patterns may reflect those biases. A high-confidence prediction is not independent fact-checking, and results should not be generalized to full articles or all domains.

## Installation and run

Use Python 3.10 or newer. In PowerShell, for example:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_dataset.py
python src/train.py
streamlit run app.py
```

Linux/macOS activation is `source .venv/bin/activate`. If the dataset is missing or network access is unavailable, place the three TSV files manually as listed above. If no model artifacts exist, Streamlit displays a model-not-trained message instead of crashing.

## Project layout

```text
app.py
data/liar/                   # ignored official TSV splits
models/                      # ignored joblib artifacts
outputs/metrics/             # generated evaluation JSON
src/dataset.py               # LIAR schema and split validation
src/download_dataset.py      # fetch only the three official splits
src/preprocessing.py         # shared text and label utilities
src/features.py              # TF-IDF configuration
src/train.py                 # tune, compare, CV, test, persist
src/evaluate.py              # six-class and binary metrics
src/explain.py               # local linear contributions
src/predict.py               # artifact loading and inference
tests/
```

## Verification

Run the unit suite with:

```bash
python -m pytest tests -q
```

Training takes longer because it evaluates four models and runs five-fold CV. Scores are generated from this dataset during training and saved in `outputs/metrics/evaluation.json`; no metric is hardcoded.

## Future scope

Evaluate on newer or non-political claims, compare source/time generalization, audit duplicate and label issues, report calibration quality, and add human-reviewed explanations before considering any real-world decision support.

## Viva questions

1. **What is truthfulness classification?** Predicting which label among annotated claim categories best matches a statement's learned text patterns.
2. **Why LIAR?** It provides a public, labeled six-class benchmark and official splits for reproducible evaluation.
3. **Why is LIAR not an article dataset?** Each example is a short political statement, not a complete article.
4. **What is TF-IDF?** A sparse weighting of terms by their importance within a document relative to the corpus.
5. **Why use n-grams?** They represent individual words and short phrases that can carry useful claim context.
6. **Why preserve negation and numbers?** Words such as “not” and values or years can alter a claim's meaning.
7. **What is Logistic Regression?** A linear classifier that models class probabilities from weighted input features.
8. **Why Linear SVM?** It works well with high-dimensional sparse text representations.
9. **What is Naive Bayes?** A probabilistic classifier using conditional word evidence under a simplifying independence assumption.
10. **Why Random Forest?** It supplies a nonlinear tree-ensemble comparison to the linear text models.
11. **What is precision?** Of predictions assigned to a class, the share that are correct.
12. **What is recall?** Of actual examples in a class, the share recovered by the model.
13. **What is macro F1?** The unweighted average of class-wise F1 scores.
14. **What is a confusion matrix?** A table of actual labels against predicted labels.
15. **What is data leakage?** Using held-out information during training or model selection.
16. **How is leakage reduced here?** Use official splits, train-only TF-IDF within folds, validation-only tuning, duplicate filtering, and one final test evaluation.
17. **How is confidence calculated?** From `predict_proba`; Linear SVM probabilities are sigmoid-calibrated using training/validation data.
18. **How does the explanation work?** It contrasts linear class coefficients multiplied by this statement's TF-IDF values.
19. **Why can't accuracy be guaranteed at 100%?** Labels and language are ambiguous; generalization is measured from real held-out examples.
20. **What are the main limitations?** Political-domain scope, annotation bias, dataset age, and no independent verification of claims.

## UI screenshots

Run the Streamlit app and capture the Analyze and Model Lab screens after training. This repository does not include mock screenshots or fabricated dashboard results.
