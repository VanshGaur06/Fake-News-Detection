# TruthLens

**AI-Powered Fake News Detection** using classical NLP and machine learning. TruthLens identifies whether a submitted article resembles examples labeled FAKE or REAL in a training dataset. It is an educational research demo, not a fact-checking service.

## Problem and aim

Misleading news spreads quickly and is difficult to review at scale. This project demonstrates a transparent text-classification workflow: clean labeled news, turn words into TF-IDF features, compare four classical classifiers, evaluate on held-out data, and inspect the words influencing a prediction.

## Features

- Streamlit dashboard with article analysis, local explanations, model performance, and methodology.
- Optional UTF-8 `.txt` upload; direct text paste is also supported.
- TF-IDF unigrams and bigrams with preprocessing shared between training and inference.
- Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest comparison.
- Majority-class baseline, held-out classification metrics, five-fold cross-validation, and confusion matrix.
- Small grid search for Logistic Regression and Linear SVM.
- Calibrated sigmoid probabilities for Linear SVM confidence; other supported estimators use their `predict_proba` output.
- Linear-model explanations show signed TF-IDF × coefficient contributions.
- Saved artifacts mean Streamlit does not retrain at startup.

## Technology

Python 3, pandas, NumPy, scikit-learn, Streamlit, matplotlib, seaborn, joblib, and pytest. No transformers or deep learning are used. No Watermelon UI integration or related component was present in the repository, so the interface uses Streamlit with custom CSS.

## Dataset

Put a real labeled CSV at `data/dataset.csv`. It must include:

- `label`: Fake/Real, FAKE/REAL, or 0/1 (0 maps to FAKE; 1 maps to REAL)
- `text`, `title`, or both. If both exist, they are combined.

`subject` and `date` are optional and are not used as model features. At least 14 usable rows, seven from each class, are needed for the stratified split and five-fold training validation. Exact duplicate cleaned articles are dropped before splitting. Review near duplicates and source overlap in your dataset when interpreting results. The repository intentionally contains no fabricated example dataset.

## Method

1. Missing title/body fields are treated as empty; HTML, URLs, punctuation and common English stopwords are removed while negation words are retained.
2. A stratified 80/20 train/test split uses seed 42. Exact duplicate cleaned documents are removed before splitting.
3. A scikit-learn pipeline fits TF-IDF only on the relevant training partition. Thus, cross-validation folds fit their own vocabularies and the held-out test data does not influence vocabulary or tuning.
4. Five-fold CV and compact C grids compare candidate models using the training partition. The held-out metrics are calculated only after fitting on the training partition.
5. The winner is selected using mean FAKE-class F1 from training-only cross-validation, with mean accuracy as a tie-breaker. The held-out test set is reserved for the final comparison and report.
6. Confidence is the selected model's predicted class probability. Linear SVM probabilities come from sigmoid calibration fitted using training data only. It is not factual certainty.

Metrics use FAKE as the positive class. The confusion matrix rows are actual FAKE/REAL and columns are predicted FAKE/REAL. The majority-class baseline is the most frequent training label's share.

## Install and run

```bash
pip install -r requirements.txt
python src/train.py
streamlit run app.py
```

Without a dataset, training exits with a readable message and the Streamlit app shows a model-not-trained notice. Place the CSV, train, then restart or refresh the app.

## Example prediction

Paste a complete article or its title and body in **Analyze**. TruthLens shows FAKE/REAL pattern, model confidence, selected model and article length. The **Explainability** page displays local feature contributions when the selected model is linear. Performance charts appear after successful training.

## Limitations and future work

Text-only labels can encode publisher, topic, time-period, and annotation bias. Accuracy on one dataset does not establish real-world reliability, and textual similarity cannot confirm claims. Add provenance-aware and time-based evaluation, deduplicate near-identical stories, test on unseen sources, report calibration measures, and obtain expert-reviewed labels before making stronger claims.

## Viva notes

- **Why TF-IDF?** It represents words and phrases by frequency adjusted for how common they are across documents, giving compact sparse features.
- **Why train/test split?** The held-out partition estimates performance on data not used to fit the feature extractor or classifier.
- **Why cross-validation?** It compares settings across several training folds; every fold gets its own TF-IDF fit through the pipeline.
- **Why F1?** It combines precision and recall and is useful when class errors both matter or class balance is uneven.
- **Why SVM?** Linear SVMs often work well with high-dimensional sparse text features.
- **How does explanation work?** For a linear model, each active TF-IDF value is multiplied by its signed coefficient; contributions show association with a class, not proof.
- **How is confidence calculated?** Use the predicted class's `predict_proba`; Linear SVM is sigmoid-calibrated using training-only data.

## Tests

Run `pytest -q`. Training and UI checks require the dependencies above; the project does not ship a training dataset.
