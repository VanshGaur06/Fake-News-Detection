# TruthLens: Fake News & Claim Truthfulness Detection

An AI/ML Mini Project for detecting fake news and classifying the truthfulness of political claims using classical Natural Language Processing (NLP) and Machine Learning.

---

## 📌 Project Overview

TruthLens addresses the challenge of misinformation by analyzing textual patterns in claims and statements. It uses the **LIAR benchmark dataset** (PolitiFact fact-checking claims) to classify claims into **FAKE** or **REAL** categories and provides word-level explainability to highlight which terms influenced the model's decision.

### Key Features
- **Interactive Web Interface**: Built with Streamlit for live claim analysis.
- **Model Comparison**: Compares Logistic Regression, Multinomial Naive Bayes, and Linear SVM.
- **Explainable AI (XAI)**: Visualizes word contributions toward FAKE or REAL.
- **Leakage-Safe Evaluation**: Evaluates on official held-out test data with TF-IDF fitted strictly on the training partition.

---

## 📊 Dataset: LIAR Benchmark

The dataset consists of 12.8k human-labeled political statements from PolitiFact across 3 official splits:
- **Training set**: ~10,240 statements
- **Validation set**: ~1,284 statements
- **Test set**: ~1,283 statements

### Binary Truthfulness Mapping
PolitiFact rates claims on a 6-point scale. TruthLens maps them into binary truthfulness:
- **FAKE**: `pants-fire`, `false`, `barely-true`
- **REAL**: `half-true`, `mostly-true`, `true`

---

## 🛠️ Project Architecture & Pipeline

1. **Text Preprocessing**: Lowercasing, URL removal, HTML stripping, punctuation filtering.
2. **Feature Extraction**: TF-IDF (Term Frequency - Inverse Document Frequency) with unigrams and bigrams.
3. **Model Training**: Evaluates candidate classifiers on the validation split and trains the primary model (Logistic Regression).
4. **Explainability**: Calculates signed feature contributions:
   $$\text{Contribution}(w) = \text{TF-IDF}(w) \times \text{Coefficient}(w)$$
5. **Dashboard**: Interactive Streamlit application for inference and model evaluation.

---

## 🚀 How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train.py
```
*This vectorizes the dataset, compares candidate models, evaluates on the test set, and saves the trained artifacts into `models/`.*

### 3. Run Sample Predictions (CLI)
```bash
python scripts/test_model.py
```

### 4. Run the Test Suite
```bash
python -m pytest tests -v
```

### 5. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```text
Fake-News-Detection/
├── data/
│   ├── train.tsv             # Official LIAR training split
│   ├── valid.tsv             # Official LIAR validation split
│   ├── test.tsv              # Official LIAR test split
│   └── README.md             # Dataset documentation
├── models/
│   ├── model.joblib          # Trained Logistic Regression classifier
│   ├── vectorizer.joblib     # Fitted TF-IDF vectorizer
│   └── metadata.json         # Evaluation metrics and class distributions
├── outputs/
│   └── metrics/
│       └── evaluation.json   # JSON evaluation metrics for dashboard
├── src/
│   ├── __init__.py           # Package init
│   ├── dataset.py            # Data loading, cleaning, and binary mapping
│   ├── features.py           # TF-IDF feature extraction
│   ├── evaluate.py           # Evaluation metric computation
│   ├── train.py              # Model training, comparison, and serialization
│   ├── predict.py            # Prediction pipeline for user claims
│   └── explain.py            # Word-level feature contributions
├── scripts/
│   └── test_model.py         # Test script on held-out test samples
├── tests/
│   └── test_project.py       # Pytest unit and integration test suite
├── app.py                    # Streamlit web application
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

---

## 🎓 Viva / Presentation Guide

- **Q: What dataset did you use?**
  *A: The LIAR dataset introduced by William Yang Wang, containing 12.8k fact-checked statements from PolitiFact.*
- **Q: Why map 6 classes into binary?**
  *A: The 6-class task is fine-grained and noisy for short text; grouping into untruthful (FAKE) vs truthful/nuanced (REAL) produces clear, actionable classification suitable for fake news detection.*
- **Q: Why did you choose Logistic Regression as the final model?**
  *A: Logistic Regression performed best among linear models on the validation split, provides calibrated probability estimates (`predict_proba`), and allows direct interpretation of feature coefficients for explainability.*
- **Q: How does explainability work in TruthLens?**
  *A: We multiply the TF-IDF weight of each word in the input by the model's corresponding coefficient. Positive values pull toward REAL, while negative values pull toward FAKE.*
- **Q: How did you prevent data leakage?**
  *A: The TF-IDF vectorizer was fit exclusively on the training split. Validation and test splits were transformed only.*
