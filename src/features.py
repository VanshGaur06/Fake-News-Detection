"""TF-IDF feature extraction for full news articles."""
from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer(max_features: int = 120_000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2,
        max_df=0.98, sublinear_tf=True, lowercase=True, strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b")
