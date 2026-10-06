"""Feature extraction configuration."""
from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer(max_features: int = 100_000) -> TfidfVectorizer:
    return TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=2,
                           max_df=0.98, sublinear_tf=True, lowercase=False,
                           token_pattern=r"(?u)\b\w\w+\b")
