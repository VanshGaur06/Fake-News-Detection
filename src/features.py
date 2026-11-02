"""Feature extraction using TF-IDF for fake news detection."""
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 20000) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer using unigrams and bigrams."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
