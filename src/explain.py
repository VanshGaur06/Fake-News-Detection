"""Word-level explainability for linear classification models."""
import numpy as np


def explain_prediction(text: str, vectorizer, model, top_k: int = 8) -> dict:
    """Calculate word contributions toward FAKE or REAL using TF-IDF weights and model coefficients."""
    if not hasattr(model, "coef_"):
        return {"words": []}

    feature_names = np.array(vectorizer.get_feature_names_out())
    tfidf_vector = vectorizer.transform([text]).toarray()[0]

    # Find words present in the input
    nonzero_idx = np.where(tfidf_vector > 0)[0]
    if len(nonzero_idx) == 0:
        return {"words": []}

    # model.coef_[0] is oriented toward classes_[1] ('REAL')
    coefs = model.coef_[0]
    contributions = tfidf_vector[nonzero_idx] * coefs[nonzero_idx]
    words = feature_names[nonzero_idx]

    # Rank by magnitude
    ranked_order = np.argsort(np.abs(contributions))[::-1][:top_k]

    explained_words = []
    for idx in ranked_order:
        val = float(contributions[idx])
        word = str(words[idx])
        direction = "REAL" if val > 0 else "FAKE"
        explained_words.append({
            "word": word,
            "weight": round(val, 4),
            "impact": direction,
            "importance": round(abs(val), 4),
        })

    return {
        "words": explained_words,
        "fake_signals": [w for w in explained_words if w["impact"] == "FAKE"],
        "real_signals": [w for w in explained_words if w["impact"] == "REAL"],
    }
