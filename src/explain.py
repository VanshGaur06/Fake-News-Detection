"""Local feature contributions for linear binary classifiers."""
from __future__ import annotations

import numpy as np


def explain_prediction(text: str, vectorizer, model, limit: int = 10) -> dict:
    """Return signed TF-IDF × coefficient contributions for this text."""
    if not hasattr(model, "coef_"):
        return {"available": False, "features": []}
    vector = vectorizer.transform([text])
    classes = list(model.classes_)
    coef = np.asarray(model.coef_).reshape(-1)
    # Binary sklearn linear models orient the coefficient toward classes_[1].
    fake_sign = 1.0 if classes[1] == "FAKE" else -1.0
    signed = vector.multiply(coef * fake_sign).tocoo()
    names = vectorizer.get_feature_names_out()
    order = np.argsort(np.abs(signed.data))[::-1][:limit]
    features = [{"feature": str(names[signed.col[i]]), "contribution": float(signed.data[i]),
                 "direction": "FAKE" if signed.data[i] > 0 else "REAL"}
                for i in order if signed.data[i] != 0]
    return {"available": True, "features": features}
