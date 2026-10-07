"""Local binary Logistic Regression feature contributions."""
from __future__ import annotations

import numpy as np


def explain_prediction(text: str, vectorizer, model, predicted_label: str, limit: int = 8) -> dict:
    """Show signed TF-IDF contributions toward the predicted binary class."""
    if not hasattr(model, "coef_"):
        return {"available": False, "features": [], "competitor": None}
    vector = vectorizer.transform([text])
    classes = list(model.classes_)
    predicted_index = classes.index(predicted_label)
    coefficients = np.asarray(model.coef_)
    if len(classes) != 2 or coefficients.shape[0] != 1:
        return {"available": False, "features": [], "competitor": None}
    competitor_index = 1 - predicted_index
    # sklearn's binary coefficient row is oriented toward classes_[1].
    contrast = coefficients[0] if predicted_index == 1 else -coefficients[0]
    contributions = vector.multiply(contrast).tocoo()
    names = vectorizer.get_feature_names_out()
    order = np.argsort(np.abs(contributions.data))[::-1][:limit]
    features = [{"feature": str(names[contributions.col[i]]),
                 "contribution": float(contributions.data[i]),
                 "direction": "supports" if contributions.data[i] > 0 else "pushes away"}
                for i in order if contributions.data[i] != 0]
    return {"available": True, "features": features, "competitor": classes[competitor_index]}
