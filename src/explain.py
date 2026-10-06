"""Local multiclass linear-model feature contributions."""
from __future__ import annotations

import numpy as np


def explain_prediction(text: str, vectorizer, model, predicted_label: str, limit: int = 8) -> dict:
    """Compare the predicted class score with its strongest competing class."""
    if not hasattr(model, "coef_"):
        return {"available": False, "features": [], "competitor": None}
    vector = vectorizer.transform([text])
    classes = list(model.classes_)
    predicted_index = classes.index(predicted_label)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(vector)).reshape(-1)
        competitor_index = int(np.argsort(scores)[-2])
    else:
        competitor_index = (predicted_index + 1) % len(classes)
    coefficients = np.asarray(model.coef_)
    contrast = coefficients[predicted_index] - coefficients[competitor_index]
    contributions = vector.multiply(contrast).tocoo()
    names = vectorizer.get_feature_names_out()
    order = np.argsort(np.abs(contributions.data))[::-1][:limit]
    features = [{"feature": str(names[contributions.col[i]]),
                 "contribution": float(contributions.data[i]),
                 "direction": "supports" if contributions.data[i] > 0 else "pushes away"}
                for i in order if contributions.data[i] != 0]
    return {"available": True, "features": features, "competitor": classes[competitor_index]}
