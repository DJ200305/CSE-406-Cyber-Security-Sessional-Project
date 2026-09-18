"""Victim model used by the local extraction experiment."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class VictimModel:
    """A deterministic Iris classifier exposed through a probability oracle."""

    def __init__(self) -> None:
        iris = load_iris()
        self.feature_count = iris.data.shape[1]
        self.class_count = len(iris.target_names)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=7),
        )
        self.model.fit(iris.data, iris.target)

    def predict(self, features: list[float], decimals: int | None = None) -> dict:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        if values.shape[1] != self.feature_count:
            raise ValueError(f"expected {self.feature_count} features")
        probabilities = self.model.predict_proba(values)[0]
        if decimals is not None:
            probabilities = np.round(probabilities, decimals=int(decimals))
        return {
            "label": int(np.argmax(probabilities)),
            "probs": probabilities.tolist(),
        }

    def label(self, features: np.ndarray) -> int:
        return int(self.model.predict(np.asarray(features).reshape(1, -1))[0])


victim = VictimModel()
