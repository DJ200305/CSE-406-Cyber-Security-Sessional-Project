"""Black-box equation-solving attacks against the prediction API."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import requests
from sklearn.datasets import load_iris

from victim import victim


@dataclass
class ExtractedModel:
    weights: np.ndarray
    intercept: np.ndarray
    reference_class: int

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=float)
        logits = features @ self.weights.T + self.intercept
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        return probabilities / probabilities.sum(axis=1, keepdims=True)


def solve_probabilities(
    features: np.ndarray,
    probabilities: np.ndarray,
    probability_floor: float | None = None,
) -> ExtractedModel:
    """Recover softmax parameters up to a reference-class shift."""
    features = np.asarray(features, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if probability_floor is None:
        probability_floor = np.finfo(float).tiny
    probabilities = np.maximum(probabilities, probability_floor)

    if np.any(probabilities <= 0):
        raise ValueError("all probabilities must be positive; use more precision")

    query_matrix = np.column_stack([features, np.ones(len(features))])
    reference = probabilities.shape[1] - 1
    recovered = np.zeros((probabilities.shape[1], features.shape[1] + 1))
    for class_index in range(reference):
        targets = np.log(probabilities[:, class_index] / probabilities[:, reference])
        recovered[class_index] = np.linalg.lstsq(query_matrix, targets, rcond=None)[0]
    return ExtractedModel(
        weights=recovered[:, :-1],
        intercept=recovered[:, -1],
        reference_class=reference,
    )


def query_oracle(url: str, features: np.ndarray) -> np.ndarray:
    payload = {"features": np.asarray(features, dtype=float).tolist()}
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return np.asarray(response.json()["probs"], dtype=float)


def collect_queries(url: str, count: int, feature_count: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(406)
    iris = load_iris()
    lower = iris.data.min(axis=0)
    upper = iris.data.max(axis=0)
    features = rng.uniform(lower, upper, size=(count, feature_count))
    probabilities = np.vstack([query_oracle(url, row) for row in features])
    return features, probabilities


def dynamic_queries(
    url: str,
    count: int,
    feature_count: int,
    domain_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    max_attempts: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate increasingly diverse queries using several realistic attacker strategies."""
    rng = np.random.default_rng(406)
    if domain_bounds is None:
        iris = load_iris()
        lower = iris.data.min(axis=0)
        upper = iris.data.max(axis=0)
        domain_bounds = (lower, upper)
    lower, upper = domain_bounds
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    features: list[np.ndarray] = []
    tries = 0
    while len(features) < count and tries < max_attempts:
        tries += 1
        candidate = rng.normal(0.0, 1.0, size=feature_count)
        candidate = np.clip(candidate, lower, upper)

        try:
            prob = query_oracle(url, candidate)
        except requests.exceptions.HTTPError:
            continue

        if np.isfinite(prob).all() and np.all(prob > 0):
            features.append(candidate)

    if len(features) < count:
        raise ValueError("Failed to generate enough valid queries with dynamic strategy")

    probabilities = np.vstack([query_oracle(url, row) for row in features])
    return np.asarray(features), probabilities


def compare(extracted: ExtractedModel, features: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predicted = extracted.predict_proba(features)
    classifier = victim.model[-1]
    scaler = victim.model[0]
    actual_weights = classifier.coef_ / scaler.scale_
    actual_intercept = classifier.intercept_ - np.sum(classifier.coef_ * scaler.mean_ / scaler.scale_, axis=1)
    actual_weight_differences = actual_weights[:-1] - actual_weights[-1]
    actual_intercept_differences = actual_intercept[:-1] - actual_intercept[-1]
    victim_predictions = victim.model.predict(features)
    attacker_predictions = np.argmax(predicted, axis=1)
    victim_accuracy = float(np.mean(victim_predictions == np.argmax(probabilities, axis=1)))
    attacker_accuracy = float(np.mean(attacker_predictions == victim_predictions))
    return {
        "mean_absolute_probability_error": float(np.mean(np.abs(predicted - probabilities))),
        "parameter_error": float(
            np.linalg.norm(extracted.weights[:-1] - actual_weight_differences)
            + np.linalg.norm(extracted.intercept[:-1] - actual_intercept_differences)
        ),
        "victim_accuracy": victim_accuracy,
        "attacker_accuracy": attacker_accuracy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract the local Iris softmax model")
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict")
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--features", type=int, default=4)
    parser.add_argument("--dynamic_queries", action="store_true", help="Use adaptive random-query generation")
    args = parser.parse_args()

    if args.dynamic_queries:
        features, probabilities = dynamic_queries(args.url, args.queries, args.features)
    else:
        features, probabilities = collect_queries(args.url, args.queries, args.features)

    extracted = solve_probabilities(features, probabilities)
    metrics = compare(extracted, features, probabilities)
    with open("extractor_results.jsonl", "a") as f:
        f.write(json.dumps({"queries": args.queries, "dynamic_queries": args.dynamic_queries, **metrics}) + "\n")


if __name__ == "__main__":
    main()
