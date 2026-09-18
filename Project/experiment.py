"""Run the rounding-defense experiment against the live API."""

from __future__ import annotations

import argparse
import json

import numpy as np

from extractor import collect_queries, solve_probabilities
from victim import victim


def run(url: str, queries: int) -> list[dict[str, float | int | None]]:
    rng = np.random.default_rng(406)
    evaluation = rng.normal(size=(200, victim.feature_count))
    results = []
    classifier = victim.model[-1]
    scaler = victim.model[0]
    actual_weights = classifier.coef_ / scaler.scale_
    actual_intercept = classifier.intercept_ - np.sum(classifier.coef_ * scaler.mean_ / scaler.scale_, axis=1)
    actual_weight_differences = actual_weights[:-1] - actual_weights[-1]
    actual_intercept_differences = actual_intercept[:-1] - actual_intercept[-1]
    for decimals in [None, 5, 4, 3, 2]:
        features, probabilities = collect_queries(url, queries, victim.feature_count, decimals)
        try:
            probability_floor = None if decimals is None else 0.5 * 10 ** (-decimals)
            extracted = solve_probabilities(features, probabilities, probability_floor)
        except ValueError:
            results.append({
                "decimals": decimals,
                "queries": queries,
                "mean_absolute_probability_error": None,
                "parameter_error": None,
            })
            continue
        actual = victim.model.predict_proba(evaluation)
        predicted = extracted.predict_proba(evaluation)
        victim_predictions = victim.model.predict(evaluation)
        attacker_predictions = np.argmax(predicted, axis=1)
        results.append({
            "decimals": decimals,
            "queries": queries,
            "mean_absolute_probability_error": float(np.mean(np.abs(predicted - actual))),
            "parameter_error": float(
                np.linalg.norm(extracted.weights[:-1] - actual_weight_differences)
                + np.linalg.norm(extracted.intercept[:-1] - actual_intercept_differences)
            ),
            "victim_accuracy": float(np.mean(victim_predictions == np.argmax(actual, axis=1))),
            "attacker_accuracy": float(np.mean(attacker_predictions == victim_predictions)),
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/predict")
    parser.add_argument("--queries", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(run(args.url, args.queries), indent=2))


if __name__ == "__main__":
    main()
