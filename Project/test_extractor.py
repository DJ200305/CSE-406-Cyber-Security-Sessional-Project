import numpy as np

from extractor import solve_probabilities


def test_equation_solver_recovers_softmax_probabilities():
    rng = np.random.default_rng(10)
    features = rng.normal(size=(12, 4))
    weights = rng.normal(size=(3, 4))
    intercept = rng.normal(size=3)
    logits = features @ weights.T + intercept
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)

    extracted = solve_probabilities(features, probabilities)

    np.testing.assert_allclose(extracted.predict_proba(features), probabilities, atol=1e-10)
