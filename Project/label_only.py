"""Label-only boundary probing for a binary class pair."""

from __future__ import annotations

import argparse
import json

import numpy as np
import requests


def query_label(url: str, features: np.ndarray) -> int:
    response = requests.post(url, json={"features": features.tolist()}, timeout=10)
    response.raise_for_status()
    return int(response.json()["label"])


def boundary_points(url: str, count: int, feature_count: int, class_a: int, class_b: int) -> np.ndarray:
    rng = np.random.default_rng(406)
    points = []
    while len(points) < count:
        left = rng.uniform([4.3, 2.0, 1.0, 0.1], [6.0, 4.0, 5.0, 1.8], feature_count)
        right = rng.uniform([5.5, 2.0, 3.5, 1.0], [8.0, 4.0, 7.0, 2.5], feature_count)
        left_label = query_label(url, left)
        right_label = query_label(url, right)
        if {left_label, right_label} != {class_a, class_b}:
            continue
        low, high = left, right
        for _ in range(30):
            middle = (low + high) / 2
            if query_label(url, middle) == left_label:
                low = middle
            else:
                high = middle
        points.append((low + high) / 2)
    return np.asarray(points)


def fit_boundary(points: np.ndarray) -> np.ndarray:
    """Return a homogeneous hyperplane through boundary points."""
    augmented = np.column_stack([points, np.ones(len(points))])
    _, _, vectors = np.linalg.svd(augmented)
    return vectors[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5000/label")
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--class-a", type=int, default=1)
    parser.add_argument("--class-b", type=int, default=2)
    args = parser.parse_args()
    points = boundary_points(args.url, args.points, 4, args.class_a, args.class_b)
    print(json.dumps({"points": len(points), "homogeneous_boundary": fit_boundary(points).tolist()}, indent=2))


if __name__ == "__main__":
    main()