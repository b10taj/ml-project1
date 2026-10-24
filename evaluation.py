"""Metrics and cross-validation helpers (NumPy only)."""

import numpy as np


def k_fold_indices(n, k, seed=0):
    """Split range(n) into k shuffled folds (list of index arrays)."""
    rng = np.random.default_rng(seed)
    return np.array_split(rng.permutation(n), k)


def f1_score(y_true, y_pred):
    """F1 score of the positive class. Labels in {0, 1}."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(y_true, scores, n_grid=100):
    """Threshold on the scores maximizing F1. Returns (threshold, f1)."""
    grid = np.quantile(scores, np.linspace(0.5, 0.995, n_grid))
    f1s = [f1_score(y_true, (scores >= t).astype(int)) for t in grid]
    best = int(np.argmax(f1s))
    return grid[best], f1s[best]


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(y_true == y_pred))
