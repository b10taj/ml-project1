"""Model comparison with k-fold CV, using our own implementations.py (NumPy only)."""

import sys
import time

import numpy as np

from evaluation import accuracy, best_threshold, f1_score, k_fold_indices
from implementations import reg_logistic_regression, ridge_regression
from preprocess import (
    apply_special_codes,
    find_special_codes,
    fit_features,
    read_column_names,
    transform_features,
)

K = 4


def fit_ridge(lambda_):
    """Return a fit function (y, tx) -> w using ridge_regression."""
    return lambda y, tx: ridge_regression(y, tx, lambda_)[0]


def fit_logistic(lambda_, max_iters, gamma):
    """Return a fit function (y, tx) -> w using reg_logistic_regression."""
    return lambda y, tx: reg_logistic_regression(
        y, tx, lambda_, np.zeros(tx.shape[1]), max_iters, gamma
    )[0]


def cross_validate(x, y01, names, fit, tune_threshold=True, **feature_kwargs):
    """Mean F1 and accuracy over K folds.

    The threshold is tuned on each training fold, or fixed at 0 (probability 0.5)
    if tune_threshold is False. feature_kwargs are passed to fit_features.
    """
    folds = k_fold_indices(len(y01), K)
    f1s, accs = [], []
    for k in range(K):
        val = folds[k]
        train = np.concatenate([folds[i] for i in range(K) if i != k])
        params = fit_features(x[train], names, y=y01[train], **feature_kwargs)
        tx_train = transform_features(x[train], params).astype(np.float64)
        tx_val = transform_features(x[val], params).astype(np.float64)
        w = fit(y01[train], tx_train)
        threshold = 0.0
        if tune_threshold:
            threshold, _ = best_threshold(y01[train], tx_train @ w)
        pred = (tx_val @ w >= threshold).astype(int)
        f1s.append(f1_score(y01[val], pred))
        accs.append(accuracy(y01[val], pred))
    return np.mean(f1s), np.mean(accs)


if __name__ == "__main__":
    d = np.load("data/cache.npz")
    x, y01 = d["x"], (d["y"] == 1).astype(int)
    names = read_column_names("data/dataset/")
    x = apply_special_codes(x, find_special_codes(x), names)

    experiments = [
        ("ridge lambda=1e-3", fit_ridge(1e-3)),
        ("ridge lambda=1e-2", fit_ridge(1e-2)),
        ("ridge lambda=1e-1", fit_ridge(1e-1)),
        ("logistic lambda=0", fit_logistic(0.0, 300, 0.1)),
        ("logistic lambda=1e-4", fit_logistic(1e-4, 300, 0.1)),
        ("logistic lambda=1e-3", fit_logistic(1e-3, 300, 0.1)),
    ]
    for label, fit in experiments:
        t = time.time()
        f1, acc = cross_validate(x, y01, names, fit)
        print(f"{label:<22} F1={f1:.4f} acc={acc:.4f} ({time.time() - t:.0f}s)")
        sys.stdout.flush()
