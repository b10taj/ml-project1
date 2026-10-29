"""Table 1 of the report: model comparison with k-fold CV (own implementations only).

Uses the initial feature set (default fit_features arguments: columns with more
than 80% missing values are dropped, no extra features).
Output: results/table1_model_comparison.txt
"""

import time

import numpy as np

from evaluation import accuracy, best_threshold, f1_score, k_fold_indices, open_log
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
    """K-fold CV. Returns (mean F1, std F1 across folds, mean accuracy, n_features).

    Feature processing and the threshold are fitted on each training fold only.
    The threshold is tuned on the training fold, or fixed at 0 (probability 0.5)
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
        dim = tx_train.shape[1]
        del tx_train, tx_val  # free memory before building the next fold
    return np.mean(f1s), np.std(f1s), np.mean(accs), dim


def load_clean_cache(keep_raw=False):
    """Load data/cache.npz and map special codes to NaN.

    Returns (x_raw, x, y01, names); x_raw is None unless keep_raw (saves memory).
    """
    d = np.load("data/cache.npz")
    x_raw, y01 = d["x"], (d["y"] == 1).astype(int)
    names = read_column_names("data/dataset/")
    x = apply_special_codes(x_raw, find_special_codes(x_raw), names)
    return (x_raw if keep_raw else None), x, y01, names


if __name__ == "__main__":
    _, x, y01, names = load_clean_cache()
    log = open_log(
        "table1_model_comparison.txt",
        f"Table 1 - model comparison, initial features, {K}-fold CV, "
        "threshold tuned on each training fold",
    )
    log(
        f"{'always negative':<26} F1=0.0000 acc={1 - y01.mean():.4f} "
        "(no model, for reference)"
    )
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
        f1, sd, acc, dim = cross_validate(x, y01, names, fit)
        log(
            f"{label:<26} F1={f1:.4f} +- {sd:.4f} acc={acc:.4f} "
            f"dim={dim} ({time.time() - t:.0f}s)"
        )
