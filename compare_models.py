"""Model and hyperparameter comparison with k-fold CV, using our own implementations.py."""

import itertools
import multiprocessing
import os
import time

WORKERS = 4  # parallel processes, ~3 GB of RAM each
if WORKERS > 1:
    # Set before numpy is imported: one BLAS thread per worker.
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")

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
RESULTS_PATH = "model_comparison_convergence.txt"
RIDGE_LAMBDAS = []
LOGISTIC_LAMBDAS = [1e-3]
LOGISTIC_GAMMAS = [0.4, 0.2]
LOGISTIC_ITERS = [3000, 1500, 600]  # longest first so the workers finish together


def fit_ridge(lambda_):
    """Return a fit function (y, tx) -> w using ridge_regression."""
    return lambda y, tx: ridge_regression(y, tx, lambda_)[0]


def fit_logistic(lambda_, max_iters, gamma):
    """Return a fit function (y, tx) -> w using reg_logistic_regression."""
    return lambda y, tx: reg_logistic_regression(
        y, tx, lambda_, np.zeros(tx.shape[1]), max_iters, gamma
    )[0]


def cross_validate(
    x, y01, names, fit, max_categories=0, max_nan_rate=0.8, tune_threshold=True
):
    """Mean F1 and accuracy over K folds.

    The threshold is tuned on each training fold, or fixed at 0 (probability 0.5)
    if tune_threshold is False.
    """
    folds = make_folds(x, y01, names, max_nan_rate, max_categories)
    return evaluate(folds, fit, tune_threshold)


def make_folds(x, y01, names, max_nan_rate=0.8, max_categories=0):
    """Per fold (tx_train, y_train, tx_val, y_val), features fitted on the training part only."""
    splits = k_fold_indices(len(y01), K)
    folds = []
    for k in range(K):
        val = splits[k]
        train = np.concatenate([splits[i] for i in range(K) if i != k])
        params = fit_features(x[train], names, max_nan_rate, max_categories)
        tx_train = transform_features(x[train], params).astype(np.float64)
        tx_val = transform_features(x[val], params).astype(np.float64)
        folds.append((tx_train, y01[train], tx_val, y01[val]))
    return folds


def evaluate(folds, fit, tune_threshold=True):
    """Mean F1 and accuracy of a fit function over prepared folds."""
    f1s, accs = [], []
    for tx_train, y_train, tx_val, y_val in folds:
        w = fit(y_train, tx_train)
        threshold = 0.0
        if tune_threshold:
            threshold, _ = best_threshold(y_train, tx_train @ w)
        pred = (tx_val @ w >= threshold).astype(int)
        f1s.append(f1_score(y_val, pred))
        accs.append(accuracy(y_val, pred))
    return np.mean(f1s), np.mean(accs)


_FOLDS = None


def make_fit(method, args):
    """Build a fit function from a picklable experiment description."""
    return fit_ridge(*args) if method == "ridge" else fit_logistic(*args)


def init_worker():
    """Compute the CV folds once per process; every experiment reuses them."""
    global _FOLDS
    t = time.time()
    d = np.load("data/cache.npz")
    x, y01 = d["x"], (d["y"] == 1).astype(int)
    names = read_column_names("data/dataset/")
    x = apply_special_codes(x, find_special_codes(x), names)
    _FOLDS = make_folds(x, y01, names)
    print(f"Folds ready ({time.time() - t:.0f}s)", flush=True)


def run_experiment(job):
    """Run one experiment on the cached folds. Returns (label, f1, acc, seconds)."""
    label, method, args = job
    t = time.time()
    f1, acc = evaluate(_FOLDS, make_fit(method, args))
    return label, f1, acc, time.time() - t


if __name__ == "__main__":
    experiments = [
        (f"ridge lambda={lam:g}", "ridge", (lam,)) for lam in RIDGE_LAMBDAS
    ] + [
        (
            f"logistic lambda={lam:g} gamma={gamma:g} iters={iters}",
            "logistic",
            (lam, iters, gamma),
        )
        for lam, gamma, iters in itertools.product(
            LOGISTIC_LAMBDAS, LOGISTIC_GAMMAS, LOGISTIC_ITERS
        )
    ]

    if WORKERS > 1:
        pool = multiprocessing.Pool(min(WORKERS, len(experiments)), init_worker)
        outcomes = pool.imap_unordered(run_experiment, experiments)
    else:
        init_worker()
        outcomes = map(run_experiment, experiments)

    results = []
    for i, (label, f1, acc, secs) in enumerate(outcomes, 1):
        results.append((f1, acc, label))
        print(
            f"[{i}/{len(experiments)}] {label:<45} F1={f1:.4f} acc={acc:.4f} "
            f"({secs:.0f}s)",
            flush=True,
        )
    if WORKERS > 1:
        pool.close()
        pool.join()

    results.sort(reverse=True)
    lines = [f"{K}-fold CV, threshold tuned on each training fold", ""]
    lines += [f"{label:<50} F1={f1:.4f} acc={acc:.4f}" for f1, acc, label in results]
    print("\nRanked by F1:")
    print("\n".join(lines))
    with open(RESULTS_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")
