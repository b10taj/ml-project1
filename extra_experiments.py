"""Extra experiments on the final feature set: stronger regularization and class weights.

Output: results/extra_experiments.txt

Class weights scale the loss of positive samples; they need a weighted gradient, which
implementations.py does not provide, so a small gradient descent is defined here.
"""

import time

import numpy as np

from compare_models import K, load_clean_cache
from evaluation import best_threshold, f1_score, k_fold_indices, open_log
from implementations import reg_logistic_regression, sigmoid
from preprocess import fit_features, transform_features
from run import FEATURES


def weighted_logistic(y, tx, lambda_, max_iters, gamma, pos_weight):
    """Gradient descent for logistic regression where positives weigh pos_weight."""
    sample_weight = np.where(y == 1, pos_weight, 1.0)
    sample_weight = sample_weight / sample_weight.mean()
    w = np.zeros(tx.shape[1])
    for _ in range(max_iters):
        p = sigmoid(tx @ w)
        grad = tx.T @ ((p - y) * sample_weight) / len(y) + 2 * lambda_ * w
        w -= gamma * grad
    return w


EXPERIMENTS = [
    (
        "lambda=1e-2",
        lambda y, tx: reg_logistic_regression(
            y, tx, 1e-2, np.zeros(tx.shape[1]), 300, 0.3
        )[0],
    ),
    (
        "lambda=1e-1",
        lambda y, tx: reg_logistic_regression(
            y, tx, 1e-1, np.zeros(tx.shape[1]), 300, 0.3
        )[0],
    ),
    ("class weight 2", lambda y, tx: weighted_logistic(y, tx, 1e-3, 600, 0.1, 2.0)),
    ("class weight 4", lambda y, tx: weighted_logistic(y, tx, 1e-3, 600, 0.1, 4.0)),
]

if __name__ == "__main__":
    _, x, y01, names = load_clean_cache()
    log = open_log(
        "extra_experiments.txt",
        f"Extra experiments on the final feature set, {K}-fold CV",
    )

    folds = k_fold_indices(len(y01), K)
    scores = {label: [] for label, _ in EXPERIMENTS}
    for k in range(K):
        t = time.time()
        val = folds[k]
        train = np.concatenate([folds[i] for i in range(K) if i != k])
        params = fit_features(x[train], names, y=y01[train], **FEATURES)
        tx_train = transform_features(x[train], params).astype(np.float64)
        tx_val = transform_features(x[val], params).astype(np.float64)
        for label, fit in EXPERIMENTS:
            w = fit(y01[train], tx_train)
            threshold, _ = best_threshold(y01[train], tx_train @ w)
            f1 = f1_score(y01[val], (tx_val @ w >= threshold).astype(int))
            scores[label].append(f1)
            log(f"fold {k} {label}: F1={f1:.4f} ({time.time() - t:.0f}s)")
        del tx_train, tx_val  # free memory before building the next fold

    log()
    log("Mean CV F1 +- std (compare with table4_convergence.txt)")
    for label, f1s in scores.items():
        log(f"{label:<16} {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
