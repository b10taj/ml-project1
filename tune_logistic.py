"""Tune the learning rate and number of iterations of logistic regression with k-fold CV.

Features are built once per fold; training continues from the previous weights, so
one run gives the F1 at several iteration counts.
"""

import sys
import time

import numpy as np

from compare_models import K
from evaluation import best_threshold, f1_score, k_fold_indices
from implementations import reg_logistic_regression
from preprocess import (
    apply_special_codes,
    find_special_codes,
    fit_features,
    read_column_names,
    transform_features,
)

LAMBDA = 1e-3
GAMMAS = [0.1, 0.3]
CHUNK, N_CHUNKS = 300, 3
FEATURES = {
    "max_nan_rate": 1.0,
    "flag_min_nan": 0.02,
    "n_top": 20,
    "squares": True,
    "pairs": True,
}

if __name__ == "__main__":
    d = np.load("data/cache.npz")
    x_raw, y01 = d["x"], (d["y"] == 1).astype(int)
    names = read_column_names("data/dataset/")
    x = apply_special_codes(x_raw, find_special_codes(x_raw), names)

    folds = k_fold_indices(len(y01), K)
    scores = {}
    for k in range(K):
        t = time.time()
        val = folds[k]
        train = np.concatenate([folds[i] for i in range(K) if i != k])
        params = fit_features(x[train], names, y=y01[train], **FEATURES)
        tx_train = transform_features(x[train], params).astype(np.float64)
        tx_val = transform_features(x[val], params).astype(np.float64)
        for gamma in GAMMAS:
            w = np.zeros(tx_train.shape[1])
            for c in range(1, N_CHUNKS + 1):
                w, loss = reg_logistic_regression(
                    y01[train], tx_train, LAMBDA, w, CHUNK, gamma
                )
                threshold, _ = best_threshold(y01[train], tx_train @ w)
                f1 = f1_score(y01[val], (tx_val @ w >= threshold).astype(int))
                scores.setdefault((gamma, c * CHUNK), []).append(f1)
                print(
                    f"fold {k} gamma={gamma} iters={c * CHUNK} loss={loss:.4f} "
                    f"F1={f1:.4f} ({time.time() - t:.0f}s)",
                    flush=True,
                )

    print("\nMean CV F1")
    for (gamma, iters), f1s in sorted(scores.items()):
        print(f"gamma={gamma} iters={iters}: {np.mean(f1s):.4f}")
