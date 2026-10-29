"""Table 4 of the report: learning rate and number of steps on the final feature set.

Features are built once per fold; training continues from the previous weights, so
one run gives the F1 at 300, 600 and 900 steps. Each finished fold is saved in
results/checkpoints/, so an interrupted run resumes at the next fold.
Output: results/table4_convergence.txt
"""

import time

import numpy as np

from compare_models import K, load_clean_cache
from evaluation import (
    best_threshold,
    f1_score,
    k_fold_indices,
    load_checkpoint,
    open_log,
    save_checkpoint,
)
from implementations import reg_logistic_regression
from preprocess import fit_features, transform_features
from run import FEATURES

LAMBDA = 1e-3
GAMMAS = [0.1, 0.3]
CHUNK, N_CHUNKS = 300, 3

if __name__ == "__main__":
    _, x, y01, names = load_clean_cache()
    log = open_log(
        "table4_convergence.txt",
        f"Table 4 - convergence on the final feature set, {K}-fold CV, "
        f"logistic lambda={LAMBDA}",
    )

    folds = k_fold_indices(len(y01), K)
    scores = {}
    for k in range(K):
        done = load_checkpoint("table4", k)
        if done is not None:
            for line in done["lines"]:
                log(line)
            for gamma, steps, f1 in done["scores"]:
                scores.setdefault((gamma, steps), []).append(f1)
            continue
        t = time.time()
        lines, fold_scores = [], []
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
                fold_scores.append((gamma, c * CHUNK, float(f1)))
                lines.append(
                    f"fold {k} dim={tx_train.shape[1]} gamma={gamma} "
                    f"steps={c * CHUNK} loss={loss:.4f} F1={f1:.4f} "
                    f"({time.time() - t:.0f}s)"
                )
                log(lines[-1])
        save_checkpoint("table4", k, {"lines": lines, "scores": fold_scores})
        del tx_train, tx_val  # free memory before building the next fold

    log()
    log("Mean CV F1 +- std across folds")
    for (gamma, steps), f1s in sorted(scores.items()):
        log(f"gamma={gamma} steps={steps}: {np.mean(f1s):.4f} +- {np.std(f1s):.4f}")
