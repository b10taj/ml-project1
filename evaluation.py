"""Metrics, cross-validation helpers and result logging (NumPy only)."""

import json
import os

import numpy as np

RESULTS_DIR = "results"


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
    """Threshold on the scores maximizing F1. Returns (threshold, f1).

    Candidates are 100 quantiles of the scores between the 50th and 99.5th
    percentile, so at most half of the samples can be predicted positive.
    """
    grid = np.quantile(scores, np.linspace(0.5, 0.995, n_grid))
    f1s = [f1_score(y_true, (scores >= t).astype(int)) for t in grid]
    best = int(np.argmax(f1s))
    return grid[best], f1s[best]


def accuracy(y_true, y_pred):
    """Fraction of correct predictions."""
    return float(np.mean(y_true == y_pred))


def open_log(filename, header):
    """Return a print-like function writing to stdout and to results/<filename>.

    The file is overwritten, so each results file always matches the last run.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w") as f:
        f.write(header + "\n\n")
    print(header + "\n", flush=True)

    def log(line=""):
        print(line, flush=True)
        with open(path, "a") as f:
            f.write(line + "\n")

    return log


def load_checkpoint(name, k):
    """Return the saved results of fold k of experiment `name`, or None.

    Long experiments save each finished fold in results/checkpoints/, so an
    interrupted run restarts at the first unfinished fold. Delete the folder
    to recompute everything.
    """
    path = os.path.join(RESULTS_DIR, "checkpoints", f"{name}_fold{k}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_checkpoint(name, k, data):
    """Save the results of fold k of experiment `name` (JSON-serializable dict)."""
    folder = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}_fold{k}.json"), "w") as f:
        json.dump(data, f)
