"""Ablation study: CV F1 of the final model when one preprocessing choice is changed."""

import sys
import time

import numpy as np

from compare_models import cross_validate, fit_logistic
from preprocess import apply_special_codes, find_special_codes, read_column_names

LAMBDA, MAX_ITERS, GAMMA = 1e-3, 300, 0.1

if __name__ == "__main__":
    d = np.load("data/cache.npz")
    x_raw, y01 = d["x"], (d["y"] == 1).astype(int)
    names = read_column_names("data/dataset/")
    x_clean = apply_special_codes(x_raw, find_special_codes(x_raw), names)
    fit = fit_logistic(LAMBDA, MAX_ITERS, GAMMA)

    experiments = [
        ("full pipeline", x_clean, {}),
        ("no special-code cleaning", x_raw, {}),
        ("no threshold tuning (0.5)", x_clean, {"tune_threshold": False}),
        ("max NaN rate 0.5", x_clean, {"max_nan_rate": 0.5}),
        ("max NaN rate 0.95", x_clean, {"max_nan_rate": 0.95}),
        ("keep all non-constant columns", x_clean, {"max_nan_rate": 1.0}),
        ("one-hot (<= 8 levels)", x_clean, {"max_categories": 8}),
    ]
    for label, x, kwargs in experiments:
        t = time.time()
        f1, acc = cross_validate(x, y01, names, fit, **kwargs)
        print(f"{label:<32} F1={f1:.4f} acc={acc:.4f} ({time.time() - t:.0f}s)")
        sys.stdout.flush()
