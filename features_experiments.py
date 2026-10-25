"""Feature engineering experiments: CV F1 with extra features on top of the baseline."""

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
    x = apply_special_codes(x_raw, find_special_codes(x_raw), names)
    fit = fit_logistic(LAMBDA, MAX_ITERS, GAMMA)

    base = {"max_nan_rate": 1.0}
    experiments = [
        ("baseline (all columns)", {}),
        ("+ missing flags (NaN rate >= 2%)", {"flag_min_nan": 0.02}),
        ("+ squares of top 30", {"n_top": 30, "squares": True}),
        ("+ pairs of top 20", {"n_top": 20, "pairs": True}),
        (
            "+ flags, squares and pairs",
            {"flag_min_nan": 0.02, "n_top": 20, "squares": True, "pairs": True},
        ),
    ]
    for label, kwargs in experiments:
        t = time.time()
        f1, acc = cross_validate(x, y01, names, fit, **base, **kwargs)
        print(f"{label:<36} F1={f1:.4f} acc={acc:.4f} ({time.time() - t:.0f}s)")
        sys.stdout.flush()
