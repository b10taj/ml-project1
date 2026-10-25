"""Feature engineering experiments: CV F1 with extra features on top of the baseline."""

import sys
import time

import numpy as np

from compare_models import cross_validate, fit_logistic
from preprocess import apply_special_codes, find_special_codes, read_column_names

LAMBDA = 1e-3

# Round 1 uses 300 iterations with gamma=0.1; round 2 (python features_experiments.py 2)
# uses gamma=0.3, as in the final model, and varies the size of the extra feature set.
ROUND_1 = [
    ("baseline (all columns)", {}),
    ("+ missing flags (NaN rate >= 2%)", {"flag_min_nan": 0.02}),
    ("+ squares of top 30", {"n_top": 30, "squares": True}),
    ("+ pairs of top 20", {"n_top": 20, "pairs": True}),
    (
        "+ flags, squares and pairs",
        {"flag_min_nan": 0.02, "n_top": 20, "squares": True, "pairs": True},
    ),
]
ROUND_2 = [
    (
        "flags >= 2%, top 20 (reference)",
        {"flag_min_nan": 0.02, "n_top": 20, "squares": True, "pairs": True},
    ),
    (
        "flags >= 0.1%, top 20",
        {"flag_min_nan": 0.001, "n_top": 20, "squares": True, "pairs": True},
    ),
    (
        "flags >= 2%, top 10",
        {"flag_min_nan": 0.02, "n_top": 10, "squares": True, "pairs": True},
    ),
    (
        "flags >= 2%, top 35",
        {"flag_min_nan": 0.02, "n_top": 35, "squares": True, "pairs": True},
    ),
]

if __name__ == "__main__":
    d = np.load("data/cache.npz")
    x_raw, y01 = d["x"], (d["y"] == 1).astype(int)
    names = read_column_names("data/dataset/")
    x = apply_special_codes(x_raw, find_special_codes(x_raw), names)

    second_round = len(sys.argv) > 1 and sys.argv[1] == "2"
    fit = fit_logistic(LAMBDA, 300, 0.3 if second_round else 0.1)
    experiments = ROUND_2 if second_round else ROUND_1
    base = {"max_nan_rate": 1.0}
    for label, kwargs in experiments:
        t = time.time()
        f1, acc = cross_validate(x, y01, names, fit, **base, **kwargs)
        print(f"{label:<36} F1={f1:.4f} acc={acc:.4f} ({time.time() - t:.0f}s)")
        sys.stdout.flush()
