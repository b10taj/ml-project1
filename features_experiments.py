"""Table 3 of the report: CV F1 with extra features on top of the all-columns baseline.

    python features_experiments.py      round 1 (Table 3): gamma=0.1, 300 steps
    python features_experiments.py 2    round 2: gamma=0.3, 300 steps, size of the
                                        extra feature set (supports the final choice)
Output: results/table3_features.txt or results/features_round2.txt
"""

import sys
import time

from compare_models import K, cross_validate, fit_logistic, load_clean_cache
from evaluation import open_log

LAMBDA, MAX_ITERS = 1e-3, 300

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
        "flags >= 2%, top 20 (final)",
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
    _, x, y01, names = load_clean_cache()

    second_round = len(sys.argv) > 1 and sys.argv[1] == "2"
    gamma = 0.3 if second_round else 0.1
    fit = fit_logistic(LAMBDA, MAX_ITERS, gamma)
    experiments = ROUND_2 if second_round else ROUND_1
    log = open_log(
        "features_round2.txt" if second_round else "table3_features.txt",
        f"{'Feature round 2' if second_round else 'Table 3 - feature engineering'}, "
        f"{K}-fold CV, logistic lambda={LAMBDA}, gamma={gamma}, {MAX_ITERS} steps, "
        "all non-constant columns kept",
    )
    base = {"max_nan_rate": 1.0}
    for label, kwargs in experiments:
        t = time.time()
        f1, sd, acc, dim = cross_validate(x, y01, names, fit, **base, **kwargs)
        log(
            f"{label:<36} F1={f1:.4f} +- {sd:.4f} acc={acc:.4f} "
            f"dim={dim} ({time.time() - t:.0f}s)"
        )
