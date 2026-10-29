"""Table 2 of the report: CV F1 when one preprocessing choice is changed.

Reference configuration: initial features, logistic lambda=1e-3, gamma=0.1, 300 steps.
Output: results/table2_ablation.txt
"""

import time

from compare_models import K, cross_validate, fit_logistic, load_clean_cache
from evaluation import open_log

LAMBDA, MAX_ITERS, GAMMA = 1e-3, 300, 0.1

if __name__ == "__main__":
    x_raw, x_clean, y01, names = load_clean_cache(keep_raw=True)
    fit = fit_logistic(LAMBDA, MAX_ITERS, GAMMA)
    log = open_log(
        "table2_ablation.txt",
        f"Table 2 - ablations, {K}-fold CV, logistic lambda={LAMBDA}, "
        f"gamma={GAMMA}, {MAX_ITERS} steps",
    )

    experiments = [
        ("initial pipeline", x_clean, {}),
        ("no special-code cleaning", x_raw, {}),
        ("no threshold tuning (0.5)", x_clean, {"tune_threshold": False}),
        ("max NaN rate 0.5", x_clean, {"max_nan_rate": 0.5}),
        ("max NaN rate 0.95", x_clean, {"max_nan_rate": 0.95}),
        ("keep all non-constant columns", x_clean, {"max_nan_rate": 1.0}),
        ("one-hot (<= 8 levels)", x_clean, {"max_categories": 8}),
    ]
    for label, x, kwargs in experiments:
        t = time.time()
        f1, sd, acc, dim = cross_validate(x, y01, names, fit, **kwargs)
        log(
            f"{label:<32} F1={f1:.4f} +- {sd:.4f} acc={acc:.4f} "
            f"dim={dim} ({time.time() - t:.0f}s)"
        )
