"""Data cleaning and feature processing (NumPy only).

Pipeline:
1. find_special_codes / apply_special_codes: BRFSS "don't know / refused" codes -> NaN
2. fit_features: choose columns, imputation values, scaling, one-hot levels (train data only)
3. transform_features: apply the fitted processing to any data (train, validation, test)
"""

import numpy as np

# Interview dates / sampling identifiers: no medical meaning
ADMIN_COLUMNS = [
    "FMONTH",
    "IDATE",
    "IMONTH",
    "IDAY",
    "IYEAR",
    "DISPCODE",
    "SEQNO",
    "_PSU",
]
# Number-of-days questions where 88 means "none"
DAYS_COLUMNS = ["PHYSHLTH", "MENTHLTH", "POORHLTH"]
# (refused, don't know) code pairs used by BRFSS
CODE_PAIRS = ((9, 7), (99, 77), (999, 777))


def read_column_names(data_path):
    """Return the feature names of x_train.csv (without the 'Id' column)."""
    with open(data_path + "x_train.csv") as f:
        return np.array(f.readline().strip().split(",")[1:])


def find_special_codes(x):
    """Find, per column, the codes meaning "refused" / "don't know".

    Heuristic: a column whose maximum is 9, 99 or 999 uses that value as "refused";
    the matching 7, 77 or 777 is "don't know" if present.

    Returns:
        dict {column index: list of codes}
    """
    rules = {}
    for j in range(x.shape[1]):
        col = x[:, j]
        values = np.unique(col[~np.isnan(col)])
        if len(values) == 0:
            continue
        codes = []
        for refused, dont_know in CODE_PAIRS:
            if values.max() == refused:
                codes.append(refused)
                if dont_know in values:
                    codes.append(dont_know)
        if codes:
            rules[j] = codes
    return rules


def apply_special_codes(x, rules, names):
    """Return a copy of x where special codes are NaN and 88 -> 0 in the days columns."""
    x = x.copy()
    for j, codes in rules.items():
        x[np.isin(x[:, j], codes), j] = np.nan
    for name in DAYS_COLUMNS:
        j = int(np.where(names == name)[0][0])
        x[x[:, j] == 88, j] = 0
    return x


def fit_features(x, names, max_nan_rate=0.8, max_categories=0):
    """Learn the feature processing on (cleaned) training data.

    Args:
        x: cleaned data, shape (N, D), NaN for missing
        names: column names
        max_nan_rate: columns with a larger fraction of NaN are dropped
        max_categories: columns with 3..max_categories distinct values are one-hot
            encoded (NaN becomes its own level). 0 disables one-hot encoding.

    Returns:
        dict of parameters for transform_features
    """
    nan_rate = np.isnan(x).mean(axis=0)
    n_unique = np.array([len(np.unique(c[~np.isnan(c)])) for c in x.T])
    keep = [
        j
        for j in range(x.shape[1])
        if nan_rate[j] <= max_nan_rate
        and n_unique[j] > 1
        and names[j] not in ADMIN_COLUMNS
    ]
    onehot = [j for j in keep if 2 < n_unique[j] <= max_categories]
    numeric = [j for j in keep if j not in onehot]

    median = np.nanmedian(x[:, numeric], axis=0)
    filled = np.where(np.isnan(x[:, numeric]), median, x[:, numeric])
    mean, std = filled.mean(axis=0), filled.std(axis=0)
    std[std == 0] = 1.0

    levels = {}
    for j in onehot:
        values = np.unique(x[:, j][~np.isnan(x[:, j])])
        levels[j] = values[1:]  # first level is the reference, avoids collinearity
    return {
        "numeric": numeric,
        "median": median,
        "mean": mean,
        "std": std,
        "levels": levels,
    }


def transform_features(x, params):
    """Apply the fitted processing. Returns float32 features with a bias column first."""
    numeric = params["numeric"]
    z = np.where(np.isnan(x[:, numeric]), params["median"], x[:, numeric])
    z = (z - params["mean"]) / params["std"]
    blocks = [np.ones((x.shape[0], 1)), z]
    for j, levels in params["levels"].items():
        col = x[:, j]
        block = (col[:, None] == levels[None, :]).astype(np.float64)
        blocks.append(block)
        blocks.append(np.isnan(col)[:, None].astype(np.float64))  # "missing" level
    return np.hstack(blocks).astype(np.float32)
