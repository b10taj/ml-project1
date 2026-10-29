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


def fit_features(
    x,
    names,
    max_nan_rate=0.8,
    max_categories=0,
    flag_min_nan=None,
    n_top=0,
    squares=False,
    pairs=False,
    y=None,
):
    """Learn the feature processing on (cleaned) training data.

    Args:
        x: cleaned data, shape (N, D), NaN for missing
        names: column names
        max_nan_rate: columns with a larger fraction of NaN are dropped
        max_categories: columns with 3..max_categories distinct values are one-hot
            encoded (NaN becomes its own level). 0 disables one-hot encoding.
        flag_min_nan: if set, add a 0/1 "is missing" feature for each kept column
            whose NaN rate is at least this value
        n_top: number of columns most correlated with y used for squares and pairs
        squares: add squares of the n_top columns
        pairs: add pairwise products of the n_top columns
        y: labels in {0, 1}, required when n_top > 0

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
    params = {
        "numeric": numeric,
        "median": median,
        "mean": mean,
        "std": std,
        "levels": levels,
        "flags": [],
        "squares": [],
        "pairs": [],
    }

    if flag_min_nan is not None:
        params["flags"] = [j for j in numeric if nan_rate[j] >= flag_min_nan]
    if n_top > 0:
        z = (filled - mean) / std
        corr = np.abs(z.T @ (y - y.mean())) / (z.std(axis=0) + 1e-12)
        top = [int(i) for i in np.argsort(corr)[-n_top:]]
        if squares:
            params["squares"] = top
        if pairs:
            params["pairs"] = [(a, b) for a in top for b in top if a < b]
    extra = _extra_features(x, params)
    if extra.shape[1] > 0:
        params["extra_mean"] = extra.mean(axis=0)
        params["extra_std"] = np.where(extra.std(axis=0) == 0, 1.0, extra.std(axis=0))
    return params


def _extra_features(x, params):
    """Missing-value flags, squares and pairwise products (before standardization)."""
    numeric = params["numeric"]
    z = np.where(np.isnan(x[:, numeric]), params["median"], x[:, numeric])
    z = (z - params["mean"]) / params["std"]
    blocks = [np.zeros((x.shape[0], 0), dtype=np.float32)]
    if params["flags"]:
        blocks.append(np.isnan(x[:, params["flags"]]).astype(np.float32))
    if params["squares"]:
        blocks.append((z[:, params["squares"]] ** 2).astype(np.float32))
    if params["pairs"]:
        a, b = zip(*params["pairs"])
        blocks.append((z[:, list(a)] * z[:, list(b)]).astype(np.float32))
    return np.hstack(blocks)


def transform_features(x, params):
    """Apply the fitted processing. Returns float32 features with a bias column first.

    Blocks are written into one preallocated float32 matrix instead of being stacked,
    which keeps the memory peak low for the 762-column feature set.
    """
    numeric = params["numeric"]
    z = np.where(np.isnan(x[:, numeric]), params["median"], x[:, numeric])
    z = (z - params["mean"]) / params["std"]

    n_onehot = sum(len(levels) + 1 for levels in params["levels"].values())
    n_extra = len(params["flags"]) + len(params["squares"]) + len(params["pairs"])
    out = np.empty(
        (x.shape[0], 1 + len(numeric) + n_onehot + n_extra), dtype=np.float32
    )
    out[:, 0] = 1.0
    out[:, 1 : 1 + len(numeric)] = z
    del z
    c = 1 + len(numeric)
    for j, levels in params["levels"].items():
        col = x[:, j]
        out[:, c : c + len(levels)] = col[:, None] == levels[None, :]
        out[:, c + len(levels)] = np.isnan(col)  # "missing" level
        c += len(levels) + 1
    if n_extra > 0:
        extra = _extra_features(x, params)
        out[:, c:] = (extra - params["extra_mean"]) / params["extra_std"]
    return out
