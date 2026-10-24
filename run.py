"""Reproduces the best submission: loads data, trains the final model, writes predictions."""

import time

import numpy as np

from evaluation import best_threshold
from helpers import create_csv_submission, load_csv_data
from implementations import reg_logistic_regression
from preprocess import (
    apply_special_codes,
    find_special_codes,
    fit_features,
    read_column_names,
    transform_features,
)

DATA_PATH = "data/dataset/"
OUTPUT_PATH = "submission.csv"
SEED = 0

LAMBDA = 1e-3
MAX_ITERS = 300
GAMMA = 0.1


def main():
    """Preprocess, train regularized logistic regression, tune the threshold, predict."""
    np.random.seed(SEED)
    start = time.time()

    print("[1/5] Loading CSV files (takes a few minutes)...", flush=True)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
    names = read_column_names(DATA_PATH)
    print(
        f"      train: {x_train.shape[0]} rows x {x_train.shape[1]} features, "
        f"test: {x_test.shape[0]} rows, "
        f"positive rate: {np.mean(y_train == 1):.2%} ({time.time() - start:.0f}s)",
        flush=True,
    )

    print("[2/5] Cleaning special codes (don't know / refused -> NaN)...", flush=True)
    rules = find_special_codes(x_train)
    x_train = apply_special_codes(x_train, rules, names)
    x_test = apply_special_codes(x_test, rules, names)
    print(
        f"      {len(rules)} columns with special codes, "
        f"{np.isnan(x_train).mean():.1%} of train values are NaN",
        flush=True,
    )

    print("[3/5] Building features (drop, impute, standardize)...", flush=True)
    params = fit_features(x_train, names)
    tx_train = transform_features(x_train, params).astype(np.float64)
    tx_test = transform_features(x_test, params).astype(np.float64)
    print(
        f"      kept {len(params['numeric'])} of {x_train.shape[1]} columns "
        f"(+ bias), feature matrix {tx_train.shape}",
        flush=True,
    )

    print(
        f"[4/5] Training regularized logistic regression "
        f"(lambda={LAMBDA}, {MAX_ITERS} iterations, gamma={GAMMA})...",
        flush=True,
    )
    y01 = (y_train == 1).astype(int)  # logistic regression expects labels in {0, 1}
    w, loss = reg_logistic_regression(
        y01, tx_train, LAMBDA, np.zeros(tx_train.shape[1]), MAX_ITERS, GAMMA
    )
    print(f"      final training loss: {loss:.4f}", flush=True)

    print("[5/5] Tuning the decision threshold and predicting...", flush=True)
    threshold, f1 = best_threshold(y01, tx_train @ w)
    print(f"      threshold {threshold:.4f} (train F1 {f1:.4f})", flush=True)
    y_pred = np.where(tx_test @ w >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(
        f"Wrote {OUTPUT_PATH}: {np.mean(y_pred == 1):.1%} predicted positive "
        f"(total {time.time() - start:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
