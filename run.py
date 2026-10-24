"""Reproduces the best submission: loads data, trains the final model, writes predictions."""

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
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
    names = read_column_names(DATA_PATH)

    rules = find_special_codes(x_train)
    x_train = apply_special_codes(x_train, rules, names)
    x_test = apply_special_codes(x_test, rules, names)

    params = fit_features(x_train, names)
    tx_train = transform_features(x_train, params).astype(np.float64)
    tx_test = transform_features(x_test, params).astype(np.float64)

    y01 = (y_train == 1).astype(int)  # logistic regression expects labels in {0, 1}
    w, _ = reg_logistic_regression(
        y01, tx_train, LAMBDA, np.zeros(tx_train.shape[1]), MAX_ITERS, GAMMA
    )
    threshold, f1 = best_threshold(y01, tx_train @ w)
    print(f"Threshold {threshold:.4f} (train F1 {f1:.4f})")

    y_pred = np.where(tx_test @ w >= threshold, 1, -1)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}: {np.mean(y_pred == 1):.1%} predicted positive")


if __name__ == "__main__":
    main()
