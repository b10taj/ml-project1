"""Reproduces the best submission: loads data, trains the final model, writes predictions."""

import numpy as np

from helpers import create_csv_submission, load_csv_data

DATA_PATH = "data/"
OUTPUT_PATH = "submission.csv"
SEED = 0


def main():
    np.random.seed(SEED)
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)
    # Pipeline to fill in: preprocessing -> training -> threshold -> predictions in {-1, 1}
    raise NotImplementedError


if __name__ == "__main__":
    main()
