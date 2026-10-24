"""Parse the CSV files once and cache them as a NumPy archive (data/cache.npz).

Parsing the CSVs takes several minutes, loading the cache takes seconds.
The cache is only used for experiments and notebooks; run.py reads the CSVs directly.
"""

import numpy as np

from helpers import load_csv_data

x, x_test, y, ids, test_ids = load_csv_data("data/dataset/")
np.savez("data/cache.npz", x=x, x_test=x_test, y=y, ids=ids, test_ids=test_ids)
