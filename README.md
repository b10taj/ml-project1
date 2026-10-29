# EPFL CS-433 Machine Learning — Project 1

**Task.** Predict self-reported coronary heart disease (MICHD) from BRFSS 2015 survey answers, using only NumPy implementations of linear models.

**Team.** Badis Taj, Zeineb Sellami, Ilias Taj — **Repository:** https://github.com/b10taj/ml-project1

## Results

| Evaluation | F1-score | Accuracy |
| --- | ---: | ---: |
| Final model, 4-fold cross-validation (mean ± std over folds) | 0.4382 ± 0.0049 | — |
| Final model, AIcrowd public leaderboard | **0.450** | 0.881 |

Cross-validation estimates generalization on the training data; the AIcrowd score is computed by the platform on the unseen test set. Only 8.83% of training respondents are positive, so F1 is the primary metric (always predicting "no disease" gives 91.2% accuracy and F1 = 0).

## Requirements

- Python 3 (tested with 3.13 and 3.14) and NumPy (tested with 2.4 and 2.5).
- The model, preprocessing and all experiments use NumPy only, as required. Matplotlib is used only to draw the report figure (`threshold_curve.py`) and the plots of the exploration notebook.

## Data

Download the data from the [AIcrowd challenge page](https://www.aicrowd.com/challenges/epfl-machine-learning-project-1) and place the three CSV files as follows (the `data/` folder is git-ignored):

```text
data/
  dataset/
    x_train.csv
    y_train.csv
    x_test.csv
```

## Reproduce the final submission

From the repository root:

```bash
python run.py
```

`run.py` loads the CSVs, cleans and transforms the data, trains the final regularized logistic regression, tunes the decision threshold and writes `submission.csv` with predictions in `{-1, +1}`. It is deterministic (zero initialization, full-batch gradient descent, seed 0) and needs no cache. It takes about 6 minutes on a laptop, about a third of it spent parsing the CSV files.

## Method

1. **Special codes.** BRFSS encodes "don't know" and "refused" as 7/9, 77/99 or 777/999. For every column whose maximum value is 9, 99 or 999, these two codes become missing values. In `PHYSHLTH`, `MENTHLTH` and `POORHLTH`, code 88 ("no days") becomes 0.
2. **Columns.** We drop 8 administrative columns (interview date, sequence numbers, sampling unit) and 6 constant columns, and keep the other 307 columns, including very incomplete ones.
3. **Imputation and scaling.** Missing values are replaced by the training median, then every column is standardized with training statistics.
4. **Extra features.** We add a 0/1 missingness indicator for the 244 columns with at least 2% missing values, plus the squares and the 190 pairwise products of the 20 columns most correlated with the label. With the bias column, the final matrix has 762 columns.
5. **Model.** We use regularized logistic regression (`reg_logistic_regression`) with `lambda = 1e-3`, `gamma = 0.3` and 600 gradient steps, starting from zero weights.
6. **Decision threshold.** Instead of using probability 0.5, we choose the threshold that maximizes F1 on the training scores (about probability 0.22 for the final model).

All steps that learn something from data (medians, means, selected columns, threshold) are fitted on the training folds only during cross-validation.

## Reproduce the experiments of the report

The experiment scripts read a NumPy cache of the CSVs, which takes about 2 minutes to build:

```bash
python make_cache.py
```

Each script then writes its output to `results/` (the files from our run are committed):

| Command | Report | Output file |
| --- | --- | --- |
| `python compare_models.py` | Table 1: ridge vs. logistic regression | `results/table1_model_comparison.txt` |
| `python ablation.py` | Table 2: preprocessing ablations | `results/table2_ablation.txt` |
| `python features_experiments.py` | Table 3: feature engineering | `results/table3_features.txt` |
| `python tune_logistic.py` | Table 4: learning rate and number of steps | `results/table4_convergence.txt` |
| `python threshold_curve.py` | Figure 1: F1, precision and recall vs. threshold | `figures/threshold_curve.pdf`, `results/threshold_curve.txt` |

Each script takes between a few minutes and about 40 minutes on a laptop. All use the same shuffled 4-fold split (seed 0). The two longest scripts (`tune_logistic.py`, `threshold_curve.py`) save each finished fold in `results/checkpoints/` (git-ignored), so an interrupted run resumes where it stopped; delete that folder to recompute from scratch.

Two additional exploratory scripts are not used in the report, and their outputs are not committed: `python features_experiments.py 2` (size of the extra feature set) and `python extra_experiments.py` (stronger regularization and class weights).

## Repository structure

| File | Content |
| --- | --- |
| `implementations.py` | The six required methods: `mean_squared_error_gd`, `mean_squared_error_sgd`, `least_squares`, `ridge_regression`, `logistic_regression`, `reg_logistic_regression` |
| `run.py` | End-to-end training and creation of `submission.csv` |
| `preprocess.py` | Special-code cleaning, column selection, imputation, scaling and extra features |
| `evaluation.py` | F1, accuracy, k-fold split, threshold search, result logging |
| `helpers.py` | CSV loading and submission writing (provided by the course) |
| `make_cache.py` | Builds `data/cache.npz` for the experiment scripts |
| `compare_models.py`, `ablation.py`, `features_experiments.py`, `tune_logistic.py`, `threshold_curve.py` | Experiments of the report (see table above) |
| `extra_experiments.py` | Exploratory experiments, not used in the report |
| `results/`, `figures/` | Outputs of the experiments |
| `notebooks/01_exploration.ipynb` | Exploratory data analysis (needs `data/cache.npz`) |

## Limitations

- The special-code rule is a heuristic. It misses other BRFSS conventions: 88 = "none" outside the three days columns (e.g. `CHILDREN`), 888 and 555 in the alcohol and diet frequency columns, 7777/9999 and metric units in `WEIGHT2`/`HEIGHT3`, and 14 = "don't know" in `_AGEG5YR`.
- Hyperparameters were chosen on the same cross-validation folds that we report, so the CV score is slightly optimistic.
- In `run.py` the threshold is tuned on the training scores of the full training set, as in each CV training fold.
