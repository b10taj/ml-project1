"""Figure 1 of the report: validation F1, precision and recall versus the decision threshold.

Final model (run.py settings), 4-fold CV. The curves are computed on the held-out
folds; the marked threshold is the one chosen on each training fold (mean over folds).
Matplotlib is only used to draw the figure.
Each finished fold is saved in results/checkpoints/, so an interrupted run resumes.
Output: figures/threshold_curve.pdf and results/threshold_curve.txt
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_models import K, load_clean_cache
from evaluation import (
    best_threshold,
    f1_score,
    k_fold_indices,
    load_checkpoint,
    open_log,
    save_checkpoint,
)
from implementations import reg_logistic_regression, sigmoid
from preprocess import fit_features, transform_features
from run import FEATURES, GAMMA, LAMBDA, MAX_ITERS

PROBS = np.linspace(0.02, 0.7, 69)


def precision_recall(y_true, y_pred):
    """Precision and recall of the positive class. Labels in {0, 1}."""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    p = tp / max(np.sum(y_pred == 1), 1)
    r = tp / max(np.sum(y_true == 1), 1)
    return p, r


if __name__ == "__main__":
    _, x, y01, names = load_clean_cache()
    log = open_log(
        "threshold_curve.txt",
        f"Figure 1 - validation metrics vs probability threshold, final model, {K}-fold CV",
    )

    folds = k_fold_indices(len(y01), K)
    f1, prec, rec, chosen = [], [], [], []
    for k in range(K):
        done = load_checkpoint("threshold", k)
        if done is not None:
            chosen.append(done["chosen"])
            f1.append(done["f1"])
            prec.append(done["prec"])
            rec.append(done["rec"])
            log(f"fold {k}: chosen threshold p={chosen[-1]:.3f}")
            continue
        val = folds[k]
        train = np.concatenate([folds[i] for i in range(K) if i != k])
        params = fit_features(x[train], names, y=y01[train], **FEATURES)
        tx_train = transform_features(x[train], params).astype(np.float64)
        tx_val = transform_features(x[val], params).astype(np.float64)
        w, _ = reg_logistic_regression(
            y01[train], tx_train, LAMBDA, np.zeros(tx_train.shape[1]), MAX_ITERS, GAMMA
        )
        t, _ = best_threshold(y01[train], tx_train @ w)
        chosen.append(float(sigmoid(t)))
        p_val = sigmoid(tx_val @ w)
        preds = [(p_val >= p).astype(int) for p in PROBS]
        f1.append([f1_score(y01[val], yp) for yp in preds])
        pr = [precision_recall(y01[val], yp) for yp in preds]
        prec.append([a for a, _ in pr])
        rec.append([b for _, b in pr])
        log(f"fold {k}: chosen threshold p={chosen[-1]:.3f}")
        save_checkpoint(
            "threshold",
            k,
            {"chosen": chosen[-1], "f1": [float(v) for v in f1[-1]],
             "prec": [float(v) for v in prec[-1]], "rec": [float(v) for v in rec[-1]]},
        )
        del tx_train, tx_val  # free memory before building the next fold

    f1, prec, rec = (np.mean(a, axis=0) for a in (f1, prec, rec))
    p_star = float(np.mean(chosen))
    i_half = int(np.argmin(np.abs(PROBS - 0.5)))
    log(f"mean chosen threshold p={p_star:.3f}")
    log(f"validation F1 at p=0.5: {f1[i_half]:.4f}")
    log(f"best validation F1 on the grid: {f1.max():.4f} at p={PROBS[f1.argmax()]:.2f}")
    log()
    log("p      F1      precision recall")
    for p, a, b, c in zip(PROBS, f1, prec, rec):
        log(f"{p:.2f}  {a:.4f}  {b:.4f}    {c:.4f}")

    os.makedirs("figures", exist_ok=True)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "mathtext.fontset": "cm"})
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    ax.plot(PROBS, f1, color="#1f4e9c", lw=1.8, label="F$_1$")
    ax.plot(PROBS, prec, color="#c0504d", lw=1.1, ls="--", label="Precision")
    ax.plot(PROBS, rec, color="#4f8f3a", lw=1.1, ls=":", label="Recall")
    ax.axvline(p_star, color="0.25", lw=0.8)
    ax.axvline(0.5, color="0.6", lw=0.8, ls="--")
    ax.annotate(
        f"tuned ({p_star:.2f})", (p_star, 0.9), xytext=(4, 0),
        textcoords="offset points", fontsize=7,
    )
    ax.annotate(
        "default (0.5)", (0.5, 0.9), xytext=(4, 0),
        textcoords="offset points", fontsize=7, color="0.4",
    )
    ax.set_xlabel("Decision threshold on predicted probability")
    ax.set_xlim(PROBS[0], PROBS[-1])
    ax.set_ylim(0, 1)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(pad=0.3)
    fig.savefig("figures/threshold_curve.pdf")
    print("Wrote figures/threshold_curve.pdf")
