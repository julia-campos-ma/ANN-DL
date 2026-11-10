"""
Shared module of the classification deliverable (Step 0 -- starting point).

Everything that more than one script needs lives here, so it is defined once:

- the data: ``load_split()`` and ``build_preprocessor()`` are imported from the EDA code,
  unchanged, so the train/test split, the seed and the 193-column pipeline are the EDA's;
- the validation set: carved out of the TRAINING set with the same temporal rule the EDA
  used for the test set (the last 20% of training arrivals, cut on a whole day);
- the main metric, the success criterion and the error costs, fixed before any model was
  trained (Step 0 of the report);
- small helpers: metrics, cost-based threshold, figure style, JSON results.

The test set is loaded here but only ``step5_test.py`` and ``step6_errors.py`` read it.

Run from the repository root:
    python docs/projects/classification/code/common.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EDA_CODE = ROOT / "docs" / "projects" / "eda" / "code"
sys.path.insert(0, str(EDA_CODE))

import data as eda_data  # noqa: E402  (EDA module, imported unchanged)
import pipeline as eda_pipeline  # noqa: E402

from data import CATEGORICAL_HIGH, CATEGORICAL_LOW, NUMERICAL, SEED, TARGET  # noqa: E402,F401

FIGURES = HERE.parent / "figures"
RESULTS = HERE.parent / "results"
FIGURES.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

COLORS = {0: "#4C72B0", 1: "#DD8452"}  # not canceled / canceled, same as the EDA
MODEL_COLORS = {"trivial": "#BBBBBB", "logistic": "#4C72B0", "boosting": "#55A868", "mlp": "#DD8452"}

# --8<-- [start:protocol]
# ---------------------------------------------------------------------------------------
# Step 0 -- decided BEFORE training any model.
# ---------------------------------------------------------------------------------------
VAL_SHARE = 0.20            # last 20% of the training arrivals -> validation (temporal)
SEEDS = [42, 43, 44, 45, 46]  # Step 5: the chosen network is retrained with 5 seeds

MAIN_METRIC = "average_precision"   # PR-AUC of the positive class (canceled)
# Success criterion: on the test set, the network's mean average precision over the 5
# seeds must beat the linear baseline (logistic regression, same pipeline) by at least
# 0.01, and the gain must be larger than 2 standard deviations across seeds.
MIN_GAIN_OVER_LINEAR = 0.01

# Error costs, in "room-nights of revenue", used to set the decision threshold on
# validation. A false positive (predicted cancel, guest shows up) is what an overbooking
# policy turns into a walked guest: a room elsewhere, transport and compensation, about
# twice the nightly rate. A false negative (predicted show, guest cancels) is, at most, an
# empty room: one nightly rate.
COST_FP = 2.0
COST_FN = 1.0
# --8<-- [end:protocol]


def load_frames():
    """(train_fit, val, test, cut_val, cut_test): the EDA split plus a temporal validation."""
    train, test, cut_test = eda_data.load_split()
    order = train.sort_values("arrival_date", kind="stable")
    cumulative = np.arange(1, len(order) + 1) / len(order)
    cut_val = order["arrival_date"].iloc[np.searchsorted(cumulative, 1 - VAL_SHARE)]
    is_val = train["arrival_date"] >= cut_val
    fit = train.loc[~is_val].reset_index(drop=True)
    val = train.loc[is_val].reset_index(drop=True)
    return fit, val, test, cut_val, cut_test


def build_preprocessor(drop=()):
    """The EDA preprocessor; with ``drop`` the listed raw features are left out.

    ``drop=()`` returns exactly ``eda_pipeline.build_preprocessor()``. The variant only
    exists for the feature ablation of Step 4 (``previous_cancellations`` and
    ``booking_changes``, the two risks the EDA asked us to measure).
    """
    if not drop:
        return eda_pipeline.build_preprocessor()
    log = [c for c in eda_pipeline.LOG_FEATURES if c not in drop]
    lin = [c for c in eda_pipeline.LINEAR_FEATURES if c not in drop]
    low = [c for c in CATEGORICAL_LOW if c not in drop]
    high = [c for c in CATEGORICAL_HIGH if c not in drop]
    return ColumnTransformer(
        transformers=[
            ("num_log", eda_pipeline.numeric_branch(log=True), log),
            ("num_lin", eda_pipeline.numeric_branch(log=False), lin),
            ("cat_low", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), low),
            ("cat_high", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(min_frequency=eda_pipeline.MIN_FREQUENCY,
                                         handle_unknown="infrequent_if_exist", sparse_output=False)),
            ]), high),
        ],
        verbose_feature_names_out=False,
    )


def prepare(drop=(), with_test=False):
    """Fit the preprocessor on the training part ONLY and transform the other splits.

    Returns a dict with float32 matrices, int labels, the raw frames and the fitted
    preprocessor. The test matrix is only built when ``with_test=True`` (Steps 5-6).
    """
    fit, val, test, cut_val, cut_test = load_frames()
    pre = build_preprocessor(drop)
    out = {
        "X_fit": pre.fit_transform(fit).astype(np.float32),
        "X_val": pre.transform(val).astype(np.float32),
        "y_fit": fit[TARGET].to_numpy(np.int64),
        "y_val": val[TARGET].to_numpy(np.int64),
        "fit": fit, "val": val, "pre": pre,
        "names": pre.get_feature_names_out(), "cut_val": cut_val, "cut_test": cut_test,
    }
    if with_test:
        out["X_test"] = pre.transform(test).astype(np.float32)
        out["y_test"] = test[TARGET].to_numpy(np.int64)
        out["test"] = test
    return out


# --8<-- [start:threshold]
def expected_cost(y, p, t):
    """Average cost per booking of flagging p >= t, with the Step 0 costs."""
    pred = p >= t
    fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1))
    return (COST_FP * fp + COST_FN * fn) / len(y)


def cost_threshold(y_val, p_val, grid=np.linspace(0.01, 0.99, 99)):
    """Threshold that minimizes the expected cost ON VALIDATION (never on test)."""
    costs = np.array([expected_cost(y_val, p_val, t) for t in grid])
    return float(grid[int(np.argmin(costs))]), costs
# --8<-- [end:threshold]


def metrics(y, p, t):
    """Every task metric of the report for probabilities p and threshold t."""
    pred = (p >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float((pred == y).mean()),
        "precision_0": float(prec[0]), "recall_0": float(rec[0]), "f1_0": float(f1[0]),
        "precision_1": float(prec[1]), "recall_1": float(rec[1]), "f1_1": float(f1[1]),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "cost": float(expected_cost(y, p, t)), "threshold": float(t),
        "log_loss": float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1)))),
        "brier": float(np.mean((p - y) ** 2)),
    }


def save_json(name, obj):
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)


def section(title):
    print("\n" + "=" * 88 + f"\n{title}\n" + "=" * 88)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.3, "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "legend.fontsize": 8,
    })
    return plt


if __name__ == "__main__":
    d = prepare()
    fit, val = d["fit"], d["val"]
    print(f"train (fit): {len(fit)} rows, arrivals {fit.arrival_date.min().date()} .. "
          f"{fit.arrival_date.max().date()}, cancel rate {fit[TARGET].mean():.4f}")
    print(f"validation:  {len(val)} rows, arrivals {val.arrival_date.min().date()} .. "
          f"{val.arrival_date.max().date()}, cancel rate {val[TARGET].mean():.4f}")
    print(f"X_fit {d['X_fit'].shape}  X_val {d['X_val'].shape}")
