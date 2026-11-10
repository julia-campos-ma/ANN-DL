"""
Step 1 -- baselines on the same pipeline, evaluated on VALIDATION only.

- trivial: most-frequent class (always "not canceled"); its AP equals the prevalence;
- linear: L2 logistic regression, C chosen on validation from a log grid;
- optional: histogram gradient boosting (scikit-learn), a strong tabular reference.

The decision threshold of each model is the one that minimizes the Step 0 cost on
validation. The chosen hyperparameters are saved for Step 5, which refits the same
baselines and evaluates them on test together with the network.

Run from the repository root:
    python docs/projects/classification/code/step1_baselines.py
"""

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

import common
from common import SEED, section

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
HGB_GRID = [  # (learning_rate, max_leaf_nodes, max_iter): small grid around the scikit-learn default
    (0.1, 31, 100), (0.1, 31, 300), (0.05, 31, 200), (0.1, 15, 200), (0.05, 63, 100),
]


def fit_logistic(X, y, C):
    return LogisticRegression(C=C, max_iter=5000).fit(X, y)


def fit_boosting(X, y, lr, leaves, n_iter, seed=SEED):
    """No internal early stopping: its random 10% hold-out would ignore the temporal drift,
    so the number of trees is a hyperparameter chosen on the temporal validation."""
    return HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=n_iter,
                                          early_stopping=False, random_state=seed).fit(X, y)


def main():
    d = common.prepare()
    Xf, yf, Xv, yv = d["X_fit"], d["y_fit"], d["X_val"], d["y_val"]
    rows, out = [], {}

    section("Trivial baseline (most frequent class)")
    dummy = DummyClassifier(strategy="most_frequent").fit(Xf, yf)
    pv = dummy.predict_proba(Xv)[:, 1]                       # all zeros
    m = common.metrics(yv, pv, 0.5)
    rows.append({"model": "trivial (sempre 'não cancela')", "hyper": "—", **m})
    print(f"val: AP {m['average_precision']:.4f}  ROC-AUC {m['roc_auc']:.4f}  macro F1 {m['macro_f1']:.4f}  "
          f"accuracy {m['accuracy']:.4f}  cost {m['cost']:.4f}")

    section("Logistic regression -- C grid on validation")
    best = None
    for C in C_GRID:
        t0 = time.perf_counter()
        lr = fit_logistic(Xf, yf, C)
        pv = lr.predict_proba(Xv)[:, 1]
        t, _ = common.cost_threshold(yv, pv)
        m = common.metrics(yv, pv, t)
        print(f"C={C:<6}  val AP {m['average_precision']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
              f"log loss {m['log_loss']:.4f}  ({time.perf_counter() - t0:.1f}s)")
        if best is None or m["average_precision"] > best[1]["average_precision"]:
            best = (C, m, lr)
    C, m, lr = best
    rows.append({"model": "regressão logística (L2)", "hyper": f"C = {C}", **m})
    out["logistic"] = {"C": C, "val": m}
    coef = pd.Series(lr.coef_[0], index=d["names"]).sort_values()
    print(f"chosen C = {C}; threshold {m['threshold']:.2f}; val macro F1 {m['macro_f1']:.4f}")
    print("largest positive coefficients:", coef.tail(6).round(2).to_dict())
    print("largest negative coefficients:", coef.head(6).round(2).to_dict())

    section("Histogram gradient boosting (optional reference)")
    best = None
    for lrate, leaves, n_iter in HGB_GRID:
        t0 = time.perf_counter()
        hgb = fit_boosting(Xf, yf, lrate, leaves, n_iter)
        pv = hgb.predict_proba(Xv)[:, 1]
        t, _ = common.cost_threshold(yv, pv)
        m = common.metrics(yv, pv, t)
        print(f"lr={lrate} leaves={leaves} trees={n_iter}: val AP {m['average_precision']:.4f}  ROC-AUC {m['roc_auc']:.4f}  "
              f"({time.perf_counter() - t0:.1f}s)")
        if best is None or m["average_precision"] > best[1]["average_precision"]:
            best = ((lrate, leaves, n_iter), m)
    (lrate, leaves, n_iter), m = best
    rows.append({"model": "gradient boosting (HGB)", "hyper": f"lr = {lrate}, folhas = {leaves}, árvores = {n_iter}", **m})
    out["boosting"] = {"learning_rate": lrate, "max_leaf_nodes": leaves, "max_iter": n_iter, "val": m}

    section("Table -- baselines on validation (threshold = minimum cost on validation)")
    table = pd.DataFrame(rows)[["model", "hyper", "average_precision", "roc_auc", "macro_f1",
                                "recall_1", "precision_1", "accuracy", "cost", "threshold"]]
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    out["table"] = table.to_dict(orient="records")
    common.save_json("step1.json", out)


if __name__ == "__main__":
    main()
