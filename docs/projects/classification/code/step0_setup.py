"""
Step 0 -- starting point: the EDA split and pipeline, the temporal validation set carved
from the training set, and the checks that nothing is fitted outside the training part.
Figure 1.

Run from the repository root:
    python docs/projects/classification/code/step0_setup.py
"""

import numpy as np
import pandas as pd

import common
from common import COLORS, TARGET, section

plt = common.style()


def main():
    fit, val, test, cut_val, cut_test = common.load_frames()
    d = common.prepare()

    section("Split sizes (EDA test split + temporal validation inside the training set)")
    total = len(fit) + len(val) + len(test)
    rows = []
    for name, df in (("train (fit)", fit), ("validation", val), ("test", test)):
        rows.append({"split": name, "rows": len(df), "share": len(df) / total,
                     "first": df.arrival_date.min().date(), "last": df.arrival_date.max().date(),
                     "cancel_rate": df[TARGET].mean(),
                     "non_refund": (df.deposit_type == "Non Refund").mean(),
                     "prt": (df.country == "PRT").mean(),
                     "prev_cancel_gt0": (df.previous_cancellations > 0).mean()})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"validation cut: {cut_val.date()}   test cut (EDA): {cut_test.date()}")

    section("Leakage checks")
    feats = [c for c in fit.columns if c != "arrival_date"]
    keys_fit = set(map(tuple, fit[feats].astype(str).to_numpy()))
    dup_val = sum(tuple(r) in keys_fit for r in val[feats].astype(str).to_numpy())
    print(f"validation rows with an exact copy (26 features + target) in train-fit: {dup_val}")
    print(f"preprocessor fitted on {len(fit)} rows only -> X_fit {d['X_fit'].shape}, X_val {d['X_val'].shape}")
    print(f"NaN after the pipeline: fit {int(np.isnan(d['X_fit']).sum())}, val {int(np.isnan(d['X_val']).sum())}")
    enc = d["pre"].named_transformers_["cat_high"].named_steps["onehot"]
    kept = {c: len(cats) - (0 if inf is None else len(inf))
            for c, cats, inf in zip(common.CATEGORICAL_HIGH, enc.categories_, enc.infrequent_categories_)}
    print(f"high-cardinality columns kept with >= 100 train-fit rows: {kept} (+3 infrequent)")
    n_val_new = {c: int((~val[c].fillna("unknown").isin(set(cats))).sum())
                 for c, cats in zip(common.CATEGORICAL_HIGH, enc.categories_)}
    print(f"validation rows with a level unseen in train-fit (-> infrequent): {n_val_new}")
    print(f"positives / negatives in train-fit: {int(d['y_fit'].sum())} / {int((1 - d['y_fit']).sum())}  "
          f"ratio {(1 - d['y_fit']).sum() / d['y_fit'].sum():.3f}")

    section("Protocol fixed before training")
    print(f"main metric: {common.MAIN_METRIC} (positive class = canceled)")
    print(f"success: mean test AP over {len(common.SEEDS)} seeds >= logistic AP + {common.MIN_GAIN_OVER_LINEAR}, "
          "gain > 2 std across seeds")
    print(f"costs: FP = {common.COST_FP}, FN = {common.COST_FN} -> calibrated optimum "
          f"p* = C_FP / (C_FP + C_FN) = {common.COST_FP / (common.COST_FP + common.COST_FN):.3f}")
    print(f"trivial AP = prevalence: val {val[TARGET].mean():.4f}, test {test[TARGET].mean():.4f}")

    # Figure 1 -- the three splits on the calendar
    allr = pd.concat([fit.assign(split="train"), val.assign(split="val"), test.assign(split="test")])
    allr["month"] = allr.arrival_date.dt.to_period("M").dt.to_timestamp()
    monthly = allr.groupby(["month", "split"]).agg(n=(TARGET, "size"), rate=(TARGET, "mean")).reset_index()
    rate = allr.groupby("month")[TARGET].mean()
    fig, ax1 = plt.subplots(figsize=(9, 3.8))
    shades = {"train": "#BBBBBB", "val": "#8DA0CB", "test": "#66C2A5"}
    labels = {"train": f"treino ({len(fit):,})", "val": f"validação ({len(val):,})", "test": f"teste ({len(test):,})"}
    bottom = pd.Series(0.0, index=sorted(allr.month.unique()))
    for s in ("train", "val", "test"):
        g = monthly[monthly.split == s].set_index("month")["n"].reindex(bottom.index, fill_value=0)
        ax1.bar(bottom.index, g, width=20, bottom=bottom, color=shades[s], label=labels[s].replace(",", "."))
        bottom += g
    ax1.set_ylabel("reservas por mês de chegada")
    ax1.set_xlabel("mês de chegada")
    ax2 = ax1.twinx()
    ax2.plot(rate.index, rate * 100, color=COLORS[1], marker="o", ms=3, label="taxa de cancelamento")
    ax2.set_ylabel("taxa de cancelamento (%)")
    ax2.set_ylim(0, 60)
    ax2.grid(False)
    for c, txt in ((cut_val, "corte validação\n2016-12-10"), (cut_test, "corte teste\n2017-04-22")):
        ax1.axvline(c, color="black", ls="--", lw=1)
        ax1.text(c, ax1.get_ylim()[1] * 0.97, " " + txt, va="top", fontsize=7)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", ncol=4, frameon=True)
    ax1.set_title("Figura 1 — Treino, validação e teste no calendário (split temporal)")
    fig.savefig(common.FIGURES / "fig01-splits.png")
    plt.close(fig)

    common.save_json("step0.json", {
        "splits": table.astype({"first": str, "last": str}).to_dict(orient="records"),
        "cut_val": str(cut_val.date()), "cut_test": str(cut_test.date()),
        "n_features": int(d["X_fit"].shape[1]), "dup_val_in_fit": int(dup_val),
        "val_unseen_levels": n_val_new, "high_card_kept": kept,
    })


if __name__ == "__main__":
    main()
