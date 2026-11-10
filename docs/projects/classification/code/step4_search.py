"""
Step 4 -- regularization and hyperparameters, all chosen on VALIDATION.

A. Regularization: the reference network (Step 3) without regularization against weight
   decay (AdamW), dropout and both, trained 60 epochs WITHOUT early stopping so the curves
   show what each technique (and the stop itself) does. 3 seeds. Figure 5.
B. Random search: 24 configurations drawn with seed 42 from the grid below, each trained
   with early stopping; sorted table by validation AP. The top 3 are re-trained with two
   more seeds, and the winner is the best MEAN, not the luckiest single run.
C. Feature ablation (the two risks left open by the EDA): the winner with and without
   ``previous_cancellations`` and ``booking_changes``, 3 seeds each. Rule fixed before
   running: a feature stays only if removing it lowers the mean validation AP by more than
   two standard deviations across seeds; otherwise the simpler, safer set wins.

The chosen configuration is saved to results/step4.json for Step 5.

Run from the repository root (about 25 minutes on 2 CPU cores):
    python docs/projects/classification/code/step4_search.py
"""

import itertools

import numpy as np
import pandas as pd

import common
import model as M
from common import SEED, section

plt = common.style()

BASE = dict(hidden=(128, 64), lr=3e-4, batch_size=256, pos_weight=1.0)
REGS = {
    "nenhuma": dict(dropout=0.0, weight_decay=0.0),
    "weight decay (AdamW, λ = 0,05)": dict(dropout=0.0, weight_decay=0.05),
    "dropout (p = 0,3)": dict(dropout=0.3, weight_decay=0.0),
    "dropout + weight decay": dict(dropout=0.3, weight_decay=0.05),
}
REG_EPOCHS = 60
REG_SEEDS = [42, 43, 44]

GRID = {
    "hidden": [(64,), (128,), (128, 64), (256, 128), (256, 128, 64), (512, 256)],
    "dropout": [0.0, 0.1, 0.3, 0.5],
    "weight_decay": [0.0, 1e-3, 1e-2, 1e-1],
    "lr": [1e-4, 3e-4, 1e-3],
    "batch_size": [256, 1024],
    "pos_weight": ["1", "n_neg/n_pos"],
}
N_CONFIGS = 24
TOP_K = 3
EXTRA_SEEDS = [43, 44]
ABLATIONS = {
    "todas as 26 features": (),
    "sem previous_cancellations": ("previous_cancellations",),
    "sem booking_changes": ("booking_changes",),
    "sem as duas": ("previous_cancellations", "booking_changes"),
}


def run(d, cfg, seed, **kw):
    pw = 1.0 if cfg["pos_weight"] == "1" else float((1 - d["y_fit"]).sum() / d["y_fit"].sum())
    net, h = M.train_model(d["X_fit"], d["y_fit"], d["X_val"], d["y_val"], hidden=cfg["hidden"],
                           dropout=cfg["dropout"], weight_decay=cfg["weight_decay"], lr=cfg["lr"],
                           batch_size=cfg["batch_size"], pos_weight=pw, seed=seed, **kw)
    b = h["best_epoch"]
    return net, h, {"val_ap": h["val_ap"][b - 1], "val_loss": h["val_loss"][b - 1], "best_epoch": b,
                    "stopped": h["stopped_epoch"], "params": M.n_params(net),
                    "train_loss": h["train_loss"][b - 1]}


def regularization(d):
    section(f"A. Regularization ({REG_EPOCHS} epochs, no early stopping, seeds {REG_SEEDS})")
    rows, curves = [], {}
    for name, reg in REGS.items():
        for seed in REG_SEEDS:
            _, h = M.train_model(d["X_fit"], d["y_fit"], d["X_val"], d["y_val"], **BASE, **reg,
                                 max_epochs=REG_EPOCHS, early_stopping=False, seed=seed)
            b = h["best_epoch"]
            rows.append({"regularização": name, "seed": seed, "melhor época": b,
                         "val loss (melhor)": h["val_loss"][b - 1], "val AP (melhor)": h["val_ap"][b - 1],
                         "val loss (época 60)": h["val_loss"][-1], "val AP (época 60)": h["val_ap"][-1],
                         "train loss (época 60)": h["train_loss"][-1],
                         "gap (época 60)": h["val_loss"][-1] - h["train_loss"][-1]})
            if seed == SEED:
                curves[name] = h
    df = pd.DataFrame(rows)
    agg = df.drop(columns="seed").groupby("regularização", sort=False).agg(["mean", "std"])
    with pd.option_context("display.width", 250):
        print(agg.round(4).to_string())
    return df, agg, curves


def search(d):
    section(f"B. Random search: {N_CONFIGS} configurations (seed {SEED}), early stopping, patience 10")
    space = list(itertools.product(*GRID.values()))
    rng = np.random.default_rng(SEED)
    picks = rng.choice(len(space), size=N_CONFIGS, replace=False)
    rows = []
    for i, k in enumerate(picks, 1):
        cfg = dict(zip(GRID.keys(), space[k]))
        _, _, r = run(d, cfg, SEED)
        rows.append({"id": i, **cfg, **r})
        print(f"[{i:2d}] {cfg}  -> val AP {r['val_ap']:.4f}  val loss {r['val_loss']:.4f}  "
              f"best epoch {r['best_epoch']}  params {r['params']:,}")
    df = pd.DataFrame(rows).sort_values("val_ap", ascending=False).reset_index(drop=True)
    print("\nsorted by validation AP:")
    with pd.option_context("display.width", 250):
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    section(f"B'. Top {TOP_K} re-trained with seeds {EXTRA_SEEDS} (3 seeds in total)")
    top = []
    for _, row in df.head(TOP_K).iterrows():
        cfg = {k: row[k] for k in GRID}
        aps = [row["val_ap"]] + [run(d, cfg, s)[2]["val_ap"] for s in EXTRA_SEEDS]
        top.append({"id": int(row["id"]), **cfg, "val_ap_mean": float(np.mean(aps)),
                    "val_ap_std": float(np.std(aps, ddof=1)), "val_ap_seeds": aps})
        print(f"[{int(row['id']):2d}] {cfg}  val AP {np.mean(aps):.4f} ± {np.std(aps, ddof=1):.4f}  {np.round(aps, 4)}")
    top = sorted(top, key=lambda r: -r["val_ap_mean"])
    return df, top


def ablation(cfg):
    section(f"C. Feature ablation with the chosen configuration, seeds {REG_SEEDS}")
    rows = []
    for name, drop in ABLATIONS.items():
        d = common.prepare(drop)
        aps = [run(d, cfg, s)[2]["val_ap"] for s in REG_SEEDS]
        rows.append({"features": name, "drop": list(drop), "n_inputs": int(d["X_fit"].shape[1]),
                     "val_ap_mean": float(np.mean(aps)), "val_ap_std": float(np.std(aps, ddof=1)),
                     "val_ap_seeds": aps})
        print(f"{name:<28} inputs {d['X_fit'].shape[1]:>3}  val AP {np.mean(aps):.4f} ± {np.std(aps, ddof=1):.4f}")
    full = rows[0]
    keep = {}
    for r, feat in ((rows[1], "previous_cancellations"), (rows[2], "booking_changes")):
        loss = full["val_ap_mean"] - r["val_ap_mean"]
        noise = 2 * max(full["val_ap_std"], r["val_ap_std"])
        keep[feat] = bool(loss > noise)
        print(f"removing {feat}: AP change {-loss:+.4f}, 2 std = {noise:.4f} -> "
              f"{'KEEP' if keep[feat] else 'DROP'}")
    drop = tuple(f for f, k in keep.items() if not k)
    return rows, drop


def figure(curves):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    palette = ["#C44E52", "#4C72B0", "#55A868", "#8172B2"]
    for (name, h), c in zip(curves.items(), palette):
        ep = np.arange(1, len(h["val_loss"]) + 1)
        a1.plot(ep, h["val_loss"], color=c, lw=1.4, label=f"{name} — validação")
        a1.plot(ep, h["train_loss"], color=c, lw=1, ls=":", label=f"{name} — treino")
        a1.scatter([h["best_epoch"]], [h["val_loss"][h["best_epoch"] - 1]], color=c, zorder=5, s=25)
        a2.plot(ep, h["val_ap"], color=c, lw=1.4, label=name)
    a1.set_xlabel("época")
    a1.set_ylabel("perda (BCE média)")
    a1.set_title("Perda: validação (linha) e treino (pontilhado)")
    a1.set_ylim(0.15, 0.55)
    a2.set_xlabel("época")
    a2.set_ylabel("average precision (validação)")
    a2.set_title("AP de validação")
    a2.set_ylim(0.815, 0.845)
    a2.legend(loc="lower left")
    handles, labels = a1.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.12), fontsize=7)
    fig.suptitle("Figura 5 — Regularização: rede 128-64 treinada 60 épocas sem early stopping (semente 42; "
                 "ponto = melhor época)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig05-regularization.png")
    plt.close(fig)


def main():
    d = common.prepare()
    reg_df, reg_agg, curves = regularization(d)
    figure(curves)
    search_df, top = search(d)
    best = {k: top[0][k] for k in GRID}
    print(f"\nchosen configuration: {best}")
    abl, drop = ablation(best)
    print(f"final feature set: drop {list(drop) or 'nothing'}")
    common.save_json("step4.json", {
        "regularization": reg_df.to_dict(orient="records"),
        "regularization_summary": [
            {"regularização": name, **{f"{col} {stat}": float(reg_agg.loc[name, (col, stat)])
                                      for col, stat in reg_agg.columns}}
            for name in reg_agg.index],
        "search": search_df.to_dict(orient="records"), "top": top,
        "ablation": abl, "chosen": {**best, "hidden": list(best["hidden"]), "drop": list(drop)},
    })


if __name__ == "__main__":
    main()
