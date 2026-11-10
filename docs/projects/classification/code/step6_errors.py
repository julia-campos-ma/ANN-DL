"""
Step 6 -- error analysis of the seed-42 network on the test predictions saved by Step 5
(no choice is made here; nothing feeds back into the model).

A. Errors by subgroup: EDA categories (deposit_type, market_segment, hotel, customer_type,
   arrival month), lead_time ranges, and rows that have an exact copy in the test set.
B. The worst examples: the most confident false positives and false negatives.
C. Permutation importance on test (each raw feature shuffled before the pipeline, so all
   one-hot columns of a categorical move together; drop in AP, 5 repeats), compared with
   the EDA association with the target (|Spearman| for numerical, bias-corrected Cramér's
   V for categorical, both on the training set).
Figures 8 and 9.

Run from the repository root, after step5_test.py:
    python docs/projects/classification/code/step6_errors.py
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

import common
import model as M
from common import CATEGORICAL_HIGH, CATEGORICAL_LOW, NUMERICAL, SEED, TARGET, section

plt = common.style()
import stage3_bivariate  # noqa: E402  (EDA module: reuses cramers_v)

LEAD_BINS = [-1, 7, 30, 90, 180, 365, 10_000]
LEAD_LABELS = ["0–7", "8–30", "31–90", "91–180", "181–365", "> 365"]
MIN_N = 200
N_REPEATS = 5


def group_table(df, col):
    rows = []
    for level, g in df.groupby(col, observed=True, sort=col in ("lead_time_faixa", "mês")):
        if len(g) < MIN_N:
            continue
        y, p, pred = g[TARGET].to_numpy(), g["p"].to_numpy(), g["pred"].to_numpy()
        neg, pos = (y == 0).sum(), (y == 1).sum()
        rows.append({"variável": col, "nível": str(level), "n": len(g), "taxa real": y.mean(),
                     "prob. média": p.mean(), "erro": (pred != y).mean(),
                     "FPR": ((pred == 1) & (y == 0)).sum() / neg if neg else np.nan,
                     "FNR": ((pred == 0) & (y == 1)).sum() / pos if pos else np.nan,
                     "AP": average_precision_score(y, p) if 20 <= pos <= len(y) - 20 else np.nan,
                     "custo": common.expected_cost(y, p, g["t"].iloc[0])})
    return rows


def main():
    s5 = common.load_json("step5.json")
    cfg = s5["config"]
    d = common.prepare(tuple(cfg["drop"]), with_test=True)
    pred_file = np.load(common.RESULTS / "test_predictions.npz")
    test = d["test"].copy()
    r42 = next(r for r in s5["rows"] if r["model"] == "MLP" and r["seed"] == SEED)
    t = r42["threshold"]
    test["p"] = pred_file[f"mlp_{SEED}"]
    test["pred"] = (test["p"] >= t).astype(int)
    test["t"] = t
    test["lead_time_faixa"] = pd.cut(test["lead_time"], LEAD_BINS, labels=LEAD_LABELS)
    feats = [c for c in test.columns if c in NUMERICAL + CATEGORICAL_LOW + CATEGORICAL_HIGH]
    test["tem_cópia"] = np.where(test.duplicated(feats + [TARGET], keep=False), "com cópia no teste", "única")
    test["mês"] = test["arrival_date"].dt.strftime("%Y-%m")
    test["país"] = np.where(test["country"] == "PRT", "PRT", "outros")
    print(f"seed {SEED}, threshold {t:.2f}; overall error {(test.pred != test[TARGET]).mean():.4f}, "
          f"AP {average_precision_score(test[TARGET], test.p):.4f}")

    section(f"A. Errors by subgroup (levels with >= {MIN_N} test rows)")
    cols = ["deposit_type", "market_segment", "hotel", "customer_type", "lead_time_faixa", "mês", "país", "tem_cópia",
            "is_repeated_guest"]
    rows = sum((group_table(test, c) for c in cols), [])
    sub = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(sub.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    worst = sub.sort_values("erro", ascending=False).iloc[0]
    print(f"\nlargest error rate: {worst['variável']} = {worst['nível']}: {worst['erro']:.4f} "
          f"(n = {worst['n']}, real rate {worst['taxa real']:.3f}, mean prob {worst['prob. média']:.3f})")
    nd = test[test.deposit_type == "No Deposit"]
    print(f"inside No Deposit ({len(nd)} rows, {len(nd) / len(test):.1%}): AP "
          f"{average_precision_score(nd[TARGET], nd.p):.4f}, prevalence {nd[TARGET].mean():.4f}, "
          f"error {(nd.pred != nd[TARGET]).mean():.4f}")
    lin = pred_file["logistic"]
    mask = (test.deposit_type == "No Deposit").to_numpy()
    print(f"logistic inside No Deposit: AP {average_precision_score(nd[TARGET], lin[mask]):.4f}")
    uq = (test["tem_cópia"] == "única").to_numpy()
    print(f"test rows without an exact copy: {uq.sum()} -> MLP AP {average_precision_score(test[TARGET][uq], test.p[uq]):.4f}, "
          f"logistic {average_precision_score(test[TARGET][uq], lin[uq]):.4f}")
    months = test.groupby("mês").apply(lambda g: pd.Series({
        "n": len(g), "rate": g[TARGET].mean(), "mean_p": g.p.mean(),
        "AP": average_precision_score(g[TARGET], g.p)}), include_groups=False)
    print("\nby month:\n", months.round(4).to_string())

    section("B. Most confident errors")
    show = ["p", TARGET, "hotel", "lead_time", "deposit_type", "market_segment", "customer_type", "country",
            "agent", "adr", "total_of_special_requests", "booking_changes", "previous_cancellations",
            "stays_in_week_nights", "arrival_date"]
    fp = test[(test.pred == 1) & (test[TARGET] == 0)].sort_values("p", ascending=False)
    fn = test[(test.pred == 0) & (test[TARGET] == 1)].sort_values("p")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("top 10 false positives (predicted cancel, guest came):")
        print(fp[show].head(10).to_string(index=False))
        print("\ntop 10 false negatives (predicted show, canceled):")
        print(fn[show].head(10).to_string(index=False))
    for name, e in (("FP with p >= 0.9", fp[fp.p >= 0.9]), ("FN with p <= 0.1", fn[fn.p <= 0.1])):
        print(f"\n{name}: {len(e)} rows; profile:")
        for c in ("deposit_type", "market_segment", "country", "customer_type", "hotel"):
            print(f"  {c}: " + ", ".join(f"{k} {v:.0%}" for k, v in e[c].value_counts(normalize=True).head(3).items()))
        print(f"  lead_time median {e.lead_time.median():.0f}; special requests > 0: "
              f"{(e.total_of_special_requests > 0).mean():.0%}; with copy: {(e['tem_cópia'] != 'única').mean():.0%}")
    hi = test[test.p >= 0.9]
    print(f"\nall bookings with p >= 0.9: {len(hi)}, of which canceled {hi[TARGET].mean():.3f}; "
          f"Non Refund among them {(hi.deposit_type == 'Non Refund').mean():.3f}")
    nr = test[test.deposit_type == "Non Refund"]
    print(f"Non Refund in test: {len(nr)} rows, canceled {nr[TARGET].mean():.4f}, predicted cancel {nr.pred.mean():.4f}")
    lo = test[test.p <= 0.1]
    print(f"all bookings with p <= 0.1: {len(lo)}, of which canceled {lo[TARGET].mean():.3f}")
    top_err = {"fp": fp[show].head(10).astype(str).to_dict(orient="records"),
               "fn": fn[show].head(10).astype(str).to_dict(orient="records")}

    section(f"C. Permutation importance on test ({N_REPEATS} repeats, drop in AP)")
    net = M.MLP(d["X_test"].shape[1], tuple(cfg["hidden"]), cfg["dropout"])
    net.load_state_dict(torch.load(common.RESULTS / "mlp_seed42.pt"))
    base_ap = average_precision_score(d["y_test"], M.predict_proba(net, d["X_test"]))
    raw = d["test"]
    rng = np.random.default_rng(SEED)
    used = [c for c in feats if c not in cfg["drop"]]      # the dropped features are not inputs
    imp = []
    for col in used:
        drops = []
        for _ in range(N_REPEATS):
            shuffled = raw.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            p = M.predict_proba(net, d["pre"].transform(shuffled).astype(np.float32))
            drops.append(base_ap - average_precision_score(d["y_test"], p))
        imp.append({"feature": col, "drop_mean": float(np.mean(drops)), "drop_std": float(np.std(drops, ddof=1))})
    imp = pd.DataFrame(imp).sort_values("drop_mean", ascending=False).reset_index(drop=True)

    fit, val = d["fit"], d["val"]
    train = pd.concat([fit, val], ignore_index=True)      # the EDA training set
    assoc = {}
    for col in used:
        if col in NUMERICAL:
            assoc[col] = abs(spearmanr(train[col], train[TARGET], nan_policy="omit")[0])
        else:
            assoc[col] = stage3_bivariate.cramers_v(train[col].fillna("unknown"), train[TARGET])
    imp["eda_assoc"] = imp.feature.map(assoc)
    imp["eda_kind"] = imp.feature.map(lambda c: "|ρ| Spearman" if c in NUMERICAL else "V de Cramér")
    imp["rank_pi"] = imp.drop_mean.rank(ascending=False)
    imp["rank_eda"] = imp.eda_assoc.rank(ascending=False)
    rho = spearmanr(imp.drop_mean, imp.eda_assoc)[0]
    print(f"base test AP {base_ap:.4f}")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nSpearman between permutation importance and EDA association (n = {len(imp)}): {rho:.3f}")
    for col in cfg["drop"]:
        a = abs(spearmanr(train[col], train[TARGET])[0])
        print(f"dropped in Step 4 (importance 0 by construction): {col}, EDA |rho| {a:.4f}")

    # Figure 8 -- errors by subgroup
    panels = [("deposit_type", "deposit_type"), ("market_segment", "market_segment"),
              ("lead_time_faixa", "lead_time (dias)"), ("mês", "mês de chegada")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=True)
    for ax, (col, title) in zip(axes, panels):
        g = sub[sub["variável"] == col]
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["FPR"], 0.4, color=common.COLORS[0], label="FPR (vem, previsto cancela)")
        ax.bar(x + 0.2, g["FNR"], 0.4, color=common.COLORS[1], label="FNR (cancela, previsto vem)")
        ax.plot(x, g["erro"], color="black", marker="o", ms=4, lw=1, label="taxa de erro")
        ax.set_xticks(x, [f"{lv}\n(n={n:,})".replace(",", ".") for lv, n in zip(g["nível"], g["n"])],
                      rotation=45 if len(g) > 4 else 0, ha="right" if len(g) > 4 else "center", fontsize=7)
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("taxa no teste (MLP, semente 42)")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle("Figura 8 — Erros por subgrupo no teste: taxa de erro, FPR e FNR no limiar de custo",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig08-subgroups.png")
    plt.close(fig)

    # Figure 9 -- permutation importance vs EDA association
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    top = imp.head(15)[::-1]
    a1.barh(top.feature, top.drop_mean, xerr=top.drop_std, color=common.MODEL_COLORS["mlp"], capsize=2)
    a1.set_xlabel("queda de AP no teste ao permutar a feature")
    a1.set_title("Importância por permutação (15 maiores)")
    for kind, marker in (("|ρ| Spearman", "o"), ("V de Cramér", "s")):
        g = imp[imp.eda_kind == kind]
        a2.scatter(g.eda_assoc, g.drop_mean, marker=marker, s=28, label=kind,
                   color=common.COLORS[0] if marker == "o" else common.COLORS[1])
    for _, r in imp.head(8).iterrows():
        a2.annotate(r.feature, (r.eda_assoc, r.drop_mean), fontsize=7, xytext=(3, 2), textcoords="offset points")
    for c in ("deposit_type",):
        if c in set(imp.feature) and c not in set(imp.head(8).feature):
            r = imp[imp.feature == c].iloc[0]
            a2.annotate(c, (r.eda_assoc, r.drop_mean), fontsize=7, xytext=(3, 2), textcoords="offset points")
    a2.set_xlabel("associação com o alvo na EDA (treino)")
    a2.set_ylabel("queda de AP no teste (permutação)")
    a2.set_title(f"Permutação × EDA (Spearman entre rankings = {rho:.2f})".replace(".", ","))
    a2.legend(loc="upper left")
    fig.suptitle("Figura 9 — O que a rede usa no teste × o que a EDA apontou no treino",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(common.FIGURES / "fig09-permutation.png")
    plt.close(fig)

    common.save_json("step6.json", {
        "threshold": t, "subgroups": sub.to_dict(orient="records"),
        "worst_group": worst.to_dict(), "months": months.reset_index().to_dict(orient="records"),
        "top_errors": top_err, "permutation": imp.to_dict(orient="records"), "rho_pi_eda": float(rho),
        "base_ap": float(base_ap),
    })


if __name__ == "__main__":
    main()
