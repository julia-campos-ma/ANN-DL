"""
Stage 3 -- bivariate and multivariate analysis on the TRAINING set: numerical x
numerical (3A), categorical x target (3B) and numerical x categorical (3C).
Draws Figures 7 to 12.

Run from the repository root:
    python docs/projects/eda/code/stage3_bivariate.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CATEGORICAL, CATEGORICAL_HIGH, CATEGORICAL_LOW, FIGURES, MONTHS, NUMERICAL, TARGET, load_split

pd.set_option("display.width", 160, "display.max_columns", 40, "display.max_rows", 100)
COLORS = {0: "#4C72B0", 1: "#DD8452"}
LABELS = {0: "não cancelada", 1: "cancelada"}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def top_pairs(corr, k=8):
    """The k strongest off-diagonal pairs of a correlation matrix, by absolute value."""
    mask = np.triu(np.ones(corr.shape, dtype=bool), 1)
    return corr.where(mask).stack().sort_values(key=np.abs, ascending=False).head(k)


# --8<-- [start:numnum]
def num_num(train):
    """3A -- Spearman between the 14 numerical features; Pearson printed next to it for the
    top pairs, so the choice is justified with numbers and not only with the skewness."""
    section("3A - NUMERICAL x NUMERICAL (train)")
    X = train[NUMERICAL]
    spearman = X.corr("spearman")
    pearson = X.corr("pearson")
    pairs = top_pairs(spearman)
    print("Strongest Spearman pairs (Pearson for the same pair in brackets):")
    for (a, b), r in pairs.items():
        print(f"  {a:<30} {b:<30} rho = {r:+.3f}   [r = {pearson.loc[a, b]:+.3f}]")
    print("\nStrongest Pearson pairs, for comparison:")
    for (a, b), r in top_pairs(pearson, 4).items():
        print(f"  {a:<30} {b:<30} r = {r:+.3f}   [rho = {spearman.loc[a, b]:+.3f}]")
    print(f"\n|rho| >= 0.5 pairs: {int((np.abs(top_pairs(spearman, 100)) >= 0.5).sum())}; "
          f"|rho| >= 0.3 pairs: {int((np.abs(top_pairs(spearman, 100)) >= 0.3).sum())}")

    with_target = train[NUMERICAL + [TARGET]].corr("spearman")[TARGET].drop(TARGET).sort_values()
    print("\nSpearman of each numerical feature with the target:")
    for col, r in with_target.items():
        print(f"  {col:<30} {r:+.3f}")

    # Figure 7 -- the matrix
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(spearman.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(NUMERICAL)), NUMERICAL, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(NUMERICAL)), NUMERICAL, fontsize=8)
    for i in range(len(NUMERICAL)):
        for j in range(len(NUMERICAL)):
            v = spearman.values[i, j]
            if i != j and abs(v) >= 0.1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    ax.set_title("Figura 7 — Correlação de Spearman entre as 14 features numéricas (treino; |ρ| ≥ 0.10 anotado)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig07-spearman.png", dpi=150)
    plt.close(fig)

    # Figure 8 -- the three strongest pairs as 2D histograms (points would overplot at n = 95k)
    shown = [("lead_time", "stays_in_week_nights"),        # strongest Spearman pair
             ("stays_in_weekend_nights", "stays_in_week_nights"),  # strongest Pearson pair
             ("adults", "adr")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (a, b) in zip(axes, shown):
        xa, xb = X[a].clip(upper=X[a].quantile(0.999)), X[b].clip(upper=X[b].quantile(0.999))
        # Small integer-valued axes get one bin per value, centred on the integer.
        bins = [np.arange(v.min() - 0.5, v.max() + 1.5) if v.nunique() <= 40 and (v % 1 == 0).all()
                else 40 for v in (xa, xb)]
        h = ax.hist2d(xa, xb, bins=bins, cmap="Blues", norm="log")
        ax.set_xlabel(a, fontsize=9)
        ax.set_ylabel(b, fontsize=9)
        ax.set_title(f"ρ = {spearman.loc[a, b]:+.2f} · r = {pearson.loc[a, b]:+.2f}", fontsize=9)
        fig.colorbar(h[3], ax=ax, label="reservas (log)")
    fig.suptitle("Figura 8 — Três pares numéricos como histogramas 2D (treino; eixos cortados no percentil 99.9)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig08-pairs.png", dpi=150)
    plt.close(fig)
    return spearman
# --8<-- [end:numnum]


def cramers_v(x, y):
    """Bias-corrected Cramér's V between two categorical series (Bergsma 2013)."""
    table = pd.crosstab(x, y).values
    n = table.sum()
    row, col = table.sum(1, keepdims=True), table.sum(0, keepdims=True)
    expected = row @ col / n
    chi2 = ((table - expected) ** 2 / expected).sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2c / max(1e-12, min(kc - 1, rc - 1))))


# --8<-- [start:cattarget]
def cat_target(train):
    """3B -- cancellation rate per level of every categorical feature, ranked by Cramér's V."""
    section("3B - CATEGORICAL x TARGET (train)")
    base = train[TARGET].mean()
    print(f"baseline cancel rate (train): {base:.2%}\n")
    strength = {col: cramers_v(train[col], train[TARGET]) for col in CATEGORICAL}
    print("Cramér's V with the target (bias-corrected):")
    for col, v in sorted(strength.items(), key=lambda kv: -kv[1]):
        print(f"  {col:<22} {v:.3f}")

    print("\nCancel rate by level (levels with >= 100 train rows; low-cardinality features in full):")
    rates = {}
    for col in CATEGORICAL:
        g = train.groupby(col)[TARGET].agg(["size", "mean"])
        g = g[g["size"] >= 100].sort_values("mean", ascending=False)
        rates[col] = g
        if col in CATEGORICAL_LOW:
            print(f"\n{col}: " + ", ".join(f"{k} {m:.1%} (n={s})" for k, (s, m) in g.iterrows()))
        else:
            print(f"\n{col} (top 5 and bottom 5 of {len(g)} levels with >= 100 rows):")
            print("  highest: " + ", ".join(f"{k} {m:.1%} (n={s})" for k, (s, m) in g.head(5).iterrows()))
            print("  lowest:  " + ", ".join(f"{k} {m:.1%} (n={s})" for k, (s, m) in g.tail(5).iterrows()))

    nr = train[train.deposit_type == "Non Refund"]
    print(f"\nWho is Non Refund: {len(nr)} train rows; Groups {(nr.market_segment == 'Groups').mean():.1%}, "
          f"Offline TA/TO {(nr.market_segment == 'Offline TA/TO').mean():.1%}, "
          f"lead_time > 90 days {(nr.lead_time > 90).mean():.1%}, City Hotel {(nr.hotel == 'City Hotel').mean():.1%}, "
          f"PRT {(nr.country == 'PRT').mean():.1%}")

    # Figure 9 -- six low-cardinality features, cancel rate per level with the baseline
    chosen = ["deposit_type", "market_segment", "customer_type", "hotel", "arrival_date_month", "is_repeated_guest"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, col in zip(axes.ravel(), chosen):
        g = train.groupby(col)[TARGET].agg(["size", "mean"])
        g = g[g["size"] >= 100]  # same rule as the printed tables: no rate from fewer than 100 rows
        g = g.reindex(MONTHS) if col == "arrival_date_month" else g.sort_values("mean")
        ax.barh(g.index.astype(str), g["mean"], color=COLORS[1])
        for i, (s, m) in enumerate(zip(g["size"], g["mean"])):
            ax.text(m + 0.01, i, f"{m:.0%} (n={s})", va="center", fontsize=7)
        ax.axvline(base, color="black", ls="--", lw=1)
        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])  # rate axis ends at 100%; the extra room is for the labels
        ax.set_title(col, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("taxa de cancelamento", fontsize=8)
    fig.suptitle(f"Figura 9 — Taxa de cancelamento por nível (treino; linha tracejada = taxa global {base:.1%})")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig09-cancel-rate-by-category.png", dpi=150)
    plt.close(fig)

    # Figure 10 -- the high-cardinality features: ten most frequent levels each
    print("\nCancel rate of the 10 most frequent levels (the data of Figure 10):")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, col in zip(axes, CATEGORICAL_HIGH):
        g = train.groupby(col)[TARGET].agg(["size", "mean"]).sort_values("size", ascending=False).head(10)
        print(f"  {col}: " + ", ".join(f"{k} {m:.1%} (n={int(s)})" for k, (s, m) in g.iterrows()))
        ax.barh(g.index.astype(str)[::-1], g["mean"][::-1], color=COLORS[1])
        for i, (s, m) in enumerate(zip(g["size"][::-1], g["mean"][::-1])):
            ax.text(m + 0.01, i, f"{m:.0%} (n={s})", va="center", fontsize=7)
        ax.axvline(base, color="black", ls="--", lw=1)
        ax.set_xlim(0, 1.3)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])  # rate axis ends at 100%; the extra room is for the labels
        ax.set_title(f"{col}: 10 níveis mais frequentes", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("taxa de cancelamento", fontsize=8)
    fig.suptitle(f"Figura 10 — Taxa de cancelamento nos 10 níveis mais frequentes de country, agent e company "
                 f"(treino; tracejado = taxa global {base:.1%})")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig10-cancel-rate-high-cardinality.png", dpi=150)
    plt.close(fig)
    return strength
# --8<-- [end:cattarget]


# --8<-- [start:numcat]
def num_cat(train):
    """3C -- grouped boxplots: do the groups differ in location, in spread, or in both?"""
    section("3C - NUMERICAL x CATEGORICAL (train)")

    def summary(col, by):
        g = train.groupby(by)[col].describe()[["count", "25%", "50%", "75%"]]
        g["IQR"] = g["75%"] - g["25%"]
        return g

    for col, by in (("lead_time", "hotel"), ("adr", "hotel"), ("lead_time", "deposit_type"),
                    ("lead_time", "market_segment"), ("adr", "market_segment"), ("lead_time", "customer_type")):
        print(f"\n{col} by {by} (count, q1, median, q3, IQR):")
        print(summary(col, by).round(1).to_string())
    print("\nlead_time by hotel x target (median / IQR):")
    g = train.groupby(["hotel", TARGET])["lead_time"].describe()[["50%", "25%", "75%"]]
    g["IQR"] = g["75%"] - g["25%"]
    print(g.round(1).to_string())
    print("\nlead_time median by market_segment x target:")
    print(train.groupby(["market_segment", TARGET])["lead_time"].median().unstack().round(1).to_string())
    print("\nadr by hotel x target (median / IQR):")
    g = train.groupby(["hotel", TARGET])["adr"].describe()[["50%", "25%", "75%"]]
    g["IQR"] = g["75%"] - g["25%"]
    print(g.round(1).to_string())

    def grouped_box(ax, col, by, order, ymax=None):
        positions, data, colors, ticks = [], [], [], []
        for i, level in enumerate(order):
            for j, y in enumerate((0, 1)):
                s = train.loc[(train[by] == level) & (train[TARGET] == y), col]
                data.append(s.values)
                positions.append(i * 3 + j)
                colors.append(COLORS[y])
            ticks.append(i * 3 + 0.5)
        bp = ax.boxplot(data, positions=positions, widths=0.8, showfliers=False, patch_artist=True,
                        medianprops={"color": "black", "linewidth": 1.4})
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticks(ticks, order, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(f"{col} ({UNITS[col]})", fontsize=9)
        ax.set_title(f"{col} por {by} e alvo", fontsize=10)
        ax.legend(handles, [LABELS[0], LABELS[1]], loc="upper left", fontsize=7)
        if ymax:
            ax.set_ylim(0, ymax)

    UNITS = {"lead_time": "dias", "adr": "€ por noite"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[y]) for y in (0, 1)]

    # Figure 11 -- lead_time and adr by hotel, split by the target
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    grouped_box(axes[0], "lead_time", "hotel", ["City Hotel", "Resort Hotel"])
    grouped_box(axes[1], "adr", "hotel", ["City Hotel", "Resort Hotel"], ymax=300)
    fig.suptitle("Figura 11 — lead_time e adr por hotel, separados pelo alvo (treino; sem outliers)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig11-box-hotel.png", dpi=150)
    plt.close(fig)

    # Figure 12 -- lead_time by deposit type and by market segment, split by the target
    segs = ["Online TA", "Offline TA/TO", "Groups", "Direct", "Corporate"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 2]})
    grouped_box(axes[0], "lead_time", "deposit_type", ["No Deposit", "Non Refund", "Refundable"])
    grouped_box(axes[1], "lead_time", "market_segment", segs)
    fig.suptitle("Figura 12 — lead_time por deposit_type e por market_segment, separados pelo alvo (treino; sem outliers)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig12-box-deposit-segment.png", dpi=150)
    plt.close(fig)
# --8<-- [end:numcat]


if __name__ == "__main__":
    train, test, cut = load_split()
    print(f"train: {train.shape[0]} rows (arrivals before {cut.date()})")
    num_num(train)
    cat_target(train)
    num_cat(train)
    print("\nFigures written to", FIGURES)
