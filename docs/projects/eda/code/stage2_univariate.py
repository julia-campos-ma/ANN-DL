"""
Stage 2 -- univariate analysis on the TRAINING set only: descriptive statistics of all
14 numerical features (2A) and frequencies/cardinality of all 12 categorical features
(2B). Draws Figures 3 to 6.

Run from the repository root:
    python docs/projects/eda/code/stage2_univariate.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import CATEGORICAL, CATEGORICAL_HIGH, CATEGORICAL_LOW, FIGURES, MONTHS, NUMERICAL, load_split

pd.set_option("display.width", 160, "display.max_columns", 40, "display.max_rows", 100)
RARE = 0.01  # a level is "rare" when it holds less than 1% of the training rows


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --8<-- [start:numerical]
def numerical(train):
    """2A -- one row per numerical feature; every column of the table is quoted in the report."""
    section("2A - NUMERICAL (train)")
    X = train[NUMERICAL]
    table = pd.DataFrame({
        "mean": X.mean(), "std": X.std(), "min": X.min(), "q1": X.quantile(0.25),
        "median": X.median(), "q3": X.quantile(0.75), "p99": X.quantile(0.99), "max": X.max(),
        "zeros%": (X == 0).mean() * 100, "skew": X.skew(), "distinct": X.nunique(),
    })
    print(table.round(2).to_string())
    print(f"\nmedian == 0 for {int((table['median'] == 0).sum())} of {len(NUMERICAL)} features; "
          f"skew > 2 for {int((table['skew'] > 2).sum())}")
    iqr = table["q3"] - table["q1"]
    upper = table["q3"] + 1.5 * iqr
    above = (X > upper).mean() * 100
    print("Rows above the Tukey upper fence (q3 + 1.5 IQR), %:")
    print(above.round(2).sort_values(ascending=False).to_string())
    print(f"\nlead_time: {(X.lead_time == 0).mean():.2%} are same-day bookings; "
          f"{(X.lead_time > 365).mean():.2%} are booked more than a year ahead")
    print(f"adr: {(X.adr <= 0).mean():.2%} at or below zero; p99.9 = {X.adr.quantile(0.999):.2f}; "
          f"max {X.adr.max()}")
    print(f"stays: {((X.stays_in_weekend_nights + X.stays_in_week_nights) == 0).mean():.2%} with zero "
          f"nights; median total nights {(X.stays_in_weekend_nights + X.stays_in_week_nights).median():.0f}")

    # Figure 3 -- every numerical feature, one histogram each, log-scaled counts
    fig, axes = plt.subplots(4, 4, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), NUMERICAL):
        s = X[col]
        # integer features with few values get one bin per value, so 0 and 1 are never merged
        bins = np.arange(s.min(), s.max() + 2) - 0.5 if s.nunique() <= 60 else 50
        ax.hist(s, bins=bins, color="#4C72B0")
        ax.set_yscale("log")
        ax.set_title(f"{col}\nmediana {s.median():g} · assimetria {s.skew():.1f}", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[len(NUMERICAL):]:
        ax.axis("off")
    fig.suptitle("Figura 3 — Histogramas das 14 features numéricas (treino; contagem em escala log)", y=0.995)
    fig.supxlabel("valor da feature", fontsize=9)
    fig.supylabel("reservas (escala log)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig03-numerical-histograms.png", dpi=150)
    plt.close(fig)

    # Figure 4 -- the two continuous features: raw scale vs log1p
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for row, col in enumerate(["lead_time", "adr"]):
        s = X[col].clip(lower=0)
        axes[row, 0].hist(s, bins=60, color="#4C72B0")
        axes[row, 0].set_title(f"{col} (escala original)", fontsize=9)
        axes[row, 1].hist(np.log1p(s), bins=60, color="#55A868")
        axes[row, 1].set_title(f"log1p({col})  · assimetria {np.log1p(s).skew():.2f}", fontsize=9)
        unit = "dias" if col == "lead_time" else "€ por noite"
        axes[row, 0].set_xlabel(f"{col} ({unit})", fontsize=8)
        axes[row, 1].set_xlabel(f"log1p({col})", fontsize=8)
        for ax in axes[row]:
            ax.set_ylabel("reservas")
    fig.suptitle("Figura 4 — lead_time e adr antes e depois de log1p (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig04-logtransform.png", dpi=150)
    plt.close(fig)
    return table
# --8<-- [end:numerical]


# --8<-- [start:categorical]
def categorical(train):
    """2B -- cardinality, dominant level and rare levels of every categorical feature."""
    section("2B - CATEGORICAL (train)")
    rows = []
    for col in CATEGORICAL:
        vc = train[col].value_counts()
        share = vc / vc.sum()
        rare = share[share < RARE]
        rows.append({
            "feature": col, "levels": len(vc), "top level": vc.index[0], "top %": share.iloc[0] * 100,
            "levels < 1%": len(rare), "rows in them %": rare.sum() * 100,
            "levels >= 100 rows": int((vc >= 100).sum()),
        })
    table = pd.DataFrame(rows).set_index("feature")
    print(table.round(2).to_string())

    print("\nFull frequency tables of the low-cardinality features:")
    for col in CATEGORICAL_LOW:
        vc = train[col].value_counts()
        if col == "arrival_date_month":
            vc = vc.reindex(MONTHS)
        print(f"\n{col}: " + ", ".join(f"{k} {v} ({v / vc.sum():.1%})" for k, v in vc.items()))

    print("\nHigh-cardinality features: coverage of the top levels")
    for col in CATEGORICAL_HIGH:
        vc = train[col].value_counts()
        cum = vc.cumsum() / vc.sum()
        n90 = int((cum < 0.90).sum()) + 1
        print(f"  {col}: {len(vc)} levels; top 1 = {vc.index[0]} ({cum.iloc[0]:.1%}); "
              f"top 5 cover {cum.iloc[4]:.1%}; {n90} levels cover 90%; "
              f"{int((vc == 1).sum())} levels appear once")

    # Figure 5 -- bars for the nine low-cardinality features
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, col in zip(axes.ravel(), CATEGORICAL_LOW):
        vc = train[col].value_counts()
        if col == "arrival_date_month":
            vc = vc.reindex(MONTHS)
        ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color="#4C72B0")
        for i, v in enumerate(vc.values[::-1]):
            ax.text(v + vc.max() * 0.01, i, f"{v}", va="center", fontsize=6)
        ax.set_xlim(0, vc.max() * 1.18)
        ax.set_title(f"{col} ({len(vc)} níveis)", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("reservas", fontsize=8)
    fig.suptitle("Figura 5 — Frequências das 9 features categóricas de baixa cardinalidade (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig05-categorical-bars.png", dpi=150)
    plt.close(fig)

    # Figure 6 -- the three high-cardinality features: top 15 and cumulative coverage
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, col in enumerate(CATEGORICAL_HIGH):
        vc = train[col].value_counts()
        top = vc.head(15)
        axes[0, j].barh(top.index.astype(str)[::-1], top.values[::-1], color="#4C72B0")
        axes[0, j].set_title(f"{col}: 15 níveis mais frequentes de {len(vc)}", fontsize=9)
        axes[0, j].set_xlabel("reservas", fontsize=8)
        axes[0, j].tick_params(labelsize=7)
        cum = (vc.cumsum() / vc.sum()).values
        axes[1, j].plot(np.arange(1, len(cum) + 1), cum, color="#DD8452")
        axes[1, j].axhline(0.9, color="gray", ls=":")
        axes[1, j].set_xscale("log")
        axes[1, j].set_xlabel("níveis (ordenados por frequência, escala log)", fontsize=8)
        axes[1, j].set_ylabel("cobertura acumulada", fontsize=8)
        axes[1, j].tick_params(labelsize=7)
    fig.suptitle("Figura 6 — country, agent e company: níveis mais frequentes e cobertura acumulada (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig06-high-cardinality.png", dpi=150)
    plt.close(fig)
    return table
# --8<-- [end:categorical]


if __name__ == "__main__":
    train, test, cut = load_split()
    print(f"train: {train.shape[0]} rows (arrivals before {cut.date()})")
    numerical(train)
    categorical(train)
    print("\nFigures written to", FIGURES)
