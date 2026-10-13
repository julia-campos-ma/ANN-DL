"""
Stage 1 -- initial inspection: data dictionary checks (1A), quality (1B), target (1C)
and the train/test split (1D). Prints every number quoted in the report and draws
Figures 1 and 2.

Run from the repository root:
    python docs/projects/eda/code/stage1_inspection.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data import (CATEGORICAL, FEATURES, FIGURES, LEAKAGE, NUMERICAL, TARGET,
                  TIME_PROXY, clean, load_raw, split)

pd.set_option("display.width", 160, "display.max_columns", 40, "display.max_rows", 100)

COLORS = {0: "#4C72B0", 1: "#DD8452"}  # not canceled / canceled, used in every figure
LABELS = {0: "não cancelada (0)", 1: "cancelada (1)"}


def section(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --8<-- [start:dictionary]
def dictionary(raw):
    """1A -- what pandas thinks each column is, against what it means."""
    section("1A - DATA DICTIONARY: dtype, distinct values and range per column")
    rows = []
    for col in raw.columns:
        s = raw[col]
        if s.dtype.kind in "if":
            rng = f"{s.min()} .. {s.max()}"
        else:
            levels = ", ".join(map(str, s.dropna().unique()[:4]))
            rng = levels + (" ..." if s.nunique() > 4 else "")
        rows.append((col, str(s.dtype), s.nunique(), rng))
    table = pd.DataFrame(rows, columns=["column", "dtype", "distinct", "range / levels"])
    print(table.to_string(index=False))
    print("\nColumns stored as numbers but that are codes/flags, not quantities:")
    for col in ("agent", "company", "is_repeated_guest", "is_canceled"):
        print(f"  {col:<20} dtype={raw[col].dtype}  distinct={raw[col].nunique()}")
# --8<-- [end:dictionary]


# --8<-- [start:quality]
def quality(raw):
    """1B -- missing values, duplicates, impossible values and leakage evidence."""
    section("1B - QUALITY")
    n = len(raw)

    print("\nMissing values (NaN) per column, count and %:")
    miss = raw.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    for col, k in miss.items():
        print(f"  {col:<10} {k:>7}  {k / n:7.2%}")
    print("Pattern behind the two big ones (share of the missing rows by market_segment):")
    for col in ("agent", "company"):
        share = raw.loc[raw[col].isna(), "market_segment"].value_counts(normalize=True).head(3)
        print(f"  {col} missing -> " + ", ".join(f"{k} {v:.1%}" for k, v in share.items()))
    share = raw.loc[raw["company"].notna(), "market_segment"].value_counts(normalize=True).head(2)
    print("  company present -> " + ", ".join(f"{k} {v:.1%}" for k, v in share.items()))
    print(f"  country missing: cancel rate {raw.loc[raw.country.isna(), TARGET].mean():.1%} "
          f"vs {raw[TARGET].mean():.1%} overall")

    print("\nMissing values disguised as a level ('Undefined'):")
    for col in ("meal", "market_segment", "distribution_channel"):
        k = (raw[col] == "Undefined").sum()
        print(f"  {col:<22} {k:>5}  {k / n:6.2%}")

    dup = raw.duplicated()
    print(f"\nExact duplicate rows: {dup.sum()} ({dup.mean():.2%}); distinct rows: {(~dup).sum()}")
    both = raw.duplicated(keep=False)
    print(f"  cancel rate inside duplicated rows {raw.loc[both, TARGET].mean():.1%} "
          f"vs {raw.loc[~both, TARGET].mean():.1%} outside")
    sizes = raw.groupby(list(raw.columns), dropna=False).size()
    groups_share = raw.loc[both, "market_segment"].isin(["Groups", "Offline TA/TO"]).mean()
    print(f"  largest identical group: {sizes.max()} rows; "
          f"share of duplicated rows that are Groups/Offline TA: {groups_share:.1%} "
          f"(vs {raw.market_segment.isin(['Groups', 'Offline TA/TO']).mean():.1%} of all rows); "
          f"rows with at least one copy: {int(both.sum())}")
    print(f"  if duplicates were dropped the cancel rate would become "
          f"{raw.drop_duplicates()[TARGET].mean():.1%}")

    print("\nImpossible or inconsistent values:")
    guests = raw["adults"] + raw["children"].fillna(0) + raw["babies"]
    print(f"  adults == 0:                    {(raw.adults == 0).sum():>6}")
    print(f"  adults+children+babies == 0:    {(guests == 0).sum():>6}  (removed in clean())")
    print(f"  adults > 4:                     {(raw.adults > 4).sum():>6}  max {raw.adults.max()}")
    print(f"  babies >= 9 or children >= 9:   {((raw.babies >= 9) | (raw.children >= 9)).sum():>6}")
    nights = raw["stays_in_weekend_nights"] + raw["stays_in_week_nights"]
    print(f"  total nights == 0:              {(nights == 0).sum():>6}  ({(nights == 0).mean():.2%})")
    print(f"  adr < 0:                        {(raw.adr < 0).sum():>6}  min {raw.adr.min()}")
    comp_zero = (raw.loc[raw.adr == 0, "market_segment"] == "Complementary").sum()
    print(f"  adr == 0:                       {(raw.adr == 0).sum():>6}  ({(raw.adr == 0).mean():.2%}); "
          f"of which Complementary: {comp_zero}")
    others = raw.loc[(raw.adr == 0) & (raw.market_segment != "Complementary")]
    print(f"    the other {len(others)} zero-rate rows by segment: "
          + ", ".join(f"{k} {v}" for k, v in others.market_segment.value_counts().head(5).items())
          + f"; by customer_type: " + ", ".join(f"{k} {v}" for k, v in others.customer_type.value_counts().head(2).items()))
    print(f"  children NaN rows: " + ", ".join(
        f"{h} {y}" for h, y in raw.loc[raw.children.isna(), ["hotel", "arrival_date_year"]].value_counts().index))
    print(f"  meal == Undefined by hotel: {raw.loc[raw.meal == 'Undefined', 'hotel'].value_counts().to_dict()}")
    two_letter = raw.country.dropna().loc[raw.country.dropna().str.len() != 3].unique()
    print(f"  country codes that are not 3 letters: {list(two_letter)}; "
          + ", ".join(f"{c} {int((raw.country == c).sum())} rows, cancel rate {raw.loc[raw.country == c, TARGET].mean():.1%}"
                      for c in list(two_letter) + ["CHN"]))
    print(f"  adr > 1000:                     {(raw.adr > 1000).sum():>6}  max {raw.adr.max()}; "
          f"next largest {raw.adr.nlargest(2).iloc[1]}")

    prev = raw.previous_cancellations > 0
    print(f"  previous_cancellations > 0 but is_repeated_guest == 0: "
          f"{int((prev & (raw.is_repeated_guest == 0)).sum())} of {int(prev.sum())} rows "
          f"({(raw.loc[prev, 'is_repeated_guest'] == 0).mean():.1%}); "
          f"of those rows, {raw.loc[prev].duplicated(keep=False).mean():.1%} are inside exact-duplicate blocks")
    print("  cancel rate when previous_cancellations > 0, by arrival year: "
          + ", ".join(f"{y} {r:.1%} (n={n})" for y, (r, n) in
                      raw.loc[prev].groupby("arrival_date_year")[TARGET].agg(["mean", "size"]).iterrows()))

    print("\nLeakage evidence:")
    print(pd.crosstab(raw.reservation_status, raw[TARGET]))
    rsd = pd.to_datetime(raw.reservation_status_date)
    arr = pd.to_datetime(raw.arrival_date_year.astype(str) + "-" + raw.arrival_date_month
                         + "-" + raw.arrival_date_day_of_month.astype(str), format="%Y-%B-%d")
    print(f"  reservation_status_date <= arrival_date in {(rsd <= arr).mean():.1%} of rows; "
          f"for canceled bookings it is the cancellation date")
    diff = raw.assigned_room_type != raw.reserved_room_type
    print(f"  assigned != reserved room: {diff.sum()} rows, cancel rate {raw.loc[diff, TARGET].mean():.1%} "
          f"vs {raw.loc[~diff, TARGET].mean():.1%} when equal "
          f"({int((diff & (raw[TARGET] == 1)).sum())} canceled bookings do have a different assigned room)")
    park = raw.required_car_parking_spaces > 0
    print(f"  required_car_parking_spaces > 0: {park.sum()} rows, "
          f"{raw.loc[park, TARGET].sum()} canceled ({raw.loc[park, TARGET].mean():.1%})")
    chg = raw.booking_changes > 0
    print(f"  booking_changes > 0: {chg.sum()} rows, cancel rate {raw.loc[chg, TARGET].mean():.1%} "
          f"vs {raw.loc[~chg, TARGET].mean():.1%} (kept, flagged as a risk)")
    print("\nDropped columns:")
    for col, why in {**LEAKAGE, **TIME_PROXY}.items():
        print(f"  {col:<28} {why}")
# --8<-- [end:quality]


# --8<-- [start:target]
def target(df, train):
    """1C -- class balance in the whole cleaned table and in the training set."""
    section("1C - TARGET")
    for name, d in (("all cleaned rows", df), ("train", train)):
        counts = d[TARGET].value_counts().sort_index()
        share = counts / counts.sum()
        print(f"{name}: " + "  ".join(f"{LABELS[k]} {counts[k]} ({share[k]:.2%})" for k in counts.index)
              + f"  | majority/minority = {counts.max() / counts.min():.2f}"
              + f"  | always-majority accuracy = {share.max():.2%}")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    counts = train[TARGET].value_counts().sort_index()
    bars = ax.bar([LABELS[k] for k in counts.index], counts.values,
                  color=[COLORS[k] for k in counts.index])
    for b, k in zip(bars, counts.index):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"{counts[k]}\n({counts[k] / counts.sum():.1%})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.set_ylabel("reservas no treino")
    ax.set_title("Figura 1 — Distribuição do alvo is_canceled (treino)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig01-target.png", dpi=150)
    plt.close(fig)
# --8<-- [end:target]


# --8<-- [start:split]
def split_report(df, train, test, cut):
    """1D -- the temporal split: sizes, dates, drift and unseen categories."""
    section("1D - TRAIN / TEST (temporal)")
    print(f"cut date {cut.date()}: train = arrivals before it, test = arrivals from it on")
    print(f"train {len(train)} rows ({len(train) / len(df):.2%}), "
          f"{train.arrival_date.min().date()} .. {train.arrival_date.max().date()}, "
          f"cancel rate {train[TARGET].mean():.2%}")
    print(f"test  {len(test)} rows ({len(test) / len(df):.2%}), "
          f"{test.arrival_date.min().date()} .. {test.arrival_date.max().date()}, "
          f"cancel rate {test[TARGET].mean():.2%}")
    same_day = df.groupby("arrival_date").size()
    print(f"on average {same_day.mean():.0f} bookings per arrival date; "
          f"no arrival date is on both sides (cut on a whole day)")
    dup_key = FEATURES + [TARGET]
    tr_keys = set(map(tuple, train[dup_key].astype(str).values))
    crossing = test[dup_key].astype(str).apply(tuple, axis=1).isin(tr_keys).sum()
    print(f"test rows whose features (incl. week and day, excl. year) match a train row: "
          f"{crossing} ({crossing / len(test):.2%})")
    r_train, r_test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=42)
    r_keys = set(map(tuple, r_train[dup_key].astype(str).values))
    r_cross = r_test[dup_key].astype(str).apply(tuple, axis=1).isin(r_keys).sum()
    print(f"  for comparison, a random stratified 80/20 split would have {r_cross} such test rows "
          f"({r_cross / len(r_test):.2%}): the copies of a group booking fall on both sides")
    print("unseen categories in test (levels absent from train):")
    for col in CATEGORICAL:
        new = set(test[col].unique()) - set(train[col].unique())
        if new:
            rows = test[col].isin(new).sum()
            print(f"  {col:<10} {len(new):>3} new levels, {rows} test rows ({rows / len(test):.2%})")

    monthly = df.groupby(df.arrival_date.dt.to_period("M")).agg(n=(TARGET, "size"), rate=(TARGET, "mean"))
    print(f"monthly cancel rate: min {monthly.rate.min():.1%} ({monthly.rate.idxmin()}), "
          f"max {monthly.rate.max():.1%} ({monthly.rate.idxmax()}); overall {df[TARGET].mean():.2%}")
    print("test months: " + ", ".join(f"{m} {r:.1%}" for m, r in monthly.rate[monthly.index >= cut.to_period('M')].items()))
    same_window_2016 = df[(df.arrival_date >= cut - pd.DateOffset(years=1)) & (df.arrival_date <= test.arrival_date.max() - pd.DateOffset(years=1))]
    print(f"same calendar window one year earlier ({same_window_2016.arrival_date.min().date()} .. "
          f"{same_window_2016.arrival_date.max().date()}): cancel rate {same_window_2016[TARGET].mean():.1%} "
          f"(n={len(same_window_2016)}) -> the drift is year-over-year, not seasonal")
    print(f"share of train rows with 2017 arrivals: {(train.arrival_date.dt.year == 2017).mean():.1%}")
    print(f"always-majority accuracy on test: {1 - test[TARGET].mean():.2%}")
    print("other shifts between train and test: "
          f"PRT share {(train.country == 'PRT').mean():.1%} -> {(test.country == 'PRT').mean():.1%}; "
          f"Non Refund share {(train.deposit_type == 'Non Refund').mean():.1%} -> {(test.deposit_type == 'Non Refund').mean():.1%}; "
          f"previous_cancellations > 0 share {(train.previous_cancellations > 0).mean():.2%} -> "
          f"{(test.previous_cancellations > 0).mean():.2%}, with cancel rate "
          f"{train.loc[train.previous_cancellations > 0, TARGET].mean():.1%} -> "
          f"{test.loc[test.previous_cancellations > 0, TARGET].mean():.1%}")
    fig, ax1 = plt.subplots(figsize=(9, 3.8))
    x = monthly.index.to_timestamp()
    ax1.bar(x, monthly["n"], width=20, color="#BBBBBB", label="reservas / mês")
    ax1.set_ylabel("reservas por mês de chegada")
    ax2 = ax1.twinx()
    ax2.plot(x, monthly["rate"] * 100, color=COLORS[1], marker="o", ms=3, label="taxa de cancelamento")
    ax2.set_ylabel("taxa de cancelamento (%)")
    ax2.set_ylim(0, 60)
    ax1.set_xlabel("mês de chegada")
    ax1.axvline(cut, color="black", ls="--")
    ax1.text(cut, monthly["n"].max() * 0.95, f"  corte {cut.date()}\n  teste →", va="top", fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    ax1.set_title("Figura 2 — Reservas e taxa de cancelamento por mês de chegada, com o corte temporal")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig02-split-timeline.png", dpi=150)
    plt.close(fig)
# --8<-- [end:split]


if __name__ == "__main__":
    raw = load_raw()
    print(f"raw: {raw.shape[0]} rows x {raw.shape[1]} columns")
    dictionary(raw)
    quality(raw)
    df = clean(raw)
    train, test, cut = split(df)
    print(f"\nclean: {df.shape[0]} rows x {df.shape[1]} columns; features used: {len(FEATURES)} "
          f"({len(NUMERICAL)} numerical, {len(CATEGORICAL)} categorical) + target + arrival_date")
    target(df, train)
    split_report(df, train, test, cut)
    print("\nFigures written to", FIGURES)
