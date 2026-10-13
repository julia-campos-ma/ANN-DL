"""
Preprocessing pipeline (items 4A and 4C): one ``build_preprocessor()`` that returns an
unfitted scikit-learn ``Pipeline`` + ``ColumnTransformer``. The classification deliverable
imports it from here and fits it on the training split only.

Every decision below points to the finding that motivated it (section numbers refer to
the EDA report):

- missing values (1B): ``children`` median; ``country`` becomes the level "unknown";
  ``agent``/``company`` NaN already became "none" in ``data.clean`` because the data
  article defines NaN there as "no agent / no company", not as unknown;
- outliers (2A): every numerical feature is clipped to the training [0.1%, 99.9%]
  quantiles, which removes the adr of -6.38 and 5,400, 55 adults, 10 babies and the
  lead times beyond 600 days without deleting rows; the seven long-tailed features
  (skew > 2, or skew 1.5 with a maximum far beyond the 99th percentile) then go
  through log1p; the three person counts are not logged because, once clipped, they
  only take the values 0..3, and adr is not logged because its zeros would become an
  isolated spike;
- encoding (2B, 1D): one-hot for everything; for the three high-cardinality features
  levels with fewer than 100 training rows are merged into one "infrequent" column, and
  the same column receives any level unseen in training (12 countries, 22 agents and
  20 companies appear only in the test set);
- scaling (2A): StandardScaler after log1p, because the target model is an MLP.

Run from the repository root to fit on train, transform both splits and print the
checks of item 4C:
    python docs/projects/eda/code/pipeline.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from data import CATEGORICAL_HIGH, CATEGORICAL_LOW, NUMERICAL, TARGET, load_split

# Long right tails in the training set (Stage 2, Table 3): five with skew > 2 plus lead_time
# and stays_in_weekend_nights (skew 1.49, max 737 and 19 against P99 426 and 4). The person
# counts (adults, children, babies) also have skew > 2 but, after clipping, only take 0..3.
LOG_FEATURES = [
    "lead_time",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
]
# adr is clipped but NOT logged: after clipping its skew is already below 1, and log1p would
# turn the 1.7% of zero rates into an isolated spike six standard deviations below the mean.
LINEAR_FEATURES = [c for c in NUMERICAL if c not in LOG_FEATURES]
MIN_FREQUENCY = 100  # a level needs 100 training rows to get its own one-hot column
CLIP = (0.001, 0.999)


# --8<-- [start:clipper]
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to the [lower, upper] quantiles learned on the training data."""

    def __init__(self, lower=0.001, upper=0.999):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
# --8<-- [end:clipper]


# --8<-- [start:build]
def numeric_branch(log):
    steps = [
        ("impute", SimpleImputer(strategy="median")),  # children: 4 NaN
        ("clip", QuantileClipper(*CLIP)),
    ]
    if log:
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def build_preprocessor():
    """Unfitted preprocessor. Fit it on the training split only."""
    return ColumnTransformer(
        transformers=[
            ("num_log", numeric_branch(log=True), LOG_FEATURES),
            ("num_lin", numeric_branch(log=False), LINEAR_FEATURES),
            ("cat_low", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), CATEGORICAL_LOW),
            ("cat_high", Pipeline([
                ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),  # country: 488 NaN
                ("onehot", OneHotEncoder(min_frequency=MIN_FREQUENCY,
                                         handle_unknown="infrequent_if_exist", sparse_output=False)),
            ]), CATEGORICAL_HIGH),
        ],
        verbose_feature_names_out=False,
    )


def fit_transform_splits(train, test):
    """Fit on train, transform both; returns (X_train, X_test, feature_names, preprocessor)."""
    pre = build_preprocessor()
    X_train = pre.fit_transform(train, train[TARGET])
    X_test = pre.transform(test)
    return X_train, X_test, pre.get_feature_names_out(), pre
# --8<-- [end:build]


def report(train, test):
    """Item 4C checks, plus the numbers behind the strategies of item 4A."""
    X_train, X_test, names, pre = fit_transform_splits(train, test)
    print(f"train: {train.shape} -> {X_train.shape}")
    print(f"test:  {test.shape} -> {X_test.shape}")
    print(f"NaN in X_train: {np.isnan(X_train).sum()}   NaN in X_test: {np.isnan(X_test).sum()}")
    print(f"{len(names)} output features: {len(LOG_FEATURES)} log+scaled, {len(LINEAR_FEATURES)} scaled, "
          f"{sum(n.startswith(tuple(c + '_' for c in CATEGORICAL_LOW)) for n in names)} one-hot (low cardinality), "
          f"{sum(n.startswith(tuple(c + '_' for c in CATEGORICAL_HIGH)) for n in names)} one-hot (high cardinality)")
    print("all feature names:", list(names))
    print("infrequent buckets:", [n for n in names if "infrequent" in n])

    print("\nOutlier strategy -- clip bounds learned on train:")
    affected = np.zeros(len(train), dtype=bool)
    for branch, cols in (("num_log", LOG_FEATURES), ("num_lin", LINEAR_FEATURES)):
        clipper = pre.named_transformers_[branch].named_steps["clip"]
        raw = train[cols].fillna(train[cols].median()).to_numpy(dtype=float)
        changed = (raw < clipper.lower_) | (raw > clipper.upper_)
        affected |= changed.any(axis=1)
        for j, col in enumerate(cols):
            if changed[:, j].any():
                print(f"  {col:<30} [{clipper.lower_[j]:g}, {clipper.upper_[j]:g}]  "
                      f"rows changed: {int(changed[:, j].sum())}")
    print(f"  rows with at least one clipped value (train): {int(affected.sum())} "
          f"({affected.mean():.2%})")
    print("  skew before clip / after clip / after clip+log1p, train:")
    for col in ("lead_time", "adr", "days_in_waiting_list", "adults"):
        s = train[col].astype(float)
        c = s.clip(s.quantile(CLIP[0]), s.quantile(CLIP[1]))
        print(f"    {col:<22} {s.skew():6.2f} / {c.skew():6.2f} / {np.log1p(c.clip(lower=0)).skew():6.2f}")
    j = list(names).index("adr")
    print(f"  adr after the pipeline: min z = {X_train[:, j].min():.2f}, max z = {X_train[:, j].max():.2f}; "
          f"largest |z| over all 14 numerical columns: "
          + ", ".join(f"{names[i]} {np.abs(X_train[:, i]).max():.1f}"
                      for i in np.argsort(-np.abs(X_train[:, :len(NUMERICAL)]).max(0))[:3]))

    print("\nEncoding -- levels per high-cardinality feature after min_frequency=100:")
    enc = pre.named_transformers_["cat_high"].named_steps["onehot"]
    for col, cats, infreq in zip(CATEGORICAL_HIGH, enc.categories_, enc.infrequent_categories_):
        kept = len(cats) - (0 if infreq is None else len(infreq))
        unseen = set(test[col].dropna().unique()) - set(cats)
        print(f"  {col:<8} {len(cats)} train levels -> {kept} columns + 1 infrequent "
              f"({0 if infreq is None else len(infreq)} levels, "
              f"{train[col].isin([] if infreq is None else infreq).sum()} train rows); "
              f"{len(unseen)} unseen test levels ({test[col].isin(unseen).sum()} rows) -> infrequent")

    k = len(NUMERICAL)
    print(f"\nScaling -- train mean/std of the {k} scaled numerical columns (should be ~0/1):")
    stats = pd.DataFrame({"train mean": X_train[:, :k].mean(0), "train std": X_train[:, :k].std(0, ddof=1),
                          "test mean": X_test[:, :k].mean(0)}, index=names[:k])
    print(stats.round(3).to_string())
    print("(the test set is not re-centred, so its means show the temporal drift)")
    return X_train, X_test, names


if __name__ == "__main__":
    train, test, cut = load_split()
    report(train, test)
