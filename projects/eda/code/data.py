"""
Shared data module for the EDA deliverable: loads the Hotel Booking Demand dataset,
applies the row-level cleaning that does not depend on any fitted statistic, and
performs the single temporal train/test split used by every other script.

Every script in this folder (and the classification deliverable later) imports
``load_raw``, ``clean`` and ``split`` from here, so the split is defined in exactly
one place.

Run from the repository root:
    python docs/projects/eda/code/data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "data" / "hotel" / "hotel_bookings.csv"
FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

TARGET = "is_canceled"

# --8<-- [start:columns]
# Columns recorded after the outcome, or that only change for guests who did arrive.
LEAKAGE = {
    "reservation_status": "equals the target: Check-Out <=> not canceled, Canceled/No-Show <=> canceled",
    "reservation_status_date": "date of the last status change, i.e. of the cancellation itself",
    "assigned_room_type": "assigned at check-in; differs from the reserved type almost only for guests who arrived",
    "required_car_parking_spaces": "0 cancellations among the bookings that have it, which only happens if it is recorded at check-in",
}
# A time proxy: the test set is the most recent slice, so the year would just label it.
TIME_PROXY = {"arrival_date_year": "used only to build arrival_date; the test set is a single year"}

NUMERICAL = [
    "lead_time",
    "arrival_date_week_number",
    "arrival_date_day_of_month",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adults",
    "children",
    "babies",
    "previous_cancellations",
    "previous_bookings_not_canceled",
    "booking_changes",
    "days_in_waiting_list",
    "adr",
    "total_of_special_requests",
]
# Low cardinality: one-hot as is.
CATEGORICAL_LOW = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "distribution_channel",
    "reserved_room_type",
    "deposit_type",
    "customer_type",
    "is_repeated_guest",
]
# High cardinality: rare levels grouped, unseen levels in the test set go to the same bucket.
CATEGORICAL_HIGH = ["country", "agent", "company"]
CATEGORICAL = CATEGORICAL_LOW + CATEGORICAL_HIGH
FEATURES = NUMERICAL + CATEGORICAL

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# --8<-- [end:columns]


def load_raw():
    """Read the CSV exactly as distributed (119,390 x 32)."""
    if not DATA.exists():
        raise SystemExit(
            f"Missing {DATA}.\nDownload hotel_bookings.csv from "
            "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand "
            f"(or the identical TidyTuesday copy) and place it at {DATA}."
        )
    return pd.read_csv(DATA)


# --8<-- [start:clean]
def clean(raw):
    """Row-level cleaning that uses no fitted statistic, so it is safe before the split.

    - builds ``arrival_date`` from the three date columns (needed for the temporal split);
    - fixes types: ``agent``/``company`` are codes, not quantities; NaN there means
      "no agent" / "no company" (the data article says so), so it becomes a level;
    - merges ``meal == "Undefined"`` into ``"SC"`` (the data article defines both as
      "no meal package") and the alpha-2 country code ``"CN"`` into ``"CHN"``;
    - removes bookings with zero guests (adults + children + babies == 0): a booking
      nobody would stay in is a recording error, not an observation;
    - drops the leakage and time-proxy columns listed above.
    """
    df = raw.copy()
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str)
        + "-" + df["arrival_date_month"]
        + "-" + df["arrival_date_day_of_month"].astype(str),
        format="%Y-%B-%d",
    )
    for col in ("agent", "company"):
        df[col] = df[col].map(lambda v: "none" if pd.isna(v) else str(int(v)))
    df["meal"] = df["meal"].replace("Undefined", "SC")
    # One country is coded twice: "CN" (ISO alpha-2) next to "CHN" (alpha-3, like every other level).
    df["country"] = df["country"].replace("CN", "CHN")
    df["is_repeated_guest"] = df["is_repeated_guest"].astype(str)

    guests = df["adults"] + df["children"].fillna(0) + df["babies"]
    df = df.loc[guests > 0].copy()

    df = df.drop(columns=list(LEAKAGE) + list(TIME_PROXY))
    return df.reset_index(drop=True)
# --8<-- [end:clean]


# --8<-- [start:split]
TEST_SHARE = 0.20


def split(df):
    """Temporal split: the last 20% of arrivals (by date) are the test set.

    The cut falls on a whole day, so every booking with the same arrival date -- in
    particular the identical rows of a group booking -- lands on one side only. The seed
    is not used to split, but it is used, with the same value, by every sampling in the
    other scripts.
    """
    order = df.sort_values(["arrival_date"], kind="stable")
    cumulative = np.arange(1, len(order) + 1) / len(order)
    cut_date = order["arrival_date"].iloc[np.searchsorted(cumulative, 1 - TEST_SHARE)]
    test_mask = df["arrival_date"] >= cut_date
    train = df.loc[~test_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)
    return train, test, cut_date


def load_split():
    """What every other script calls: cleaned train and test frames plus the cut date."""
    return split(clean(load_raw()))
# --8<-- [end:split]


if __name__ == "__main__":
    raw = load_raw()
    print(f"raw: {raw.shape[0]} rows x {raw.shape[1]} columns")
    df = clean(raw)
    print(f"clean: {df.shape[0]} rows x {df.shape[1]} columns "
          f"({raw.shape[0] - df.shape[0]} zero-guest rows removed, "
          f"{len(LEAKAGE) + len(TIME_PROXY)} columns dropped, arrival_date added)")
    train, test, cut = split(df)
    print(f"cut date: {cut.date()}  train: {len(train)} ({len(train)/len(df):.2%})  "
          f"test: {len(test)} ({len(test)/len(df):.2%})")
    print(f"train arrivals: {train.arrival_date.min().date()} .. {train.arrival_date.max().date()}")
    print(f"test arrivals:  {test.arrival_date.min().date()} .. {test.arrival_date.max().date()}")
    print(f"cancel rate train: {train[TARGET].mean():.4f}  test: {test[TARGET].mean():.4f}")
    print(f"features: {len(FEATURES)} = {len(NUMERICAL)} numerical + {len(CATEGORICAL)} categorical")
