"""
feature_engineering.py

Reusable, callable version of the feature-engineering logic that
lives in Favorita_GlobalModel.py (store merge, holiday/calendar
features, lag/rolling/momentum/spike features, and the direct
multi-horizon dataset construction). Favorita_GlobalModel.py itself
is left untouched -- this is a deliberate, documented duplication so
the research script and the production pipeline can evolve
independently. See README for the reasoning.

Only functions here; no top-level script execution.
"""

import numpy as np
import pandas as pd

HORIZON = 15

BASE_FEATURES = [
    "store_nbr", "family", "store_type", "city", "state", "store_cluster",
]

ROLLING_FEATURES = [
    "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
    "rolling_std_7", "rolling_std_14", "rolling_std_28", "rolling_cv_28",
]

TREND_MOMENTUM_FEATURES = [
    "momentum_7_28", "momentum_14_28", "trend_7_28", "yearly_ratio",
]

SPIKE_FEATURES = ["historical_spike_rate"]

MODEL_FEATURES = (
    BASE_FEATURES
    + ["day_of_week", "month", "lag_28", "lag_365"]
    + ROLLING_FEATURES
    + ["horizon"]
    + TREND_MOMENTUM_FEATURES
    + SPIKE_FEATURES
)


def add_store_and_holiday_features(history_df: pd.DataFrame, stores_df: pd.DataFrame,
                                    holidays_df: pd.DataFrame) -> pd.DataFrame:
    """Merge store attributes and add binary holiday/event calendar features."""
    df = history_df.merge(stores_df, on="store_nbr", how="left")
    df = df.rename(columns={"type": "store_type", "cluster": "store_cluster"})

    df["is_holiday"] = 0

    national = holidays_df[
        (holidays_df["type"] == "Holiday")
        & (holidays_df["locale"] == "National")
        & (holidays_df["transferred"] == False)
    ][["date"]].drop_duplicates()
    df.loc[df["date"].isin(national["date"]), "is_holiday"] = 1

    regional = holidays_df[
        (holidays_df["type"] == "Holiday")
        & (holidays_df["locale"] == "Regional")
        & (holidays_df["transferred"] == False)
    ][["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": "state"})
    df = df.merge(regional.assign(is_regional_holiday=1), on=["date", "state"], how="left")
    df["is_holiday"] = (
        df["is_holiday"].astype(bool) | df["is_regional_holiday"].fillna(0).astype(bool)
    ).astype("int8")
    df = df.drop(columns="is_regional_holiday")

    local = holidays_df[
        (holidays_df["type"] == "Holiday")
        & (holidays_df["locale"] == "Local")
        & (holidays_df["transferred"] == False)
    ][["date", "locale_name"]].drop_duplicates().rename(columns={"locale_name": "city"})
    df = df.merge(local.assign(is_local_holiday=1), on=["date", "city"], how="left")
    df["is_holiday"] = (
        df["is_holiday"].astype(bool) | df["is_local_holiday"].fillna(0).astype(bool)
    ).astype("int8")
    df = df.drop(columns="is_local_holiday")

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day_of_week / month calendar features for the row's own date."""
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def add_historical_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag / rolling / momentum / trend / spike features. Mirrors the
    'Historical Sales Features' region of Favorita_GlobalModel.py.
    All rolling/mean features use shift(1) so the current row's own
    sales never leaks into its own features.
    """
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    group = df.groupby(["store_nbr", "family"])["sales"]

    df["lag_7"] = group.shift(7)
    df["lag_14"] = group.shift(14)
    df["lag_28"] = group.shift(28)
    df["lag_365"] = group.shift(365)

    df["rolling_mean_7"] = group.transform(lambda x: x.shift(1).rolling(7).mean())
    df["rolling_mean_14"] = group.transform(lambda x: x.shift(1).rolling(14).mean())
    df["rolling_mean_28"] = group.transform(lambda x: x.shift(1).rolling(28).mean())

    df["rolling_std_7"] = group.transform(lambda x: x.shift(1).rolling(7).std())
    df["rolling_std_14"] = group.transform(lambda x: x.shift(1).rolling(14).std())
    df["rolling_std_28"] = group.transform(lambda x: x.shift(1).rolling(28).std())

    df["rolling_cv_28"] = df["rolling_std_28"] / (df["rolling_mean_28"] + 1e-6)

    df["momentum_7_28"] = df["rolling_mean_7"] - df["rolling_mean_28"]
    df["momentum_14_28"] = df["rolling_mean_14"] - df["rolling_mean_28"]
    df["trend_7_28"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-6)
    df["yearly_ratio"] = df["rolling_mean_28"] / (df["lag_365"] + 1e-6)

    df["is_spike_day"] = (
        df["sales"] > (df["rolling_mean_28"] + 2 * df["rolling_std_28"])
    ).astype("int8")
    df["historical_spike_rate"] = df.groupby(["store_nbr", "family"])["is_spike_day"].transform(
        lambda x: x.shift(1).expanding(min_periods=30).mean()
    )

    return df


def build_origin_dataset(prepared_df: pd.DataFrame, origin_date: str,
                          horizons: range = range(1, HORIZON + 1)) -> pd.DataFrame:
    """
    Build one inference-ready row per (store, family, horizon) for a
    single forecast origin date. Mirrors build_direct_dataset() from
    Favorita_GlobalModel.py, but without a `target` column (the
    future is unknown at inference time) and restricted to a single
    origin date instead of the whole date range.

    `prepared_df` must already have gone through
    add_store_and_holiday_features -> add_calendar_features ->
    add_historical_sales_features, using history up to and including
    origin_date.
    """
    origin_ts = pd.to_datetime(origin_date)
    group_keys = ["store_nbr", "family"]
    sales_group = prepared_df.groupby(group_keys)["sales"]

    rows = []
    for horizon in horizons:
        result = prepared_df[BASE_FEATURES + ["date"]].copy()
        result["target_date"] = result["date"] + pd.to_timedelta(horizon, unit="D")
        result["day_of_week"] = result["target_date"].dt.dayofweek
        result["month"] = result["target_date"].dt.month

        result["lag_28"] = sales_group.shift(28 - horizon)
        result["lag_365"] = sales_group.shift(365 - horizon)

        for feature in ROLLING_FEATURES + TREND_MOMENTUM_FEATURES + SPIKE_FEATURES:
            result[feature] = prepared_df[feature]

        result["horizon"] = horizon

        origin_row = result.loc[result["date"] == origin_ts].copy()
        rows.append(origin_row)

    origin_dataset = pd.concat(rows, ignore_index=True)
    return origin_dataset[["date", "target_date"] + MODEL_FEATURES]
