# region Imports
import os
os.system('cls')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# endregion

# region Artifact Configuration

from pathlib import Path
import os


PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)


# RUN_TAG = "1yr_h15"
RUN_TAG = "1yr_h15_spikefeatures"

# ---------------------------------------------------------
# Output base directory
#
# Defaults to a folder INSIDE the project (portable -- works
# for anyone who clones the repo with no setup required).
#
# On a personal machine, this can optionally be redirected
# elsewhere (e.g. to avoid syncing large files through cloud
# storage) by setting the FAVORITA_OUTPUT_DIR environment
# variable once. If it's not set, the relative default is
# used automatically.
# ---------------------------------------------------------

OUTPUT_BASE_DIR = Path(
    os.environ.get(
        "FAVORITA_OUTPUT_DIR",
        str(PROJECT_ROOT / "outputs")
    )
)

OUTPUT_DIR = OUTPUT_BASE_DIR / RUN_TAG

# ---------------------------------------------------------
# Persistent Output Directories
# ---------------------------------------------------------

DATASET_OUTPUT_DIR = (
    OUTPUT_DIR
    / "datasets"
)

MODEL_OUTPUT_DIR = (
    OUTPUT_DIR
    / "models"
)

PREDICTION_OUTPUT_DIR = (
    OUTPUT_DIR
    / "predictions"
)


DATASET_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# XGBoost Dataset Artifacts
# ---------------------------------------------------------

XGB_TRAIN_DF_ARTIFACT = (
    DATASET_OUTPUT_DIR
    / "xgb_train_df.pkl"
)

XGB_VALID_DF_ARTIFACT = (
    DATASET_OUTPUT_DIR
    / "xgb_valid_df.pkl"
)

FIGURES_OUTPUT_DIR = (
    OUTPUT_DIR
    / "figures"
)

FIGURES_OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SHAP_SUMMARY_PLOT_ARTIFACT = (
    FIGURES_OUTPUT_DIR
    / "shap_summary_plot.png"
)

# ---------------------------------------------------------
# XGBoost Model Artifact
# ---------------------------------------------------------

XGB_MODEL_ARTIFACT = (
    MODEL_OUTPUT_DIR
    / "xgb_model.json"
)


# ---------------------------------------------------------
# XGBoost Validation Prediction Artifact
# ---------------------------------------------------------

XGB_VALID_RESULTS_ARTIFACT = (
    PREDICTION_OUTPUT_DIR
    / "xgb_valid_results.pkl"
)


# endregion

# region Read Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

def load_data(file_name):
    return pd.read_csv(DATA_PATH / file_name)

train = load_data("train.csv")
test = load_data("test.csv")
stores = load_data("stores.csv")
oil = load_data("oil.csv")
holidays = load_data("holidays_events.csv")
transactions = load_data("transactions.csv")

# endregion

# region Data Understanding

datasets = {
    "train": train,
    "test": test,
    "stores": stores,
    "oil": oil,
    "holidays": holidays,
    "transactions": transactions
}
# =====================================
# # Dataset Overview:

# for name, df in datasets.items():
#     print("=" * 50)
#     print(name.upper())
#     print("=" * 50)

#     print("\nShape:")
#     print(df.shape)

#     print("\nInfo:")
#     df.info()

#     print("\n")

# # =====================================
# # Data Quality Assessment:

# for name, df in datasets.items():
#     print("=" * 50)
#     print(name.upper())
#     print("=" * 50)

#     print("\nMissing Values:")
#     print(df.isnull().sum())

#     print("\nTotal Missing Values:")
#     print(df.isnull().sum().sum())

#     print("\nDuplicate Rows:")
#     print(df.duplicated().sum())

#     print("\n")

# =====================================
# # Target Variable Understanding:

# # Summary Statistics:
# print(train["sales"].describe())

# # Zero Sales:
# zero_sales = (train["sales"] == 0).sum()

# print(f"Zero Sales: {zero_sales}")
# print(f"Percentage : {zero_sales / len(train) * 100:.2f}%")

# endregion

# region Data Preparation

# =====================================
# Convert Date Columns:

for df in datasets.values():
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

# =====================================
# Memory Check (Before Optimization):

# print(train.memory_usage(deep=True))
# print("\nTotal Memory (MB):")
# print(train.memory_usage(deep=True).sum() / 1024**2)

# =====================================
# Optimize Data Types:

# Convert categorical columns:
train["family"] = train["family"].astype("category")
test["family"] = test["family"].astype("category")

# Reduce integer memory usage:
train["store_nbr"] = train["store_nbr"].astype("int8")
test["store_nbr"] = test["store_nbr"].astype("int8")
stores["store_nbr"] = stores["store_nbr"].astype("int8")
transactions["store_nbr"] = transactions["store_nbr"].astype("int8")

train["onpromotion"] = train["onpromotion"].astype("int16")
test["onpromotion"] = test["onpromotion"].astype("int16")

# =====================================
# Memory Check (After Optimization):

# print(train.memory_usage(deep=True))
# print("\nTotal Memory (MB):")
# print(train.memory_usage(deep=True).sum() / 1024**2)

# endregion

# region EDA - Target Variable (Sales)

# =====================================
# Business Question
# What are the statistical characteristics of the target variable (sales)?
#
# Why this analysis?
# Understanding the target distribution helps identify
# skewness, zero-inflation, outliers and potential modeling challenges.
# =====================================
# Basic Statistics:

# print(train["sales"].describe())

# # =====================================
# # Zero Sales

# zero_sales = (train["sales"] == 0).sum()

# print(f"\nZero Sales : {zero_sales:,}")
# print(f"Zero Sales Percentage : {zero_sales / len(train) * 100:.2f}%")

# # =====================================
# # Negative Sales

# negative_sales = (train["sales"] < 0).sum()

# print(f"\nNegative Sales : {negative_sales:,}")

# # =====================================
# # Important Values

# print(f"\nMedian Sale : {train['sales'].median():,.2f}")
# print(f"Maximum Sale : {train['sales'].max():,.2f}")

# # =====================================
# # Distribution

# plt.figure(figsize=(10,5))

# plt.hist(train["sales"], bins=100)

# plt.title("Sales Distribution")
# plt.xlabel("Sales")
# plt.ylabel("Frequency")

# plt.tight_layout()
# plt.show()

# # =====================================
# # Boxplot

# plt.figure(figsize=(10,2))

# plt.boxplot(train["sales"], vert=False)

# plt.title("Sales Boxplot")

# plt.tight_layout()
# plt.show()

# endregion

# region EDA - Trend Analysis

# =====================================
# Business Question
# How has the overall sales volume changed over time?
#
# Analysis Level
# Daily Total Sales (Aggregated across all stores and families)
# =====================================
daily_sales = (
    train.groupby("date", as_index=False)["sales"]
         .sum()
)

# print(daily_sales.head())
# print(daily_sales.tail())
# print(f"\nNumber of Days: {len(daily_sales)}")

# =====================================
# Business Question
# How has total daily sales changed over time?
#
# Why this analysis?
# Before building a forecasting model, we need to understand
# whether sales exhibit a long-term trend or major structural changes.
#
# Analysis Level
# Daily sales aggregated across all stores and product families.
# =====================================
# plt.figure(figsize=(16, 6))

# plt.plot(daily_sales["date"], daily_sales["sales"])

# plt.title("Total Daily Sales")
# plt.xlabel("Date")
# plt.ylabel("Sales")

# plt.tight_layout()
# plt.show()

# endregion

# region EDA - Seasonality Analysis

# =====================================
# Business Question
# Does sales exhibit recurring seasonal patterns?
#
# Why this analysis?
# Forecasting models rely heavily on seasonal behavior.
# In this section we investigate sales patterns across
# different time scales:
#
# • Yearly
# • Monthly
# • Weekly
# =====================================
# Yearly Seasonality:

# Extract Year:
daily_sales["year"] = daily_sales["date"].dt.year

# plt.figure(figsize=(16,6))

# for year, group in daily_sales.groupby("year"):
#     plt.plot(
#         group["date"].dt.dayofyear,
#         group["sales"],
#         label=year
#     )

# plt.title("Daily Sales by Year")
# plt.xlabel("Day of Year")
# plt.ylabel("Total Sales")
# plt.legend()

# plt.tight_layout()
# plt.show()

# =====================================
# Monthly Seasonality:

monthly_sales = (
    daily_sales.assign(month=daily_sales["date"].dt.month)
              .groupby("month", as_index=False)["sales"]
              .mean()
)

# month_labels = [
#     "Jan", "Feb", "Mar", "Apr", "May", "Jun",
#     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
# ]

# plt.figure(figsize=(12,5))

# plt.plot(monthly_sales["month"], monthly_sales["sales"])

# plt.xticks(range(1, 13), month_labels)

# plt.title("Average Daily Sales by Month")
# plt.xlabel("Month")
# plt.ylabel("Average Daily Sales")

# plt.tight_layout()
# plt.show()

# =====================================
# Weekly Seasonality:

weekly_sales = (
    daily_sales.assign(weekday=daily_sales["date"].dt.dayofweek)
              .groupby("weekday", as_index=False)["sales"]
              .mean()
)

# weekday_labels = [
#     "Mon",
#     "Tue",
#     "Wed",
#     "Thu",
#     "Fri",
#     "Sat",
#     "Sun"
# ]

# plt.figure(figsize=(10,5))

# plt.plot(weekly_sales["weekday"], weekly_sales["sales"])

# plt.xticks(range(7), weekday_labels)

# plt.title("Average Daily Sales by Weekday")
# plt.xlabel("Weekday")
# plt.ylabel("Average Daily Sales")

# plt.tight_layout()
# plt.show()

# endregion

# region Merge Stores Features

# =====================================
# Business Question
# Can store characteristics help explain differences in sales?
#
# Why this analysis?
# The stores dataset contains additional information about
# each store, such as location, type and cluster. These
# attributes may improve forecasting performance.
# =====================================
# Merge Store Information:
df = train.copy()
df = df.merge(
    stores,
    on="store_nbr",
    how="left"
)

df = df.rename(
    columns={
        "type": "store_type",
        "cluster": "store_cluster"
    }
)

# # Verify Merge:
# print(df[[
#     "store_nbr",
#     "city",
#     "state",
#     "store_type",
#     "store_cluster"
# ]].head())

# print()

# print(df[[
#     "city",
#     "state",
#     "store_type",
#     "store_cluster"
# ]].isnull().sum())

# endregion

# region Merge Holiday Features

# =====================================
# Business Question
#
# Can holiday-related information improve
# sales forecasting?
#
# Decision:
#
# Create four binary calendar features:
#
# - is_holiday
# - is_black_friday
# - is_cyber_monday
# - is_mothers_day
#
# National, Regional and Local holidays are
# mapped to the stores they affect.
#
# Unpredictable external events are excluded
# because they are not available at prediction
# time.
#
# Business-driven events with known future
# dates are retained as forecasting features.
# =====================================


# =====================================
# Create Holiday Feature
#
# National holidays apply to all stores.
# Regional holidays apply to stores within
# the corresponding state.
# Local holidays apply to stores within
# the corresponding city.
#
# Transferred holidays are excluded from the
# original date because the transferred date
# is represented separately in the dataset.
# =====================================

df["is_holiday"] = 0


# ----- National Holidays -----

national_holidays = holidays[
    (holidays["type"] == "Holiday") &
    (holidays["locale"] == "National") &
    (holidays["transferred"] == False)
][["date"]].drop_duplicates()

df.loc[
    df["date"].isin(national_holidays["date"]),
    "is_holiday"
] = 1


# ----- Regional Holidays -----

regional_holidays = holidays[
    (holidays["type"] == "Holiday") &
    (holidays["locale"] == "Regional") &
    (holidays["transferred"] == False)
][[
    "date",
    "locale_name"
]].drop_duplicates()

regional_holidays = regional_holidays.rename(
    columns={"locale_name": "state"}
)

df = df.merge(
    regional_holidays.assign(is_regional_holiday=1),
    on=["date", "state"],
    how="left"
)

df["is_holiday"] = (
    df["is_holiday"].astype(bool) |
    df["is_regional_holiday"].fillna(0).astype(bool)
).astype("int8")

df = df.drop(columns="is_regional_holiday")


# ----- Local Holidays -----

local_holidays = holidays[
    (holidays["type"] == "Holiday") &
    (holidays["locale"] == "Local") &
    (holidays["transferred"] == False)
][[
    "date",
    "locale_name"
]].drop_duplicates()

local_holidays = local_holidays.rename(
    columns={"locale_name": "city"}
)

df = df.merge(
    local_holidays.assign(is_local_holiday=1),
    on=["date", "city"],
    how="left"
)

df["is_holiday"] = (
    df["is_holiday"].astype(bool) |
    df["is_local_holiday"].fillna(0).astype(bool)
).astype("int8")

df = df.drop(columns="is_local_holiday")


# =====================================
# Create Special Event Features
#
# Only business-relevant events with known
# future dates are retained.
# =====================================

# ----- Black Friday -----

black_friday_dates = holidays[
    holidays["description"].str.contains(
        "Black Friday",
        case=False,
        na=False
    )
]["date"].drop_duplicates()

df["is_black_friday"] = (
    df["date"]
    .isin(black_friday_dates)
    .astype("int8")
)


# ----- Cyber Monday -----

cyber_monday_dates = holidays[
    holidays["description"].str.contains(
        "Cyber Monday",
        case=False,
        na=False
    )
]["date"].drop_duplicates()

df["is_cyber_monday"] = (
    df["date"]
    .isin(cyber_monday_dates)
    .astype("int8")
)


# ----- Mother's Day -----

mothers_day_dates = holidays[
    holidays["description"].str.contains(
        "Dia de la Madre",
        case=False,
        na=False
    )]["date"].drop_duplicates()

df["is_mothers_day"] = (
    df["date"]
    .isin(mothers_day_dates)
    .astype("int8")
)


# =====================================
# Verify Calendar Features
#
# All four features must be binary int8
# variables.
# =====================================

# calendar_features = [
#     "is_holiday",
#     "is_black_friday",
#     "is_cyber_monday",
#     "is_mothers_day"
# ]

# print("===== Calendar Features =====")

# print(
#     df[calendar_features].sum()
# )

# print()

# print("===== Data Types =====")

# print(
#     df[calendar_features].dtypes
# )

# print()

# print("===== Unique Values =====")

# for feature in calendar_features:
#     print(
#         f"{feature}: "
#         f"{df[feature].unique()}"
#     )

# endregion

# region Calendar Features

df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# endregion

# region Historical Sales Features

df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

group = df.groupby(
    ["store_nbr", "family"]
)["sales"]

# Lag Features

df["lag_7"] = group.shift(7)
df["lag_14"] = group.shift(14)
df["lag_28"] = group.shift(28)
df["lag_365"] = group.shift(365)

# Rolling Mean Features

df["rolling_mean_7"] = (
    group.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
)

df["rolling_mean_14"] = (
    group.transform(
        lambda x: x.shift(1).rolling(14).mean()
    )
)

df["rolling_mean_28"] = (
    group.transform(
        lambda x: x.shift(1).rolling(28).mean()
    )
)

# =====================================
# Rolling Std Features (NEW)
#
# Purpose:
# Capture volatility, not just level. A series with high
# variance relative to its mean is more prone to sudden
# spikes even when its recent average looks unremarkable.
# =====================================

df["rolling_std_7"] = (
    group.transform(
        lambda x: x.shift(1).rolling(7).std()
    )
)

df["rolling_std_14"] = (
    group.transform(
        lambda x: x.shift(1).rolling(14).std()
    )
)

df["rolling_std_28"] = (
    group.transform(
        lambda x: x.shift(1).rolling(28).std()
    )
)

# =====================================
# Coefficient of Variation (NEW)
#
# Normalized volatility, comparable across series with very
# different sales levels (a CLEANING series averaging 200/day
# and a BEVERAGES series averaging 2000/day are not
# comparable on raw std alone).
# =====================================

df["rolling_cv_28"] = (
    df["rolling_std_28"]
    / (df["rolling_mean_28"] + 1e-6)
)

#==================
# Momentum Features

df["momentum_7_28"] = (
    df["rolling_mean_7"]
    / df["rolling_mean_28"].replace(0, np.nan)
)

df["momentum_14_28"] = (
    df["rolling_mean_14"]
    / df["rolling_mean_28"].replace(0, np.nan)
)

# Trend Feature

df["trend_7_28"] = (
    df["rolling_mean_7"] - df["rolling_mean_28"]
)

# Yearly Ratio

df["yearly_ratio"] = (
    df["sales"]
    / df["lag_365"].replace(0, np.nan)
)
# =====================================
# Trend & Momentum Features
# =====================================

df["momentum_7_28"] = (
    df["rolling_mean_7"] - df["rolling_mean_28"]
)

df["momentum_14_28"] = (
    df["rolling_mean_14"] - df["rolling_mean_28"]
)

df["trend_7_28"] = (
    df["rolling_mean_7"] /
    (df["rolling_mean_28"] + 1e-6)
)

df["yearly_ratio"] = (
    df["rolling_mean_28"] /
    (df["lag_365"] + 1e-6)
)

# =====================================
# Historical Spike Rate Feature (NEW)
#
# Purpose:
# For each store-family series, measure how often it has
# historically produced a "spike day" -- a day where sales
# went well beyond that series' own recent mean+std.
#
# Leakage safety:
# is_spike_day for a given row uses rolling_mean_28 /
# rolling_std_28, which are already shift(1)-based (i.e. they
# do not include the current day). historical_spike_rate then
# uses an EXPANDING window additionally shifted by 1 day, so
# a day's own spike status never leaks into its own feature.
# =====================================

df["is_spike_day"] = (
    df["sales"]
    > (df["rolling_mean_28"] + 2 * df["rolling_std_28"])
).astype("int8")

df["historical_spike_rate"] = (
    df.groupby(["store_nbr", "family"])["is_spike_day"]
    .transform(
        lambda x: x.shift(1).expanding(min_periods=30).mean()
    )
)

# endregion

# region Train / Validation Split

split_date = "2017-07-01"

# Use only the most recent 1 years for training
TRAIN_HISTORY_YEARS = 1

# endregion

# region Direct Forecast Dataset

HORIZON = 15

BASE_FEATURES = [
    "store_nbr",
    "family",
    "store_type",
    "city",
    "state",
    "store_cluster"
]

ROLLING_FEATURES = [
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_cv_28"
]

TREND_MOMENTUM_FEATURES = [
    "momentum_7_28",
    "momentum_14_28",
    "trend_7_28",
    "yearly_ratio"
]

SPIKE_FEATURES = [
    "historical_spike_rate"
]


def build_direct_dataset(data, horizon):

    group_keys = ["store_nbr", "family"]

    result = data[
        BASE_FEATURES + ["date", "sales"]
    ].copy()

    sales_group = (
        data
        .groupby(group_keys)["sales"]
    )

    # Target
    result["target"] = (
        sales_group.shift(-horizon)
    )

    # Target date
    result["target_date"] = (
        result["date"]
        + pd.to_timedelta(horizon, unit="D")
    )

    # Calendar features of target date
    result["day_of_week"] = (
        result["target_date"].dt.dayofweek
    )

    result["month"] = (
        result["target_date"].dt.month
    )

    # Horizon-specific historical lags
    if horizon <= 7:
        lag = 7 - horizon
        result["lag_7"] = (
            sales_group.shift(lag)
        )

    if horizon <= 14:
        lag = 14 - horizon
        result["lag_14"] = (
            sales_group.shift(lag)
        )

    lag = 28 - horizon
    result["lag_28"] = (
        sales_group.shift(lag)
    )

    lag = 365 - horizon
    result["lag_365"] = (
        sales_group.shift(lag)
    )

    # Rolling features based only on information
    # available at the forecast origin
    for feature in ROLLING_FEATURES:
        result[feature] = data[feature]

    # Trend & Momentum features
    for feature in TREND_MOMENTUM_FEATURES:
        result[feature] = data[feature]

    # Historical spike-rate feature
    for feature in SPIKE_FEATURES:
        result[feature] = data[feature]

    result["horizon"] = horizon

    # Remove rows where required information
    # is not available
    required_columns = [
        "target",
        "lag_28",
        "lag_365",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
        "rolling_cv_28",
        "momentum_7_28",
        "momentum_14_28",
        "trend_7_28",
        "yearly_ratio",
        "historical_spike_rate"
    ]

    if horizon <= 7:
        required_columns.append("lag_7")

    if horizon <= 14:
        required_columns.append("lag_14")

    result = result.dropna(
        subset=required_columns
    )

    # Rename origin sales explicitly
    result = result.rename(
        columns={
            "sales": "origin_sales"
        }
    )

    return result


# Build Direct Forecast Dataset for all horizons

direct_datasets = []

for horizon in range(1, HORIZON + 1):

    horizon_df = build_direct_dataset(
        df,
        horizon
    )

    direct_datasets.append(
        horizon_df
    )

direct_df = pd.concat(
    direct_datasets,
    ignore_index=True
)

# endregion

# region Evaluation Metrics

def calculate_metrics(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mae = np.mean(
        np.abs(y_true - y_pred)
    )

    rmse = np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )

    denominator = np.sum(
        np.abs(y_true)
    )

    if denominator == 0:
        wape = np.nan
    else:
        wape = (
            np.sum(
                np.abs(y_true - y_pred)
            )
            / denominator
        )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape
    }

# endregion

# region Baseline - Naive

FORECAST_ORIGINS = [
    "2017-06-30",
    "2017-07-15",
    "2017-07-30"
]

baseline_results = []

for origin in FORECAST_ORIGINS:

    origin = pd.Timestamp(origin)

    for horizon in range(1, HORIZON + 1):

        direct_data = build_direct_dataset(
            df,
            horizon
        )

        validation_data = direct_data[
            direct_data["date"] == origin
        ].copy()

        if validation_data.empty:
            continue

        validation_data["prediction"] = (
            validation_data["origin_sales"]
        )

        metrics = calculate_metrics(
            validation_data["target"],
            validation_data["prediction"]
        )

        baseline_results.append({
            "origin": origin,
            "horizon": horizon,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"]
        })


baseline_results_df = pd.DataFrame(
    baseline_results
)

print("\nNaive Baseline Results")
print("=" * 70)

print(
    baseline_results_df.to_string(
        index=False
    )
)

# endregion

# region Baseline - Weekly Seasonal Naive

def weekly_seasonal_naive(
    data,
    origin,
    horizons=15
):

    origin = pd.Timestamp(origin)

    # Observations available at forecast origin
    history = data[
        data["date"] <= origin
    ].copy()

    # Last actual observation for each
    # store-family-day_of_week combination
    history["day_of_week"] = (
        history["date"].dt.dayofweek
    )

    weekly_reference = (
        history
        .sort_values("date")
        .groupby(
            [
                "store_nbr",
                "family",
                "day_of_week"
            ],
            as_index=False
        )
        .last()
    )

    results = []

    for horizon in range(1, horizons + 1):

        target_date = (
            origin
            + pd.Timedelta(days=horizon)
        )

        target_day_of_week = (
            target_date.dayofweek
        )

        forecast = data[
            data["date"] == target_date
        ][
            [
                "date",
                "store_nbr",
                "family",
                "sales"
            ]
        ].copy()

        forecast = forecast.rename(
            columns={
                "date": "target_date",
                "sales": "target"
            }
        )

        forecast["horizon"] = horizon
        forecast["day_of_week"] = (
            target_day_of_week
        )

        forecast = forecast.merge(
            weekly_reference[
                [
                    "store_nbr",
                    "family",
                    "day_of_week",
                    "sales"
                ]
            ],
            on=[
                "store_nbr",
                "family",
                "day_of_week"
            ],
            how="left"
        )

        forecast = forecast.rename(
            columns={
                "sales": "prediction"
            }
        )

        results.append(
            forecast[
                [
                    "target_date",
                    "store_nbr",
                    "family",
                    "horizon",
                    "target",
                    "prediction"
                ]
            ]
        )

    return pd.concat(
        results,
        ignore_index=True
    )

# ===========================================
# Baseline - Weekly Seasonal Naive Evaluation

weekly_seasonal_results = []

for origin in FORECAST_ORIGINS:

    forecast = weekly_seasonal_naive(
        df,
        origin,
        horizons=HORIZON
    )

    for horizon in range(1, HORIZON + 1):

        horizon_data = forecast[
            forecast["horizon"] == horizon
        ]

        metrics = calculate_metrics(
            horizon_data["target"],
            horizon_data["prediction"]
        )

        weekly_seasonal_results.append({
            "origin": pd.Timestamp(origin),
            "horizon": horizon,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"]
        })


weekly_seasonal_results_df = pd.DataFrame(
    weekly_seasonal_results
)

print("\nWeekly Seasonal Naive Baseline Results")
print("=" * 75)

print(
    weekly_seasonal_results_df.to_string(
        index=False
    )
)

# endregion

# region XGBoost Model Dataset

# ---------------------------------------------------------
# Artifact-aware XGBoost Dataset Configuration
# ---------------------------------------------------------

TRAIN_START_DATE = (
    pd.Timestamp(split_date)
    - pd.DateOffset(years=TRAIN_HISTORY_YEARS)
)

TRAIN_END_DATE = (
    pd.Timestamp(split_date)
    - pd.Timedelta(days=1)
)

VALIDATION_ORIGINS = [
    pd.Timestamp("2017-06-30"),
    pd.Timestamp("2017-07-15"),
    pd.Timestamp("2017-07-30"),
    # Held-out origin: not inspected during feature/threshold
    # decisions. Only look at this one at the very end, as a
    # final check that improvements generalize.
    pd.Timestamp("2017-07-10")
]

MODEL_FEATURES = [
    "store_nbr",
    "family",
    "store_type",
    "city",
    "state",
    "store_cluster",
    "day_of_week",
    "month",
    "lag_28",
    "lag_365",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "rolling_cv_28",
    "horizon",
    "momentum_7_28",
    "momentum_14_28",
    "trend_7_28",
    "yearly_ratio",
    "historical_spike_rate"
]

TARGET_COLUMN = "target"


# ---------------------------------------------------------
# Load existing artifacts when available
# ---------------------------------------------------------

if (
    XGB_TRAIN_DF_ARTIFACT.exists()
    and
    XGB_VALID_DF_ARTIFACT.exists()
):

    print(
        "\nLoading XGBoost datasets "
        "from persistent artifacts..."
    )

    xgb_train_df = pd.read_pickle(
        XGB_TRAIN_DF_ARTIFACT
    )

    xgb_valid_df = pd.read_pickle(
        XGB_VALID_DF_ARTIFACT
    )

    print(
        "XGBoost datasets loaded."
    )


# ---------------------------------------------------------
# Build datasets only when artifacts are unavailable
# ---------------------------------------------------------

else:

    print(
        "\nXGBoost dataset artifacts "
        "not found."
    )

    print(
        "Building XGBoost datasets..."
    )


    xgb_train_df = direct_df[
        (
            direct_df["date"] >= TRAIN_START_DATE
        )
        &
        (
            direct_df["date"] <= TRAIN_END_DATE
        )
    ][
        MODEL_FEATURES + [TARGET_COLUMN]
    ].copy()


    xgb_valid_df = direct_df[
        direct_df["date"].isin(
            VALIDATION_ORIGINS
        )
    ][
        ["date"] + MODEL_FEATURES + [TARGET_COLUMN]
    ].copy()


    # -----------------------------------------------------
    # Persist newly built datasets
    # -----------------------------------------------------

    xgb_train_df.to_pickle(
        XGB_TRAIN_DF_ARTIFACT
    )

    xgb_valid_df.to_pickle(
        XGB_VALID_DF_ARTIFACT
    )

    print(
        "XGBoost datasets built and persisted."
    )


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print(
    "\nXGBoost Model Dataset"
)

print("=" * 60)

print(
    "Train Shape      :",
    xgb_train_df.shape
)

print(
    "Validation Shape :",
    xgb_valid_df.shape
)


# endregion

# region XGBoost Categorical Preparation

CATEGORICAL_FEATURES = [
    "store_nbr",
    "family",
    "store_type",
    "city",
    "state",
    "store_cluster"
]


for column in CATEGORICAL_FEATURES:

    train_categories = (
        xgb_train_df[column]
        .astype("category")
        .cat.categories
    )

    xgb_train_df[column] = pd.Categorical(
        xgb_train_df[column],
        categories=train_categories
    )

    xgb_valid_df[column] = pd.Categorical(
        xgb_valid_df[column],
        categories=train_categories
    )


# Safety checks

for column in CATEGORICAL_FEATURES:

    assert (
        xgb_train_df[column].dtype
        == xgb_valid_df[column].dtype
    )

    assert (
        xgb_train_df[column]
        .cat
        .categories
        .equals(
            xgb_valid_df[column]
            .cat
            .categories
        )
    )


# endregion

# region XGBoost Training Dataset

X_train = (
    xgb_train_df[
        MODEL_FEATURES
    ]
    .copy()
)

y_train = (
    xgb_train_df[
        TARGET_COLUMN
    ]
    .copy()
)

X_valid = (
    xgb_valid_df[
        MODEL_FEATURES
    ]
    .copy()
)

y_valid = (
    xgb_valid_df[
        TARGET_COLUMN
    ]
    .copy()
)


# Safety checks

assert list(
    X_train.columns
) == list(
    X_valid.columns
)

for column in CATEGORICAL_FEATURES:

    assert (
        X_train[column].dtype
        ==
        X_valid[column].dtype
    )

    assert (
        X_train[column]
        .cat
        .categories
        .equals(
            X_valid[column]
            .cat
            .categories
        )
    )

# endregion

# region XGBoost Model Training

from xgboost import XGBRegressor


# ---------------------------------------------------------
# Load existing model when available
# ---------------------------------------------------------

if XGB_MODEL_ARTIFACT.exists():

    print(
        "\nLoading XGBoost model "
        "from artifact..."
    )

    xgb_model = XGBRegressor()

    xgb_model.load_model(
        XGB_MODEL_ARTIFACT
    )

    print(
        "XGBoost model loaded successfully."
    )


# ---------------------------------------------------------
# Train model when artifact is unavailable
# ---------------------------------------------------------

else:

    print(
        "\nXGBoost model artifact "
        "not found."
    )

    print(
        "Training XGBoost model..."
    )


    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        enable_categorical=True,
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1
    )


    xgb_model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_valid, y_valid)
        ],
        verbose=50
    )


    # -----------------------------------------------------
    # Save newly trained model
    # -----------------------------------------------------

    xgb_model.save_model(
        XGB_MODEL_ARTIFACT
    )

    print(
        "\nXGBoost model trained "
        "and saved successfully."
    )


# endregion

# region Tail Diagnostic Dataset

TAIL_THRESHOLD = 1000

TAIL_TRAIN_START = (
    pd.Timestamp(split_date)
    - pd.DateOffset(years=1)
)

TAIL_TRAIN_END = (
    pd.Timestamp(split_date)
    - pd.Timedelta(days=1)
)

TAIL_VALID_ORIGINS = [
    pd.Timestamp("2017-06-30"),
    pd.Timestamp("2017-07-15"),
    pd.Timestamp("2017-07-30")
]


# ---------------------------------------------------------
# Tail Training Dataset
# ---------------------------------------------------------

tail_train_df = direct_df[
    (
        direct_df["date"] >= TAIL_TRAIN_START
    )
    &
    (
        direct_df["date"] <= TAIL_TRAIN_END
    )
    &
    (
        direct_df["target"] > TAIL_THRESHOLD
    )
].copy()


# ---------------------------------------------------------
# Tail Validation Dataset
# ---------------------------------------------------------

tail_valid_df = direct_df[
    (
        direct_df["date"].isin(
            TAIL_VALID_ORIGINS
        )
    )
    &
    (
        direct_df["target"] > TAIL_THRESHOLD
    )
].copy()


# ---------------------------------------------------------
# Diagnostic Information
# ---------------------------------------------------------

print("\nTail Diagnostic Dataset")
print("=" * 60)

print(
    "Tail threshold:",
    TAIL_THRESHOLD
)

print(
    "Tail train rows:",
    len(tail_train_df)
)

print(
    "Tail validation rows:",
    len(tail_valid_df)
)

# endregion

# region XGBoost Validation Prediction

# ---------------------------------------------------------
# Prepare validation features for XGBoost prediction
# ---------------------------------------------------------

xgb_prediction_features = (
    xgb_valid_df[
        MODEL_FEATURES
    ]
    .copy()
)


# ---------------------------------------------------------
# Match training dtypes and categorical levels
# ---------------------------------------------------------

for column in MODEL_FEATURES:

    train_dtype = (
        xgb_train_df[column].dtype
    )

    if isinstance(
        train_dtype,
        pd.CategoricalDtype
    ):

        xgb_prediction_features[column] = (
            pd.Categorical(
                xgb_prediction_features[column],
                categories=(
                    xgb_train_df[column]
                    .cat
                    .categories
                ),
                ordered=(
                    xgb_train_df[column]
                    .cat
                    .ordered
                )
            )
        )

    else:

        xgb_prediction_features[column] = (
            xgb_prediction_features[column]
            .astype(train_dtype)
        )


# ---------------------------------------------------------
# Safety checks
# ---------------------------------------------------------

assert list(
    xgb_prediction_features.columns
) == MODEL_FEATURES


for column in MODEL_FEATURES:

    assert (
        xgb_prediction_features[column].dtype
        ==
        xgb_train_df[column].dtype
    )

    if isinstance(
        xgb_train_df[column].dtype,
        pd.CategoricalDtype
    ):

        assert (
            xgb_prediction_features[column]
            .cat
            .categories
            .equals(
                xgb_train_df[column]
                .cat
                .categories
            )
        )


# ---------------------------------------------------------
# Generate validation predictions
# ---------------------------------------------------------

xgb_valid_predictions = (
    xgb_model.predict(
        xgb_prediction_features
    )
)


# ---------------------------------------------------------
# Build complete validation results
# ---------------------------------------------------------

xgb_valid_results = (
    xgb_valid_df[
        [
            "date",
            "target"
        ]
        +
        MODEL_FEATURES
    ]
    .reset_index(drop=True)
    .copy()
)


xgb_valid_results["prediction"] = (
    xgb_valid_predictions
)


# ---------------------------------------------------------
# Final safety checks
# ---------------------------------------------------------

assert len(
    xgb_valid_results
) == len(
    xgb_valid_predictions
)

assert (
    xgb_valid_results[
        "prediction"
    ].notna().all()
)


# ---------------------------------------------------------
# Save validation results
# ---------------------------------------------------------

xgb_valid_results.to_pickle(
    XGB_VALID_RESULTS_ARTIFACT
)


print(
    "\nXGBoost validation predictions "
    "generated successfully."
)

print(
    "Validation rows:",
    len(xgb_valid_results)
)

print(
    "Validation results saved."
)


# endregion

# region Prediction Confidence Layer

# ---------------------------------------------------------
# Purpose:
# Attach a confidence label to each prediction, based on the
# series' own historical volatility (rolling_cv_28). This
# turns the model's known weak spot (high-volatility series
# are harder to predict) into an explicit, actionable signal
# instead of a silent failure mode.
#
# This does NOT change the model or its predictions. It only
# adds a post-hoc reliability annotation.
# ---------------------------------------------------------

print("\nPrediction Confidence Layer")
print("=" * 70)

# ---------------------------------------------------------
# 1. Define confidence tiers from rolling_cv_28
#
# Thresholds are based on the tercile split of rolling_cv_28
# in the validation set itself, so tiers are data-driven
# rather than arbitrary fixed cutoffs.
# ---------------------------------------------------------

cv_low_threshold = xgb_valid_results["rolling_cv_28"].quantile(1 / 3)
cv_high_threshold = xgb_valid_results["rolling_cv_28"].quantile(2 / 3)

print(f"rolling_cv_28 tercile thresholds: "
      f"low <= {cv_low_threshold:.3f} < medium <= "
      f"{cv_high_threshold:.3f} < high")


def assign_confidence(cv_value):

    if pd.isna(cv_value):
        return "unknown"

    if cv_value <= cv_low_threshold:
        return "high"

    if cv_value <= cv_high_threshold:
        return "medium"

    return "low"


xgb_valid_results["prediction_confidence"] = (
    xgb_valid_results["rolling_cv_28"]
    .apply(assign_confidence)
)

# ---------------------------------------------------------
# 2. Validate: does confidence actually correlate with error?
#
# If "low confidence" rows do NOT show worse WAPE than "high
# confidence" rows, this label is meaningless and should not
# ship. This check is the whole point of the layer.
# ---------------------------------------------------------

xgb_valid_results["abs_error"] = (
    (xgb_valid_results["prediction"] - xgb_valid_results["target"])
    .abs()
)

confidence_validation = []

for tier in ["high", "medium", "low"]:

    tier_data = xgb_valid_results[
        xgb_valid_results["prediction_confidence"] == tier
    ]

    if len(tier_data) == 0:
        continue

    tier_metrics = calculate_metrics(
        tier_data["target"], tier_data["prediction"]
    )

    severe_rate = (
        (tier_data["prediction"] < 0.5 * tier_data["target"])
        & (tier_data["target"] > TAIL_THRESHOLD)
    ).sum() / max((tier_data["target"] > TAIL_THRESHOLD).sum(), 1) * 100

    confidence_validation.append(
        {
            "confidence_tier": tier,
            "count": len(tier_data),
            "MAE": tier_metrics["MAE"],
            "WAPE": tier_metrics["WAPE"] * 100,
            "severe_underprediction_rate_pct": severe_rate,
        }
    )

confidence_validation_df = pd.DataFrame(confidence_validation)

print("\nError by Confidence Tier")
print("-" * 70)

print(
    confidence_validation_df
    .round({"MAE": 2, "WAPE": 2, "severe_underprediction_rate_pct": 2})
    .to_string(index=False)
)

# ---------------------------------------------------------
# 3. Save annotated results
# ---------------------------------------------------------

xgb_valid_results.to_pickle(
    XGB_VALID_RESULTS_ARTIFACT
)

print(
    "\nValidation results with confidence labels saved."
)

# endregion

# region Tail Error Dataset Artifact

TAIL_ERROR_ARTIFACT = (
    XGB_VALID_RESULTS_ARTIFACT.parent
    / "tail_error_df.pkl"
)


# ---------------------------------------------------------
# Load existing Tail Error Dataset when available
# ---------------------------------------------------------

if TAIL_ERROR_ARTIFACT.exists():

    print(
        "\nLoading Tail Error Dataset "
        "from artifact..."
    )

    tail_error_df = pd.read_pickle(
        TAIL_ERROR_ARTIFACT
    )

    print(
        "Tail Error Dataset loaded successfully."
    )


# ---------------------------------------------------------
# Build Tail Error Dataset when artifact is unavailable
# ---------------------------------------------------------

else:

    print(
        "\nTail Error Dataset artifact "
        "not found."
    )

    print(
        "Building Tail Error Dataset..."
    )

    tail_error_df = (
        xgb_valid_results[
            xgb_valid_results["target"]
            > TAIL_THRESHOLD
        ]
        .copy()
    )

    tail_error_df["error"] = (
        tail_error_df["prediction"]
        - tail_error_df["target"]
    )

    tail_error_df["absolute_error"] = (
        np.abs(
            tail_error_df["error"]
        )
    )

    tail_error_df["prediction_ratio"] = (
        tail_error_df["prediction"]
        /
        tail_error_df["target"]
    )

    tail_error_df.to_pickle(
        TAIL_ERROR_ARTIFACT
    )

    print(
        "Tail Error Dataset built "
        "and persisted."
    )


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

print(
    "\nTail Error Dataset"
)

print("=" * 60)

print(
    "Tail threshold :",
    TAIL_THRESHOLD
)

print(
    "Tail rows      :",
    len(tail_error_df)
)

print(
    "Artifact       :",
    TAIL_ERROR_ARTIFACT
)

# endregion

# region Tail Error Analysis

# ---------------------------------------------------------
# 6. Overall Tail Metrics
# ---------------------------------------------------------

tail_metrics = calculate_metrics(
    tail_error_df["target"],
    tail_error_df["prediction"]
)

print("\nTail Error Analysis")
print("=" * 60)

print(
    f"Tail Threshold : > {TAIL_THRESHOLD}"
)

print(
    f"Tail Rows      : "
    f"{len(tail_error_df)}"
)

print(
    f"MAE            : "
    f"{tail_metrics['MAE']:.3f}"
)

print(
    f"RMSE           : "
    f"{tail_metrics['RMSE']:.3f}"
)

print(
    f"WAPE           : "
    f"{tail_metrics['WAPE'] * 100:.2f}%"
)

print(
    f"Mean Error     : "
    f"{tail_error_df['error'].mean():.3f}"
)

print(
    f"Median Error   : "
    f"{tail_error_df['error'].median():.3f}"
)


# ---------------------------------------------------------
# 7. Tail Performance by Horizon
# ---------------------------------------------------------

tail_horizon_results = []

for horizon in sorted(
    tail_error_df["horizon"].unique()
):

    horizon_data = (
        tail_error_df[
            tail_error_df["horizon"]
            == horizon
        ]
    )

    metrics = calculate_metrics(
        horizon_data["target"],
        horizon_data["prediction"]
    )

    tail_horizon_results.append(
        {
            "horizon": horizon,
            "count": len(horizon_data),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE":
                metrics["WAPE"] * 100,
            "Mean_Error":
                horizon_data["error"].mean(),
            "Mean_Ratio":
                horizon_data[
                    "prediction_ratio"
                ].mean()
        }
    )

tail_horizon_results = (
    pd.DataFrame(
        tail_horizon_results
    )
)

print("\nTail Performance by Horizon")
print("-" * 60)

print(
    tail_horizon_results.round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2,
            "Mean_Error": 3,
            "Mean_Ratio": 3
        }
    )
)


# ---------------------------------------------------------
# 8. Tail Performance by Validation Origin
# ---------------------------------------------------------

tail_origin_results = []

for origin in sorted(
    tail_error_df["date"].unique()
):

    origin_data = (
        tail_error_df[
            tail_error_df["date"]
            == origin
        ]
    )

    metrics = calculate_metrics(
        origin_data["target"],
        origin_data["prediction"]
    )

    tail_origin_results.append(
        {
            "origin": origin,
            "count": len(origin_data),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE":
                metrics["WAPE"] * 100,
            "Mean_Error":
                origin_data["error"].mean(),
            "Mean_Ratio":
                origin_data[
                    "prediction_ratio"
                ].mean()
        }
    )

tail_origin_results = (
    pd.DataFrame(
        tail_origin_results
    )
)

print("\nTail Performance by Validation Origin")
print("-" * 60)

print(
    tail_origin_results.round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2,
            "Mean_Error": 3,
            "Mean_Ratio": 3
        }
    )
)


# ---------------------------------------------------------
# 9. Spike Severity
# ---------------------------------------------------------

tail_error_df["spike_level"] = pd.cut(
    tail_error_df["target"],
    bins=[
        1000,
        2000,
        5000,
        10000,
        20000,
        np.inf
    ],
    labels=[
        "1000-2000",
        "2000-5000",
        "5000-10000",
        "10000-20000",
        ">20000"
    ]
)


spike_severity_results = []

for level in (
    tail_error_df[
        "spike_level"
    ].cat.categories
):

    level_data = (
        tail_error_df[
            tail_error_df["spike_level"]
            == level
        ]
    )

    if len(level_data) == 0:
        continue

    metrics = calculate_metrics(
        level_data["target"],
        level_data["prediction"]
    )

    spike_severity_results.append(
        {
            "spike_level": level,
            "count": len(level_data),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE":
                metrics["WAPE"] * 100,
            "Mean_Error":
                level_data["error"].mean(),
            "Mean_Ratio":
                level_data[
                    "prediction_ratio"
                ].mean()
        }
    )

spike_severity_results = (
    pd.DataFrame(
        spike_severity_results
    )
)

print("\nPerformance by Spike Severity")
print("-" * 60)

print(
    spike_severity_results.round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2,
            "Mean_Error": 3,
            "Mean_Ratio": 3
        }
    )
)


# ---------------------------------------------------------
# 10. Largest Tail Underpredictions
# ---------------------------------------------------------

tail_underpredictions = (
    tail_error_df
    .sort_values(
        "error",
        ascending=True
    )
    .head(30)
)

print("\nLargest Tail Underpredictions")
print("-" * 60)

print(
    tail_underpredictions[
        [
            "date",
            "horizon",
            "store_nbr",
            "family",
            "target",
            "prediction",
            "error",
            "prediction_ratio"
        ]
    ]
    .round(3)
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 11. Prediction Coverage
# ---------------------------------------------------------

tail_error_df["coverage_level"] = pd.cut(
    tail_error_df["prediction_ratio"],
    bins=[
        -np.inf,
        0.25,
        0.50,
        0.75,
        1.00,
        1.25,
        1.50,
        np.inf
    ],
    labels=[
        "<25%",
        "25%-50%",
        "50%-75%",
        "75%-100%",
        "100%-125%",
        "125%-150%",
        ">150%"
    ]
)


coverage_results = (
    tail_error_df[
        "coverage_level"
    ]
    .value_counts(
        sort=False
    )
    .rename_axis(
        "prediction_coverage"
    )
    .reset_index(
        name="count"
    )
)

coverage_results["percentage"] = (
    coverage_results["count"]
    /
    len(tail_error_df)
    *
    100
)


print("\nTail Prediction Coverage")
print("-" * 60)

print(
    coverage_results.round(
        {
            "percentage": 2
        }
    )
)


# ---------------------------------------------------------
# 12. Underprediction Rate
# ---------------------------------------------------------

underprediction_rate = (
    (
        tail_error_df["prediction"]
        <
        tail_error_df["target"]
    )
    .mean()
    * 100
)

severe_underprediction_rate = (
    (
        tail_error_df["prediction_ratio"]
        < 0.50
    )
    .mean()
    * 100
)


print("\nTail Underprediction Diagnostics")
print("-" * 60)

print(
    f"Underprediction Rate     : "
    f"{underprediction_rate:.2f}%"
)

print(
    f"Prediction < 50% Actual : "
    f"{severe_underprediction_rate:.2f}%"
)


# endregion

# region Severe Tail Underprediction Diagnostic

# =========================================================
# Purpose
# =========================================================
#
# Investigate severe tail underpredictions without changing
# the XGBoost model.
#
# Main questions:
# 1. Which tail observations are severely underpredicted?
# 2. Are these concentrated in specific families?
# 3. Are they concentrated in specific stores?
# 4. Are they concentrated in specific horizons?
# 5. Did the historical lag / rolling features already
#    contain evidence of the spike?
# 6. How do severe underpredictions differ from well-
#    predicted tail observations?
#
# No model retraining is performed in this region.
# =========================================================


# ---------------------------------------------------------
# 1. Create Severe Underprediction Dataset
# ---------------------------------------------------------

SEVERE_UNDERPREDICTION_RATIO = 0.50

severe_tail_df = (
    tail_error_df[
        tail_error_df["prediction_ratio"]
        <
        SEVERE_UNDERPREDICTION_RATIO
    ]
    .copy()
)


print("\nSevere Tail Underprediction Diagnostic")
print("=" * 70)

print(
    "Tail rows:",
    len(tail_error_df)
)

print(
    "Severe underprediction threshold:",
    f"< {SEVERE_UNDERPREDICTION_RATIO:.0%}"
)

print(
    "Severe underprediction rows:",
    len(severe_tail_df)
)

print(
    "Severe underprediction rate:",
    f"{len(severe_tail_df) / len(tail_error_df) * 100:.2f}%"
)

# ---------------------------------------------------------
# 5. Severe Underprediction by Family
# ---------------------------------------------------------

severe_family_results = []

for family in sorted(
    severe_tail_df["family"].unique()
):

    family_data = (
        severe_tail_df[
            severe_tail_df["family"]
            == family
        ]
    )

    all_tail_family_data = (
        tail_error_df[
            tail_error_df["family"]
            == family
        ]
    )

    severe_family_results.append(
        {
            "family": family,
            "severe_count":
                len(family_data),
            "tail_count":
                len(all_tail_family_data),
            "severe_rate":
                (
                    len(family_data)
                    /
                    len(all_tail_family_data)
                    *
                    100
                ),
            "mean_ratio":
                family_data[
                    "prediction_ratio"
                ].mean(),
            "mean_error":
                family_data[
                    "error"
                ].mean()
        }
    )


severe_family_results = (
    pd.DataFrame(
        severe_family_results
    )
    .sort_values(
        "severe_count",
        ascending=False
    )
)


print("\nSevere Underprediction by Family")
print("-" * 70)

print(
    severe_family_results
    .round(
        {
            "severe_rate": 2,
            "mean_ratio": 3,
            "mean_error": 3
        }
    )
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 6. Severe Underprediction by Store
# ---------------------------------------------------------

severe_store_results = []

for store in sorted(
    severe_tail_df["store_nbr"].unique()
):

    store_data = (
        severe_tail_df[
            severe_tail_df["store_nbr"]
            == store
        ]
    )

    all_tail_store_data = (
        tail_error_df[
            tail_error_df["store_nbr"]
            == store
        ]
    )

    severe_store_results.append(
        {
            "store_nbr": store,
            "severe_count":
                len(store_data),
            "tail_count":
                len(all_tail_store_data),
            "severe_rate":
                (
                    len(store_data)
                    /
                    len(all_tail_store_data)
                    *
                    100
                ),
            "mean_ratio":
                store_data[
                    "prediction_ratio"
                ].mean(),
            "mean_error":
                store_data[
                    "error"
                ].mean()
        }
    )


severe_store_results = (
    pd.DataFrame(
        severe_store_results
    )
    .sort_values(
        "severe_count",
        ascending=False
    )
)


print("\nSevere Underprediction by Store")
print("-" * 70)

print(
    severe_store_results
    .head(10)
    .round(
        {
            "severe_rate": 2,
            "mean_ratio": 3,
            "mean_error": 3
        }
    )
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 7. Severe Underprediction by Horizon
# ---------------------------------------------------------

severe_horizon_results = []

for horizon in sorted(
    severe_tail_df["horizon"].unique()
):

    horizon_data = (
        severe_tail_df[
            severe_tail_df["horizon"]
            == horizon
        ]
    )

    all_tail_horizon_data = (
        tail_error_df[
            tail_error_df["horizon"]
            == horizon
        ]
    )

    severe_horizon_results.append(
        {
            "horizon": horizon,
            "severe_count":
                len(horizon_data),
            "tail_count":
                len(all_tail_horizon_data),
            "severe_rate":
                (
                    len(horizon_data)
                    /
                    len(all_tail_horizon_data)
                    *
                    100
                ),
            "mean_ratio":
                horizon_data[
                    "prediction_ratio"
                ].mean(),
            "mean_error":
                horizon_data[
                    "error"
                ].mean()
        }
    )


severe_horizon_results = (
    pd.DataFrame(
        severe_horizon_results
    )
)


print("\nSevere Underprediction by Horizon")
print("-" * 70)

print(
    severe_horizon_results
    .round(
        {
            "severe_rate": 2,
            "mean_ratio": 3,
            "mean_error": 3
        }
    )
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 8. Severe Underprediction by Validation Origin
# ---------------------------------------------------------

severe_origin_results = []

for origin in sorted(
    severe_tail_df["date"].unique()
):

    origin_data = (
        severe_tail_df[
            severe_tail_df["date"]
            == origin
        ]
    )

    all_tail_origin_data = (
        tail_error_df[
            tail_error_df["date"]
            == origin
        ]
    )

    severe_origin_results.append(
        {
            "origin": origin,
            "severe_count":
                len(origin_data),
            "tail_count":
                len(all_tail_origin_data),
            "severe_rate":
                (
                    len(origin_data)
                    /
                    len(all_tail_origin_data)
                    *
                    100
                ),
            "mean_ratio":
                origin_data[
                    "prediction_ratio"
                ].mean(),
            "mean_error":
                origin_data[
                    "error"
                ].mean()
        }
    )


severe_origin_results = (
    pd.DataFrame(
        severe_origin_results
    )
)


print("\nSevere Underprediction by Validation Origin")
print("-" * 70)

print(
    severe_origin_results
    .round(
        {
            "severe_rate": 2,
            "mean_ratio": 3,
            "mean_error": 3
        }
    )
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 9. Historical Feature Diagnostics
# ---------------------------------------------------------

diagnostic_feature_columns = [
    column
    for column in MODEL_FEATURES
    if any(
        keyword in column.lower()
        for keyword in [
            "lag",
            "rolling",
            "ma_",
            "mean",
            "max",
            "min",
            "std"
        ]
    )
]


if len(diagnostic_feature_columns) > 0:

    historical_comparison = []

    for column in diagnostic_feature_columns:

        severe_values = pd.to_numeric(
            severe_tail_df[column],
            errors="coerce"
        )

        all_values = pd.to_numeric(
            tail_error_df[column],
            errors="coerce"
        )

        historical_comparison.append(
            {
                "feature": column,

                "severe_mean":
                    severe_values.mean(),

                "tail_mean":
                    all_values.mean(),

                "severe_median":
                    severe_values.median(),

                "tail_median":
                    all_values.median()
            }
        )


    historical_comparison = (
        pd.DataFrame(
            historical_comparison
        )
    )


    historical_comparison[
        "mean_difference"
    ] = (
        historical_comparison[
            "severe_mean"
        ]
        -
        historical_comparison[
            "tail_mean"
        ]
    )


    historical_comparison[
        "median_difference"
    ] = (
        historical_comparison[
            "severe_median"
        ]
        -
        historical_comparison[
            "tail_median"
        ]
    )


    historical_comparison = (
        historical_comparison
        .sort_values(
            "mean_difference",
            key=np.abs,
            ascending=False
        )
    )


    print(
        "\nHistorical Feature Comparison:"
    )

    print("-" * 70)

    print(
        historical_comparison
        .round(3)
        .to_string(
            index=False
        )
    )
    
# ---------------------------------------------------------
# 10. Worst Severe Underpredictions
# ---------------------------------------------------------

print(
    "\nWorst Severe Underpredictions"
)

print("-" * 70)


worst_severe_columns = [
    "date",
    "horizon",
    "store_nbr",
    "family",
    "target",
    "prediction",
    "error",
    "prediction_ratio"
]


worst_severe_columns = [
    column
    for column in worst_severe_columns
    if column in severe_tail_df.columns
]


worst_severe = (
    severe_tail_df[
        worst_severe_columns
    ]
    .sort_values(
        "prediction_ratio",
        ascending=True
    )
    .head(30)
)


print(
    worst_severe.to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 11. Compare Severe vs Well-Predicted Tail
# ---------------------------------------------------------
#
# Well-predicted tail:
# 0.75 <= prediction / actual <= 1.25
#
# This creates a reference population for identifying
# characteristics that distinguish severe failures.
# ---------------------------------------------------------

well_predicted_tail_df = (
    tail_error_df[
        (
            tail_error_df[
                "prediction_ratio"
            ]
            >= 0.75
        )
        &
        (
            tail_error_df[
                "prediction_ratio"
            ]
            <= 1.25
        )
    ]
    .copy()
)


print(
    "\nSevere vs Well-Predicted Tail"
)

print("-" * 70)

print(
    "Severe underpredictions:",
    len(severe_tail_df)
)

print(
    "Well-predicted tail:",
    len(well_predicted_tail_df)
)


# ---------------------------------------------------------
# 12. Key Series Concentration
# ---------------------------------------------------------
#
# Store + family combinations.
#
# This is particularly important because a global model
# may perform well overall while struggling with specific
# series.
# ---------------------------------------------------------

severe_tail_df[
    "series"
] = (
    severe_tail_df[
        "store_nbr"
    ].astype(str)
    +
    " | "
    +
    severe_tail_df[
        "family"
    ].astype(str)
)


tail_error_df[
    "series"
] = (
    tail_error_df[
        "store_nbr"
    ].astype(str)
    +
    " | "
    +
    tail_error_df[
        "family"
    ].astype(str)
)


series_results = (
    severe_tail_df[
        "series"
    ]
    .value_counts()
    .rename_axis(
        "series"
    )
    .reset_index(
        name="severe_count"
    )
)


series_tail_counts = (
    tail_error_df[
        "series"
    ]
    .value_counts()
    .rename_axis(
        "series"
    )
    .reset_index(
        name="tail_count"
    )
)


series_results = (
    series_results
    .merge(
        series_tail_counts,
        on="series",
        how="left"
    )
)


series_results[
    "severe_rate"
] = (
    series_results[
        "severe_count"
    ]
    /
    series_results[
        "tail_count"
    ]
    *
    100
)


series_results = (
    series_results
    .sort_values(
        [
            "severe_count",
            "severe_rate"
        ],
        ascending=False
    )
)


print(
    "\nSeries Concentration of Severe Underpredictions"
)

print("-" * 70)

print(
    series_results
    .head(30)
    .round(
        {
            "severe_rate": 2
        }
    )
    .to_string(
        index=False
    )
)


# ---------------------------------------------------------
# 13. Final Diagnostic Summary
# ---------------------------------------------------------

print(
    "\nDiagnostic Summary"
)

print("=" * 70)

print(
    f"Severe tail cases      : "
    f"{len(severe_tail_df)}"
)

print(
    f"Severe rate            : "
    f"{len(severe_tail_df) / len(tail_error_df) * 100:.2f}%"
)

print(
    f"Well-predicted cases   : "
    f"{len(well_predicted_tail_df)}"
)

print(
    f"Historical features    : "
    f"{len(diagnostic_feature_columns)}"
)

print(
    f"Unique severe series   : "
    f"{severe_tail_df['series'].nunique()}"
)


# endregion

# region SEVERE SPIKE: FULL MODEL FEATURE SIGNAL ANALYSIS
#
# Purpose:
# Compare ALL features available to XGBoost at prediction
# time between:
#
#   1. Severe underpredicted tail observations
#   2. Well-predicted tail observations
#
# This determines whether the information needed to predict
# severe spikes already exists in the current feature set.
#
# No model retraining is performed.
# =========================================================


print(
    "\nSEVERE SPIKE FULL MODEL FEATURE SIGNAL ANALYSIS"
)

print("=" * 70)


# ---------------------------------------------------------
# Reference populations
# ---------------------------------------------------------

severe_signal_df = (
    severe_tail_df[
        MODEL_FEATURES
    ]
    .copy()
)


well_predicted_signal_df = (
    well_predicted_tail_df[
        MODEL_FEATURES
    ]
    .copy()
)


print(
    "Severe spike rows       :",
    len(severe_signal_df)
)

print(
    "Well-predicted rows     :",
    len(well_predicted_signal_df)
)

print(
    "Model features analyzed:",
    len(MODEL_FEATURES)
)


# ---------------------------------------------------------
# Separate numeric and categorical features
# ---------------------------------------------------------

numeric_model_features = [
    column
    for column in MODEL_FEATURES
    if pd.api.types.is_numeric_dtype(
        xgb_valid_results[column]
    )
]


categorical_model_features = [
    column
    for column in MODEL_FEATURES
    if column not in numeric_model_features
]


# =========================================================
# 1. Numeric Feature Comparison
# =========================================================

numeric_feature_comparison = []


for column in numeric_model_features:

    severe_values = pd.to_numeric(
        severe_signal_df[column],
        errors="coerce"
    )

    well_values = pd.to_numeric(
        well_predicted_signal_df[column],
        errors="coerce"
    )


    severe_mean = (
        severe_values.mean()
    )

    well_mean = (
        well_values.mean()
    )

    severe_median = (
        severe_values.median()
    )

    well_median = (
        well_values.median()
    )


    numeric_feature_comparison.append(
        {
            "feature": column,

            "severe_mean":
                severe_mean,

            "well_mean":
                well_mean,

            "mean_difference":
                severe_mean
                -
                well_mean,

            "severe_median":
                severe_median,

            "well_median":
                well_median,

            "median_difference":
                severe_median
                -
                well_median
        }
    )


numeric_feature_comparison = (
    pd.DataFrame(
        numeric_feature_comparison
    )
)


# Relative difference based on well-predicted mean

numeric_feature_comparison[
    "relative_mean_difference_pct"
] = (
    (
        numeric_feature_comparison[
            "mean_difference"
        ]
        /
        numeric_feature_comparison[
            "well_mean"
        ].replace(
            0,
            np.nan
        )
    )
    * 100
)


numeric_feature_comparison = (
    numeric_feature_comparison
    .sort_values(
        "relative_mean_difference_pct",
        key=np.abs,
        ascending=False
    )
)


print(
    "\nNumeric Feature Comparison"
)

print("-" * 70)

print(
    numeric_feature_comparison
    .round(3)
    .to_string(
        index=False
    )
)


# =========================================================
# 2. Categorical Feature Comparison
# =========================================================

print(
    "\nCategorical Feature Comparison"
)

print("-" * 70)


for column in categorical_model_features:

    severe_distribution = (
        severe_signal_df[column]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .rename(
            "severe_pct"
        )
    )


    well_distribution = (
        well_predicted_signal_df[column]
        .value_counts(
            normalize=True
        )
        .mul(100)
        .rename(
            "well_pct"
        )
    )


    categorical_comparison = (
        pd.concat(
            [
                severe_distribution,
                well_distribution
            ],
            axis=1
        )
        .fillna(0)
    )


    categorical_comparison[
        "difference_pct"
    ] = (
        categorical_comparison[
            "severe_pct"
        ]
        -
        categorical_comparison[
            "well_pct"
        ]
    )


    categorical_comparison = (
        categorical_comparison
        .sort_values(
            "difference_pct",
            key=np.abs,
            ascending=False
        )
    )


    print(
        f"\nFeature: {column}"
    )

    print(
        categorical_comparison
        .round(2)
        .to_string()
    )


# =========================================================
# 3. Strongest Numeric Signal
# =========================================================

if len(
    numeric_feature_comparison
) > 0:

    strongest_numeric = (
        numeric_feature_comparison
        .iloc[0]
    )


    print(
        "\nStrongest Numeric Signal"
    )

    print("-" * 70)

    print(
        "Feature:",
        strongest_numeric[
            "feature"
        ]
    )

    print(
        "Relative mean difference:",
        f"{strongest_numeric['relative_mean_difference_pct']:.2f}%"
    )


# =========================================================
# 4. Diagnostic Interpretation
# =========================================================

print(
    "\nFull Feature Signal Diagnostic"
)

print("-" * 70)

print(
    "This analysis compares severe spike cases against "
    "well-predicted tail cases using every feature available "
    "to the current XGBoost model."
)

print(
    "A strong and consistent difference suggests that "
    "predictive information exists in the current feature "
    "space but may not be adequately exploited by the model."
)

print(
    "A weak difference across most features suggests that "
    "the severe spikes are poorly observable from the "
    "information currently supplied to the model."
)

# endregion

# region XGBoost Error Analysis
#
# Diagnostic analysis of validation errors.
#
# This region does not retrain the model.
# It analyzes the predictions already generated by XGBoost.
# ---------------------------------------------------------

print("\nXGBoost Error Analysis")
print("=" * 60)


# ---------------------------------------------------------
# 1. Prepare error columns
# ---------------------------------------------------------

error_analysis_df = (
    xgb_valid_results[
        [
            "date",
            "horizon",
            "store_nbr",
            "family",
            "target",
            "prediction"
        ]
    ]
    .copy()
)

error_analysis_df["error"] = (
    error_analysis_df["prediction"]
    - error_analysis_df["target"]
)

error_analysis_df["absolute_error"] = (
    np.abs(
        error_analysis_df["error"]
    )
)


# ---------------------------------------------------------
# 2. Error by Horizon
# ---------------------------------------------------------

horizon_error_results = []

for horizon in sorted(
    error_analysis_df["horizon"].unique()
):

    horizon_data = (
        error_analysis_df[
            error_analysis_df["horizon"]
            == horizon
        ]
    )

    metrics = calculate_metrics(
        horizon_data["target"],
        horizon_data["prediction"]
    )

    mean_error = (
        horizon_data["error"].mean()
    )

    horizon_error_results.append(
        {
            "horizon": horizon,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"] * 100,
            "Mean_Error": mean_error
        }
    )

horizon_error_results = (
    pd.DataFrame(
        horizon_error_results
    )
)

print("\nError by Horizon")
print("-" * 60)

print(
    horizon_error_results.round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2,
            "Mean_Error": 3
        }
    )
)


# ---------------------------------------------------------
# 3. Error by Validation Origin
# ---------------------------------------------------------

origin_error_results = []

for origin in sorted(
    error_analysis_df["date"].unique()
):

    origin_data = (
        error_analysis_df[
            error_analysis_df["date"]
            == origin
        ]
    )

    metrics = calculate_metrics(
        origin_data["target"],
        origin_data["prediction"]
    )

    mean_error = (
        origin_data["error"].mean()
    )

    origin_error_results.append(
        {
            "origin": origin,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"] * 100,
            "Mean_Error": mean_error
        }
    )

origin_error_results = (
    pd.DataFrame(
        origin_error_results
    )
)

print("\nError by Validation Origin")
print("-" * 60)

print(
    origin_error_results.round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2,
            "Mean_Error": 3
        }
    )
)


# ---------------------------------------------------------
# 4. Error by Actual Sales Level
# ---------------------------------------------------------

error_analysis_df["sales_level"] = (
    pd.cut(
        error_analysis_df["target"],
        bins=[
            -np.inf,
            0,
            10,
            50,
            200,
            1000,
            np.inf
        ],
        labels=[
            "0",
            "0 < sales <= 10",
            "10 < sales <= 50",
            "50 < sales <= 200",
            "200 < sales <= 1000",
            "> 1000"
        ],
        right=True
    )
)


sales_level_results = []

for level in (
    error_analysis_df[
        "sales_level"
    ].cat.categories
):

    level_data = (
        error_analysis_df[
            error_analysis_df["sales_level"]
            == level
        ]
    )

    if len(level_data) == 0:
        continue

    metrics = calculate_metrics(
        level_data["target"],
        level_data["prediction"]
    )

    mean_error = (
        level_data["error"].mean()
    )

    sales_level_results.append(
        {
            "sales_level": level,
            "count": len(level_data),
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"] * 100,
            "Mean_Error": mean_error
        }
    )

sales_level_results = (
    pd.DataFrame(
        sales_level_results
    )
)

print("\nError by Actual Sales Level")
print("-" * 60)

print(
    sales_level_results.round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2,
            "Mean_Error": 3
        }
    )
)


# ---------------------------------------------------------
# 5. Overall Bias
# ---------------------------------------------------------

overall_mean_error = (
    error_analysis_df["error"].mean()
)

overall_median_error = (
    error_analysis_df["error"].median()
)

print("\nOverall Prediction Bias")
print("-" * 60)

print(
    f"Mean Error   : "
    f"{overall_mean_error:.3f}"
)

print(
    f"Median Error : "
    f"{overall_median_error:.3f}"
)

if overall_mean_error < 0:

    print(
        "Interpretation: "
        "The model tends to underpredict."
    )

elif overall_mean_error > 0:

    print(
        "Interpretation: "
        "The model tends to overpredict."
    )

else:

    print(
        "Interpretation: "
        "No overall directional bias."
    )


# ---------------------------------------------------------
# 6. Bias by Horizon
# ---------------------------------------------------------

bias_by_horizon = (
    error_analysis_df
    .groupby("horizon")["error"]
    .agg(
        mean="mean",
        median="median"
    )
    .reset_index()
)

print("\nPrediction Bias by Horizon")
print("-" * 60)

print(
    bias_by_horizon.round(3)
)


# ---------------------------------------------------------
# 7. Worst Individual Predictions
# ---------------------------------------------------------

worst_predictions = (
    error_analysis_df
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(20)
)

print("\nWorst 20 Predictions")
print("-" * 60)

print(
    worst_predictions[
        [
            "date",
            "horizon",
            "target",
            "prediction",
            "error",
            "absolute_error"
        ]
    ]
    .round(3)
    .to_string(index=False)
)


# ---------------------------------------------------------
# 8. Overall Error Summary
# ---------------------------------------------------------

overall_metrics = calculate_metrics(
    error_analysis_df["target"],
    error_analysis_df["prediction"]
)

print("\nOverall Error Summary")
print("-" * 60)

print(
    f"MAE  : "
    f"{overall_metrics['MAE']:.3f}"
)

print(
    f"RMSE : "
    f"{overall_metrics['RMSE']:.3f}"
)

print(
    f"WAPE : "
    f"{overall_metrics['WAPE'] * 100:.2f}%"
)

print(
    f"Mean Error : "
    f"{overall_mean_error:.3f}"
)


# endregion

# region XGBoost Model Interpretation

# ---------------------------------------------------------
# Optional Model Interpretation
#
# This region is intentionally optional because SHAP analysis
# can be computationally expensive.
#
# Run this region only when model interpretation is required.
# It does NOT retrain the model.
# ---------------------------------------------------------

RUN_XGB_INTERPRETATION = True


if RUN_XGB_INTERPRETATION:

    # -----------------------------------------------------
    # 1. Built-in Feature Importance
    # -----------------------------------------------------

    xgb_feature_importance = (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "importance":
                    xgb_model.feature_importances_
            }
        )
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )


    print(
        "\nXGBoost Feature Importance"
    )

    print("=" * 60)

    print(
        xgb_feature_importance
        .to_string(index=False)
    )


    # -----------------------------------------------------
    # 2. Prepare Representative SHAP Sample
    # -----------------------------------------------------

    SHAP_SAMPLE_SIZE = 3000

    shap_sample = (
        xgb_valid_results[
            MODEL_FEATURES +
            ["target", "prediction"]
        ]
        .copy()
    )


    if len(shap_sample) > SHAP_SAMPLE_SIZE:

        shap_sample = (
            shap_sample
            .sample(
                n=SHAP_SAMPLE_SIZE,
                random_state=42
            )
        )


    X_shap = (
        shap_sample[
            MODEL_FEATURES
        ]
        .copy()
    )


    print(
        "\nSHAP Analysis"
    )

    print("=" * 60)

    print(
        "SHAP Sample Size:",
        len(X_shap)
    )


    # -----------------------------------------------------
    # 3. SHAP Tree Explainer
    # -----------------------------------------------------

    import shap

    explainer = shap.TreeExplainer(
        xgb_model
    )


    shap_values = (
        explainer(
            X_shap
        )
    )


    # -----------------------------------------------------
    # 4. SHAP Global Feature Importance
    # -----------------------------------------------------

    shap_importance = (
        pd.DataFrame(
            {
                "feature": MODEL_FEATURES,
                "mean_abs_shap":
                    np.abs(
                        shap_values.values
                    ).mean(axis=0)
            }
        )
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )


    print(
        "\nSHAP Global Feature Importance"
    )

    print("=" * 60)

    print(
        shap_importance
        .to_string(index=False)
    )


    # -----------------------------------------------------
    # 5. SHAP Summary Plot -- saved to file, not shown
    #
    # plt.show() blocks script execution until the plot
    # window is closed manually. Saving to file keeps the
    # script fully non-interactive and produces a reusable
    # artifact for the README / report.
    # -----------------------------------------------------

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_shap,
        show=False
    )

    plt.title(
        "XGBoost SHAP Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        SHAP_SUMMARY_PLOT_ARTIFACT,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSHAP summary plot saved to:",
        SHAP_SUMMARY_PLOT_ARTIFACT
    )


# endregion

# region XGBoost Evaluation

print("\nXGBoost Validation Results")
print("=" * 70)


# ---------------------------------------------------------
# Evaluate by Validation Origin and Horizon
# ---------------------------------------------------------

xgb_evaluation_results = []

for (origin, horizon), group in (
    xgb_valid_results
    .groupby(
        ["date", "horizon"]
    )
):

    metrics = calculate_metrics(
        y_true=group["target"],
        y_pred=group["prediction"]
    )

    xgb_evaluation_results.append(
        {
            "origin": origin,
            "horizon": horizon,
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "WAPE": metrics["WAPE"]
        }
    )


xgb_evaluation_results = (
    pd.DataFrame(
        xgb_evaluation_results
    )
    .sort_values(
        ["origin", "horizon"]
    )
    .reset_index(drop=True)
)


# ---------------------------------------------------------
# Display Evaluation Results
# ---------------------------------------------------------

print(
    xgb_evaluation_results
    .to_string(index=False)
)


# ---------------------------------------------------------
# Overall Validation Metrics
# ---------------------------------------------------------

xgb_overall_metrics = (
    calculate_metrics(
        y_true=xgb_valid_results["target"],
        y_pred=xgb_valid_results["prediction"]
    )
)


print("\nOverall XGBoost Metrics")
print("-" * 70)

print(
    f"MAE  : "
    f"{xgb_overall_metrics['MAE']:.3f}"
)

print(
    f"RMSE : "
    f"{xgb_overall_metrics['RMSE']:.3f}"
)

print(
    f"WAPE : "
    f"{xgb_overall_metrics['WAPE'] * 100:.2f}%"
)


# ---------------------------------------------------------
# Average Performance by Horizon
# ---------------------------------------------------------

xgb_horizon_summary = (
    xgb_evaluation_results
    .groupby("horizon")[
        [
            "MAE",
            "RMSE",
            "WAPE"
        ]
    ]
    .mean()
    .reset_index()
)


print(
    "\nAverage XGBoost Performance by Horizon"
)

print("-" * 70)

print(
    xgb_horizon_summary
    .assign(
        WAPE=lambda df:
            df["WAPE"] * 100
    )
    .round(
        {
            "MAE": 3,
            "RMSE": 3,
            "WAPE": 2
        }
    )
    .to_string(index=False)
)


# endregion