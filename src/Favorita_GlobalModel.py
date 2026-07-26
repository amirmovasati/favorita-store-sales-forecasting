# region Imports
import os
os.system('cls')
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score

# endregion

# region Read Data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data"

train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
stores = pd.read_csv(DATA_PATH / "stores.csv")
oil = pd.read_csv(DATA_PATH / "oil.csv")
holidays = pd.read_csv(DATA_PATH / "holidays_events.csv")
transactions = pd.read_csv(DATA_PATH / "transactions.csv")

# endregion

# region Initial Data Inspection

datasets = {
    "train": train,
    "test": test,
    "stores": stores,
    "oil": oil,
    "holidays": holidays,
    "transactions": transactions
}

# for name, df in datasets.items():
#     print("=" * 70)
#     print(name.upper())
#     print("=" * 70)

#     print("\nShape:")
#     print(df.shape)

#     print("\nInfo:")
#     print(df.info())

#     print("\nMissing Values:")
#     print(df.isnull().sum())

#     print("\nUnique Values:")
#     print(df.nunique())

#     print("\nHead:")
#     print(df.head())

#     print("\n")

# endregion

# region Data Preparation

# Convert Date Columns
train["date"] = pd.to_datetime(train["date"])
test["date"] = pd.to_datetime(test["date"])
oil["date"] = pd.to_datetime(oil["date"])
holidays["date"] = pd.to_datetime(holidays["date"])
transactions["date"] = pd.to_datetime(transactions["date"])

# for name, df in datasets.items():
#     if "date" in df.columns:
#         print(f"\n{name}")
#         print(df["date"].min())
#         print(df["date"].max())

# print(train.memory_usage(deep=True))
# print("\nTotal Memory (MB):")
# print(train.memory_usage(deep=True).sum() / 1024**2)


# Optimize Data Types
train["family"] = train["family"].astype("category")
test["family"] = test["family"].astype("category")

train["store_nbr"] = train["store_nbr"].astype("int8")
test["store_nbr"] = test["store_nbr"].astype("int8")
stores["store_nbr"] = stores["store_nbr"].astype("int8")
transactions["store_nbr"] = transactions["store_nbr"].astype("int8")

train["onpromotion"] = train["onpromotion"].astype("int16")
test["onpromotion"] = test["onpromotion"].astype("int16")

# print(train.memory_usage(deep=True))
# print("\nTotal Memory (MB):")
# print(train.memory_usage(deep=True).sum() / 1024**2)

# endregion

# region EDA - Target Variable (Sales)

# # Basic Statistics
# print(train["sales"].describe())

# # Number of zero sales
# zero_sales = (train["sales"] == 0).sum()
# print(f"\nZero Sales : {zero_sales:,}")

# print(f"Zero Sales Percentage : {zero_sales/len(train)*100:.2f}%")

# # Negative Sales
# negative_sales = (train["sales"] < 0).sum()
# print(f"\nNegative Sales : {negative_sales:,}")

# # Maximum Sale
# print(f"\nMaximum Sale : {train['sales'].max():,.2f}")

# # Median Sale
# print(f"Median Sale : {train['sales'].median():,.2f}")

# # Histogram
# plt.figure(figsize=(10,5))
# plt.hist(train["sales"], bins=100)
# plt.title("Sales Distribution")
# plt.xlabel("Sales")
# plt.ylabel("Frequency")
# plt.show()

# # Boxplot
# plt.figure(figsize=(10,2))
# plt.boxplot(train["sales"], vert=False)
# plt.title("Sales Boxplot")
# plt.show()

# endregion

# region EDA - Overall Sales Trend

# Daily Total Sales
daily_sales = (
    train.groupby("date", as_index=False)["sales"]
         .sum()
)

# print(daily_sales.head())
# print(daily_sales.tail())
# print(f"\nNumber of Days: {len(daily_sales)}")

# plt.figure(figsize=(16, 6))

# plt.plot(daily_sales["date"], daily_sales["sales"])

# plt.title("Total Daily Sales")
# plt.xlabel("Date")
# plt.ylabel("Sales")

# plt.tight_layout()
# plt.show()

# endregion

# region Choose a Representative Time Series

df = train.copy()

df = df.sort_values(
    ["store_nbr", "family", "date"]
).reset_index(drop=True)

# endregion

# region Merge Stores Features

# Explore Stores Dataset
# print(stores.head())

# print("\nColumns")
# print(stores.columns)

# print("\nMissing Values")
# print(stores.isnull().sum())

# print("\nStore Types")
# print(stores["type"].value_counts())

# print("\nClusters")
# print(stores["cluster"].value_counts().sort_index())

# print("\nCities")
# print(stores["city"].nunique())

# print("\nStates")
# print(stores["state"].nunique())

# Merge
df = df.merge(
    stores,
    on="store_nbr",
    how="left"
)

# Verify Merge
# print(df[[
#     "store_nbr",
#     "city",
#     "state",
#     "type",
#     "cluster"
# ]].head())

# print()

# print(df[[
#     "city",
#     "state",
#     "type",
#     "cluster"
# ]].isnull().sum())

# endregion

# region One-Hot Encoding for Store Features

df = pd.get_dummies(
    df,
    columns=[
        "city",
        "state",
        "type"
    ],
    drop_first=True,
    dtype="int8"
)

# endregion

# region Merge Holiday Features

# Check Holiday Columns
# print(holidays.columns)

# print("\nHoliday Types")
# print(holidays["type"].value_counts())

# print("\nHoliday Locales")
# print(holidays["locale"].value_counts())

# print("\nTransferred")
# print(holidays["transferred"].value_counts())

# Keep Only Valid National Holidays
holiday_df = holidays[
    (holidays["type"] == "Holiday") &
    (holidays["locale"] == "National") &
    (holidays["transferred"] == False)
].copy()

holiday_df = holiday_df[
    [
        "date",
        "type"
    ]
]

# Rename Holiday Type
holiday_df = holiday_df.rename(
    columns={
        "type": "holiday_type"
    }
)

holiday_df["is_holiday"] = 1

# Merge
df = df.merge(
    holiday_df,
    on="date",
    how="left"
)

# Fill Missing Holiday Values
df["is_holiday"] = (
    df["is_holiday"]
    .fillna(0)
    .astype("int8")
)

df["holiday_type"] = (
    df["holiday_type"]
    .fillna("None")
)

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["holiday_type"],
    drop_first=True,
    dtype="int8"
)

# Verify Holiday Merge
# print(df["is_holiday"].value_counts())

# print()

# print(df[["date", "is_holiday"]].head(20))

# endregion

# region Merge Transactions Features

# Merge
# df = df.merge(
#     transactions,
#     on=["date", "store_nbr"],
#     how="left"
# )

# Verify Merge
# print(df["transactions"].isnull().sum())

# print()

# print(
#     df[
#         ["date", "store_nbr", "transactions"]
#     ].head(20)
# )

# endregion

# region Feature Engineering

# Calendar Features
df["day_name"] = df["date"].dt.day_name()
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month

# Long-Term Lag Features
df["lag_90"] = (
    df.groupby(["store_nbr", "family"])["sales"]
      .shift(90)
)

df["lag_180"] = (
    df.groupby(["store_nbr", "family"])["sales"]
      .shift(180)
)

df["lag_365"] = (
    df.groupby(["store_nbr", "family"])["sales"]
      .shift(365)
)

# One-Hot Encoding
df = pd.get_dummies(
    df,
    columns=["store_nbr", "family"],
    drop_first=True,
    dtype="int8"
)

# endregion

# region Advanced Calendar Features

# Calendar Features

df["year"] = df["date"].dt.year

df["day"] = df["date"].dt.day

df["week_of_year"] = (
    df["date"]
    .dt.isocalendar()
    .week
    .astype("int16")
)

df["is_month_start"] = (
    df["date"]
    .dt.is_month_start
    .astype("int8")
)

df["is_month_end"] = (
    df["date"]
    .dt.is_month_end
    .astype("int8")
)

df["is_weekend"] = (
    (df["day_of_week"] >= 5)
    .astype("int8")
)

# endregion

# region Clean Feature Names

df.columns = (
    df.columns
      .str.replace(r"[^A-Za-z0-9_]", "_", regex=True)
)

# endregion

# region Train / Validation Split

split_date = "2017-01-01"

train_df = df[
    df["date"] < split_date
].copy()

valid_df = df[
    df["date"] >= split_date
].copy()

# print("Train Shape :", train_df.shape)
# print("Validation Shape :", valid_df.shape)

# print("\nTrain Range")
# print(train_df["date"].min(), " --> ", train_df["date"].max())

# print("\nValidation Range")
# print(valid_df["date"].min(), " --> ", valid_df["date"].max())

# endregion

# region Baseline Model (Lag-7)

# valid_df["baseline_pred"] = valid_df["lag_365"]
# baseline = valid_df.dropna(subset=["baseline_pred"])
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# mae = mean_absolute_error(
#     baseline["sales"],
#     baseline["baseline_pred"]
# )

# rmse = np.sqrt(
#     mean_squared_error(
#         baseline["sales"],
#         baseline["baseline_pred"]
#     )
# )

# print(f"Baseline MAE  : {mae:.2f}")
# print(f"Baseline RMSE : {rmse:.2f}")

# endregion

# region Preparing Features for the First ML Model

feature_cols = [
    col for col in df.columns
    if col not in ["date", "sales", "id", "day_name"]
]

X_train = train_df[feature_cols]
y_train = train_df["sales"]

X_valid = valid_df[feature_cols]
y_valid = valid_df["sales"]

# print(X_train.shape)
# print(X_valid.shape)

# ==================================================
# Remove Missing Values
train_model = train_df.dropna().copy()
valid_model = valid_df.dropna().copy()

# print("Train Shape :", train_model.shape)
# print("Validation Shape :", valid_model.shape)

# ==================================================
# Final Training Dataset
X_train = train_model[feature_cols]
y_train = train_model["sales"]

X_valid = valid_model[feature_cols]
y_valid = valid_model["sales"]

# print(X_train.shape)
# print(X_valid.shape)

# endregion

# region Metric
def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print("=" * 40)
    print(name)
    print("-" * 40)
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²  : {r2:.4f}")
# endregion

# region Linear Regression

# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred_Linear = model.predict(X_valid)

# evaluate("Linear Regression:", y_valid, y_pred_Linear)

# endregion

# region Decision Tree Regressor

# from sklearn.tree import DecisionTreeRegressor

# tree = DecisionTreeRegressor(max_depth=5, random_state=42)
# tree.fit(X_train, y_train)
# y_pred_tree = tree.predict(X_valid)
# evaluate("Decision Tree:", y_valid, y_pred_tree)

# endregion

# region Random Forest

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import TimeSeriesSplit

# rf = RandomForestRegressor(
#     n_estimators=50,
#     max_depth=3,
#     random_state=42
# )

# rf.fit(X_train, y_train)

# y_pred_rf = rf.predict(X_valid)

# evaluate("Random Forest:", y_valid, y_pred_rf)

# tscv = TimeSeriesSplit(n_splits=5)

# #Grid Search
# param_grid = {
#     'n_estimators':[50, 100, 200],
#     'max_depth':[3, 5, 10, None]
# }

# grid = GridSearchCV(
#     RandomForestRegressor(random_state=42),
#     param_grid = param_grid,
#     cv = tscv,
#     scoring = 'neg_mean_absolute_error',
#     n_jobs=-1
# )

# grid.fit(X_train, y_train)

# rf = grid.best_estimator_
# y_pred_rf = rf.predict(X_valid)
# evaluate("Random Forest(gridsearch):", y_valid, y_pred_rf)

# print('-'*15)
# print(grid.best_params_)
# print(grid.best_score_)

# endregion

# region LightGBM Regressor

# from lightgbm import LGBMRegressor

# lgbm = LGBMRegressor(
#     n_estimators=50,
#     max_depth=3,
#     learning_rate=0.1,
#     random_state=42
# )

# lgbm.fit(
#     X_train,
#     y_train
# )

# y_pred_lgbm = lgbm.predict(X_valid)

# evaluate(
#     "LightGBM:",
#     y_valid,
#     y_pred_lgbm
# )

# endregion

# region CatBoost Regressor

# from catboost import CatBoostRegressor

# cat = CatBoostRegressor(
#     iterations=500,
#     depth=3,
#     learning_rate=0.05,
#     loss_function="RMSE",
#     random_state=42,
#     verbose=False
# )

# cat.fit(
#     X_train,
#     y_train
# )

# y_pred_cat = cat.predict(X_valid)

# evaluate("CatBoost:", y_valid, y_pred_cat)

# endregion

# region XGBoost Regressor

# from xgboost import XGBRegressor
# from xgboost.callback import EarlyStopping

# xgb = XGBRegressor(
#     n_estimators=50,
#     max_depth=3,
#     random_state=42
# )

# xgb.fit(X_train, y_train)

# y_pred_xgb = xgb.predict(X_valid)

# evaluate("XGBoost:", y_valid, y_pred_xgb)

# tscv = TimeSeriesSplit(n_splits=5)

# # Grid Search
# param_grid = {
#     'n_estimators':[50, 100, 200],
#     'max_depth':[3, 5, 10, None]
# }

# grid = GridSearchCV(
#     XGBRegressor(random_state=42),
#     param_grid = param_grid,
#     cv = tscv,
#     scoring = 'neg_mean_absolute_error',
#     n_jobs=-1
# )
# grid.fit(X_train, y_train)

# xgb = grid.best_estimator_
# y_pred_xgb = xgb.predict(X_valid)
# evaluate("XGBoost(gridsearch):", y_valid, y_pred_xgb)

# print('-'*15)
# print(grid.best_params_)
# print(grid.best_score_)

# endregion

# region XGBoost - Early Stopping

# from xgboost import XGBRegressor

# xgb_es = XGBRegressor(
#     random_state=42,
#     max_depth=3,
#     learning_rate=0.05,
#     n_estimators=500,
#     eval_metric="mae",
#     early_stopping_rounds=20
# )

# xgb_es.fit(
#     X_train,
#     y_train,
#     eval_set=[(X_valid, y_valid)],
#     verbose=False
# )

# y_pred_es = xgb_es.predict(X_valid)

# evaluate("XGBoost (Early Stopping):", y_valid, y_pred_es)

# print("-"*20)
# print("Best Iteration :", xgb_es.best_iteration)
# print("Best Score     :", xgb_es.best_score)

# endregion

# region XGBoost - Feature Importance

# importance = pd.Series(
#     xgb_es.feature_importances_,
#     index=X_train.columns
# ).sort_values(ascending=False)

# print(importance)

# plt.figure(figsize=(8,6))

# importance.sort_values().plot(kind="barh")

# plt.title("Feature Importance - XGBoost")
# plt.xlabel("Importance")
# plt.tight_layout()

# plt.show()

# endregion

#================================

# region ARIMA Dataset

arima_df = train[
    (train["store_nbr"] == 1) &
    (train["family"] == "BEVERAGES")
].copy()

arima_df = arima_df.sort_values("date")

print(arima_df.shape)

print(arima_df.head())

# endregion

# region ADF Test
from statsmodels.tsa.stattools import adfuller
result = adfuller(arima_df["sales"])

print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.6f}")

print("\nCritical Values")

for key, value in result[4].items():
    print(f"{key} : {value:.4f}")

# endregion

# region ARIMA Dataset

ts = train[
    (train["store_nbr"] == 1) &
    (train["family"] == "BEVERAGES")
].copy()

ts = ts.sort_values("date")

ts = ts.set_index("date")

ts = ts["sales"]

print(ts.head())

# endregion

# region ARIMA
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(
    ts,
    order=(1,0,1)
)

result = model.fit()

print(result.summary())

# endregion

# region Prophet

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ===============================
# Prepare Dataset
# ===============================

prophet_df = train[
    (train["store_nbr"] == 1) &
    (train["family"] == "BEVERAGES")
][["date", "sales"]].copy()

prophet_df = prophet_df.rename(
    columns={
        "date": "ds",
        "sales": "y"
    }
)

# ===============================
# Train / Validation Split
# ===============================

train_prophet = prophet_df[
    prophet_df["ds"] < "2017-01-01"
].copy()

valid_prophet = prophet_df[
    prophet_df["ds"] >= "2017-01-01"
].copy()

# ===============================
# Model
# ===============================

model = Prophet()

model.fit(train_prophet)

# ===============================
# Forecast
# ===============================

future = valid_prophet[["ds"]].copy()

forecast = model.predict(future)

# ===============================
# Evaluation
# ===============================

mae = mean_absolute_error(
    valid_prophet["y"],
    forecast["yhat"]
)

rmse = np.sqrt(
    mean_squared_error(
        valid_prophet["y"],
        forecast["yhat"]
    )
)

r2 = r2_score(
    valid_prophet["y"],
    forecast["yhat"]
)

print("=" * 40)
print("Prophet")
print("-" * 40)
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²  : {r2:.4f}")

# ===============================
# Plot Forecast
# ===============================

fig = model.plot(forecast)

# ===============================
# Trend + Seasonality
# ===============================

fig2 = model.plot_components(forecast)

# endregion

# raise SystemExit