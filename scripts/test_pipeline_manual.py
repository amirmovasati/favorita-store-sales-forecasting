"""
test_pipeline_manual.py

Manual, throwaway smoke test for the pipeline modules built so far
(ingestion, validation, history_store, feature_engineering,
inference). NOT part of the final pipeline -- just a quick way to
verify the modules work correctly against the REAL project data and
the REAL trained model artifact, before wiring everything into
run_pipeline.py.

Run this from the project root:

    python test_pipeline_manual.py --origin-date 2017-07-20

If --origin-date is omitted, it defaults to 20 days before the last
date in train.csv (leaves enough real "future" days in the data to
sanity-check the forecast against, while still keeping the origin
close to the end of the dataset -- for maximum historical depth).
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Make src/pipeline importable when running from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import ingestion, validation, history_store, feature_engineering as fe, inference as inf  # noqa: E402

# ---------------------------------------------------------------
# Same artifact configuration as Favorita_GlobalModel.py
# ---------------------------------------------------------------

RUN_TAG = "1yr_h15_spikefeatures"

OUTPUT_BASE_DIR = Path(
    os.environ.get("FAVORITA_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
)
OUTPUT_DIR = OUTPUT_BASE_DIR / RUN_TAG

MODEL_ARTIFACT = OUTPUT_DIR / "models" / "xgb_model.json"
TRAIN_DF_ARTIFACT = OUTPUT_DIR / "datasets" / "xgb_train_df.pkl"

DATA_PATH = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_PATH / "train.csv"
STORES_CSV = DATA_PATH / "stores.csv"
HOLIDAYS_CSV = DATA_PATH / "holidays_events.csv"

MIN_HISTORY_DAYS = 393  # 365 (yearly lag) + up to 28 (horizon-adjusted lag)


def main(origin_date: str | None) -> None:
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model artifact: {MODEL_ARTIFACT}")
    print(f"Train df artifact: {TRAIN_DF_ARTIFACT}")

    for path in (TRAIN_CSV, STORES_CSV, HOLIDAYS_CSV, MODEL_ARTIFACT, TRAIN_DF_ARTIFACT):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # -------------------------------------------------------
    # 1. Load raw data
    # -------------------------------------------------------
    print("\nLoading data...")
    full_df = ingestion.load_full_dataset(str(TRAIN_CSV))
    stores_df = pd.read_csv(STORES_CSV)
    holidays_df = pd.read_csv(HOLIDAYS_CSV, parse_dates=["date"])

    last_available_date = full_df["date"].max()
    if origin_date is None:
        origin_ts = last_available_date - pd.Timedelta(days=20)
        origin_date = str(origin_ts.date())
    else:
        origin_ts = pd.to_datetime(origin_date)

    days_of_history = (origin_ts - full_df["date"].min()).days
    print(f"Origin date  : {origin_date}")
    print(f"Days of history available before origin: {days_of_history}")
    if days_of_history < MIN_HISTORY_DAYS:
        print(
            f"WARNING: fewer than {MIN_HISTORY_DAYS} days of history before "
            f"the origin date -- lag_365 will likely be NaN for early rows."
        )

    # -------------------------------------------------------
    # 2. Simulate daily ingestion + validation up to origin_date
    #
    # NOTE: for this smoke test, batches are validated one day at a
    # time (as run_pipeline.py will do), but appended in bulk rather
    # than one-by-one, to keep the test fast. run_pipeline.py itself
    # will do this incrementally, one real day at a time.
    # -------------------------------------------------------
    print("\nSimulating daily ingestion + validation...")
    history_df = history_store.init_history(full_df.columns.tolist())

    all_dropped_missing = 0
    all_dropped_invalid = 0
    sim_dates = sorted(d for d in full_df["date"].unique() if d <= origin_ts)

    clean_batches = []
    for d in sim_dates:
        batch = ingestion.get_daily_batch(full_df, str(pd.Timestamp(d).date()))
        clean_batch, report = validation.validate_batch(batch)
        all_dropped_missing += report["dropped_missing_keys"]
        all_dropped_invalid += report["dropped_invalid_values"]
        clean_batches.append(clean_batch)

    history_df = pd.concat(clean_batches, ignore_index=True)
    history_df = history_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    print(f"Simulated days           : {len(sim_dates)}")
    print(f"Total rows in history    : {len(history_df)}")
    print(f"Dropped (missing keys)   : {all_dropped_missing}")
    print(f"Dropped (invalid values) : {all_dropped_invalid}")

    # -------------------------------------------------------
    # 3. Feature engineering
    # -------------------------------------------------------
    print("\nBuilding features...")
    prepared_df = fe.add_store_and_holiday_features(history_df, stores_df, holidays_df)
    prepared_df = fe.add_calendar_features(prepared_df)
    prepared_df = fe.add_historical_sales_features(prepared_df)

    origin_dataset = fe.build_origin_dataset(prepared_df, origin_date)
    print(f"Origin dataset shape: {origin_dataset.shape}")

    null_counts = origin_dataset[fe.MODEL_FEATURES].isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if len(null_counts) > 0:
        print("\nWARNING: NaNs found in model features:")
        print(null_counts)

    # -------------------------------------------------------
    # 4. Inference
    # -------------------------------------------------------
    print("\nLoading model and running inference...")
    model = inf.load_model(str(MODEL_ARTIFACT))
    categorical_reference = inf.load_categorical_reference(str(TRAIN_DF_ARTIFACT))
    result = inf.predict(model, origin_dataset, categorical_reference)

    print("\nSample forecasts:")
    print(
        result[["target_date", "store_nbr", "family", "horizon", "forecast"]]
        .head(15)
        .to_string(index=False)
    )

    print(f"\nAny NaN forecasts: {result['forecast'].isnull().sum()}")
    print(f"Forecast range: {result['forecast'].min():.2f} to {result['forecast'].max():.2f}")

    # -------------------------------------------------------
    # 5. If the origin date has enough "future" days left in the
    #    dataset, compare forecasts against actual sales as a rough
    #    sanity check (NOT a full evaluation -- just a smoke test).
    # -------------------------------------------------------
    actuals = full_df.rename(columns={"date": "target_date"})[
        ["target_date", "store_nbr", "family", "sales"]
    ]
    comparison = result.merge(actuals, on=["target_date", "store_nbr", "family"], how="left")
    comparison = comparison.dropna(subset=["sales"])

    if len(comparison) > 0:
        comparison["abs_error"] = (comparison["forecast"] - comparison["sales"]).abs()
        print(f"\nRows with known actuals to compare against: {len(comparison)}")
        print(f"Mean absolute error (rough smoke check): {comparison['abs_error'].mean():.2f}")
    else:
        print(
            "\nNo actuals available yet to compare against "
            "(origin date is too close to the end of the dataset)."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-date", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.origin_date)
