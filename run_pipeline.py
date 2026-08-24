"""
run_pipeline.py

Single entry point for the Favorita deployment pipeline. Runs the
full chain -- ingestion, validation, feature engineering, inference,
and business decisioning -- for one simulated forecast origin date,
and writes one business-consumable report file.

Usage (from the project root):

    python run_pipeline.py --date 2017-07-20
    python run_pipeline.py --date 2017-07-20 --inventory data/sample_inventory.csv

If --inventory is omitted, the report is still produced, with
decision_status = "no_inventory_data" for every row.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pipeline import ingestion, validation, feature_engineering as fe, inference as inf, decision as dec, monitoring  # noqa: E402

RUN_TAG = "1yr_h15_spikefeatures"

OUTPUT_BASE_DIR = Path(
    os.environ.get("FAVORITA_OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
)
ARTIFACT_DIR = OUTPUT_BASE_DIR / RUN_TAG

MODEL_ARTIFACT = ARTIFACT_DIR / "models" / "xgb_model.json"
TRAIN_DF_ARTIFACT = ARTIFACT_DIR / "datasets" / "xgb_train_df.pkl"
VALID_RESULTS_ARTIFACT = ARTIFACT_DIR / "predictions" / "xgb_valid_results.pkl"

DATA_PATH = PROJECT_ROOT / "data"
TRAIN_CSV = DATA_PATH / "train.csv"
STORES_CSV = DATA_PATH / "stores.csv"
HOLIDAYS_CSV = DATA_PATH / "holidays_events.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"

MIN_HISTORY_DAYS = 393

# ---------------------------------------------------------------
# Defaults for running directly from an editor (e.g. VSCode's Run
# button), with no command-line arguments. Edit these two lines to
# change what a plain `python run_pipeline.py` does. Command-line
# arguments (--date / --inventory), if given, always override these.
# ---------------------------------------------------------------
DEFAULT_ORIGIN_DATE = "2017-07-20"
DEFAULT_INVENTORY_PATH = None  # e.g. "data/sample_inventory.csv"


def run(origin_date: str, inventory_path: str | None) -> Path:
    """Public entry point: runs the pipeline and logs the outcome
    (success or failure) via monitoring.py, then re-raises any error
    so callers (CLI, API) still see it and can react."""
    start_time = monitoring.start_run(origin_date)
    try:
        report_path, run_stats = _run_pipeline(origin_date, inventory_path, start_time)
    except Exception as exc:
        monitoring.log_failure(origin_date, start_time, exc)
        raise

    monitoring.log_success(
        origin_date, start_time,
        rows=run_stats["rows"],
        dropped_missing=run_stats["dropped_missing"],
        dropped_invalid=run_stats["dropped_invalid"],
        decision_status_counts=run_stats["decision_status_counts"],
    )
    return report_path


def _run_pipeline(origin_date: str, inventory_path: str | None, start_time: float):
    start_time = time.time()

    print(f"Favorita Pipeline -- origin date: {origin_date}")
    print("=" * 60)

    for path in (TRAIN_CSV, STORES_CSV, HOLIDAYS_CSV, MODEL_ARTIFACT, TRAIN_DF_ARTIFACT):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    # -----------------------------------------------------------
    # 1. Ingestion + Validation (simulated day-by-day up to origin)
    # -----------------------------------------------------------
    print("\n[1/5] Ingestion + Validation")
    full_df = ingestion.load_full_dataset(str(TRAIN_CSV))
    origin_ts = pd.to_datetime(origin_date)

    days_of_history = (origin_ts - full_df["date"].min()).days
    if days_of_history < MIN_HISTORY_DAYS:
        raise ValueError(
            f"Not enough history before {origin_date}: {days_of_history} days "
            f"available, {MIN_HISTORY_DAYS} required (needed for yearly features)."
        )

    sim_dates = sorted(d for d in full_df["date"].unique() if d <= origin_ts)
    clean_batches = []
    dropped_missing = dropped_invalid = 0
    for d in sim_dates:
        batch = ingestion.get_daily_batch(full_df, str(pd.Timestamp(d).date()))
        clean_batch, report = validation.validate_batch(batch)
        dropped_missing += report["dropped_missing_keys"]
        dropped_invalid += report["dropped_invalid_values"]
        clean_batches.append(clean_batch)

    history_df = pd.concat(clean_batches, ignore_index=True)
    history_df = history_df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    print(f"  Days simulated: {len(sim_dates)} | rows: {len(history_df)} "
          f"| dropped (missing keys / invalid values): {dropped_missing} / {dropped_invalid}")

    # -----------------------------------------------------------
    # 2. Feature Engineering
    # -----------------------------------------------------------
    print("\n[2/5] Feature Engineering")
    stores_df = pd.read_csv(STORES_CSV)
    holidays_df = pd.read_csv(HOLIDAYS_CSV, parse_dates=["date"])

    prepared_df = fe.add_store_and_holiday_features(history_df, stores_df, holidays_df)
    prepared_df = fe.add_calendar_features(prepared_df)
    prepared_df = fe.add_historical_sales_features(prepared_df)
    origin_dataset = fe.build_origin_dataset(prepared_df, origin_date)
    print(f"  Origin dataset shape: {origin_dataset.shape}")

    # -----------------------------------------------------------
    # 3. Inference
    # -----------------------------------------------------------
    print("\n[3/5] Inference")
    model = inf.load_model(str(MODEL_ARTIFACT))
    categorical_reference = inf.load_categorical_reference(str(TRAIN_DF_ARTIFACT))
    predictions = inf.predict(model, origin_dataset, categorical_reference)
    print(f"  Forecast range (raw): {predictions['forecast'].min():.2f} to "
          f"{predictions['forecast'].max():.2f}")

    # -----------------------------------------------------------
    # 4. Business Decision Layer
    # -----------------------------------------------------------
    print("\n[4/5] Business Decision Layer")
    result = dec.clip_negative_forecasts(predictions)

    if VALID_RESULTS_ARTIFACT.exists():
        cv_low, cv_high = dec.load_confidence_thresholds(str(VALID_RESULTS_ARTIFACT))
        result = dec.assign_confidence(result, cv_low, cv_high)
    else:
        print("  WARNING: validation results artifact not found -- skipping confidence labels")
        result["prediction_confidence"] = "unknown"

    inventory_df = None
    if inventory_path:
        inventory_df = pd.read_csv(inventory_path)
        print(f"  Inventory supplied: {len(inventory_df)} store/family rows")
    else:
        print("  No inventory supplied -- decision_status will be 'no_inventory_data'")

    result = dec.add_business_decision(result, inventory_df=inventory_df)
    print(f"  Decision status counts:\n{result['decision_status'].value_counts().to_string()}")

    # -----------------------------------------------------------
    # 5. Write business-consumable report
    # -----------------------------------------------------------
    print("\n[5/5] Writing report")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"report_{origin_date}.csv"

    report_columns = [
        "target_date", "store_nbr", "family", "horizon",
        "forecast", "cumulative_forecast", "prediction_confidence",
        "current_inventory", "decision_status",
    ]
    result[report_columns].to_csv(report_path, index=False)

    elapsed = time.time() - start_time
    print(f"  Report written to: {report_path}")
    print(f"\nDone in {elapsed:.1f}s")

    run_stats = {
        "rows": len(result),
        "dropped_missing": dropped_missing,
        "dropped_invalid": dropped_invalid,
        "decision_status_counts": result["decision_status"].value_counts().to_dict(),
    }
    return report_path, run_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=DEFAULT_ORIGIN_DATE, help="Forecast origin date, YYYY-MM-DD")
    parser.add_argument("--inventory", default=DEFAULT_INVENTORY_PATH, help="Path to an inventory CSV (optional)")
    args = parser.parse_args()
    run(args.date, args.inventory)
