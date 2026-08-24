"""
decision.py

Turns raw model forecasts into business-usable output:
    1. Clips negative forecasts to zero (a negative sales forecast
       is not a meaningful business quantity).
    2. Attaches a prediction_confidence label, using tercile
       thresholds already computed on the validation set during
       training (Favorita_GlobalModel.py) -- not recomputed here,
       since a single day's batch is too small to define reliable
       thresholds on its own.
    3. Optionally compares cumulative forecasted demand (summed
       across horizons, per store/family) against a supplied
       inventory level, to flag stockout / overstock risk. Inventory
       is treated as an external input the business must supply --
       never fabricated by this pipeline.
"""

from pathlib import Path

import pandas as pd

DEFAULT_OVERSTOCK_MULTIPLIER = 1.5  # inventory > 1.5x cumulative demand -> overstock flag


def clip_negative_forecasts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the raw model output in `forecast_raw`; clip `forecast` at zero."""
    result = df.copy()
    result["forecast_raw"] = result["forecast"]
    result["forecast"] = result["forecast"].clip(lower=0)
    return result


def load_confidence_thresholds(valid_results_artifact_path: str) -> tuple[float, float]:
    """
    Load the tercile thresholds for rolling_cv_28 from the existing
    validation results artifact (xgb_valid_results.pkl), the same
    way Favorita_GlobalModel.py computes them. Thresholds are
    data-driven from the validation set, not recomputed per batch.
    """
    path = Path(valid_results_artifact_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation results artifact not found: {path}")

    valid_results = pd.read_pickle(path)
    cv_low = valid_results["rolling_cv_28"].quantile(1 / 3)
    cv_high = valid_results["rolling_cv_28"].quantile(2 / 3)
    return cv_low, cv_high


def assign_confidence(df: pd.DataFrame, cv_low_threshold: float, cv_high_threshold: float) -> pd.DataFrame:
    """Attach a prediction_confidence label (high / medium / low / unknown)."""
    def _tier(cv_value):
        if pd.isna(cv_value):
            return "unknown"
        if cv_value <= cv_low_threshold:
            return "high"
        if cv_value <= cv_high_threshold:
            return "medium"
        return "low"

    result = df.copy()
    result["prediction_confidence"] = result["rolling_cv_28"].apply(_tier)
    return result


def add_business_decision(df: pd.DataFrame, inventory_df: pd.DataFrame | None = None,
                           overstock_multiplier: float = DEFAULT_OVERSTOCK_MULTIPLIER) -> pd.DataFrame:
    """
    Attach cumulative_forecast (running sum of `forecast` across
    horizons, per store/family) and, if `inventory_df` is supplied
    (columns: store_nbr, family, current_inventory), a
    decision_status per row:

        - "no_inventory_data": no inventory was supplied
        - "stockout_risk": cumulative demand exceeds current inventory
        - "overstock_risk": inventory exceeds demand by more than
           `overstock_multiplier`
        - "optimal": inventory is in between

    Inventory is expected from an external source (e.g. an ERP /
    warehouse system) -- this function never invents inventory data.
    """
    result = df.sort_values(["store_nbr", "family", "horizon"]).copy()
    result["cumulative_forecast"] = (
        result.groupby(["store_nbr", "family"])["forecast"].cumsum()
    )

    if inventory_df is None:
        result["current_inventory"] = pd.NA
        result["decision_status"] = "no_inventory_data"
        return result

    result = result.merge(
        inventory_df[["store_nbr", "family", "current_inventory"]],
        on=["store_nbr", "family"],
        how="left",
    )

    def _status(row):
        if pd.isna(row["current_inventory"]):
            return "no_inventory_data"
        if row["current_inventory"] < row["cumulative_forecast"]:
            return "stockout_risk"
        if row["current_inventory"] > row["cumulative_forecast"] * overstock_multiplier:
            return "overstock_risk"
        return "optimal"

    result["decision_status"] = result.apply(_status, axis=1)
    return result
