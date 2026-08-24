"""
validation.py

Lightweight, targeted validation for each daily batch before it enters
feature engineering. Deliberately not exhaustive: checks schema, key
missing values, and simple invalid values. Real-world error scope
would be scoped together with the data owner rather than defended
against exhaustively here.
"""

import pandas as pd

REQUIRED_COLUMNS = ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
KEY_COLUMNS = ["date", "store_nbr", "family"]


class SchemaError(Exception):
    """Raised when the batch is missing required columns."""
    pass


def validate_batch(batch_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Validate a daily batch.

    Returns
    -------
    clean_df : pd.DataFrame
        Validated data, ready for feature engineering.
    report : dict
        Counts of rows dropped per reason, for logging.

    Raises
    ------
    SchemaError
        If required columns are missing (fail-fast: the data source
        has likely changed).
    """
    report = {
        "initial_rows": len(batch_df),
        "dropped_missing_keys": 0,
        "dropped_invalid_values": 0,
        "final_rows": None,
    }

    missing_cols = set(REQUIRED_COLUMNS) - set(batch_df.columns)
    if missing_cols:
        raise SchemaError(f"Missing required column(s): {sorted(missing_cols)}")

    clean_df = batch_df.copy()

    # Missing values in key columns -> drop row
    before = len(clean_df)
    clean_df = clean_df.dropna(subset=KEY_COLUMNS)
    report["dropped_missing_keys"] = before - len(clean_df)

    # Simple invalid values -> drop row
    before = len(clean_df)
    valid_mask = (clean_df["sales"] >= 0) & (clean_df["onpromotion"] >= 0)
    clean_df = clean_df.loc[valid_mask]
    report["dropped_invalid_values"] = before - len(clean_df)

    clean_df.reset_index(drop=True, inplace=True)
    report["final_rows"] = len(clean_df)

    return clean_df, report
