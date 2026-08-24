"""
ingestion.py

Reads the raw dataset and extracts a single day's slice, simulating
how new sales data would arrive daily in a real deployment (rather
than the model seeing the full history at once).
"""

from pathlib import Path
import pandas as pd


def load_full_dataset(train_csv_path: str) -> pd.DataFrame:
    """Load the full training CSV once, with `date` parsed as datetime."""
    path = Path(train_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    return pd.read_csv(path, parse_dates=["date"])


def get_daily_batch(full_df: pd.DataFrame, target_date: str) -> pd.DataFrame:
    """
    Return only the rows for `target_date`, simulating a daily batch
    arriving from stores. Empty DataFrame (same columns) if no data
    exists for that date.
    """
    target_ts = pd.to_datetime(target_date)
    batch = full_df.loc[full_df["date"] == target_ts].copy()
    batch.reset_index(drop=True, inplace=True)
    return batch
