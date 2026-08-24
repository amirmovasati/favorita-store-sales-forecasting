"""
history_store.py

Keeps a running (accumulating) history of validated daily batches.
This models the real-world situation where, on a given simulated
"today", only the data ingested and validated up to that day is
actually available -- not the full dataset.

Kept intentionally simple: an in-memory pandas DataFrame that grows
one validated batch at a time. No database is used at this scope.
"""

import pandas as pd


def init_history(columns: list[str]) -> pd.DataFrame:
    """Create an empty history frame with the given columns."""
    return pd.DataFrame(columns=columns)


def append_batch(history_df: pd.DataFrame, clean_batch_df: pd.DataFrame) -> pd.DataFrame:
    """
    Append one validated daily batch to the running history and
    return the updated, date-sorted history.
    """
    updated = pd.concat([history_df, clean_batch_df], ignore_index=True)
    updated["date"] = pd.to_datetime(updated["date"])
    updated = updated.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return updated
