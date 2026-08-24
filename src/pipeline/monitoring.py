"""
monitoring.py

Lightweight, structured logging for pipeline runs. Not a full
monitoring stack (no dashboards, no alerting) -- just a durable,
machine-readable record of what happened on each run: when, how
long it took, how many predictions were produced, and whether it
succeeded or failed and why.

Each run appends one JSON line to logs/pipeline_runs.jsonl. JSON
Lines (one JSON object per line) is used instead of a single JSON
array so the file can be appended to safely and read line-by-line
without ever needing to parse the whole file.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "pipeline_runs.jsonl"


def _write_entry(entry: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def start_run(origin_date: str) -> float:
    """Call at the start of a pipeline run. Returns a start timestamp
    to pass into log_success/log_failure."""
    return time.time()


def log_success(origin_date: str, start_time: float, rows: int,
                 dropped_missing: int, dropped_invalid: int,
                 decision_status_counts: dict) -> None:
    """Record a successful pipeline run."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "origin_date": origin_date,
        "status": "success",
        "duration_seconds": round(time.time() - start_time, 2),
        "rows": rows,
        "dropped_missing_keys": dropped_missing,
        "dropped_invalid_values": dropped_invalid,
        "decision_status_counts": decision_status_counts,
    }
    _write_entry(entry)


def log_failure(origin_date: str, start_time: float, error: Exception) -> None:
    """Record a failed pipeline run."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "origin_date": origin_date,
        "status": "failure",
        "duration_seconds": round(time.time() - start_time, 2),
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    _write_entry(entry)


def read_recent_runs(limit: int = 20) -> list[dict]:
    """Read the most recent N run log entries, newest first."""
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE, encoding="utf-8") as f:
        lines = f.readlines()
    entries = [json.loads(line) for line in lines[-limit:]]
    return list(reversed(entries))
