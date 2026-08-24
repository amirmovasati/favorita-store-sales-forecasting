"""
api.py

A minimal FastAPI service that exposes the pipeline through HTTP:

    GET /            -- a simple, non-technical web page (date picker,
                         a "Get Forecast" button, a short summary, a
                         small table of items needing attention, and
                         a download link for the full report)
    GET /forecast     -- JSON API used by the page above
    GET /download     -- downloads the full CSV report already
                         produced by a prior /forecast call
    GET /health       -- liveness check
    GET /monitoring   -- recent pipeline run history (success/failure log)

No pipeline logic lives here -- this file only wraps run_pipeline.run().
"""

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pipeline import monitoring  # noqa: E402
from run_pipeline import REPORTS_DIR, run

app = FastAPI(title="Favorita Forecasting Pipeline")

# Rows shown inline in the web page are capped, so the browser never
# has to render a huge table -- speed matters more than completeness
# for the in-page view. The full report is always available via /download.
MAX_PRIORITY_ROWS_SHOWN = 100


@app.get("/health")
def health_check():
    """Simple liveness check -- confirms the service is up."""
    return {"status": "ok"}


@app.get("/forecast")
def forecast(origin_date: str, inventory_path: str | None = None):
    """
    Run the full pipeline for a given origin date and return a JSON
    summary: counts per decision status, plus a small table of only
    the rows that need attention (stockout/overstock risk).

    Parameters
    ----------
    origin_date : str
        Forecast origin date, format YYYY-MM-DD.
    inventory_path : str, optional
        Path to an inventory CSV, relative to the project root.
    """
    try:
        report_path = run(origin_date, inventory_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    report_df = pd.read_csv(report_path)

    priority_df = report_df[
        report_df["decision_status"].isin(["stockout_risk", "overstock_risk"])
    ].head(MAX_PRIORITY_ROWS_SHOWN)
    priority_df = priority_df.astype(object).where(priority_df.notna(), None)

    return {
        "origin_date": origin_date,
        "rows": len(report_df),
        "decision_status_counts": report_df["decision_status"].value_counts().to_dict(),
        "priority_items": priority_df.to_dict(orient="records"),
        "priority_items_truncated": len(priority_df) == MAX_PRIORITY_ROWS_SHOWN,
    }


@app.get("/download")
def download(origin_date: str):
    """Download the full CSV report for a date already forecast."""
    report_path = REPORTS_DIR / f"report_{origin_date}.csv"
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No report found for {origin_date}. Call /forecast for this date first.",
        )
    return FileResponse(
        path=report_path,
        media_type="text/csv",
        filename=report_path.name,
    )


@app.get("/monitoring")
def monitoring_history(limit: int = 20):
    """Recent pipeline run history: success/failure, duration, counts."""
    return {"runs": monitoring.read_recent_runs(limit=limit)}


@app.get("/", response_class=HTMLResponse)
def home():
    """A minimal, non-technical web page for running the pipeline."""
    return HTML_PAGE


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Favorita Sales Forecast</title>
<style>
  body { font-family: -apple-system, Segoe UI, Arial, sans-serif; max-width: 720px;
         margin: 40px auto; padding: 0 16px; color: #1a1a1a; }
  h1 { font-size: 22px; }
  .controls { display: flex; gap: 8px; align-items: center; margin: 16px 0; }
  input[type=date] { padding: 8px; font-size: 15px; }
  button { padding: 8px 16px; font-size: 15px; background: #2563eb; color: white;
           border: none; border-radius: 6px; cursor: pointer; }
  button:disabled { background: #93a3b8; cursor: default; }
  #summary { font-size: 17px; margin: 16px 0; padding: 12px; background: #f3f4f6;
             border-radius: 6px; min-height: 24px; }
  table { border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 12px; }
  th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
  th { background: #f9fafb; }
  .stockout { background: #fee2e2; }
  .overstock { background: #fef9c3; }
  #downloadLink { display: none; margin-top: 14px; }
  #note { color: #6b7280; font-size: 13px; margin-top: 8px; }
</style>
</head>
<body>
  <h1>Favorita Store Sales Forecast</h1>
  <div class="controls">
    <input type="date" id="originDate" value="2017-07-20">
    <label style="font-size: 14px;">
      <input type="checkbox" id="useInventory"> Use sample inventory data
    </label>
    <button id="runBtn" onclick="runForecast()">Get Forecast</button>
  </div>
  <div id="summary">Pick a date and click "Get Forecast".</div>
  <div id="tableContainer"></div>
  <a id="downloadLink" href="#">Download full report (CSV)</a>
  <div id="note"></div>

<script>
async function runForecast() {
  const date = document.getElementById('originDate').value;
  const btn = document.getElementById('runBtn');
  const summary = document.getElementById('summary');
  const tableContainer = document.getElementById('tableContainer');
  const downloadLink = document.getElementById('downloadLink');
  const note = document.getElementById('note');

  btn.disabled = true;
  summary.textContent = 'Running... this can take about a minute.';
  tableContainer.innerHTML = '';
  downloadLink.style.display = 'none';
  note.textContent = '';

  try {
    const useInventory = document.getElementById('useInventory').checked;
    const invParam = useInventory ? '&inventory_path=data/sample_inventory.csv' : '';
    const res = await fetch(`/forecast?origin_date=${date}${invParam}`);
    const data = await res.json();
    if (!res.ok) {
      summary.textContent = 'Error: ' + (data.detail || 'unknown error');
      btn.disabled = false;
      return;
    }

    const counts = data.decision_status_counts;
    const stockout = counts.stockout_risk || 0;
    const overstock = counts.overstock_risk || 0;
    const optimal = counts.optimal || 0;

    summary.innerHTML =
      `<b>${data.rows}</b> forecasts generated for <b>${data.origin_date}</b>. ` +
      `<b>${stockout}</b> item(s) at stockout risk, ` +
      `<b>${overstock}</b> at overstock risk, <b>${optimal}</b> optimal.`;

    if (data.priority_items.length > 0) {
      let html = '<table><tr><th>Store</th><th>Product</th><th>Target Date</th>' +
                 '<th>Forecast</th><th>Status</th></tr>';
      for (const row of data.priority_items) {
        const cls = row.decision_status === 'stockout_risk' ? 'stockout' : 'overstock';
        html += `<tr class="${cls}"><td>${row.store_nbr}</td><td>${row.family}</td>` +
                `<td>${row.target_date}</td><td>${Number(row.forecast).toFixed(1)}</td>` +
                `<td>${row.decision_status}</td></tr>`;
      }
      html += '</table>';
      tableContainer.innerHTML = html;
      if (data.priority_items_truncated) {
        note.textContent = 'Showing the first 100 items needing attention. Download the full report for everything.';
      }
    } else {
      tableContainer.innerHTML = '<p>No items need attention for this date (or no inventory data was supplied).</p>';
    }

    downloadLink.href = `/download?origin_date=${date}`;
    downloadLink.style.display = 'inline-block';
  } catch (err) {
    summary.textContent = 'Something went wrong: ' + err;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
"""
