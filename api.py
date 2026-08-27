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
        "confidence_counts": report_df["prediction_confidence"].value_counts().to_dict(),
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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  :root {
    --bg: #0B1420;
    --surface: #14202F;
    --surface-2: #1A2B3E;
    --border: #263449;
    --text: #EAF1F8;
    --text-dim: #8FA3B8;
    --gold: #E8A33D;
    --gold-dim: #6B4F22;
    --red: #E5533D;
    --red-bg: #2E1712;
    --mustard: #C9932E;
    --mustard-bg: #2A2210;
    --teal: #3FBF8F;
    --teal-bg: #12271F;
  }
  * { box-sizing: border-box; }
  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    max-width: 1320px;
    margin: 0 auto;
    padding: 40px 24px 64px;
  }
  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 0 0 6px;
  }
  h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 28px;
    margin: 0;
    letter-spacing: -0.01em;
  }
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    border-bottom: 1px solid var(--border);
    padding-bottom: 20px;
    margin-bottom: 28px;
  }
  .status {
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--text-dim);
    letter-spacing: 0.04em;
  }
  .pulse {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--teal);
    box-shadow: 0 0 0 0 rgba(63,191,143,0.6);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%   { box-shadow: 0 0 0 0 rgba(63,191,143,0.5); }
    70%  { box-shadow: 0 0 0 7px rgba(63,191,143,0); }
    100% { box-shadow: 0 0 0 0 rgba(63,191,143,0); }
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px 22px;
    margin-bottom: 24px;
  }
  .controls {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
  }
  label.field {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    display: block;
    margin-bottom: 6px;
  }
  input[type=date] {
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 9px 12px;
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
  }
  .checkbox-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    color: var(--text-dim);
  }
  button#runBtn {
    background: var(--gold);
    color: #1a1204;
    border: none;
    padding: 11px 22px;
    border-radius: 6px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: background 0.15s ease;
    margin-top: 18px;
  }
  button#runBtn:hover { background: #f2b358; }
  button#runBtn:disabled { background: var(--gold-dim); color: var(--text-dim); cursor: default; }

  .metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 24px;
  }
  .metric {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--edge, var(--border));
    border-radius: 8px;
    padding: 16px 18px;
  }
  .metric .num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 30px;
    font-weight: 500;
    line-height: 1;
  }
  .metric .lbl {
    font-size: 12px;
    color: var(--text-dim);
    margin-top: 8px;
  }
  .metric.total { --edge: var(--gold); }
  .metric.stockout { --edge: var(--red); }
  .metric.overstock { --edge: var(--mustard); }

  #tableContainer { margin-bottom: 4px; }
  table { border-collapse: collapse; width: 100%; }
  th {
    text-align: left;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-dim);
    border-bottom: 1px solid var(--border);
    padding: 8px 10px;
  }
  td {
    padding: 9px 10px;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
  }
  td.num { font-family: 'IBM Plex Mono', monospace; text-align: right; }
  .pill {
    display: inline-block;
    padding: 3px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .pill.stockout_risk { background: var(--red-bg); color: #F0897A; }
  .pill.overstock_risk { background: var(--mustard-bg); color: #E0B65C; }

  #downloadLink {
    display: none;
    margin-top: 16px;
    color: var(--gold);
    font-size: 13px;
    text-decoration: none;
    font-family: 'IBM Plex Mono', monospace;
  }
  #downloadLink:hover { text-decoration: underline; }
  #note { color: var(--text-dim); font-size: 12px; margin-top: 10px; }
  #summaryEmpty { color: var(--text-dim); font-size: 14px; }
    .chart-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin: 0 0 14px;
  }
  .chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1.4fr;
    gap: 16px;
  }
  .chart-panel canvas { max-height: 220px; }
</style>
</head>
<body>

  <div class="topbar">
    <div>
      <p class="eyebrow">Favorita &middot; Forecast Ops</p>
      <h1>Store sales forecast</h1>
    </div>
    <div class="status"><span class="pulse"></span>PIPELINE ONLINE</div>
  </div>

  <div class="panel">
    <div class="controls">
      <div>
        <label class="field">Origin date</label>
        <input type="date" id="originDate" value="2017-07-20">
      </div>
      <div class="checkbox-row" style="margin-top: 20px;">
        <input type="checkbox" id="useInventory">
        <label for="useInventory">Use sample inventory data</label>
      </div>
    </div>
    <button id="runBtn" onclick="runForecast()">Run forecast</button>
  </div>

  <div id="metricsContainer"></div>
  <div id="chartsContainer"></div>
  <div id="summaryEmpty" class="panel">Pick a date and run a forecast to see results.</div>
  <div id="tableContainer"></div>
  <a id="downloadLink" href="#">&darr; Download full report (CSV)</a>
  <div id="note"></div>

<script>
async function runForecast() {
  const date = document.getElementById('originDate').value;
  const btn = document.getElementById('runBtn');
  const metricsContainer = document.getElementById('metricsContainer');
  const summaryEmpty = document.getElementById('summaryEmpty');
  const tableContainer = document.getElementById('tableContainer');
  const downloadLink = document.getElementById('downloadLink');
  const note = document.getElementById('note');

  btn.disabled = true;
  summaryEmpty.textContent = 'Running the pipeline -- this can take about a minute.';
  summaryEmpty.style.display = 'block';
  metricsContainer.innerHTML = '';
  tableContainer.innerHTML = '';
  downloadLink.style.display = 'none';
  note.textContent = '';

  try {
    const useInventory = document.getElementById('useInventory').checked;
    const invParam = useInventory ? '&inventory_path=data/sample_inventory.csv' : '';
    const res = await fetch(`/forecast?origin_date=${date}${invParam}`);
    const data = await res.json();
    if (!res.ok) {
      summaryEmpty.textContent = 'Error: ' + (data.detail || 'unknown error');
      btn.disabled = false;
      return;
    }

    summaryEmpty.style.display = 'none';

    const counts = data.decision_status_counts;
    const stockout = counts.stockout_risk || 0;
    const overstock = counts.overstock_risk || 0;

    metricsContainer.innerHTML = `
      <div class="metrics">
        <div class="metric total"><div class="num">${data.rows}</div><div class="lbl">Forecasts generated</div></div>
        <div class="metric stockout"><div class="num">${stockout}</div><div class="lbl">Stockout risk</div></div>
        <div class="metric overstock"><div class="num">${overstock}</div><div class="lbl">Overstock risk</div></div>
      </div>`;
    renderCharts(data);
    if (data.priority_items.length > 0) {
      let html = '<table><tr><th>Store</th><th>Product</th><th>Target date</th><th style="text-align:right">Forecast</th><th>Status</th></tr>';
      for (const row of data.priority_items) {
        const label = row.decision_status === 'stockout_risk' ? 'Stockout risk' : 'Overstock risk';
        html += `<tr><td>${row.store_nbr}</td><td>${row.family}</td>` +
                `<td class="num">${row.target_date}</td><td class="num">${Number(row.forecast).toFixed(1)}</td>` +
                `<td><span class="pill ${row.decision_status}">${label}</span></td></tr>`;
      }
      html += '</table>';
      tableContainer.innerHTML = html;
      if (data.priority_items_truncated) {
        note.textContent = 'Showing the first 100 items needing attention. Download the full report for everything.';
      }
    } else {
      tableContainer.innerHTML = '<div class="panel" style="color: var(--text-dim); font-size:14px;">No items need attention for this date (or no inventory data was supplied).</div>';
    }

    downloadLink.href = `/download?origin_date=${date}`;
    downloadLink.style.display = 'inline-block';
  } catch (err) {
    summaryEmpty.style.display = 'block';
    summaryEmpty.textContent = 'Something went wrong: ' + err;
  } finally {
    btn.disabled = false;
  }
}
let statusChart, confidenceChart, priorityChart;
const statusLabels = { optimal: "Optimal", stockout_risk: "Stockout risk", overstock_risk: "Overstock risk", no_inventory_data: "No inventory data" };
const statusColors = { optimal: "#3FBF8F", stockout_risk: "#E5533D", overstock_risk: "#C9932E", no_inventory_data: "#45607C" };
const confLabels = { high: "High", medium: "Medium", low: "Low", unknown: "Unknown" };
const confColors = { high: "#3FBF8F", medium: "#E8A33D", low: "#E5533D", unknown: "#45607C" };

Chart.defaults.color = '#8FA3B8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.borderColor = '#263449';

function renderCharts(data) {
  if (statusChart) statusChart.destroy();
  if (confidenceChart) confidenceChart.destroy();
  if (priorityChart) priorityChart.destroy();

  document.getElementById('chartsContainer').innerHTML = `
    <div class="chart-grid">
      <div class="panel chart-panel"><p class="chart-title">Decision status</p><canvas id="statusChart"></canvas></div>
      <div class="panel chart-panel"><p class="chart-title">Model confidence</p><canvas id="confidenceChart"></canvas></div>
      <div class="panel chart-panel"><p class="chart-title">Top items needing attention</p><canvas id="priorityChart"></canvas></div>
    </div>
  `;

  const statusCounts = data.decision_status_counts;
  statusChart = new Chart(document.getElementById('statusChart'), {
    type: 'doughnut',
    data: {
      labels: Object.keys(statusCounts).map(k => statusLabels[k] || k),
      datasets: [{ data: Object.values(statusCounts),
                   backgroundColor: Object.keys(statusCounts).map(k => statusColors[k] || '#45607C'),
                   borderColor: '#14202F', borderWidth: 2 }]
    },
    options: { plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } } }
  });

  const confCounts = data.confidence_counts;
  confidenceChart = new Chart(document.getElementById('confidenceChart'), {
    type: 'doughnut',
    data: {
      labels: Object.keys(confCounts).map(k => confLabels[k] || k),
      datasets: [{ data: Object.values(confCounts),
                   backgroundColor: Object.keys(confCounts).map(k => confColors[k] || '#45607C'),
                   borderColor: '#14202F', borderWidth: 2 }]
    },
    options: { plugins: { legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } } } }
  });

  const sortedItems = [...data.priority_items].sort((a, b) => b.forecast - a.forecast).slice(0, 10);
  priorityChart = new Chart(document.getElementById('priorityChart'), {
    type: 'bar',
    data: {
      labels: sortedItems.map(r => `Store ${r.store_nbr} · ${r.family}`),
      datasets: [{ data: sortedItems.map(r => r.forecast),
                   backgroundColor: sortedItems.map(r => statusColors[r.decision_status] || '#45607C'),
                   borderRadius: 4 }]
    },
    options: {
      indexAxis: 'y',
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#263449' }, ticks: { font: { family: "'IBM Plex Mono', monospace", size: 11 } } },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    }
  });
}
</script>
</body>
</html>
"""