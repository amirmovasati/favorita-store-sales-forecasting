🌐 [فارسی / Persian](README.fa.md)
# Favorita Store Sales Forecasting

Multi-horizon retail demand forecasting for Corporación Favorita (Ecuador) — built as an end-to-end case study in diagnosing *why* a forecasting model fails on specific segments, not just in building one that scores well on average, **and in turning that model into a working, deployable system a non-technical business user can actually run.**

## Why This Project

Most portfolio forecasting projects stop at a trained model in a notebook. This one goes further: **a validated model, a documented diagnosis of its failure modes (see below), and a working deployment pipeline** — data ingestion, a business decision layer, a REST API, a Docker container, lightweight monitoring, and a simple web page a non-technical user can run without touching any code.

- 📊 **Model**: 12.44% WAPE, 64% lower error than a naive baseline, with a validated confidence-tagging layer for low-reliability predictions (details below)
- ⚙️ **Pipeline**: ingestion → validation → feature engineering → inference → business decision (stockout/overstock risk), runnable end-to-end with one command
- 🌐 **Serving**: FastAPI + Docker, with a no-code web page for business users — pick a date, click a button, get a plain-language forecast summary and a downloadable report
- 📈 **Monitoring**: every pipeline run is logged (success/failure, duration, output counts)

👉 See [Production Deployment Pipeline](#production-deployment-pipeline) for the full breakdown, or keep reading for the modeling work.

## The Business Problem

Retailers need accurate demand forecasts to plan inventory, staffing, and promotions. A model that is accurate *on average* can still be dangerously wrong for the subset of days that matter most operationally — large-volume sales spikes, where under-forecasting means stockouts and lost revenue.

This project forecasts daily sales, 1 to 15 days ahead, for every store–product-family combination in the Favorita chain, using a direct multi-horizon XGBoost model. The focus of the analysis is not just "build a model," but **finding, explaining, and quantifying where and why the model is least reliable** — and building a mechanism to flag that unreliability automatically.

## Key Results

- **Overall validation WAPE: 12.44%** (MAE 59.2, RMSE 211.7) across a 15-day forecast horizon, evaluated on four held-out origins.
- **64% lower error than a naive persistence baseline** (34.22% WAPE) and **27% lower error than a weekly-seasonal-naive baseline** (16.99% WAPE) — see the table below.
- **High-confidence predictions (67% of tail volume) achieve a WAPE of 10.46%** and a severe-underprediction rate of just **0.42%**.
- **Low-confidence predictions carry a severe-underprediction rate of 15.22%** — 36x higher than high-confidence predictions — identified automatically by a post-hoc reliability layer (see below), without changing the model itself.
- Diagnosed that large sales spikes are concentrated in specific store/family combinations (mainly `CLEANING`, store type `C`) and ruled out promotions as the cause through targeted hypothesis testing.

### Baseline Comparison

| Model | Average WAPE (across 3 forecast origins, 15-day horizon) |
|---|---|
| Naive (persistence) | 34.22% |
| Weekly Seasonal Naive | 16.99% |
| **XGBoost (this project)** | **12.44%** |

The weekly-seasonal baseline (last observed value for the same day of week) is already a reasonably strong baseline for retail data, since it captures weekly demand cycles. The XGBoost model still cuts its error by more than a quarter, primarily by learning cross-series patterns (store, family, cluster) and longer-term trend/volatility signals that a per-series seasonal lookup cannot capture.

## Approach

### 1. Exploratory Data Analysis
Sales distribution, trend, and seasonality (yearly / monthly / weekly) were analyzed to understand the demand patterns before any modeling decision was made.

### 2. Feature Engineering
- **Calendar features**: day of week, month, national/regional/local holidays, Black Friday, Cyber Monday, Mother's Day (mapped correctly by locale to avoid leaking store-irrelevant holidays).
- **Historical demand features**: lags (7/14/28/365 days) and rolling means (7/14/28 days), all leakage-safe (`shift(1)` before rolling).
- **Volatility features** (added after diagnosing the spike problem — see below): rolling standard deviation and a coefficient-of-variation feature (`rolling_cv_28`) that captures how volatile a series is, independent of its sales level.
- **Historical spike-rate feature**: how often a given store–family series has produced an outlier day in its own history, computed with a leakage-safe expanding window.

### 3. Direct Multi-Horizon Forecasting
Rather than a single one-step model iterated forward, a **separate direct-forecast dataset is built for each horizon (1 to 15 days)**, each with horizon-appropriate lag features. This avoids compounding forecast errors across the horizon.

### 4. Model
XGBoost (`reg:squarederror`, 1000 trees, early stopping) with native categorical feature support (`store_nbr`, `family`, `store_type`, `city`, `state`, `store_cluster`).

### 5. Validation Strategy
Evaluated across **four forecast origins** spanning June–July 2017. One origin (`2017-07-10`) was deliberately **held out from all feature-engineering and threshold decisions** and only inspected once, at the end, as a final generalization check — to guard against unintentionally overfitting design choices to the validation set.

## The Spike Problem: A Diagnostic Case Study

Initial validation showed the model performing well overall but underpredicting a small subset of large-volume days (~1.3% of high-volume rows had predictions less than half the actual value). Rather than immediately tuning hyperparameters, the investigation followed a structured diagnostic path:

1. **Ruled out promotions** — severe-underprediction rows and well-predicted rows had nearly identical promotion rates (98.4% vs 96.0%), so `onpromotion` was not the missing signal.
2. **Found concentration by segment** — 44–56% of severe cases fell in the `CLEANING` family (vs. ~13% of the well-predicted population), and store type `C` was similarly over-represented.
3. **Found an information gap** — severely underpredicted rows had systematically *lower* historical lag/rolling-mean values than well-predicted rows, meaning the spike was genuinely not visible in the features available at prediction time for those specific rows.
4. **Added volatility features** (`rolling_std_*`, `rolling_cv_28`) to give the model a way to recognize *inherently volatile* series, even when their recent average looked unremarkable.
5. **Validated with SHAP** — confirmed the new volatility feature carries real signal (severe cases show 76% higher `rolling_cv_28` than well-predicted cases) even though the model's built-in feature-importance ranking under-weights it relative to level-based features (`lag_28`, `rolling_mean_*`).

This is a deliberate finding, not a limitation glossed over: **the model's objective function (squared error) has little incentive to fit a rare, high-variance subset of the data well**, even when the model has features that could help. Rather than force-fitting this with an aggressive re-weighting scheme (which risks overfitting to ~120 rows), the project instead built a way to **flag** these cases explicitly.

## Prediction Confidence Layer

Every prediction is tagged `high` / `medium` / `low` confidence based on the tercile of the series' own `rolling_cv_28` (volatility) in the validation set — a fully data-driven threshold, not an arbitrary cutoff. This is validated, not just asserted:

| Confidence Tier | WAPE | Severe Underprediction Rate |
|---|---|---|
| High | 10.46% | 0.42% |
| Medium | 14.31% | 1.88% |
| Low | 40.18% | 15.22% |

This turns a known model weakness into an actionable signal: a downstream planning process can treat `low`-confidence forecasts differently (e.g. wider safety stock, manual review) instead of trusting every number equally.

## Model Interpretation

Feature importance (gain-based) and SHAP values both confirm the model relies primarily on recent demand level (`lag_28`, `rolling_mean_7/14/28`, `lag_365`) — consistent with a squared-error objective optimizing for the bulk of the data rather than rare tail events.

![SHAP Feature Importance](assets/shap_summary_plot.png)

## Limitations

- The model underpredicts roughly 1.1% of high-volume (>1000 units) days by more than 50%, concentrated in a small number of store–family series with inherently volatile demand histories.
- This is treated as a **known, quantified, and flagged** limitation (via the confidence layer above) rather than an unaddressed gap — a deliberate choice over chasing diminishing-return fixes (aggressive re-weighting or unconstrained hyperparameter search) that risk overfitting to a few hundred rows.

---

## Production Deployment Pipeline

Training a good model is one problem. Getting its output into the hands of someone who plans inventory — and who has never opened a terminal — is a different, and in practice just as important, problem. This section covers the second one.

### Architecture

```
train.csv
   │
   ▼  ingestion.py (simulates daily data arrival)
history (accumulated day by day, validated on arrival)
   │
   ▼  validation.py (schema / missing-key / invalid-value checks)
   ▼  feature_engineering.py (calendar, lag, rolling, volatility features)
   ▼  inference.py (loads the trained XGBoost model, predicts)
   ▼  decision.py (clips negatives, attaches confidence, flags stockout/overstock risk)
   │
   ▼
report_<date>.csv  — a business-usable file, plus a JSON API and a web page
```

Each stage is an independent, tested Python module under `src/pipeline/`. `run_pipeline.py` is the single entry point that wires them together — one command runs the whole chain for a given date.

### Business Decision Layer

The pipeline's output is not a raw forecast number. Each prediction is compared against an (externally supplied) inventory level to flag `stockout_risk`, `overstock_risk`, or `optimal` — the pipeline never invents inventory data; it expects it as input, the way a real system would pull it from a warehouse/ERP feed. A small sample inventory file (`data/sample_inventory.csv`) is included for demonstration.

### No-Code Web Interface

A business user with zero technical background can use this system directly:

1. Open the web page.
2. Pick a date.
3. Click "Get Forecast."
4. Read a one-line summary ("159 items at stockout risk...") and a small color-coded table of only the items that need attention.
5. Download the full report as CSV if more detail is needed.

No command line, no code, no file paths to type.

### API

Built with FastAPI:

| Endpoint | Purpose |
|---|---|
| `GET /` | The no-code web page described above |
| `GET /forecast?origin_date=YYYY-MM-DD` | Runs the pipeline, returns a JSON summary + priority items |
| `GET /download?origin_date=YYYY-MM-DD` | Downloads the full CSV report |
| `GET /health` | Liveness check |
| `GET /monitoring` | Recent pipeline run history (see Monitoring below) |

### Containerization

The service is packaged with Docker, so it runs identically regardless of the host machine's Python version or installed packages:

```bash
docker build -t favorita-pipeline .
docker run -p 8000:8000 \
  -v ${PWD}/data:/app/data \
  -v ${PWD}/outputs:/app/outputs \
  -v ${PWD}/reports:/app/reports \
  -v ${PWD}/logs:/app/logs \
  favorita-pipeline
```

Then open `http://localhost:8000`.

### Monitoring

Every pipeline run — whether triggered from the command line, the API, or the web page — is logged to `logs/pipeline_runs.jsonl` with its timestamp, duration, row counts, and outcome (success or the exact error on failure). `GET /monitoring` surfaces the most recent runs without needing to open the log file directly.

### Design Choices Worth Noting

- **Rolling/walk-forward simulation, not a single static split.** No genuinely new data exists beyond the original Kaggle dataset, so the pipeline treats a chosen date as "today," simulates daily data arrival up to that point, and forecasts forward — closer to how a real deployed system experiences data than a one-time train/test split.
- **Deliberately excluded**: Kubernetes, complex CI/CD, cloud infrastructure, microservices. None of these solve a problem this project actually has (single-container load, no multi-service traffic); adding them would be complexity for its own sake, not engineering judgment.
- **`Favorita_GlobalModel.py` (the research/EDA script) was left untouched.** The pipeline's feature-engineering logic is a deliberate, documented duplication tuned for reuse and production use, rather than a refactor of the research script — keeping the two concerns (research vs. production) cleanly separated.

## Project Structure

```
Favorita Store Sales Forecasting/
├── src/
│   ├── Favorita_GlobalModel.py   # Full modeling pipeline: EDA → features → model → diagnostics
│   └── pipeline/                 # Production deployment pipeline (see above)
│       ├── ingestion.py
│       ├── validation.py
│       ├── history_store.py
│       ├── feature_engineering.py
│       ├── inference.py
│       ├── decision.py
│       └── monitoring.py
├── run_pipeline.py                # Single entry point for the deployment pipeline
├── api.py                         # FastAPI service + no-code web page
├── Dockerfile
├── .dockerignore
├── scripts/
│   └── test_pipeline_manual.py    # Manual smoke test against real project data
├── data/                          # Raw Kaggle data + sample_inventory.csv (not tracked in git)
├── outputs/                       # Trained model artifacts (not tracked in git)
├── reports/                       # Pipeline output reports (not tracked in git)
├── logs/                          # Pipeline run history (not tracked in git)
├── notebooks/                     # (reserved for exploratory notebooks)
├── assets/                        # Images used in this README
├── requirements.txt
└── README.md
```

## Data Source

[Store Sales - Time Series Forecasting (Kaggle)](https://www.kaggle.com/c/store-sales-time-series-forecasting) — daily sales data for Corporación Favorita, an Ecuador-based grocery retailer, including store metadata, oil prices, holidays, and promotions.

## How to Run

### 1. Train the model

```bash
pip install -r requirements.txt
```

Place the Kaggle CSV files in `data/`, then run:

```bash
python src/Favorita_GlobalModel.py
```

By default, model artifacts (trained model, processed datasets, predictions) are saved to `outputs/<run_tag>/` inside the project — no configuration needed. To redirect large artifacts elsewhere (e.g. to avoid cloud-sync storage limits), set the `FAVORITA_OUTPUT_DIR` environment variable once; the script falls back to the in-project default automatically if it's not set.

### 2. Run the deployment pipeline

```bash
python run_pipeline.py --date 2017-07-20
python run_pipeline.py --date 2017-07-20 --inventory data/sample_inventory.csv
```

### 3. Run the API + web page (with Docker)

```bash
docker build -t favorita-pipeline .
docker run -p 8000:8000 -v ${PWD}/data:/app/data -v ${PWD}/outputs:/app/outputs -v ${PWD}/reports:/app/reports -v ${PWD}/logs:/app/logs favorita-pipeline
```

Open `http://localhost:8000`.

## Tech Stack

Python · pandas · NumPy · XGBoost · SHAP · scikit-learn (metrics) · matplotlib · FastAPI · uvicorn · Docker
