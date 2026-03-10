# 🔥 P2-ETF-HAWKES

> **Self-Exciting Point Process ETF Rotation Model**
> Hawkes Process · Exponential & Power-Law Kernels · Hurst Exponent · Daily Signals

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://p2-etf-hawkes.streamlit.app)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hawkes-data)
[![GitHub Actions](https://img.shields.io/badge/Pipeline-Daily-green?logo=githubactions)](https://github.com/P2SAMAPA/P2-ETF-HAWKES/actions)

---

## 📋 Table of Contents

- [Overview](#overview)
- [The Science — Hawkes Processes](#the-science--hawkes-processes)
- [ETF Universe](#etf-universe)
- [Model Architecture](#model-architecture)
  - [Event Detection](#event-detection)
  - [Kernel Selection](#kernel-selection)
  - [Option A — Hawkes Only](#option-a--hawkes-only)
  - [Option B — Hawkes + Hurst](#option-b--hawkes--hurst)
- [Signal Generation](#signal-generation)
- [Project Structure](#project-structure)
- [Data Pipeline](#data-pipeline)
- [Deployment Guide](#deployment-guide)
- [Configuration & Secrets](#configuration--secrets)
- [GitHub Actions Workflows](#github-actions-workflows)
- [HuggingFace Dataset](#huggingface-dataset)
- [Streamlit App](#streamlit-app)
- [Relationship to Other P2 Projects](#relationship-to-other-p2-projects)
- [Limitations & Caveats](#limitations--caveats)

---

## Overview

P2-ETF-HAWKES applies **Hawkes self-exciting point processes** to daily ETF rotation across six fixed-income and commodity ETFs. Unlike momentum or machine-learning models that treat returns as independent draws, the Hawkes process explicitly models the feedback loop inside markets: a large move in GLD today increases the probability of further large GLD moves tomorrow.

The model answers two questions:

1. **Which ETF is currently in a self-reinforcing burst of activity?** (Option A — Hawkes only)
2. **Which ETF combines current self-excitation with long-run trend persistence?** (Option B — Hawkes + Hurst)

Both signals are generated daily via GitHub Actions, stored on HuggingFace, and displayed in a Streamlit app with zero training in the UI.

---

## The Science — Hawkes Processes

### Intuition

Financial markets are not memoryless. When a large trade occurs, algorithms react, liquidity shifts, and volatility clusters. A standard Poisson process assumes events arrive independently at a fixed rate — this is demonstrably wrong for real markets. Hawkes processes fix this by making the arrival rate itself a function of past events.

### Mathematical Foundation

A univariate Hawkes process is a counting process N(t) with **conditional intensity**:

```
λ*(t) = μ + Σᵢ h(t − Tᵢ)
         ↑        ↑
     baseline   excitation from past events at times Tᵢ
```

Where:
- **μ** (mu) — baseline event rate (events per day in the absence of excitation)
- **h(·)** — excitation kernel: how much influence a past event has at lag t
- The sum runs over all past event times Tᵢ < t

The **branching ratio** `||h||₁ = ∫₀^∞ h(t) dt` must be < 1 for stability (subcritical process). It represents the average number of offspring events triggered by each immigrant event. A branching ratio of 0.7 means each event triggers on average 0.7 further events.

### Excitation Ratio

The key signal metric used in this model is:

```
Excitation Ratio = λ*(t) / μ
```

- **= 1.0** → currently at baseline, no self-excitation
- **> 1.0** → currently above baseline, self-exciting burst in progress
- **>> 1.0** → intense cluster, strong momentum signal

### Cluster Representation

Equivalently, a Hawkes process can be viewed as a **branching (immigration-birth) process**:
- Immigrants arrive at baseline rate μ (exogenous events — news, macro releases)
- Each immigrant spawns offspring events according to kernel h(·)
- Each offspring can itself spawn further offspring
- The total process is the superposition of all generations

This is exactly the feedback loop observed in real markets.

---

## ETF Universe

| ETF | Asset Class | Role |
|-----|------------|------|
| **TLT** | Long-term US Treasuries | Rate risk / flight to safety |
| **LQD** | Investment-grade corporate bonds | Credit quality |
| **HYG** | High-yield corporate bonds | Risk appetite |
| **VNQ** | US REITs | Real estate / rate sensitivity |
| **GLD** | Gold | Inflation hedge / crisis hedge |
| **SLV** | Silver | Industrial metals / inflation |

**Benchmarks (comparison only, not in signal universe):**

| ETF | Asset Class |
|-----|------------|
| **SPY** | US equities (S&P 500) |
| **AGG** | US aggregate bonds |

---

## Model Architecture

### Event Detection

Before fitting the Hawkes process, we must define what counts as a **significant market event**. Three definitions are implemented and tested empirically:

| Method | Definition | Best when |
|--------|-----------|-----------|
| `return_only` | \|return\| > 1σ (rolling 63-day std dev) | Returns dominate signal |
| `volume_only` | Volume > 20-day rolling mean | Volume leads price |
| `combined` | Both conditions simultaneously | Strictest, filters noise |

**Automatic selection:** At the first pipeline run (or when `--reselect-event-def` is passed), all three definitions are evaluated on a 252-day out-of-sample window using a forward-return hit ratio test. The definition with the highest OOS hit ratio is saved to `metadata.json` and reused on all subsequent daily runs — avoiding unnecessary recomputation.

### Kernel Selection

Two kernel types are fitted for each ETF, and the one with the lower **AIC (Akaike Information Criterion)** is selected:

#### Exponential Kernel
```
h(t) = α · exp(−β · t)
```
- **α** — excitation magnitude (how much each event raises intensity)
- **β** — decay rate (how quickly the excitation fades)
- **Branching ratio** = α/β
- **Key advantage:** Markov property — intensity can be computed recursively in O(n), making MLE fast
- **Best for:** Short-memory markets, intraday-style clustering at daily resolution

#### Power-Law Kernel
```
h(t) = k / (t + c)^(1 + θ)
```
- **k** — excitation magnitude
- **c** — offset (prevents singularity at t=0)
- **θ** — tail decay exponent
- **Branching ratio** = k · c^(−θ) / θ
- **Key advantage:** Slower decay — captures excitation that persists over days/weeks
- **Best for:** Long-memory regimes, post-crisis clustering

Both kernels are fitted via **Maximum Likelihood Estimation (MLE)** using `scipy.optimize.minimize` with L-BFGS-B, with multiple random starting points to avoid local minima.

### Option A — Hawkes Only

**Signal logic:**
1. Fit Hawkes model per ETF (best kernel by AIC)
2. Compute current intensity λ*(t) at the last available date
3. Compute excitation ratio λ*(t)/μ per ETF
4. Normalise ratios to [0, 1] across the 6 ETFs
5. ETF with highest normalised ratio = next-day signal
6. Conviction = normalised excitation ratio of top ETF

**Interpretation:** Which ETF is currently experiencing the most intense self-reinforcing burst of activity relative to its own historical baseline?

### Option B — Hawkes + Hurst

**Conviction formula:**
```
Combined Score = 0.65 × Hawkes Excitation Score + 0.35 × Hurst Score
```

**Hurst component:**
- Computed via R/S (Rescaled Range) analysis on a 252-day rolling window
- H > 0.5 → persistent, trending series → higher Hurst score
- H = 0.5 → random walk → neutral score
- H < 0.5 → mean-reverting → penalises the signal

**Rationale for the combination:**
- Hawkes captures *right now* — is activity self-reinforcing at this moment?
- Hurst captures *structural character* — is this ETF in a trending regime over the past year?
- When both agree (high excitation AND H > 0.5), conviction is strongest
- When they disagree (high excitation but H < 0.5), conviction is reduced — the burst may not persist

| Hawkes | Hurst | Interpretation |
|--------|-------|----------------|
| High λ*(t)/μ | H > 0.6 | ✅ Strong signal — self-exciting AND trending |
| High λ*(t)/μ | H ≈ 0.5 | ⚠️ Moderate — burst may revert |
| High λ*(t)/μ | H < 0.4 | ⚠️ Cautious — mean-reverting regime |
| Low λ*(t)/μ | H > 0.6 | ℹ️ Trending but quiet — wait for excitation |
| Low λ*(t)/μ | H < 0.4 | ❌ Avoid — mean-reverting and quiet |

---

## Signal Generation

### Conviction Labels

| Score | Label |
|-------|-------|
| ≥ 0.75 | Very High |
| ≥ 0.55 | High |
| ≥ 0.35 | Moderate |
| < 0.35 | Low |

### Conviction Gate
A minimum conviction threshold (default 0.30, configurable in sidebar) determines whether to take a position or hold CASH. Below the gate, the model holds CASH earning the risk-free rate.

### Cross-ETF Excitation Heatmap
The app displays a 6×6 heatmap showing how much activity in one ETF predicts increased activity in another on the following day. Entry [i, j] = correlation between ETF j's event indicator and ETF i's next-day absolute return. This is a computationally tractable proxy for the full multivariate Hawkes Γ matrix.

---

## Project Structure

```
P2-ETF-HAWKES/
│
├── app.py               # Streamlit UI — tabs A, B, About
├── hawkes.py            # Core Hawkes model: fitting, intensity, events, cross-matrix
├── hurst.py             # Hurst exponent: R/S analysis, rolling computation
├── strategy.py          # Signal generation Option A & B, backtesting, metrics
├── train.py             # Daily pipeline orchestrator (called by GitHub Actions)
├── data_manager.py      # HuggingFace read/write, yfinance/Stooq OHLCV fetch
├── reseed.py            # One-time full OHLCV seed from 2008
├── requirements.txt     # Python dependencies
│
└── .github/
    └── workflows/
        ├── daily.yml    # Daily pipeline at 9pm EST weekdays
        └── reseed.yml   # One-time manual reseed trigger
```

### Module Responsibilities

| File | Responsibility |
|------|---------------|
| `hawkes.py` | All Hawkes mathematics: event detection, MLE fitting, recursive intensity computation, cross-excitation matrix, signal ranking |
| `hurst.py` | R/S Hurst exponent: rolling window, conviction conversion, colour coding |
| `strategy.py` | Option A/B signal blending, conviction labels, next trading day calculation |
| `train.py` | Orchestrates all steps in order, saves results to HF, handles CLI arguments |
| `data_manager.py` | All I/O: yfinance fetch, Stooq fallback, HF read/write, incremental update logic |
| `app.py` | Streamlit UI: loads from HF, renders hero banner, charts, heatmap, audit trail |
| `reseed.py` | Standalone one-time script, does not import other project modules |

---

## Data Pipeline

### Daily Flow (GitHub Actions, 9pm EST)

```
1. Load existing ohlcv_data.parquet from HF
2. Fetch new OHLCV rows since last update (yfinance → Stooq fallback)
3. Concat + dedup + ffill → updated ohlcv_data.parquet
4. Load best_event_def from metadata (or reselect if flag set)
5. Fit Hawkes models: all 6 ETFs, both kernels, best by AIC
6. Compute rolling Hurst (252-day window) for all 6 ETFs
7. Generate Option A signal (Hawkes excitation ratio)
8. Generate Option B signal (0.65×Hawkes + 0.35×Hurst)
9. Compute 6×6 cross-ETF excitation matrix
10. Build intensity history DataFrame
11. Save to HuggingFace in single commit:
    - ohlcv_data.parquet
    - signals_latest.parquet
    - intensity_history.parquet
    - hurst_history.parquet
    - cross_excitation.parquet
    - hawkes_params.json
    - metadata.json
```

### Data Sources

| Source | Used for | Fallback |
|--------|---------|---------|
| Yahoo Finance | Primary OHLCV | — |
| Stooq | OHLCV fallback | Triggered if YF fails |

No FRED API required — this project does not use macro data.

### yfinance → Stooq Fallback Logic

```python
df = fetch_ohlcv_yf(ticker, start, end)   # try YF first
if df is None:
    df = fetch_ohlcv_stooq(ticker, start, end)  # fall back to Stooq
```

Both sources implement exponential backoff with random jitter on rate-limit errors.

---

## Deployment Guide

### Prerequisites
- GitHub account with the `P2SAMAPA/P2-ETF-HAWKES` repo created
- HuggingFace account with `P2SAMAPA/p2-etf-hawkes-data` dataset repo created (set to Public or Private with token access)
- Streamlit Community Cloud account

### Step 1 — Push code to GitHub

```
P2-ETF-HAWKES/
├── app.py
├── hawkes.py
├── hurst.py
├── strategy.py
├── train.py
├── data_manager.py
├── reseed.py
├── requirements.txt
└── .github/workflows/
    ├── daily.yml
    └── reseed.yml
```

### Step 2 — Add GitHub Secrets

Go to **GitHub → Settings → Secrets and variables → Actions → New repository secret**:

| Secret Name | Value |
|-------------|-------|
| `HF_TOKEN` | Your HuggingFace write-access token |

### Step 3 — Run Reseed (once only)

Go to **GitHub → Actions → Reseed Dataset → Run workflow**

This fetches full OHLCV history from 2008 for all 8 tickers and uploads to HuggingFace. Runtime: ~10-20 minutes.

### Step 4 — Run Daily Pipeline manually (first time)

Go to **GitHub → Actions → Daily Training Pipeline → Run workflow**

This fits all Hawkes models and pushes signals to HF for the first time. Runtime: ~10-15 minutes.

### Step 5 — Connect Streamlit

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. New app → connect `P2SAMAPA/P2-ETF-HAWKES` → main branch → `app.py`
3. Add secrets (Settings → Secrets):

```toml
HF_TOKEN = "your-huggingface-token"
GH_PAT = "your-github-pat-token"
GITHUB_REPO = "P2SAMAPA/P2-ETF-HAWKES"
```

4. Deploy

### Step 6 — Open the app

Click **🚀 Run Model** in the sidebar. Results load from HuggingFace — no training in the UI.

---

## Configuration & Secrets

### GitHub Actions Secrets

| Secret | Purpose | Required |
|--------|---------|---------|
| `HF_TOKEN` | Read/write HuggingFace dataset | ✅ Yes |

### Streamlit Secrets

| Secret | Purpose | Required |
|--------|---------|---------|
| `HF_TOKEN` | Read model outputs from HF | ✅ Yes |
| `GH_PAT` | Trigger pipeline from app (Force Data Refresh button) | ✅ Yes |
| `GITHUB_REPO` | Repo name for pipeline dispatch | ✅ Yes |

### Sidebar Controls

| Control | Default | Description |
|---------|---------|-------------|
| Backtest Start Year | 2012 | Earliest year for display (model trains on full history) |
| Min Conviction Gate | 0.30 | Below this score → CASH |
| Transaction Fee (bps) | 5 | One-way cost per rotation |
| Benchmark | SPY | Comparison for equity curve |
| Show intensity history | On | Toggle λ*(t) chart |

### train.py CLI Arguments

```bash
# Normal daily run
python train.py

# Force full data rebuild from scratch
python train.py --force-refresh

# Local testing — skip HuggingFace write
python train.py --local

# Re-run event definition selection (OOS comparison, ~10 min extra)
python train.py --reselect-event-def

# Different start year for backtest window
python train.py --start-year 2015
```

---

## GitHub Actions Workflows

### `daily.yml` — Daily Training Pipeline

| Property | Value |
|----------|-------|
| Schedule | 02:00 UTC (9pm EST) Mon-Fri |
| Trigger | Schedule + manual dispatch |
| Timeout | 30 minutes |
| Secrets needed | `HF_TOKEN` |

### `reseed.yml` — One-Time OHLCV Seed

| Property | Value |
|----------|-------|
| Schedule | Manual only (no cron) |
| Timeout | 45 minutes |
| Secrets needed | `HF_TOKEN` |
| When to run | Once after repo creation, or after a full data reset |

---

## HuggingFace Dataset

**Repo:** `P2SAMAPA/p2-etf-hawkes-data`

### Files stored

| File | Contents | Updated |
|------|---------|---------|
| `ohlcv_data.parquet` | Full OHLCV history for all 8 tickers, MultiIndex columns (ticker, field) | Daily |
| `signals_latest.parquet` | Latest signals for Option A and B | Daily |
| `intensity_history.parquet` | Per-ETF λ*(t) time series | Daily |
| `hurst_history.parquet` | Rolling Hurst H per ETF | Daily |
| `cross_excitation.parquet` | 6×6 cross-ETF excitation correlation matrix | Daily |
| `hawkes_params.json` | Fitted μ, α, β (or k, c, θ), AIC, branching ratio per ETF | Daily |
| `metadata.json` | Last update dates, best event definition, current signals | Daily |

### `metadata.json` schema

```json
{
  "last_data_update":  "2026-03-10",
  "last_model_fit":    "2026-03-10",
  "best_event_def":    "combined",
  "signal_a":          "GLD",
  "signal_b":          "TLT",
  "next_date":         "2026-03-11",
  "dataset_version":   42,
  "tickers":           ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV", "SPY", "AGG"],
  "fields":            ["Open", "High", "Low", "Close", "Volume"]
}
```

---

## Streamlit App

**URL:** https://p2-etf-hawkes.streamlit.app

### Tab: 📊 Option A — Hawkes Only
- White hero banner with next-day signal, conviction label and score
- Per-ETF excitation ratio cards (λ*(t)/μ)
- ETF ranking bar chart
- λ*(t) intensity history chart (toggle in sidebar)
- 6×6 cross-ETF excitation heatmap
- Fitted parameters table (μ, α/β or k/c/θ, branching ratio, AIC)
- Signal history audit trail

### Tab: 📈 Option B — Hawkes + Hurst
- Same layout as Option A
- Additional per-ETF Hurst exponent cards with trend label
- Combined conviction = 65% Hawkes + 35% Hurst
- Colour-coded Hurst labels (Strong Trend / Mild Trend / Random Walk / Mean-Rev)

### Tab: ℹ️ About
- Full methodology explanation
- Mathematical foundations
- Event definition comparison table
- Kernel comparison table

### Buttons
| Button | Action |
|--------|--------|
| 🚀 Run Model | Load data from HF and render results |
| 🔄 Force Data Refresh | Trigger GitHub Actions pipeline via API dispatch |

---

## Relationship to Other P2 Projects

| Project | Model Type | Signal Basis | Timescale |
|---------|-----------|-------------|-----------|
| P2-ETF-RNN-LSTM | Deep learning (LSTM) | Directional accuracy + Hurst + vote share | Medium-term |
| P2-ETF-CNN-LSTM-ALTERNATIVE-APPROACHES | CNN-LSTM variants | Return/Sharpe/Z-score consensus | Medium-term |
| P2-ETF-REGIME-PREDICTOR | Wasserstein k-means + Momentum | Regime + momentum ranking | Short-medium |
| **P2-ETF-HAWKES** | **Hawkes Process (statistical)** | **Self-excitation clustering + Hurst** | **Short-term** |

P2-ETF-HAWKES is the only fully **non-ML, mathematically interpretable** model in the suite. Every parameter (μ, α, β, H) has a direct statistical meaning. It is designed to complement the other models rather than replace them — particularly useful during regime transitions and volatility clustering episodes where the ML models may lag.

---

## Limitations & Caveats

**Daily frequency limitation**
Hawkes processes are most powerful at intraday tick frequency where self-excitation decays within minutes. At daily resolution, excitation may decay within hours (intraday), meaning inter-day clustering is the primary signal captured. Results are meaningful but less precise than a tick-level implementation.

**MLE convergence**
For ETFs with few events (e.g. GLD in low-volatility periods), MLE may produce unreliable parameter estimates. The model logs a warning when fewer than 10 events are detected in the fitting window.

**Branching ratio near 1.0**
A branching ratio close to 1.0 indicates a near-critical process that is mathematically valid but practically fragile — small perturbations could make it explosive. The model enforces `branching < 1` via the stability constraint in the log-likelihood.

**Cross-excitation approximation**
The 6×6 heatmap uses a correlation proxy rather than a full multivariate Hawkes MLE (which would require jointly fitting 36 kernel parameters). The proxy is computationally tractable and directionally correct but not a formal multivariate Hawkes estimate.

**No transaction costs in intensity computation**
The Hawkes model itself does not account for market impact or liquidity. Transaction costs are applied in the strategy backtest layer only.

**Past performance**
This is a research tool. Backtested results are optimistic relative to live trading. Results shown in the app use in-sample fitted parameters. Not investment advice.

---

## Author

**P2SAMAPA**

Part of the P2 ETF Quantitative Research Suite.

---

*Built with Python · Streamlit · HuggingFace · GitHub Actions*
