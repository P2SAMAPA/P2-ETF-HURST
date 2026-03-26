markdown
# 📐 P2-ETF-HURST

**Hurst Confluence (HRC) ETF Rotation Signal**

Two independent modules:
- **Option A**: Fixed Income / Commodities (6 ETFs: TLT, LQD, HYG, VNQ, GLD, SLV)
- **Option B**: Equity Sectors (12 ETFs: SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME)

**Benchmarks:** SPY · AGG  
**Daily cron:** 02:00 UTC Mon–Fri (both options run sequentially)

---

## 🔬 Concept: Hurst Confluence (HRC)

The Hurst exponent H measures long-range memory in a time series:

| H Range | Zone | Meaning |
|---------|------|---------|
| H ≥ 0.65 | Strong Trend | Past direction highly likely to continue |
| 0.55 ≤ H < 0.65 | Mild Trend | Persistent — buy signal zone |
| 0.50 ≤ H < 0.55 | Weak Trend | Near-trending — monitor closely |
| 0.45 ≤ H < 0.50 | Random Walk | No memory — avoid |
| H < 0.45 | Mean-Reverting | Past direction likely to reverse |

Rather than using H as a simple filter, HRC combines **three distinct Hurst-based signals** into one conviction score, further blended with an optimised momentum overlay.

---

## 🏗️ Signal Architecture

### Final Score = (1 − mom_w) × HRC + mom_w × Momentum

**HRC Conviction Score = Weighted Sum of 3 Components:**

| Component | Weight | Logic |
|-----------|--------|-------|
| **MTF Alignment + Velocity** | 40% | H63 level + H126 confirmation + H63 velocity (rate of change). Velocity boosts score when regime is accelerating |
| **Hurst Divergence** | 40% | Blend of: (a) H risen vs 6m ago, (b) H above own 2yr baseline, (c) H recently crossed 1yr mean |
| **Cross-Asset Sync** | 20% | Reward ETFs whose H diverges positively from the cross-ETF cluster mean |

### Hurst Windows

- **H63d** (~1 quarter) — core signal window, most reliable for DFA
- **H126d** (~6 months) — medium-term regime confirmation
- **H63 Velocity** — rate of change of H63 over past 63 days (replaces unreliable short window)

> **Why DFA, not R/S?** Classic R/S analysis systematically over‑estimates H on short windows (n < 100), producing spurious values of 0.7–0.9 that wash out cross‑ETF differentiation. DFA (Detrended Fluctuation Analysis) is unbiased down to ~32 sample points.

> **Why velocity, not a short window?** The 42d (and original 21d) DFA windows were unreliable — at those lengths DFA produced extreme values (0.07–0.83) driven by bond microstructure noise rather than genuine regime persistence. H63 velocity correctly captures whether a regime is accelerating or decelerating.

### Momentum Overlay (Walk-Forward Optimised)

- **3m + 6m cross-sectional rank momentum** blended with HRC score
- Weights `(mom_weight, w3m)` grid-searched every fold on trailing in-sample Sharpe
- Grid: `mom_weight ∈ {0.10, 0.20, 0.30}` × `w3m ∈ {0.30, 0.50, 0.70}`
- Genuine OOS — weights re‑optimised every walk‑forward fold, no look‑ahead

---

## 🏗️ Pipeline Architecture
HuggingFace: P2SAMAPA/p2-etf-hurst-data
ohlcv_data.parquet (contains all 18 ETFs)
│
▼

Incremental OHLCV update (yfinance, flat column format)
│
▼

DFA Hurst at H63 + H126 + Velocity per ETF (for each option separately)
│
▼

Divergence Scores (momentum + persistence + transition)
│
▼

Cross-Asset Sync Score
│
▼

Momentum weight optimisation (grid search, last 252d)
│
▼

Conviction Score + Momentum Blend → Signal (per option)
│
▼

MTF History (rolling, step=5) – per option
│
▼

Walk-Forward Backtest (252d train, 21d step, per-fold optimisation) – per option
│
▼
HuggingFace outputs:
ohlcv_data.parquet (shared)
mtf_history_a.parquet | mtf_history_b.parquet
signals_latest_a.parquet | signals_latest_b.parquet
walkforward_returns_a.parquet | walkforward_returns_b.parquet
metadata_a.json | metadata_b.json

text

---

## 📁 Repository Structure
P2-ETF-HURST/
├── .github/workflows/daily.yml # Cron: 02:00 UTC Mon–Fri (runs both options)
├── train.py # Pipeline orchestrator with --option a/b
├── hurst_core.py # DFA engine: velocity, MTF, divergence, sync, momentum, conviction
├── walkforward.py # Walk-forward backtest with per-fold weight optimisation
├── data_manager.py # HF I/O + OHLCV helpers (option‑aware)
├── app.py # Streamlit dashboard (tabs for Option A and B)
├── reseed.py # Full reseed from 2008 for all 18 ETFs
├── config.py # ETF lists and parameters
├── requirements.txt
└── README.md

text

---

## ⚙️ Setup

### GitHub Secrets

| Secret | Value |
|--------|-------|
| `HF_TOKEN` | HuggingFace write token |
| `FRED_API_KEY` | (not used) |
| `GH_PAT` | GitHub personal access token (for triggering workflow from Streamlit) |
| `GITHUB_REPO` | `P2SAMAPA/P2-ETF-HURST` |

### HuggingFace Dataset

`P2SAMAPA/p2-etf-hurst-data` — public dataset repo containing OHLCV for all 18 ETFs.

### Streamlit Secrets
HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxx
GH_PAT = ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO = P2SAMAPA/P2-ETF-HURST

text

---

## 🔄 Daily Pipeline

The workflow `daily.yml` runs at 02:00 UTC Monday–Friday and executes:

1. **Data update** – fetches new OHLCV for all 18 ETFs (if any new trading days).
2. **Option A training** – runs Hurst analysis on the 6 FI/commodities ETFs, saves results to `*_a.parquet`.
3. **Option B training** – runs Hurst analysis on the 12 equity ETFs, saves results to `*_b.parquet`.

Both options run sequentially, so the pipeline takes about twice as long as before (~2× the original 6‑minute run). You can adjust the schedule if needed.

---

## 📊 Output Files (per option)

| File | Description |
|------|-------------|
| `signals_latest_{option}.parquet` | Daily signal log with all conviction + momentum scores |
| `mtf_history_{option}.parquet` | Rolling H63/H126/Velocity history (step=5 days) |
| `walkforward_returns_{option}.parquet` | OOS backtest returns, per‑fold optimised weights |
| `metadata_{option}.json` | Last run info, latest signal, optimised weights |

---

## 🚀 First Run

1. **Seed the dataset** (once):
   ```bash
   python reseed.py
This will fetch OHLCV for all 18 ETFs and upload to HuggingFace.

Run the pipeline for both options:

bash
python train.py --option a   # Option A
python train.py --option b   # Option B
Deploy Streamlit (the app will show two tabs).

📈 Walk-Forward Backtest (per option)
Train window: 252 trading days (~1 year)

Step size: 21 days (monthly rebalance)

Signal: Top ETF by HRC conviction score computed on train window

Fee: 5bps per rotation

No look‑ahead: Each day's signal uses only data available up to that point

📜 Disclaimer
Educational and research purposes only. Not financial advice.

text

---

Let me know if you need any adjustments or additional files (e.g., `hurst_core.py` might need minor changes to handle different ETF lists, but it already works with any list of tickers). Also, the workflow `daily.yml` should be updated to run `train.py` for both options. If you need that, I can provide the updated workflow file as well.
