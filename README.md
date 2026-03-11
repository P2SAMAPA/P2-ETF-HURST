# 📐 P2-ETF-HURST

**Hurst Confluence (HRC) ETF Rotation Signal**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-HURST/actions/workflows/daily.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HURST/actions)
[![Hugging Face](https://img.shields.io/badge/🤗%20HF-p2--etf--hurst--data-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hurst-data)
[![Streamlit](https://img.shields.io/badge/Streamlit-p2--etf--hurst-FF4B4B)](https://p2-etf-hurst.streamlit.app)

**ETF Universe:** TLT · LQD · HYG · VNQ · GLD · SLV  
**Benchmarks:** SPY · AGG  
**Cron:** 02:00 UTC Mon–Fri

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

> **Why DFA, not R/S?** Classic R/S analysis systematically over-estimates H on short windows (n < 100), producing spurious values of 0.7–0.9 that wash out cross-ETF differentiation. DFA (Detrended Fluctuation Analysis) is unbiased down to ~32 sample points.

> **Why velocity, not a short window?** The 42d (and original 21d) DFA windows were unreliable — at those lengths DFA produced extreme values (0.07–0.83) driven by bond microstructure noise rather than genuine regime persistence. H63 velocity correctly captures whether a regime is accelerating or decelerating.

### MTF + Velocity Scoring

| Condition | Score |
|-----------|-------|
| H63 trending + H126 trending + velocity ≥ 0 | 1.00 |
| H63 trending + H126 trending + velocity < 0 | 0.85 |
| H63 trending only + velocity ≥ 0 | 0.75 |
| H63 trending only + velocity < 0 | 0.55 |
| H126 trending only + velocity ≥ 0 | 0.50 |
| H126 trending only + velocity < 0 | 0.30 |
| Neither trending + velocity ≥ 0 | 0.15 |
| Neither trending + velocity < 0 | 0.00 |

### Hurst Divergence (3-Part Blend)
- **(a) Momentum**: H has risen most vs its own 6-month ago value
- **(b) Absolute persistence**: H is furthest above 0.5 relative to own 2yr baseline
- **(c) Fresh transition**: H recently crossed above its 1-year mean (tradeable regime change)

### Momentum Overlay (Walk-Forward Optimised)
- **3m + 6m cross-sectional rank momentum** blended with HRC score
- Weights `(mom_weight, w3m)` grid-searched every fold on trailing in-sample Sharpe
- Grid: `mom_weight ∈ {0.10, 0.20, 0.30}` × `w3m ∈ {0.30, 0.50, 0.70}`
- Genuine OOS — weights re-optimised every walk-forward fold, no look-ahead

### Cross-Asset Synchronisation
- High sync across ETFs = risk-off cluster → lower differentiation
- Low sync = dispersion → reward ETFs whose H is above the cluster mean

---

## 🏗️ Pipeline Architecture

```
HuggingFace: P2SAMAPA/p2-etf-hurst-data
  ohlcv_data.parquet
          │
          ▼
  1. Incremental OHLCV update (yfinance, flat column format)
          │
          ▼
  2. DFA Hurst at H63 + H126 + Velocity per ETF
          │
          ▼
  3. Divergence Scores (momentum + persistence + transition)
          │
          ▼
  4. Cross-Asset Sync Score
          │
          ▼
  5. Momentum weight optimisation (grid search, last 252d)
          │
          ▼
  6. Conviction Score + Momentum Blend → Signal
          │
          ▼
  7. MTF History (rolling, step=5)
          │
          ▼
  8. Walk-Forward Backtest (252d train, 21d step, per-fold optimisation)
          │
          ▼
HuggingFace outputs:
  ohlcv_data.parquet | mtf_history.parquet
  signals_latest.parquet | walkforward_returns.parquet | metadata.json
```

---

## 📁 Repository Structure

```
P2-ETF-HURST/
├── .github/workflows/daily.yml   # Cron: 02:00 UTC Mon–Fri
├── train.py                       # Pipeline orchestrator
├── hurst_core.py                  # DFA engine: velocity, MTF, divergence, sync, momentum, conviction
├── walkforward.py                 # Walk-forward backtest with per-fold weight optimisation
├── data_manager.py                # HF I/O + OHLCV helpers (multi-format column handling)
├── app.py                         # Streamlit dashboard
├── reseed.py                      # Emergency full reseed from 2008
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### GitHub Secrets
```
HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxx
```

### HuggingFace Dataset
`P2SAMAPA/p2-etf-hurst-data` — public dataset repo.

### Streamlit Secrets
```
HF_TOKEN    = hf_xxxxxxxxxxxxxxxxxxxx
GH_PAT      = ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO = P2SAMAPA/P2-ETF-HURST
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `ohlcv_data.parquet` | Full OHLCV history (2008–present), flat column format |
| `mtf_history.parquet` | Rolling H63/H126/Velocity history (step=5 days) |
| `signals_latest.parquet` | Daily signal log with all conviction + momentum scores |
| `walkforward_returns.parquet` | OOS backtest returns, per-fold optimised weights |
| `metadata.json` | Last run info, latest signal, optimised weights |

---

## 🔄 Changelog

### 2026-03-11 — DFA + Velocity + Momentum Overlay
- **Replaced R/S with DFA** — unbiased Hurst estimator, eliminates short-window inflation
- **Replaced short window with H63 Velocity** — rate of change of H63 over 63 days; captures regime acceleration/deceleration rather than noisy short-window DFA
- **Windows changed**: 21d/63d/252d → H63 Velocity · H63d · H126d
- **Added Weak Trend zone** (H 0.50–0.55) — distinct yellow-green label/colour between Random Walk and Mild Trend, eliminates misleading hard-cutoff cliff at 0.55
- **Added dual momentum overlay** — 3m + 6m cross-sectional rank momentum blended with HRC conviction score
- **Per-fold momentum weight optimisation** — walk-forward grid search over `mom_weight × w3m` every fold, no look-ahead bias
- **Fixed OHLCV column parsing** — `data_manager.py` now handles 6 column formats: real MultiIndex tuples, stringified tuples (parquet round-trip artifact), flat `TICKER_close`, `TICKER_close_ticker` (yfinance multi-ticker), `close_TICKER`, and fallback
- **Heatmap redesigned** — velocity column uses diverging colour scale centred on 0 (red = decelerating, green = accelerating); H columns use 5-zone colour scale; rendered as subplots with independent scales
- **Walk-forward performance** (42d/63d/126d + momentum): +9.9% ann, Sharpe 0.37, vs +1.1% / Sharpe −0.66 for AGG

### Initial Build — Hurst Confluence (HRC)
- Multi-timeframe R/S Hurst at 21d/63d/252d
- Divergence scores (3-part: momentum, persistence, transition)
- Cross-asset synchronisation score
- Walk-forward backtest: 252d train / 21d step / 5bps fee
- Streamlit dashboard with heatmap, regime timeline, signal history
- Initial performance: +4.1% ann, Sharpe 0.06

---

## ⚠️ Disclaimer
Educational and research purposes only. Not financial advice.
