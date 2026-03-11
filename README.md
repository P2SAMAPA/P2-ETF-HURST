# 📐 P2-ETF-HURST

**Hurst Confluence (HRC) ETF Rotation Signal**

[![GitHub Actions](https://github.com/P2SAMAPA/P2-ETF-HURST/actions/workflows/daily.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-HURST/actions)
[![Hugging Face](https://img.shields.io/badge/🤗%20HF-p2--etf--hurst--data-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-hurst-data)

---

## 🔬 Concept: Hurst Confluence (HRC)

The Hurst exponent H measures long-range memory in a time series:
- **H > 0.55** → Trending / persistent — past direction likely to continue
- **H ≈ 0.50** → Random walk — no memory
- **H < 0.45** → Anti-persistent / mean-reverting — past direction likely to reverse

Rather than using H as a simple filter (as in the companion ARMA-RNN-LSTM project), HRC uses **three distinct Hurst-based signals** combined into one conviction score.

---

## 🏗️ Signal Architecture

### Conviction Score = Weighted Sum of 3 Components

| Component | Weight | Logic |
|-----------|--------|-------|
| **Multi-Timeframe (MTF) Alignment** | 40% | H at 21d, 63d, 252d windows. Strong when 2-of-3 timeframes trending (H > 0.55) |
| **Hurst Divergence** | 40% | Blend of: (a) H risen vs 6m ago, (b) H above own 2yr baseline, (c) H recently crossed 1yr mean |
| **Cross-Asset Sync** | 20% | Reward ETFs whose H diverges positively from the cross-ETF cluster |

### Multi-Timeframe Rules
- **2-of-3 sufficient**: short (21d) + medium (63d) align → signal triggered, long (252d) confirms
- Scores: 1.0 (all 3) → 0.75 (short+medium) → 0.5 (any 2) → 0.25 (1 only) → 0.0

### Hurst Divergence (3-Part Blend)
- **(a) Momentum**: H has risen most vs its own 6-month ago value
- **(b) Absolute persistence**: H is furthest above 0.5 relative to own 2yr baseline
- **(c) Fresh transition**: H recently crossed above its 1-year mean (tradeable regime change)

### Cross-Asset Synchronisation
- High sync across ETFs = risk-off cluster → lower differentiation opportunity
- Low sync = dispersion → reward ETFs whose H is above the cluster mean

---

## 🏗️ Pipeline Architecture

```
HuggingFace: P2SAMAPA/p2-etf-hurst-data
  ohlcv_data.parquet
          │
          ▼
  1. Incremental OHLCV update
          │
          ▼
  2. Multi-Timeframe Hurst (21d/63d/252d) per ETF
          │
          ▼
  3. Divergence Scores (momentum + persistence + transition)
          │
          ▼
  4. Cross-Asset Sync Score
          │
          ▼
  5. Conviction Score → Signal
          │
          ▼
  6. MTF History (rolling, step=5)
          │
          ▼
  7. Walk-Forward Backtest (252d train, 21d step)
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
├── hurst_core.py                  # Hurst engine: MTF, divergence, sync, conviction
├── walkforward.py                 # Walk-forward backtest
├── data_manager.py                # HF I/O + OHLCV helpers
├── app.py                         # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### GitHub Secret
```
HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxx
```

### HuggingFace Dataset
Create `P2SAMAPA/p2-etf-hurst-data` as a **public** dataset repo.

### Streamlit Secrets
```
HF_TOKEN = hf_xxxxxxxxxxxxxxxxxxxx
GH_PAT   = ghp_xxxxxxxxxxxxxxxxxxxx
GITHUB_REPO = P2SAMAPA/P2-ETF-HURST
```

---

## 📊 Output Files

| File | Description |
|------|-------------|
| `ohlcv_data.parquet` | Full OHLCV history (2008–present) |
| `mtf_history.parquet` | Rolling MTF Hurst history (step=5 days) |
| `signals_latest.parquet` | Daily signal log with all conviction scores |
| `walkforward_returns.parquet` | OOS backtest returns + cumulative |
| `metadata.json` | Last run info, latest signal |

---

## ⚠️ Disclaimer
Educational and research purposes only. Not financial advice.
