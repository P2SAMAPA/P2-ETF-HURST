"""
hurst_core.py — P2-ETF-HURST
==============================
Core Hurst exponent computation and multi-timeframe analysis.

Three timeframe windows:
  SHORT  = 21  trading days (~1 month)
  MEDIUM = 63  trading days (~1 quarter)
  LONG   = 252 trading days (~1 year)

R/S Analysis (Hurst, 1951):
  H > 0.55 → Trending / persistent
  H ≈ 0.50 → Random walk
  H < 0.45 → Anti-persistent / mean-reverting
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

log = logging.getLogger(__name__)

# ── Timeframe windows ─────────────────────────────────────────────────────────
SHORT_WINDOW  = 21
MEDIUM_WINDOW = 63
LONG_WINDOW   = 252

# ── Regime thresholds ─────────────────────────────────────────────────────────
H_TRENDING    = 0.55
H_RANDOM      = 0.45
H_STRONG_TREND = 0.65

ETF_UNIVERSE  = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARKS    = ["SPY", "AGG"]


def hurst_rs(series: np.ndarray) -> float:
    """
    Compute Hurst exponent via R/S (rescaled range) analysis.
    Returns H in [0, 1]. Returns 0.5 if series is too short or degenerate.
    """
    n = len(series)
    if n < 20:
        return 0.5

    series = np.array(series, dtype=float)
    series = series[~np.isnan(series)]
    if len(series) < 20:
        return 0.5

    lags  = []
    rs_vals = []

    min_window = max(4, n // 16)
    max_window = n // 2

    for lag in range(min_window, max_window + 1, max(1, (max_window - min_window) // 20)):
        chunks  = [series[i:i+lag] for i in range(0, n - lag + 1, lag)]
        chunks  = [c for c in chunks if len(c) == lag]
        if not chunks:
            continue
        rs_chunk = []
        for chunk in chunks:
            mean_c = np.mean(chunk)
            dev    = np.cumsum(chunk - mean_c)
            r      = np.max(dev) - np.min(dev)
            s      = np.std(chunk, ddof=1)
            if s > 1e-10:
                rs_chunk.append(r / s)
        if rs_chunk:
            lags.append(np.log(lag))
            rs_vals.append(np.log(np.mean(rs_chunk)))

    if len(lags) < 4:
        return 0.5

    try:
        h, _ = np.polyfit(lags, rs_vals, 1)
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


def hurst_label(h: float) -> str:
    """Human-readable regime label."""
    if h >= H_STRONG_TREND:
        return "Strong Trend"
    elif h >= H_TRENDING:
        return "Mild Trend"
    elif h >= H_RANDOM:
        return "Random Walk"
    else:
        return "Mean-Reverting"


def hurst_regime_colour(h: float) -> str:
    """Colour code for regime."""
    if h >= H_TRENDING:
        return "#16a34a"   # green — trending
    elif h >= H_RANDOM:
        return "#d97706"   # amber — random walk
    else:
        return "#dc2626"   # red — mean-reverting


# ── Multi-timeframe Hurst ─────────────────────────────────────────────────────

def compute_mtf_hurst(
    returns: pd.Series,
    date:    Optional[pd.Timestamp] = None,
) -> dict:
    """
    Compute Hurst at SHORT/MEDIUM/LONG windows for a single ETF.
    Returns dict with h_short, h_medium, h_long and derived scores.
    """
    s = returns.dropna()
    n = len(s)

    h_short  = hurst_rs(s.values[-SHORT_WINDOW:])  if n >= SHORT_WINDOW  else 0.5
    h_medium = hurst_rs(s.values[-MEDIUM_WINDOW:]) if n >= MEDIUM_WINDOW else 0.5
    h_long   = hurst_rs(s.values[-LONG_WINDOW:])   if n >= LONG_WINDOW   else 0.5

    # MTF alignment score: how many timeframes are trending (H > 0.55)
    trending = sum([h_short >= H_TRENDING,
                    h_medium >= H_TRENDING,
                    h_long   >= H_TRENDING])

    # 2-of-3 rule: short + medium align, long confirms
    short_med_align = (h_short  >= H_TRENDING and h_medium >= H_TRENDING)
    long_confirms   = (h_long   >= H_TRENDING)
    mtf_strong      = short_med_align or (trending >= 2)

    mtf_score = (
        1.0 if (short_med_align and long_confirms) else
        0.75 if short_med_align else
        0.5  if (trending >= 2)  else
        0.25 if (trending == 1)  else
        0.0
    )

    return {
        "h_short":        h_short,
        "h_medium":       h_medium,
        "h_long":         h_long,
        "mtf_score":      mtf_score,
        "mtf_strong":     mtf_strong,
        "trending_count": trending,
        "label_short":    hurst_label(h_short),
        "label_medium":   hurst_label(h_medium),
        "label_long":     hurst_label(h_long),
    }


def compute_all_mtf(
    returns_df: pd.DataFrame,
) -> dict:
    """Compute MTF Hurst for all ETFs. Returns dict keyed by ticker."""
    results = {}
    for ticker in ETF_UNIVERSE:
        if ticker not in returns_df.columns:
            continue
        results[ticker] = compute_mtf_hurst(returns_df[ticker])
    return results


# ── Rolling multi-timeframe history ──────────────────────────────────────────

def build_mtf_history(
    returns_df: pd.DataFrame,
    step:       int = 5,
) -> pd.DataFrame:
    """
    Build rolling MTF Hurst history for all ETFs.
    Computed every `step` days to reduce runtime.
    Returns DataFrame with MultiIndex columns (ticker, window).
    """
    dates  = returns_df.index
    n      = len(dates)
    min_rows = LONG_WINDOW + 1

    records = []
    for i in range(min_rows, n, step):
        row = {"date": dates[i]}
        slice_ = returns_df.iloc[:i+1]
        for ticker in ETF_UNIVERSE:
            if ticker not in slice_.columns:
                continue
            s = slice_[ticker].dropna()
            row[f"{ticker}_h_short"]  = hurst_rs(s.values[-SHORT_WINDOW:])  if len(s) >= SHORT_WINDOW  else np.nan
            row[f"{ticker}_h_medium"] = hurst_rs(s.values[-MEDIUM_WINDOW:]) if len(s) >= MEDIUM_WINDOW else np.nan
            row[f"{ticker}_h_long"]   = hurst_rs(s.values[-LONG_WINDOW:])   if len(s) >= LONG_WINDOW   else np.nan
        records.append(row)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index("date")
    return df


# ── Hurst Divergence scores ───────────────────────────────────────────────────

def compute_divergence_scores(
    mtf_today:   dict,
    mtf_history: pd.DataFrame,
    lookback:    int = 504,   # 2 years for baseline
) -> dict:
    """
    Compute 3-part divergence score per ETF:
      (a) H risen most vs own history (momentum of regime)
      (b) H furthest above 0.5 relative to own baseline (absolute persistence)
      (c) H recently crossed above its 1-year mean (fresh transition)

    Returns dict: ticker → {div_a, div_b, div_c, div_score}
    """
    scores = {}

    if mtf_history is None or mtf_history.empty:
        for ticker in ETF_UNIVERSE:
            if ticker not in mtf_today:
                continue
            h = mtf_today[ticker]["h_medium"]
            scores[ticker] = {
                "div_a": 0.0, "div_b": max(0.0, h - 0.5),
                "div_c": 0.0, "div_score": max(0.0, h - 0.5),
            }
        return scores

    hist = mtf_history.tail(lookback)

    for ticker in ETF_UNIVERSE:
        if ticker not in mtf_today:
            continue

        col_m  = f"{ticker}_h_medium"
        col_l  = f"{ticker}_h_long"
        h_now  = mtf_today[ticker]["h_medium"]
        h_long = mtf_today[ticker]["h_long"]

        if col_m not in hist.columns:
            scores[ticker] = {"div_a": 0.0, "div_b": 0.0, "div_c": 0.0, "div_score": 0.0}
            continue

        hist_m = hist[col_m].dropna()
        hist_l = hist[col_l].dropna() if col_l in hist.columns else hist_m

        # (a) Momentum: how much has H risen vs its own 6-month ago value
        h_6m_ago  = float(hist_m.iloc[-126]) if len(hist_m) >= 126 else float(hist_m.iloc[0])
        div_a     = float(np.clip(h_now - h_6m_ago, -1, 1))

        # (b) Absolute persistence: H above 0.5 relative to own 2yr baseline
        h_baseline = float(hist_m.mean()) if len(hist_m) > 0 else 0.5
        div_b      = float(np.clip(h_now - h_baseline, -1, 1))

        # (c) Fresh transition: H crossed above 1yr mean in last 21 days
        h_1yr_mean = float(hist_m.tail(252).mean()) if len(hist_m) >= 63 else 0.5
        h_21d_ago  = float(hist_m.iloc[-21]) if len(hist_m) >= 21 else float(hist_m.iloc[0])
        crossed    = (h_21d_ago < h_1yr_mean) and (h_now >= h_1yr_mean)
        div_c      = 1.0 if crossed else float(np.clip((h_now - h_1yr_mean) / 0.1, -1, 1))

        # Combined divergence score (equal weight of 3 components)
        div_score = (div_a + div_b + div_c) / 3.0

        scores[ticker] = {
            "div_a":       round(div_a,     4),
            "div_b":       round(div_b,     4),
            "div_c":       round(div_c,     4),
            "div_score":   round(div_score, 4),
            "h_baseline":  round(h_baseline, 4),
            "h_1yr_mean":  round(h_1yr_mean, 4),
            "crossed":     crossed,
        }

    return scores


# ── Cross-asset synchronisation ───────────────────────────────────────────────

def compute_sync_score(mtf_today: dict) -> dict:
    """
    Cross-asset Hurst synchronisation.

    When all ETFs have similar H values → regime is synchronised (risk-off cluster).
    When ETFs diverge → dispersion opportunity → reward outliers.

    Returns:
      sync_level: 0 (fully dispersed) to 1 (fully synchronised)
      per-ETF deviation from mean H (normalised)
    """
    h_vals = {}
    for ticker in ETF_UNIVERSE:
        if ticker in mtf_today:
            h_vals[ticker] = mtf_today[ticker]["h_medium"]

    if not h_vals:
        return {"sync_level": 0.5, "scores": {t: 0.0 for t in ETF_UNIVERSE}}

    arr       = np.array(list(h_vals.values()))
    h_mean    = float(np.mean(arr))
    h_std     = float(np.std(arr))

    # Sync level: low std = synchronised; normalise to [0,1]
    sync_level = float(np.exp(-h_std / 0.05))   # decays quickly with dispersion

    # Per-ETF deviation score: positive = ETF H is above cluster mean
    max_dev = float(np.max(np.abs(arr - h_mean))) + 1e-9
    dev_scores = {t: float((v - h_mean) / max_dev) for t, v in h_vals.items()}

    return {
        "sync_level":  round(sync_level, 4),
        "h_mean":      round(h_mean, 4),
        "h_std":       round(h_std, 4),
        "scores":      {t: round(v, 4) for t, v in dev_scores.items()},
    }


# ── Master conviction score ───────────────────────────────────────────────────

# Component weights
W_MTF  = 0.40   # multi-timeframe alignment
W_DIV  = 0.40   # divergence (momentum + persistence + transition)
W_SYNC = 0.20   # cross-asset sync (reward outliers)


def compute_conviction_scores(
    mtf_today:   dict,
    div_scores:  dict,
    sync:        dict,
) -> dict:
    """
    Compute final conviction score per ETF.
    Returns dict: ticker → {mtf, div, sync_component, total, label}
    """
    results = {}

    sync_dev = sync.get("scores", {})

    for ticker in ETF_UNIVERSE:
        if ticker not in mtf_today:
            continue

        mtf_sc  = mtf_today[ticker]["mtf_score"]           # 0–1
        div_sc  = div_scores.get(ticker, {}).get("div_score", 0.0)

        # Normalise div to [0,1]: raw range is [-1, 1]
        div_norm = float(np.clip((div_sc + 1.0) / 2.0, 0.0, 1.0))

        # Sync: reward ETFs whose H deviates positively from cluster
        sync_sc  = sync_dev.get(ticker, 0.0)               # -1 to +1
        sync_norm = float(np.clip((sync_sc + 1.0) / 2.0, 0.0, 1.0))

        total = W_MTF * mtf_sc + W_DIV * div_norm + W_SYNC * sync_norm

        results[ticker] = {
            "mtf_score":   round(mtf_sc,   4),
            "div_score":   round(div_norm, 4),
            "sync_score":  round(sync_norm, 4),
            "total":       round(total,    4),
            "h_short":     mtf_today[ticker]["h_short"],
            "h_medium":    mtf_today[ticker]["h_medium"],
            "h_long":      mtf_today[ticker]["h_long"],
            "label":       hurst_label(mtf_today[ticker]["h_medium"]),
            "trending_count": mtf_today[ticker]["trending_count"],
        }

    return results


def conviction_label(score: float) -> str:
    if score >= 0.75: return "Very High"
    if score >= 0.60: return "High"
    if score >= 0.45: return "Moderate"
    return "Low"


def generate_signal(conviction_scores: dict) -> dict:
    """Pick top ETF by total conviction score."""
    if not conviction_scores:
        return {"signal": "CASH", "conviction": 0.0, "label": "Low", "ranked": []}

    ranked = sorted(conviction_scores.items(), key=lambda x: -x[1]["total"])
    top    = ranked[0]

    # Normalise top score to [0,1] relative to spread
    scores = np.array([v["total"] for _, v in ranked])
    span   = scores.max() - scores.min() + 1e-9
    norm   = (top[1]["total"] - scores.min()) / span

    return {
        "signal":     top[0],
        "conviction": round(float(norm), 4),
        "score":      top[1]["total"],
        "label":      conviction_label(top[1]["total"]),
        "ranked":     [(t, round(v["total"], 4)) for t, v in ranked],
        "scores":     conviction_scores,
    }
