"""
hurst.py — P2-ETF-HAWKES
==========================
Hurst exponent computation for Option B (Hawkes + Hurst combined signal).

Methods:
  - R/S analysis (Rescaled Range) — classical Hurst
  - Rolling Hurst over a configurable window

Interpretation:
  H > 0.5 → persistent (trending) series
  H = 0.5 → random walk
  H < 0.5 → mean-reverting series

Author: P2SAMAPA
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

ETF_UNIVERSE = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]


def hurst_rs(series: np.ndarray, min_window: int = 20) -> float:
    """
    Compute Hurst exponent using R/S (Rescaled Range) analysis.

    Algorithm:
      1. Divide series into sub-series of varying lengths n
      2. For each n: compute R/S = (max - min of cumulative deviation) / std
      3. Regress log(R/S) on log(n) — slope = H

    Parameters
    ----------
    series     : 1D array of returns
    min_window : minimum sub-series length

    Returns
    -------
    float : Hurst exponent H ∈ (0, 1)
    """
    series = np.array(series, dtype=float)
    series = series[~np.isnan(series)]
    N = len(series)

    if N < min_window * 2:
        return 0.5   # not enough data

    # Generate candidate sub-series lengths (log-spaced)
    max_k  = int(np.floor(np.log2(N / min_window)))
    if max_k < 2:
        return 0.5

    ns     = [min_window * (2 ** k) for k in range(max_k + 1) if min_window * (2 ** k) <= N]
    rs_vals = []
    n_vals  = []

    for n in ns:
        n_chunks = N // n
        if n_chunks < 1:
            continue

        rs_list = []
        for i in range(n_chunks):
            chunk = series[i * n: (i + 1) * n]
            mean  = np.mean(chunk)
            dev   = np.cumsum(chunk - mean)
            R     = np.max(dev) - np.min(dev)
            S     = np.std(chunk, ddof=1)
            if S > 1e-12:
                rs_list.append(R / S)

        if rs_list:
            rs_vals.append(np.mean(rs_list))
            n_vals.append(n)

    if len(n_vals) < 2:
        return 0.5

    log_n  = np.log(n_vals)
    log_rs = np.log(rs_vals)

    # Linear regression: slope = H
    slope, _ = np.polyfit(log_n, log_rs, 1)
    return float(np.clip(slope, 0.01, 0.99))


def rolling_hurst(
    returns: pd.Series,
    window:  int = 252,
    step:    int = 1,
) -> pd.Series:
    """
    Compute rolling Hurst exponent over a sliding window.

    Parameters
    ----------
    returns : daily return series
    window  : rolling window in days (default 252 = 1 year)
    step    : compute every `step` days (1 = every day, slower)

    Returns
    -------
    pd.Series of Hurst values, same index as returns (NaN for early dates)
    """
    values = np.full(len(returns), np.nan)
    arr    = returns.values

    for i in range(window, len(arr) + 1, step):
        chunk     = arr[i - window: i]
        h         = hurst_rs(chunk)
        # Fill forward for step > 1
        end_idx   = i - 1
        start_idx = max(0, end_idx - step + 1)
        values[start_idx: end_idx + 1] = h

    return pd.Series(values, index=returns.index, name="Hurst")


def compute_all_hurst(
    returns_df: pd.DataFrame,
    window:     int = 252,
) -> pd.DataFrame:
    """
    Compute rolling Hurst for all ETFs in ETF_UNIVERSE.

    Returns DataFrame with one column per ETF.
    """
    results = {}
    for ticker in ETF_UNIVERSE:
        if ticker not in returns_df.columns:
            continue
        try:
            h = rolling_hurst(returns_df[ticker].dropna(), window=window)
            results[ticker] = h.reindex(returns_df.index)
            log.info(f"Hurst {ticker}: current={h.dropna().iloc[-1]:.3f} "
                     f"mean={h.dropna().mean():.3f}")
        except Exception as e:
            log.error(f"Hurst failed for {ticker}: {e}")

    return pd.DataFrame(results)


def hurst_conviction(h: float) -> float:
    """
    Convert Hurst H to a [0, 1] conviction score for Option B blending.

    H > 0.5 → persistent → score > 0.5 (supports trend-following signal)
    H = 0.5 → random walk → score = 0.5 (neutral)
    H < 0.5 → mean-reverting → score < 0.5 (penalises trend signal)

    Mapping: score = H (direct, already in [0, 1])
    """
    return float(np.clip(h, 0.0, 1.0))


def hurst_label(h: float) -> str:
    if h >= 0.65:
        return "Strong Trend"
    elif h >= 0.55:
        return "Mild Trend"
    elif h >= 0.45:
        return "Random Walk"
    elif h >= 0.35:
        return "Mild Mean-Rev"
    else:
        return "Strong Mean-Rev"


def hurst_colour(h: float) -> str:
    """Return a hex colour for display based on Hurst value."""
    if h >= 0.65:
        return "#16a34a"   # green — strong trend
    elif h >= 0.55:
        return "#84cc16"   # lime — mild trend
    elif h >= 0.45:
        return "#d97706"   # amber — random walk
    elif h >= 0.35:
        return "#f97316"   # orange — mild mean-rev
    else:
        return "#dc2626"   # red — strong mean-rev
