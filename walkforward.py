"""
walkforward.py — P2-ETF-HAWKES
================================
Proper walk-forward backtest engine.

Design:
  - Train window : 252 trading days (1 year)
  - Step size    : 21 trading days (1 month)
  - For each step: fit Hawkes on train window → generate signal for next 21 days
  - No look-ahead: each day's signal uses only data available up to that point
  - Parallelised: each fold fits 6 ETFs in parallel via ProcessPoolExecutor

Output: DataFrame with columns:
  date, signal_A, signal_B, ret_A, ret_B, ret_SPY, ret_AGG
  plus cumulative return columns: cum_A, cum_B, cum_SPY, cum_AGG

Called by train.py Step 8. Stores walkforward_returns.parquet on HF.

Author: P2SAMAPA
"""

import logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

log = logging.getLogger(__name__)

from hawkes import (
    fit_all_etfs, get_signal, ETF_UNIVERSE,
    detect_events, get_event_times, compute_intensity,
)
from hurst import compute_all_hurst, hurst_conviction
from strategy import HAWKES_WEIGHT, HURST_WEIGHT

TRAIN_WINDOW = 252    # trading days in train window
STEP_SIZE    = 21     # trading days between refits (monthly)
FEE          = 5 / 10_000   # 5bps per rotation


def _signal_from_fit(
    fit_results: dict,
    hurst_df:    Optional[pd.DataFrame],
    option:      str,
) -> str:
    """Pick the top ETF from fit_results for Option A or B."""
    # Excitation ratios
    ratios = {}
    for ticker, res in fit_results.items():
        params    = res["params"]
        intensity = res["intensity"]
        current   = float(intensity[-1]) if len(intensity) > 0 else params.mu
        ratios[ticker] = current / params.mu if params.mu > 1e-9 else 1.0

    if option == "A" or hurst_df is None:
        return max(ratios, key=ratios.get) if ratios else ETF_UNIVERSE[0]

    # Option B: Hawkes + Hurst
    vals    = np.array(list(ratios.values()))
    ex_max  = vals.max() if vals.max() > 0 else 1.0
    span    = vals.max() - vals.min()
    hw = 0.20 if span < 0.05 else HAWKES_WEIGHT
    rw = 0.80 if span < 0.05 else HURST_WEIGHT

    ex_norm = {t: v / ex_max for t, v in ratios.items()}

    scores = {}
    for t in ETF_UNIVERSE:
        if t not in ex_norm:
            continue
        h_ser = hurst_df[t].dropna() if t in hurst_df.columns else pd.Series([0.5])
        h_val = float(h_ser.iloc[-1]) if len(h_ser) > 0 else 0.5
        scores[t] = hw * ex_norm[t] + rw * float(np.clip(h_val, 0, 1))

    return max(scores, key=scores.get) if scores else ETF_UNIVERSE[0]


def run_walkforward(
    returns_df:   pd.DataFrame,
    volume_df:    pd.DataFrame,
    bm_returns:   pd.DataFrame,
    event_def:    str = "combined",
    train_window: int = TRAIN_WINDOW,
    step_size:    int = STEP_SIZE,
    n_workers:    int = 6,
) -> pd.DataFrame:
    """
    Run a proper walk-forward backtest.

    Parameters
    ----------
    returns_df   : daily returns for ETF universe (full history)
    volume_df    : daily volume for ETF universe
    bm_returns   : benchmark returns DataFrame (columns: SPY, AGG or subset)
    event_def    : Hawkes event definition
    train_window : number of days in each training fold
    step_size    : number of days between refits
    n_workers    : parallel workers for ETF fitting within each fold

    Returns
    -------
    DataFrame with one row per trading day in the OOS period:
      date, signal_A, signal_B, ret_A, ret_B, cum_A, cum_B,
      + one cum_* column per benchmark
    """
    etf_cols = [t for t in ETF_UNIVERSE if t in returns_df.columns]
    n        = len(returns_df)
    dates    = returns_df.index

    if n < train_window + step_size:
        raise ValueError(f"Not enough data: need {train_window + step_size} rows, have {n}")

    log.info(f"Walk-forward backtest: {n} days, train={train_window}, step={step_size}")
    log.info(f"Folds: {(n - train_window) // step_size} × {step_size} days OOS")

    records   = []
    prev_A    = None
    prev_B    = None

    # Iterate over folds
    fold_starts = range(train_window, n - 1, step_size)
    total_folds = len(range(train_window, n - 1, step_size))

    for fold_i, fold_start in enumerate(fold_starts):
        train_end  = fold_start                          # exclusive
        oos_start  = fold_start                          # inclusive
        oos_end    = min(fold_start + step_size, n - 1)  # exclusive

        train_ret = returns_df.iloc[train_end - train_window: train_end][etf_cols]
        train_vol = volume_df.iloc[train_end - train_window: train_end][etf_cols] \
                    if volume_df is not None else None

        if fold_i % 10 == 0:
            log.info(f"  Fold {fold_i+1}/{total_folds}: "
                     f"train [{dates[train_end - train_window].date()} → "
                     f"{dates[train_end - 1].date()}] "
                     f"OOS [{dates[oos_start].date()} → "
                     f"{dates[min(oos_end, n-1) - 1].date()}]")

        # ── Fit Hawkes on train window ─────────────────────────────────────
        try:
            fit_results = fit_all_etfs(train_ret, train_vol,
                                       event_def=event_def)
        except Exception as e:
            log.warning(f"Fold {fold_i}: fit failed ({e}), skipping")
            continue

        # ── Rolling Hurst on train window ──────────────────────────────────
        try:
            hurst_window = min(train_window, len(train_ret))
            hurst_df_fold = compute_all_hurst(train_ret,
                                              window=min(252, hurst_window))
        except Exception:
            hurst_df_fold = None

        # ── Generate signal for this fold ──────────────────────────────────
        signal_A = _signal_from_fit(fit_results, None,           "A")
        signal_B = _signal_from_fit(fit_results, hurst_df_fold,  "B")

        # ── Apply signal over OOS period ───────────────────────────────────
        for day_i in range(oos_start, oos_end):
            if day_i + 1 >= n:
                break
            date     = dates[day_i]
            next_day = dates[day_i + 1]

            # Strategy returns
            def get_ret(signal, prev):
                if signal in returns_df.columns:
                    raw = float(returns_df.loc[next_day, signal])
                    fee = FEE if (prev is not None and prev != signal) else 0.0
                    return raw - fee
                return 0.045 / 252   # RF fallback

            ret_A = get_ret(signal_A, prev_A)
            ret_B = get_ret(signal_B, prev_B)
            prev_A = signal_A
            prev_B = signal_B

            # Benchmark returns
            bm_rets = {}
            for bm in bm_returns.columns:
                bm_rets[f"ret_{bm}"] = float(bm_returns.loc[next_day, bm]) \
                    if next_day in bm_returns.index else 0.0

            records.append({
                "date":     date,
                "signal_A": signal_A,
                "signal_B": signal_B,
                "ret_A":    ret_A,
                "ret_B":    ret_B,
                **bm_rets,
            })

    if not records:
        raise RuntimeError("Walk-forward produced no records.")

    df = pd.DataFrame(records).set_index("date")
    df = df.sort_index()

    # ── Cumulative returns ─────────────────────────────────────────────────
    df["cum_A"] = np.cumprod(1 + df["ret_A"].values)
    df["cum_B"] = np.cumprod(1 + df["ret_B"].values)
    for bm in bm_returns.columns:
        col = f"ret_{bm}"
        if col in df.columns:
            df[f"cum_{bm}"] = np.cumprod(1 + df[col].values)

    log.info(f"Walk-forward complete: {len(df)} OOS days, "
             f"cum_A={df['cum_A'].iloc[-1]:.3f}, "
             f"cum_B={df['cum_B'].iloc[-1]:.3f}")
    return df


def compute_wf_metrics(wf_df: pd.DataFrame, option: str = "A",
                       rf_rate: float = 0.045) -> dict:
    """Compute annualised performance metrics from walk-forward results."""
    ret_col = f"ret_{option}"
    cum_col = f"cum_{option}"
    if ret_col not in wf_df.columns:
        return {}

    rets    = wf_df[ret_col].values
    cum     = wf_df[cum_col].values
    n       = len(rets)
    rf_day  = rf_rate / 252
    ann_ret = float(cum[-1] ** (252 / n) - 1) if n > 0 else 0.0
    excess  = rets - rf_day
    sharpe  = float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))
    cum_max = np.maximum.accumulate(cum)
    max_dd  = float(np.min((cum - cum_max) / (cum_max + 1e-9)))
    wins    = rets[rets > rf_day]
    hit     = len(wins) / n if n > 0 else 0.0
    calmar  = ann_ret / (abs(max_dd) + 1e-9)

    return {
        "ann_return": ann_ret,
        "sharpe":     sharpe,
        "max_dd":     max_dd,
        "hit_ratio":  hit,
        "calmar":     calmar,
        "n_days":     n,
        "cum_final":  float(cum[-1]),
    }
