"""
walkforward.py — P2-ETF-HURST
================================
Walk-forward backtest engine using Hurst Confluence signals.
No model fitting — pure Hurst computation, very fast (~3-5 min total).

Train window : 252 days (needed for divergence baseline)
Step size    : 21 days (monthly rebalance)
Fee          : 5bps per rotation
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

log = logging.getLogger(__name__)

from hurst_core import (
    compute_all_mtf, compute_divergence_scores,
    compute_sync_score, compute_conviction_scores,
    generate_signal, build_mtf_history,
    ETF_UNIVERSE, BENCHMARKS,
)

TRAIN_WINDOW = 252
STEP_SIZE    = 21
FEE          = 5 / 10_000


def run_walkforward(
    returns_df: pd.DataFrame,
    bm_returns: pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    step_size:    int = STEP_SIZE,
) -> pd.DataFrame:
    """
    Walk-forward backtest using Hurst Confluence scores.
    Each fold: compute MTF Hurst on train window → signal for next 21 days.
    """
    etf_cols = [t for t in ETF_UNIVERSE if t in returns_df.columns]
    n        = len(returns_df)
    dates    = returns_df.index

    if n < train_window + step_size:
        raise ValueError(f"Need {train_window + step_size} rows, have {n}")

    log.info(f"Walk-forward: {n} days, train={train_window}, step={step_size}")
    total_folds = (n - train_window) // step_size
    log.info(f"Folds: {total_folds}")

    records  = []
    prev_sig = None

    for fold_i, fold_start in enumerate(range(train_window, n - 1, step_size)):
        oos_end = min(fold_start + step_size, n - 1)

        train_ret = returns_df.iloc[fold_start - train_window: fold_start][etf_cols]

        if fold_i % 20 == 0:
            log.info(f"  Fold {fold_i+1}/{total_folds}: "
                     f"train [{dates[fold_start - train_window].date()} → "
                     f"{dates[fold_start - 1].date()}] "
                     f"OOS [{dates[fold_start].date()} → "
                     f"{dates[min(oos_end, n-1) - 1].date()}]")

        # ── Compute Hurst on train window ──────────────────────────────────
        try:
            mtf_today  = compute_all_mtf(train_ret)
            # Build a mini MTF history for divergence (last 252 days of train)
            mtf_hist   = build_mtf_history(train_ret, step=5)
            div_scores = compute_divergence_scores(mtf_today, mtf_hist)
            sync       = compute_sync_score(mtf_today)
            conviction = compute_conviction_scores(mtf_today, div_scores, sync)
            sig        = generate_signal(conviction)
            signal     = sig["signal"]
        except Exception as e:
            log.warning(f"Fold {fold_i}: Hurst failed ({e}), holding previous")
            signal = prev_sig if prev_sig else etf_cols[0]

        # ── Apply signal over OOS period ───────────────────────────────────
        for day_i in range(fold_start, oos_end):
            if day_i + 1 >= n:
                break
            date     = dates[day_i]
            next_day = dates[day_i + 1]

            if signal in returns_df.columns and next_day in returns_df.index:
                raw = float(returns_df.loc[next_day, signal])
                fee = FEE if (prev_sig is not None and prev_sig != signal) else 0.0
                ret = raw - fee
            else:
                ret = 0.045 / 252

            bm_rets = {}
            for bm in bm_returns.columns:
                bm_rets[f"ret_{bm}"] = float(bm_returns.loc[next_day, bm]) \
                    if next_day in bm_returns.index else 0.0

            records.append({"date": date, "signal": signal, "ret": ret, **bm_rets})
            prev_sig = signal

    if not records:
        raise RuntimeError("Walk-forward produced no records.")

    df = pd.DataFrame(records).set_index("date").sort_index()
    df["cum_strategy"] = np.cumprod(1 + df["ret"].values)
    for bm in bm_returns.columns:
        col = f"ret_{bm}"
        if col in df.columns:
            df[f"cum_{bm}"] = np.cumprod(1 + df[col].values)

    log.info(f"Walk-forward complete: {len(df)} OOS days, "
             f"cum={df['cum_strategy'].iloc[-1]:.3f}")
    return df


def compute_wf_metrics(wf_df: pd.DataFrame, rf_rate: float = 0.045) -> dict:
    rets    = wf_df["ret"].values
    cum     = wf_df["cum_strategy"].values
    n       = len(rets)
    rf_day  = rf_rate / 252
    ann_ret = float(cum[-1] ** (252 / n) - 1) if n > 0 else 0.0
    excess  = rets - rf_day
    sharpe  = float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))
    cum_max = np.maximum.accumulate(cum)
    max_dd  = float(np.min((cum - cum_max) / (cum_max + 1e-9)))
    hit     = float(np.mean(rets > rf_day))
    calmar  = ann_ret / (abs(max_dd) + 1e-9)
    return {
        "ann_return": ann_ret, "sharpe": sharpe,
        "max_dd": max_dd,     "hit_ratio": hit,
        "calmar": calmar,     "n_days": n,
        "cum_final": float(cum[-1]),
    }
