"""
walkforward.py — P2-ETF-HURST
================================
Walk-forward backtest with DFA Hurst + in-fold momentum weight optimisation.

Each fold:
  1. Compute DFA MTF Hurst on train window
  2. Grid-search (mom_weight x w3m) to maximise in-sample Sharpe
  3. Apply best blend to OOS period (21 days)

Train window : 252 days
Step size    : 21 days
Fee          : 5bps per rotation
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

from hurst_core import (
    compute_all_mtf, compute_divergence_scores,
    compute_sync_score, compute_conviction_scores,
    compute_momentum_scores, optimise_momentum_weights,
    generate_signal, build_mtf_history,
    ETF_UNIVERSE, BENCHMARKS,
)

TRAIN_WINDOW = 252
STEP_SIZE    = 21
FEE          = 5 / 10_000


def run_walkforward(
    returns_df:   pd.DataFrame,
    bm_returns:   pd.DataFrame,
    train_window: int = TRAIN_WINDOW,
    step_size:    int = STEP_SIZE,
) -> pd.DataFrame:
    """
    Walk-forward backtest using DFA Hurst Confluence + optimised momentum blend.
    """
    etf_cols = [t for t in ETF_UNIVERSE if t in returns_df.columns]
    n        = len(returns_df)
    dates    = returns_df.index

    if n < train_window + step_size:
        raise ValueError(f"Need {train_window + step_size} rows, have {n}")

    total_folds = (n - train_window) // step_size
    log.info(f"Walk-forward: {n} days, train={train_window}, step={step_size}, folds={total_folds}")

    records   = []
    prev_sig  = None
    # Re-optimise every fold — diagnostic mode to check recency overfitting
    cached_mom_w = 0.20
    cached_w3m   = 0.50
    last_opt_fold = -999

    for fold_i, fold_start in enumerate(range(train_window, n - 1, step_size)):
        oos_end   = min(fold_start + step_size, n - 1)
        train_ret = returns_df.iloc[fold_start - train_window: fold_start][etf_cols]

        if fold_i % 20 == 0:
            log.info(f"  Fold {fold_i+1}/{total_folds}: "
                     f"[{dates[fold_start - train_window].date()} -> "
                     f"{dates[fold_start - 1].date()}] "
                     f"OOS [{dates[fold_start].date()}]")

        try:
            # -- Hurst conviction scores --
            mtf_today  = compute_all_mtf(train_ret)
            mtf_hist   = build_mtf_history(train_ret, step=5)
            div_scores = compute_divergence_scores(mtf_today, mtf_hist)
            sync       = compute_sync_score(mtf_today)
            conviction = compute_conviction_scores(mtf_today, div_scores, sync)

            # -- Optimise momentum weights (quarterly re-optimisation) --
            if (fold_i - last_opt_fold) >= 1:  # every fold — diagnostic
                cached_mom_w, cached_w3m = optimise_momentum_weights(
                    train_ret, conviction, train_window=train_window
                )
                last_opt_fold = fold_i
                log.info(f"    Optimised: mom_w={cached_mom_w:.2f} w3m={cached_w3m:.2f}")

            # -- Momentum scores with optimised w3m --
            mom_scores = compute_momentum_scores(train_ret, w3m=cached_w3m)

            # -- Final signal --
            sig_dict = generate_signal(
                conviction, mom_scores,
                mom_weight=cached_mom_w, w3m=cached_w3m,
            )
            signal = sig_dict["signal"]

        except Exception as e:
            log.warning(f"Fold {fold_i}: failed ({e}), holding previous")
            signal = prev_sig if prev_sig else etf_cols[0]

        # -- Apply signal over OOS period --
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

            records.append({
                "date": date, "signal": signal, "ret": ret,
                "mom_weight": cached_mom_w, "w3m": cached_w3m,
                **bm_rets,
            })
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
    # Handle both new (ret/cum_strategy) and old Hawkes (ret_A/cum_A) column names
    if "ret" in wf_df.columns:
        ret_col = "ret"
        cum_col = "cum_strategy"
    elif "ret_A" in wf_df.columns:
        ret_col = "ret_A"
        cum_col = "cum_A"
    else:
        return {}

    if cum_col not in wf_df.columns:
        cum_vals = np.cumprod(1 + wf_df[ret_col].values)
    else:
        cum_vals = wf_df[cum_col].values

    rets   = wf_df[ret_col].values
    n      = len(rets)
    rf_day = rf_rate / 252
    ann_ret = float(cum_vals[-1] ** (252 / n) - 1) if n > 0 else 0.0
    excess  = rets - rf_day
    sharpe  = float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))
    cum_max = np.maximum.accumulate(cum_vals)
    max_dd  = float(np.min((cum_vals - cum_max) / (cum_max + 1e-9)))
    hit     = float(np.mean(rets > rf_day))
    calmar  = ann_ret / (abs(max_dd) + 1e-9)

    return {
        "ann_return": ann_ret, "sharpe": sharpe,
        "max_dd": max_dd,     "hit_ratio": hit,
        "calmar": calmar,     "n_days": n,
        "cum_final": float(cum_vals[-1]),
        "ret_col": ret_col,   "cum_col": cum_col,
    }
