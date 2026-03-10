"""
strategy.py — P2-ETF-HAWKES
=============================
Signal generation, backtesting, and performance metrics.

Option A — Hawkes only:
  Conviction = normalised excitation ratio λ*(t)/μ

Option B — Hawkes + Hurst:
  Conviction = 0.65 * excitation_score + 0.35 * hurst_score
  Rationale: Hawkes captures current self-excitation (short-term clustering);
  Hurst captures long-run persistence (is this ETF in a trending regime?).
  Combined conviction is stronger when both agree.

Author: P2SAMAPA
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import pytz

from hawkes import (
    HawkesParams, get_signal, ETF_UNIVERSE,
    fit_all_etfs, compute_cross_excitation_matrix,
    build_intensity_history,
)
from hurst import compute_all_hurst, hurst_conviction, hurst_label, hurst_colour

log = logging.getLogger(__name__)
_EST = pytz.timezone("US/Eastern")

BENCHMARKS   = ["SPY", "AGG"]

# Conviction weights for Option B
HAWKES_WEIGHT = 0.65
HURST_WEIGHT  = 0.35


# ── Next trading day ──────────────────────────────────────────────────────────

def next_trading_day(from_date: pd.Timestamp) -> pd.Timestamp:
    """Next NYSE trading day after from_date."""
    try:
        import pandas_market_calendars as mcal
        nyse  = mcal.get_calendar("NYSE")
        start = (from_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (from_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        sched = nyse.schedule(start_date=start, end_date=end)
        if not sched.empty:
            return pd.Timestamp(sched.index[0])
    except ImportError:
        pass
    nxt = from_date + pd.Timedelta(days=1)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return nxt


def next_trading_day_from_today() -> pd.Timestamp:
    """Next NYSE trading day from today's EST clock."""
    now_est   = datetime.now(_EST)
    today_ts  = pd.Timestamp(now_est.date())
    return next_trading_day(today_ts)


# ── Option A — Hawkes only ────────────────────────────────────────────────────

def generate_signal_option_a(
    fit_results: dict,
    returns_df:  pd.DataFrame,
    event_def:   str = "combined",
) -> dict:
    """
    Option A: pure Hawkes signal.
    Returns signal dict with conviction = normalised excitation ratio.
    """
    sig = get_signal(fit_results, returns_df, event_def=event_def)

    conv_label = _conviction_label(sig["conviction"])

    return {
        "option":       "A",
        "signal":       sig["signal"],
        "conviction":   sig["conviction"],
        "label":        conv_label,
        "top_ratio":    sig["top_ratio"],
        "ranked":       sig["ranked"],
        "excitation":   sig["excitation_ratios"],
        "method":       "Hawkes only",
    }


# ── Option B — Hawkes + Hurst ─────────────────────────────────────────────────

def generate_signal_option_b(
    fit_results:  dict,
    returns_df:   pd.DataFrame,
    hurst_df:     pd.DataFrame,
    event_def:    str = "combined",
) -> dict:
    """
    Option B: Hawkes excitation + Hurst persistence combined conviction.

    For each ETF:
      combined_score = HAWKES_WEIGHT * excitation_norm + HURST_WEIGHT * hurst_score

    Uses min-max normalisation with a guaranteed minimum spread so that
    Hurst can differentiate ETFs even when Hawkes scores are nearly identical.
    """
    sig = get_signal(fit_results, returns_df, event_def=event_def)
    ex  = sig["excitation_ratios"]   # raw λ*(t)/μ ratios, e.g. {VNQ:1.58, TLT:1.00, ...}

    # Detect if Hawkes scores are flat (all near 1.0x — no meaningful excitation)
    vals  = np.array(list(ex.values()))
    span  = vals.max() - vals.min()
    hawkes_flat = span < 0.05

    if hawkes_flat:
        effective_hawkes_weight = 0.20
        effective_hurst_weight  = 0.80
        log.info("Option B: Hawkes scores flat — boosting Hurst weight to 80%")
    else:
        effective_hawkes_weight = HAWKES_WEIGHT
        effective_hurst_weight  = HURST_WEIGHT

    # Normalise raw excitation ratios to [0,1] using global max
    # (divide by max rather than min-max so that 1.00x ratios stay non-zero)
    ex_max  = vals.max() if vals.max() > 0 else 1.0
    ex_norm = {t: v / ex_max for t, v in ex.items()}

    # Get latest Hurst per ETF
    hurst_latest = {}
    for t in ETF_UNIVERSE:
        if t in hurst_df.columns:
            h_series = hurst_df[t].dropna()
            hurst_latest[t] = float(h_series.iloc[-1]) if len(h_series) > 0 else 0.5
        else:
            hurst_latest[t] = 0.5

    # Combined scores
    combined = {}
    details  = {}
    for t in ETF_UNIVERSE:
        if t not in ex_norm:
            continue
        h     = hurst_latest.get(t, 0.5)
        h_sc  = hurst_conviction(h)
        ex_sc = ex_norm[t]
        score = effective_hawkes_weight * ex_sc + effective_hurst_weight * h_sc
        combined[t] = score
        details[t]  = {
            "hawkes_score":  round(ex_sc, 4),
            "hurst_score":   round(h_sc,  4),
            "hurst_H":       round(h,     4),
            "hurst_label":   hurst_label(h),
            "combined":      round(score, 4),
            "hawkes_weight": effective_hawkes_weight,
            "hurst_weight":  effective_hurst_weight,
        }

    ranked   = sorted(combined.items(), key=lambda x: -x[1])
    top_etf  = ranked[0][0]
    top_score = ranked[0][1]

    # Normalise top score to conviction [0, 1]
    all_scores = np.array(list(combined.values()))
    conviction = float((top_score - all_scores.min()) /
                       (all_scores.max() - all_scores.min() + 1e-9))

    conv_label = _conviction_label(conviction)

    return {
        "option":       "B",
        "signal":       top_etf,
        "conviction":   round(conviction, 4),
        "label":        conv_label,
        "ranked":       [(t, round(v, 4)) for t, v in ranked],
        "excitation":   sig["excitation_ratios"],
        "details":      details,
        "hawkes_flat":  hawkes_flat,
        "method":       (
            f"Hurst-led ({effective_hurst_weight:.0%}) — Hawkes scores flat"
            if hawkes_flat else
            f"Hawkes ({effective_hawkes_weight:.0%}) + Hurst ({effective_hurst_weight:.0%})"
        ),
    }


# ── Backtesting ───────────────────────────────────────────────────────────────

def backtest(
    returns_df:      pd.DataFrame,
    signals_df:      pd.DataFrame,
    option:          str = "A",
    conviction_gate: float = 0.3,
    fee_bps:         int   = 5,
    rf_rate:         float = 0.045,
) -> Tuple[np.ndarray, List[dict], pd.DatetimeIndex]:
    """
    Simple daily backtest: hold the signalled ETF if conviction >= gate,
    else hold CASH earning risk-free rate.

    signals_df must have columns: signal_{option}, conviction_{option}

    Returns (strat_rets, audit_trail, dates)
    """
    sig_col  = f"signal_{option}"
    conv_col = f"conviction_{option}"
    daily_rf = rf_rate / 252
    fee      = fee_bps / 10_000

    if sig_col not in signals_df.columns:
        log.warning(f"Signal column {sig_col} not found in signals_df")
        return np.array([]), [], pd.DatetimeIndex([])

    common = signals_df.index.intersection(returns_df.index)
    sig_bt = signals_df.loc[common]
    ret_bt = returns_df.loc[common]

    strat_rets  = []
    audit_trail = []
    prev_signal = None

    for i, date in enumerate(common[:-1]):
        signal     = sig_bt.loc[date, sig_col]
        conviction = float(sig_bt.loc[date, conv_col]) if conv_col in sig_bt.columns else 0.5
        next_date  = common[i + 1]

        if conviction < conviction_gate or signal not in ret_bt.columns:
            net_ret    = daily_rf
            trade_sig  = "CASH"
        else:
            realized   = float(ret_bt.loc[next_date, signal]) if signal in ret_bt.columns else 0.0
            rotation   = (prev_signal is not None and prev_signal != signal)
            net_ret    = realized - (fee if rotation else 0.0)
            trade_sig  = signal

        strat_rets.append(net_ret)
        prev_signal = trade_sig

        # ETF returns for audit
        etf_rets = {}
        for t in ETF_UNIVERSE:
            if t in ret_bt.columns:
                etf_rets[f"{t}_Ret%"] = round(float(ret_bt.loc[next_date, t]) * 100, 3) \
                    if next_date in ret_bt.index else 0.0

        audit_trail.append({
            "Date":       date.strftime("%Y-%m-%d"),
            "Signal":     trade_sig,
            "Conviction": round(conviction, 3),
            "Net_Ret%":   round(net_ret * 100, 3),
            **etf_rets,
        })

    return np.array(strat_rets), audit_trail, common[:-1]


def calculate_metrics(strat_rets: np.ndarray, rf_rate: float = 0.045) -> dict:
    """Full performance metrics."""
    if len(strat_rets) == 0:
        return {}
    cum      = np.cumprod(1 + strat_rets)
    n        = len(strat_rets)
    ann_ret  = float(cum[-1] ** (252 / n) - 1)
    daily_rf = rf_rate / 252
    excess   = strat_rets - daily_rf
    sharpe   = float(np.mean(excess)) / (float(np.std(excess)) + 1e-9) * np.sqrt(252)
    cum_max  = np.maximum.accumulate(cum)
    dd       = (cum - cum_max) / (cum_max + 1e-9)
    max_dd   = float(np.min(dd))
    tol      = daily_rf * 0.01
    active   = strat_rets[np.abs(strat_rets - daily_rf) > tol]
    hit      = float(np.mean(active > 0)) if len(active) > 0 else 0.0
    wins     = strat_rets[strat_rets > 0]
    losses   = strat_rets[strat_rets < 0]
    return {
        "cum_returns": cum,
        "cum_max":     cum_max,
        "ann_return":  ann_ret,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "hit_ratio":   hit,
        "calmar":      ann_ret / (abs(max_dd) + 1e-9),
        "avg_win":     float(np.mean(wins))   if len(wins)   > 0 else 0.0,
        "avg_loss":    float(np.mean(losses)) if len(losses) > 0 else 0.0,
        "n_days":      n,
    }


def calculate_benchmark_metrics(bench_rets: np.ndarray, rf_rate: float = 0.045) -> dict:
    if len(bench_rets) == 0:
        return {}
    cum     = np.cumprod(1 + bench_rets)
    ann_ret = float(cum[-1] ** (252 / len(bench_rets)) - 1)
    sharpe  = ((np.mean(bench_rets) - rf_rate / 252) /
               (np.std(bench_rets) + 1e-9) * np.sqrt(252))
    cum_max = np.maximum.accumulate(cum)
    dd      = (cum - cum_max) / (cum_max + 1e-9)
    return {
        "cum_returns": cum,
        "ann_return":  ann_ret,
        "sharpe":      float(sharpe),
        "max_dd":      float(np.min(dd)),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _conviction_label(c: float) -> str:
    if c >= 0.75:
        return "Very High"
    elif c >= 0.55:
        return "High"
    elif c >= 0.35:
        return "Moderate"
    else:
        return "Low"


def conviction_colour(label: str) -> str:
    return {
        "Very High": "#16a34a",
        "High":      "#84cc16",
        "Moderate":  "#d97706",
        "Low":       "#dc2626",
    }.get(label, "#6b7280")


def etf_colour(ticker: str) -> str:
    return {
        "TLT":  "#4e79a7",
        "LQD":  "#59a14f",
        "HYG":  "#e15759",
        "VNQ":  "#76b7b2",
        "GLD":  "#b07aa1",
        "SLV":  "#edc948",
        "CASH": "#aaaaaa",
        "SPY":  "#888888",
        "AGG":  "#bbbbbb",
    }.get(ticker, "#888888")
