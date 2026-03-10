"""
train.py — P2-ETF-HAWKES
==========================
Daily training pipeline orchestrator.
Called by GitHub Actions at market close (9pm EST) on weekdays.

Steps:
  1. Load / update OHLCV data from HuggingFace
  2. Select best event definition (or load from metadata)
  3. Fit Hawkes models (all ETFs, best kernel by AIC)
  4. Compute rolling Hurst exponents
  5. Generate Option A and Option B signals
  6. Compute cross-ETF excitation matrix
  7. Build intensity history
  8. Save everything to HuggingFace

Author: P2SAMAPA
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from data_manager import (
    load_ohlcv_from_hf, load_metadata_from_hf,
    incremental_update, build_full_dataset,
    get_returns, get_volume, save_to_hf,
    ETF_UNIVERSE, BENCHMARKS, ALL_TICKERS,
)
from hawkes import (
    fit_all_etfs, get_signal, select_best_event_def,
    compute_cross_excitation_matrix, build_intensity_history,
    EVENT_DEFINITIONS,
)
from hurst import compute_all_hurst
from strategy import (
    generate_signal_option_a, generate_signal_option_b,
    next_trading_day, calculate_metrics, backtest,
)


def run_pipeline(
    force_refresh:     bool = False,
    skip_hf_write:     bool = False,
    reselect_event_def: bool = False,
    start_year:        int  = 2008,
) -> dict:
    results = {}

    log.info("=" * 60)
    log.info("P2-ETF-HAWKES — Daily Training Pipeline")
    log.info(f"Started : {datetime.utcnow().isoformat()}Z")
    log.info("=" * 60)

    # ── Step 1: Load / update OHLCV ──────────────────────────────────────────
    log.info("Step 1: Loading OHLCV data...")
    metadata = load_metadata_from_hf()

    if force_refresh:
        log.info("  Force refresh — rebuilding from scratch")
        ohlcv = build_full_dataset()
    else:
        existing = load_ohlcv_from_hf()
        if existing is not None:
            ohlcv = incremental_update(existing)
        else:
            log.warning("  No existing data found — running full build")
            ohlcv = build_full_dataset()

    log.info(f"  OHLCV: {ohlcv.shape}, "
             f"{ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")

    # Filter to start_year
    cutoff = pd.Timestamp(f"{start_year}-01-01")
    ohlcv  = ohlcv[ohlcv.index >= cutoff]

    returns_df = get_returns(ohlcv)
    volume_df  = get_volume(ohlcv)

    # Separate ETF vs benchmark returns
    etf_returns = returns_df[[t for t in ETF_UNIVERSE if t in returns_df.columns]]
    bm_returns  = returns_df[[t for t in BENCHMARKS  if t in returns_df.columns]]

    results["data_rows"]  = len(ohlcv)
    results["date_range"] = f"{ohlcv.index[0].date()} → {ohlcv.index[-1].date()}"

    # ── Step 2: Select best event definition ─────────────────────────────────
    log.info("Step 2: Event definition selection...")

    # Use cached event_def unless forced to reselect
    saved_event_def = metadata.get("best_event_def")
    if saved_event_def and not reselect_event_def:
        event_def = saved_event_def
        log.info(f"  Using saved event definition: {event_def}")
    else:
        log.info("  Comparing all three event definitions on OOS hit ratio...")
        event_def = select_best_event_def(etf_returns, volume_df, n_days_oos=252)
        log.info(f"  Selected: {event_def}")

    results["event_def"] = event_def

    # ── Step 3: Fit Hawkes models ─────────────────────────────────────────────
    log.info("Step 3: Fitting Hawkes models...")
    fit_results = fit_all_etfs(etf_returns, volume_df, event_def=event_def)

    for ticker, res in fit_results.items():
        p = res["params"]
        log.info(f"  {ticker}: kernel={p.kernel} μ={p.mu:.4f} "
                 f"branching={p.branching:.3f} AIC={p.aic:.2f} "
                 f"events={p.n_events}")

    # ── Step 4: Hurst exponents ───────────────────────────────────────────────
    log.info("Step 4: Computing rolling Hurst exponents...")
    hurst_df = compute_all_hurst(etf_returns, window=252)
    log.info(f"  Hurst computed for {len(hurst_df.columns)} ETFs")

    # ── Step 5: Generate signals ──────────────────────────────────────────────
    log.info("Step 5: Generating signals...")
    sig_a = generate_signal_option_a(fit_results, etf_returns, event_def)
    sig_b = generate_signal_option_b(fit_results, etf_returns, hurst_df, event_def)

    next_date = next_trading_day(ohlcv.index[-1])

    log.info(f"  Option A: {sig_a['signal']} (conviction={sig_a['conviction']:.3f} {sig_a['label']})")
    log.info(f"  Option B: {sig_b['signal']} (conviction={sig_b['conviction']:.3f} {sig_b['label']})")
    log.info(f"  Next trading day: {next_date.date()}")

    results.update({
        "signal_a":       sig_a["signal"],
        "conviction_a":   sig_a["conviction"],
        "label_a":        sig_a["label"],
        "signal_b":       sig_b["signal"],
        "conviction_b":   sig_b["conviction"],
        "label_b":        sig_b["label"],
        "next_date":      str(next_date.date()),
    })

    # ── Step 6: Cross-ETF excitation matrix ───────────────────────────────────
    log.info("Step 6: Computing cross-ETF excitation matrix...")
    cross_matrix = compute_cross_excitation_matrix(etf_returns, volume_df, event_def)
    log.info(f"  Cross-excitation matrix:\n{cross_matrix.round(3).to_string()}")

    # ── Step 7: Build intensity history ───────────────────────────────────────
    log.info("Step 7: Building intensity history...")
    intensity_df = build_intensity_history(fit_results, etf_returns.index)
    log.info(f"  Intensity history: {intensity_df.shape}")

    # ── Step 8: Build signals DataFrame ──────────────────────────────────────
    log.info("Step 8: Building signals DataFrame...")

    # Build per-day signal history using rolling window approach
    # (simplified: just store today's signal for audit trail)
    signal_row = {
        "signal_A":      sig_a["signal"],
        "conviction_A":  sig_a["conviction"],
        "label_A":       sig_a["label"],
        "top_ratio_A":   sig_a["top_ratio"],
        "signal_B":      sig_b["signal"],
        "conviction_B":  sig_b["conviction"],
        "label_B":       sig_b["label"],
        "event_def":     event_def,
    }
    # Add per-ETF excitation ratios
    for t, v in sig_a["excitation"].items():
        signal_row[f"{t}_excitation"] = v
    # Add per-ETF Hurst
    for t in ETF_UNIVERSE:
        if t in hurst_df.columns:
            h_series = hurst_df[t].dropna()
            signal_row[f"{t}_hurst"] = round(float(h_series.iloc[-1]), 4) if len(h_series) > 0 else 0.5

    signals_df = pd.DataFrame([signal_row], index=[next_date])

    # ── Hawkes params dict ────────────────────────────────────────────────────
    params_dict = {
        t: res["params"].to_dict()
        for t, res in fit_results.items()
    }
    params_dict["best_event_def"] = event_def
    params_dict["fit_date"]       = str(datetime.utcnow().date())

    # ── Step 9: Save to HuggingFace ───────────────────────────────────────────
    if not skip_hf_write:
        log.info("Step 9: Saving to HuggingFace...")

        # Update metadata
        metadata.update({
            "last_data_update":  str(ohlcv.index[-1].date()),
            "last_model_fit":    str(datetime.utcnow().date()),
            "best_event_def":    event_def,
            "signal_a":          sig_a["signal"],
            "signal_b":          sig_b["signal"],
            "next_date":         str(next_date.date()),
            "dataset_version":   metadata.get("dataset_version", 1) + 1,
        })

        ok = save_to_hf(
            files={
                "ohlcv_data.parquet":        ohlcv,
                "signals_latest.parquet":    signals_df,
                "intensity_history.parquet": intensity_df,
                "hurst_history.parquet":     hurst_df,
                "cross_excitation.parquet":  cross_matrix,
                "hawkes_params.json":        params_dict,
                "metadata.json":             metadata,
            },
            commit_message=(
                f"Daily update {datetime.utcnow().date()} — "
                f"A:{sig_a['signal']} B:{sig_b['signal']}"
            ),
        )
        results["hf_saved"] = ok
        log.info(f"  HuggingFace save: {'✅ OK' if ok else '❌ FAILED'}")
    else:
        log.info("Step 9: Skipping HF write (--local flag)")
        results["hf_saved"] = False

    log.info("=" * 60)
    log.info("Pipeline complete.")
    log.info(f"  Option A signal : {sig_a['signal']} ({sig_a['label']})")
    log.info(f"  Option B signal : {sig_b['signal']} ({sig_b['label']})")
    log.info(f"  Next trade date : {next_date.date()}")
    log.info(f"  Event def used  : {event_def}")
    log.info("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-ETF-HAWKES Training Pipeline")
    parser.add_argument("--force-refresh",      action="store_true",
                        help="Rebuild full OHLCV dataset from scratch")
    parser.add_argument("--local",              action="store_true",
                        help="Skip HuggingFace write (local testing)")
    parser.add_argument("--reselect-event-def", action="store_true",
                        help="Re-run event definition selection (takes ~10 min)")
    parser.add_argument("--start-year",         type=int, default=2008)
    args = parser.parse_args()

    results = run_pipeline(
        force_refresh      = args.force_refresh,
        skip_hf_write      = args.local,
        reselect_event_def = args.reselect_event_def,
        start_year         = args.start_year,
    )
    sys.exit(0)
