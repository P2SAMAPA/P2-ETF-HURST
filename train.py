"""
train.py — P2-ETF-HURST
=========================
Daily pipeline orchestrator.

Steps:
  1. Load / update OHLCV from HuggingFace
  2. Compute multi-timeframe Hurst for all ETFs
  3. Compute divergence scores
  4. Compute cross-asset sync
  5. Generate today's conviction scores + signal
  6. Build MTF history (rolling)
  7. Walk-forward backtest
  8. Save all outputs to HuggingFace
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def run_pipeline(skip_hf_write: bool = False) -> dict:
    from data_manager import (
        load_ohlcv_from_hf, load_metadata_from_hf,
        incremental_update, build_full_dataset,
        save_to_hf, get_returns, get_volume,
        ETF_UNIVERSE, BENCHMARKS,
    )
    from hurst_core import (
        compute_all_mtf, compute_divergence_scores,
        compute_sync_score, compute_conviction_scores,
        compute_momentum_scores, optimise_momentum_weights,
        generate_signal, build_mtf_history,
        conviction_label,
    )
    from walkforward import run_walkforward

    results  = {}

    # ── Step 1: Load OHLCV ────────────────────────────────────────────────────
    log.info("Step 1: Loading OHLCV data...")
    metadata = load_metadata_from_hf() or {}
    ohlcv    = load_ohlcv_from_hf()

    if ohlcv is not None:
        log.info(f"  Existing data: {ohlcv.shape}, updating...")
        ohlcv = incremental_update(ohlcv)
    else:
        log.info("  No existing data — full reseed from 2008...")
        ohlcv = build_full_dataset()

    log.info(f"  OHLCV: {ohlcv.shape}, "
             f"{ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")

    returns_df  = get_returns(ohlcv)
    etf_returns = returns_df[[t for t in ETF_UNIVERSE if t in returns_df.columns]]
    bm_returns  = returns_df[[t for t in BENCHMARKS   if t in returns_df.columns]]

    results["data_rows"]  = len(ohlcv)
    results["date_range"] = f"{ohlcv.index[0].date()} → {ohlcv.index[-1].date()}"

    # ── Step 2: Multi-timeframe Hurst ─────────────────────────────────────────
    log.info("Step 2: Computing multi-timeframe Hurst...")
    mtf_today = compute_all_mtf(etf_returns)
    for ticker, mtf in mtf_today.items():
        log.info(f"  {ticker}: H21={mtf['h_short']:.3f} H63={mtf['h_medium']:.3f} "
                 f"H252={mtf['h_long']:.3f} trending={mtf['trending_count']}/3 "
                 f"mtf_score={mtf['mtf_score']:.3f}")

    # ── Step 3: Divergence scores ─────────────────────────────────────────────
    log.info("Step 3: Computing divergence scores...")
    log.info("  Building MTF history for divergence baseline (step=5)...")
    mtf_history = build_mtf_history(etf_returns, step=5)
    log.info(f"  MTF history: {mtf_history.shape}")
    div_scores  = compute_divergence_scores(mtf_today, mtf_history)
    for ticker, d in div_scores.items():
        log.info(f"  {ticker}: div_a={d['div_a']:.3f} div_b={d['div_b']:.3f} "
                 f"div_c={d['div_c']:.3f} total={d['div_score']:.3f} "
                 f"crossed={d.get('crossed', False)}")

    # ── Step 4: Cross-asset sync ──────────────────────────────────────────────
    log.info("Step 4: Computing cross-asset synchronisation...")
    sync = compute_sync_score(mtf_today)
    log.info(f"  Sync level={sync['sync_level']:.3f} "
             f"H_mean={sync['h_mean']:.3f} H_std={sync['h_std']:.3f}")

    # ── Step 5: Conviction scores + signal ───────────────────────────────────
    log.info("Step 5: Generating conviction scores and signal...")
    conviction = compute_conviction_scores(mtf_today, div_scores, sync)

    # Optimise momentum blend weights on last 252 days
    log.info("  Optimising momentum weights...")
    mom_w, w3m = optimise_momentum_weights(etf_returns, conviction, train_window=252)
    log.info(f"  Optimised: mom_weight={mom_w:.2f} w3m={w3m:.2f}")

    mom_scores = compute_momentum_scores(etf_returns, w3m=w3m)
    log.info(f"  Momentum scores: { {t: round(v,3) for t,v in mom_scores.items()} }")

    signal = generate_signal(conviction, mom_scores, mom_weight=mom_w, w3m=w3m)

    log.info(f"  Signal: {signal['signal']} "
             f"conviction={signal['conviction']:.3f} "
             f"label={signal['label']} "
             f"mom_w={mom_w:.2f} w3m={w3m:.2f}")
    log.info("  Rankings:")
    for etf, score in signal["ranked"]:
        log.info(f"    {etf}: {score:.4f}")

    results["signal"]     = signal["signal"]
    results["conviction"] = signal["conviction"]

    # ── Step 6: Build signals DataFrame ──────────────────────────────────────
    log.info("Step 6: Building signals DataFrame...")
    today = pd.Timestamp(datetime.utcnow().date())
    signal_row = {
        "signal":     signal["signal"],
        "conviction": signal["conviction"],
        "label":      signal["label"],
    }
    for ticker, c in conviction.items():
        signal_row[f"{ticker}_total"]   = c["total"]
        signal_row[f"{ticker}_mtf"]     = c["mtf_score"]
        signal_row[f"{ticker}_div"]     = c["div_score"]
        signal_row[f"{ticker}_sync"]    = c["sync_score"]
        signal_row[f"{ticker}_h_short"] = c["h_short"]
        signal_row[f"{ticker}_h_med"]   = c["h_medium"]
        signal_row[f"{ticker}_h_long"]  = c["h_long"]

    signal_row_df = pd.DataFrame([signal_row], index=[today])

    # Append to existing signals history
    existing_signals = None
    try:
        from data_manager import load_signals_from_hf
        existing_signals = load_signals_from_hf()
    except Exception:
        pass

    if existing_signals is not None and not existing_signals.empty:
        signals_df = pd.concat([existing_signals, signal_row_df])
        signals_df = signals_df[~signals_df.index.duplicated(keep="last")].sort_index()
    else:
        signals_df = signal_row_df

    # ── Step 7: Walk-forward backtest ─────────────────────────────────────────
    log.info("Step 7: Running walk-forward backtest...")
    try:
        wf_df = run_walkforward(
            returns_df   = etf_returns,
            bm_returns   = bm_returns,
            train_window = 252,
            step_size    = 21,
        )
        log.info(f"  Walk-forward: {len(wf_df)} OOS days, "
                 f"cum={wf_df['cum_strategy'].iloc[-1]:.3f}")
    except Exception as e:
        log.error(f"  Walk-forward failed: {e}")
        wf_df = None

    # ── Step 8: Save to HuggingFace ───────────────────────────────────────────
    metadata.update({
        "last_data_update": str(ohlcv.index[-1].date()),
        "last_model_fit":   str(datetime.utcnow().date()),
        "signal":           signal["signal"],
        "conviction":       signal["conviction"],
        "dataset_version":  metadata.get("dataset_version", 0) + 1,
    })

    save_files = {
        "ohlcv_data.parquet":      ohlcv,
        "mtf_history.parquet":     mtf_history,
        "signals_latest.parquet":  signals_df,
        "metadata.json":           metadata,
    }
    if wf_df is not None:
        save_files["walkforward_returns.parquet"] = wf_df

    if not skip_hf_write:
        log.info("Step 8: Saving to HuggingFace...")
        ok = save_to_hf(
            files=save_files,
            commit_message=(
                f"Daily update {datetime.utcnow().date()} — "
                f"Signal: {signal['signal']} ({signal['label']})"
            ),
        )
        results["hf_saved"] = ok
        log.info(f"  HF save: {'✅ OK' if ok else '❌ FAILED'}")

    log.info("Pipeline complete.")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-hf", action="store_true")
    args = parser.parse_args()
    run_pipeline(skip_hf_write=args.skip_hf)
