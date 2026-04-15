"""
train_mfdfa.py — MFDFA Daily Pipeline
======================================
Standalone orchestrator for the Multifractal DFA module.
Runs AFTER the existing HURST pipeline (called from daily.yml after train.py).

Reads:  ohlcv_data.parquet from HuggingFace (same shared dataset as HURST)
Writes: mfdfa_signals_{option}.parquet
        mfdfa_history_{option}.parquet
        mfdfa_metadata_{option}.json

Usage
-----
  python train_mfdfa.py --option a    # FI / Commodities
  python train_mfdfa.py --option b    # Equity Sectors
  python train_mfdfa.py --option a --skip-hf   # dry run

Design rules
------------
* ZERO changes to hurst_core.py, train.py, walkforward.py, data_manager.py
* Uses data_manager helpers only for OHLCV loading and HF upload
* All MFDFA logic lives in mfdfa_core.py
* Output files are uniquely named (mfdfa_*) — no collision with HURST outputs
"""

import os
import json
import logging
import argparse
from datetime import datetime

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def run_mfdfa_pipeline(option: str = "a", skip_hf_write: bool = False) -> dict:
    """
    Run MFDFA pipeline for the given option (a = FI, b = Equity).
    Returns result summary dict.
    """
    # ── Imports (same pattern as train.py) ───────────────────────────────────
    from data_manager import (
        load_ohlcv_from_hf,
        incremental_update,
        build_full_dataset,
        get_returns,
    )
    from mfdfa_core import (
        compute_all_mfdfa,
        build_mfdfa_history,
        generate_mfdfa_signal,
    )

    try:
        from config import OPTION_A_ETFS, OPTION_B_ETFS
    except ImportError:
        raise RuntimeError("config.py not found")

    # ── Select universe ───────────────────────────────────────────────────────
    if option == "a":
        etf_list = OPTION_A_ETFS
    elif option == "b":
        etf_list = OPTION_B_ETFS
    else:
        raise ValueError(f"Unknown option: {option!r}")

    log.info(f"MFDFA pipeline — Option {option.upper()}: {etf_list}")

    # ── Step 1: OHLCV ─────────────────────────────────────────────────────────
    log.info("Step 1: Loading OHLCV...")
    ohlcv = load_ohlcv_from_hf()
    if ohlcv is not None:
        ohlcv = incremental_update(ohlcv)
    else:
        ohlcv = build_full_dataset()

    returns_df = get_returns(ohlcv)
    etf_returns = returns_df[[t for t in etf_list if t in returns_df.columns]]
    log.info(f"  Returns: {etf_returns.shape}, ETFs: {etf_returns.columns.tolist()}")

    if etf_returns.empty or len(etf_returns.columns) == 0:
        raise RuntimeError(f"No return columns for option {option}")

    # ── Step 2: Today's MFDFA ────────────────────────────────────────────────
    log.info("Step 2: Computing MFDFA for all ETFs (window=252d)...")
    mfdfa_today = compute_all_mfdfa(etf_returns, etf_list=etf_returns.columns.tolist())

    # ── Step 3: Signal ───────────────────────────────────────────────────────
    log.info("Step 3: Generating MFDFA signal...")
    signal = generate_mfdfa_signal(mfdfa_today, etf_list=etf_returns.columns.tolist())
    log.info(
        f"  Signal: {signal['signal']} | conviction={signal['conviction']:.3f} "
        f"| label={signal['label']}"
    )
    log.info("  Rankings:")
    for etf, score in signal["ranked"]:
        h    = mfdfa_today.get(etf, {}).get("H_mono", float("nan"))
        da   = mfdfa_today.get(etf, {}).get("delta_alpha", float("nan"))
        lbl  = mfdfa_today.get(etf, {}).get("width_label", "?")
        log.info(f"    {etf}: score={score:.4f}  H_mono={h:.3f}  Δα={da:.3f}  [{lbl}]")

    # ── Step 4: Build signals DataFrame ──────────────────────────────────────
    log.info("Step 4: Building signals DataFrame...")
    today = pd.Timestamp(datetime.utcnow().date())

    signal_row = {
        "signal":     signal["signal"],
        "conviction": signal["conviction"],
        "label":      signal["label"],
    }
    for ticker, res in mfdfa_today.items():
        signal_row[f"{ticker}_H_mono"]      = res.get("H_mono", float("nan"))
        signal_row[f"{ticker}_delta_alpha"] = res.get("delta_alpha", float("nan"))
        signal_row[f"{ticker}_delta_f"]     = res.get("delta_f", float("nan"))
        signal_row[f"{ticker}_width_label"] = res.get("width_label", "?")
        signal_row[f"{ticker}_conviction"]  = signal["ranked"][
            [r[0] for r in signal["ranked"]].index(ticker)
        ][1] if ticker in [r[0] for r in signal["ranked"]] else 0.0

    signal_row_df = pd.DataFrame([signal_row], index=[today])

    # Load existing signals and append
    existing = _load_mfdfa_signals(option)
    if existing is not None and not existing.empty:
        signals_df = pd.concat([existing, signal_row_df])
        signals_df = signals_df[~signals_df.index.duplicated(keep="last")].sort_index()
    else:
        signals_df = signal_row_df

    # ── Step 5: Rolling MFDFA history ────────────────────────────────────────
    log.info("Step 5: Building MFDFA history (step=5)...")
    mfdfa_history = build_mfdfa_history(
        etf_returns,
        etf_list=etf_returns.columns.tolist(),
        step=5,
        window=252,
    )
    log.info(f"  History: {mfdfa_history.shape}")

    # ── Step 6: Save to HuggingFace ──────────────────────────────────────────
    metadata = {
        "last_run":    str(datetime.utcnow().date()),
        "signal":      signal["signal"],
        "conviction":  signal["conviction"],
        "label":       signal["label"],
        "option":      option,
        "etfs":        etf_returns.columns.tolist(),
    }

    if not skip_hf_write:
        log.info("Step 6: Saving MFDFA outputs to HuggingFace...")
        _save_mfdfa_to_hf(signals_df, mfdfa_history, metadata, option)
    else:
        log.info("Step 6: Skipping HF write (--skip-hf mode)")

    log.info("MFDFA pipeline complete.")
    return {
        "signal":     signal["signal"],
        "conviction": signal["conviction"],
        "label":      signal["label"],
        "etfs_run":   len(mfdfa_today),
    }


# ── HF I/O helpers (MFDFA-specific filenames) ────────────────────────────────

def _hf_repo() -> str:
    """Same HF dataset repo used by HURST."""
    return os.environ.get("HF_DATASET_REPO", "P2SAMAPA/p2-etf-hurst-data")


def _load_mfdfa_signals(option: str) -> pd.DataFrame | None:
    """Load existing MFDFA signals parquet from HF if present."""
    try:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")
        path = hf_hub_download(
            repo_id=_hf_repo(),
            filename=f"mfdfa_signals_{option}.parquet",
            repo_type="dataset",
            token=token,
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.info(f"No existing MFDFA signals for option {option}: {e}")
        return None


def _save_mfdfa_to_hf(signals_df: pd.DataFrame,
                       history_df: pd.DataFrame,
                       metadata: dict,
                       option: str) -> None:
    """Upload MFDFA outputs to HuggingFace dataset."""
    try:
        from huggingface_hub import HfApi
        import tempfile

        token   = os.environ.get("HF_TOKEN")
        repo_id = _hf_repo()
        api     = HfApi()

        with tempfile.TemporaryDirectory() as tmp:
            # Signals
            sig_path = os.path.join(tmp, f"mfdfa_signals_{option}.parquet")
            signals_df.to_parquet(sig_path)
            api.upload_file(
                path_or_fileobj=sig_path,
                path_in_repo=f"mfdfa_signals_{option}.parquet",
                repo_id=repo_id, repo_type="dataset", token=token,
            )
            log.info(f"  ✅ mfdfa_signals_{option}.parquet uploaded")

            # History
            if not history_df.empty:
                hist_path = os.path.join(tmp, f"mfdfa_history_{option}.parquet")
                history_df.to_parquet(hist_path)
                api.upload_file(
                    path_or_fileobj=hist_path,
                    path_in_repo=f"mfdfa_history_{option}.parquet",
                    repo_id=repo_id, repo_type="dataset", token=token,
                )
                log.info(f"  ✅ mfdfa_history_{option}.parquet uploaded")

            # Metadata
            meta_path = os.path.join(tmp, f"mfdfa_metadata_{option}.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2, default=str)
            api.upload_file(
                path_or_fileobj=meta_path,
                path_in_repo=f"mfdfa_metadata_{option}.json",
                repo_id=repo_id, repo_type="dataset", token=token,
            )
            log.info(f"  ✅ mfdfa_metadata_{option}.json uploaded")

    except Exception as e:
        log.error(f"HF save failed for MFDFA option {option}: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MFDFA daily pipeline")
    parser.add_argument("--option", choices=["a", "b"], default="a",
                        help="a = FI/Commodities, b = Equity Sectors")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace upload (dry run)")
    args = parser.parse_args()
    run_mfdfa_pipeline(option=args.option, skip_hf_write=args.skip_hf)
