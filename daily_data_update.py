"""
daily_data_update.py — P2-ETF-HURST
====================================
Standalone incremental OHLCV data update.
Updates the shared ohlcv_data.parquet for all 18 ETFs (both Option A and B).

This script ONLY updates data — no training, no signal generation.
"""

import os
import sys
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

def update_data() -> bool:
    """
    Incrementally update OHLCV data for all ETFs.
    Returns True if successful, False otherwise.
    """
    from data_manager import (
        load_ohlcv_from_hf, 
        load_metadata_from_hf,
        incremental_update,
        save_to_hf,
        ALL_TICKERS,
    )

    log.info("=" * 60)
    log.info("DAILY DATA UPDATE — P2-ETF-HURST")
    log.info(f"Tickers: {ALL_TICKERS}")
    log.info("=" * 60)

    # Load existing data
    log.info("Step 1: Loading existing OHLCV from HuggingFace...")
    ohlcv = load_ohlcv_from_hf()
    metadata = load_metadata_from_hf() or {}

    if ohlcv is not None:
        log.info(f"Existing data found: {ohlcv.shape}")
        log.info(f"Date range: {ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")
        
        # Incremental update
        log.info("Step 2: Running incremental update...")
        ohlcv = incremental_update(ohlcv)
    else:
        log.warning("No existing data found — running full reseed...")
        from data_manager import build_full_dataset
        ohlcv = build_full_dataset()

    log.info(f"Updated data: {ohlcv.shape}")
    log.info(f"Date range: {ohlcv.index[0].date()} → {ohlcv.index[-1].date()}")

    # Update metadata
    metadata.update({
        "last_data_update": str(ohlcv.index[-1].date()),
        "dataset_version": metadata.get("dataset_version", 0) + 1,
        "tickers": ALL_TICKERS,
        "update_timestamp": str(datetime.utcnow().isoformat()),
    })

    # Save to HF
    log.info("Step 3: Saving to HuggingFace...")
    files = {
        "ohlcv_data.parquet": ohlcv,
        "metadata.json": metadata,
    }
    
    ok = save_to_hf(
        files=files,
        commit_message=f"Daily data update {datetime.utcnow().date()} — {ohlcv.index[-1].date()}"
    )
    
    if ok:
        log.info("✅ Data update complete")
        log.info(f"Latest data: {ohlcv.index[-1].date()}")
        return True
    else:
        log.error("❌ Failed to save to HuggingFace")
        return False

if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        log.error("HF_TOKEN not set — cannot read/write HF Dataset")
        sys.exit(1)
    
    success = update_data()
    sys.exit(0 if success else 1)
