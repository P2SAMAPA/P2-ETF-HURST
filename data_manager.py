"""
data_manager.py — P2-ETF-HURST
================================
HuggingFace data loading and saving.
"""

import os
import json
import logging
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

HF_DATASET_REPO = "P2SAMAPA/p2-etf-hurst-data"
ETF_UNIVERSE    = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARKS      = ["SPY", "AGG"]
ALL_TICKERS     = ETF_UNIVERSE + BENCHMARKS


def _hf_token() -> str:
    return os.environ.get("HF_TOKEN", "")


def _load_parquet(filename: str) -> pd.DataFrame | None:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO, filename=filename,
            repo_type="dataset", token=_hf_token(),
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.warning(f"Could not load {filename}: {e}")
        return None


def _load_json(filename: str) -> dict | None:
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO, filename=filename,
            repo_type="dataset", token=_hf_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load {filename}: {e}")
        return None


def load_ohlcv_from_hf()       -> pd.DataFrame | None: return _load_parquet("ohlcv_data.parquet")
def load_mtf_history_from_hf() -> pd.DataFrame | None: return _load_parquet("mtf_history.parquet")
def load_signals_from_hf()     -> pd.DataFrame | None: return _load_parquet("signals_latest.parquet")
def load_walkforward_from_hf() -> pd.DataFrame | None: return _load_parquet("walkforward_returns.parquet")
def load_metadata_from_hf()    -> dict | None:         return _load_json("metadata.json")


def get_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Extract close prices and compute log returns.
    Handles all observed column formats:
      - (TICKER, 'Close')        real MultiIndex tuples
      - "('TLT', 'Close')"       stringified tuples (parquet round-trip artifact)
      - TLT_close                yfinance flat
      - TLT_close_tlt            yfinance multi-ticker artifact
      - Close_TLT                inverted format
    Priority: stringified tuples first (most common in this dataset), then flat.
    """
    import re
    cols = ohlcv.columns.tolist()
    if not cols:
        return pd.DataFrame()

    # 1. Real MultiIndex tuples: (TICKER, 'Close')
    if isinstance(cols[0], tuple):
        close_cols = [c for c in cols if str(c[1]).lower() == 'close']
        if close_cols:
            close = ohlcv[close_cols].copy()
            close.columns = [str(c[0]).upper() for c in close_cols]
            return np.log(close / close.shift(1))

    # 2. Stringified tuples: "('TLT', 'Close')" or "('TLT', 'Close')"
    if str(cols[0]).startswith("("):
        close_cols = [c for c in cols
                      if str(c).replace("'","").replace('"','').split(",")[-1]
                         .strip().rstrip(") ").lower() == "close"]
        if close_cols:
            close = ohlcv[close_cols].copy()
            close.columns = [
                str(c).replace("'","").replace('"','')
                      .lstrip("( ").split(",")[0].strip().upper()
                for c in close_cols
            ]
            return np.log(close / close.shift(1))

    # 3. TICKER_close_ticker: TLT_close_tlt (yfinance multi-ticker artifact)
    close_cols = [c for c in cols if re.match(r'^[A-Za-z]+_close_[a-z]+$', str(c))]
    if close_cols:
        close = ohlcv[close_cols].copy()
        close.columns = [str(c).split("_")[0].upper() for c in close_cols]
        return np.log(close / close.shift(1))

    # 4. TICKER_close: TLT_close
    close_cols = [c for c in cols if re.match(r'^[A-Za-z]+_close$', str(c), re.IGNORECASE)]
    if close_cols:
        close = ohlcv[close_cols].copy()
        close.columns = [str(c).split("_")[0].upper() for c in close_cols]
        return np.log(close / close.shift(1))

    # 5. close_TICKER: close_TLT
    close_cols = [c for c in cols if str(c).lower().startswith("close_")]
    if close_cols:
        close = ohlcv[close_cols].copy()
        close.columns = [str(c).split("_", 1)[1].upper() for c in close_cols]
        return np.log(close / close.shift(1))

    # 6. Fallback
    return np.log(ohlcv / ohlcv.shift(1))


def get_volume(ohlcv: pd.DataFrame) -> pd.DataFrame:
    cols = ohlcv.columns.tolist()
    if isinstance(cols[0], tuple):
        vol_cols = [c for c in cols if c[1].lower() == 'volume']
        if not vol_cols:
            return pd.DataFrame()
        vol = ohlcv[vol_cols].copy()
        vol.columns = [c[0] for c in vol_cols]
        return vol
    vol_cols = [c for c in cols if c.endswith("_volume")]
    if not vol_cols:
        return pd.DataFrame()
    vol = ohlcv[vol_cols].copy()
    vol.columns = [c.replace("_volume", "") for c in vol_cols]
    return vol


def fetch_ticker_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(c).strip().lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        df.columns = [f"{ticker}_{c}" for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.error(f"Failed to fetch {ticker}: {e}")
        return None


def build_full_dataset(start: str = "2008-01-01") -> pd.DataFrame:
    import time, random
    end    = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for ticker in ALL_TICKERS:
        log.info(f"Fetching {ticker}...")
        df = fetch_ticker_ohlcv(ticker, start, end)
        if df is not None:
            frames.append(df)
        time.sleep(random.uniform(1.0, 2.5))
    if not frames:
        raise RuntimeError("No data fetched.")
    out = pd.concat(frames, axis=1)
    return out[~out.index.duplicated(keep="last")].sort_index().ffill()


def incremental_update(existing: pd.DataFrame) -> pd.DataFrame:
    import time, random
    last_date = existing.index[-1]
    start_new = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = datetime.today().strftime("%Y-%m-%d")
    if pd.Timestamp(start_new) > pd.Timestamp(today_str):
        log.info("Data already up to date.")
        return existing
    frames = []
    for ticker in ALL_TICKERS:
        df = fetch_ticker_ohlcv(ticker, start_new,
                                (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d"))
        if df is not None:
            frames.append(df)
        time.sleep(random.uniform(0.5, 1.5))
    if not frames:
        return existing
    new_df  = pd.concat(frames, axis=1)
    updated = pd.concat([existing, new_df])
    return updated[~updated.index.duplicated(keep="last")].sort_index().ffill()


def save_to_hf(files: dict, commit_message: str = "Daily update") -> bool:
    try:
        from huggingface_hub import HfApi
        api   = HfApi()
        token = _hf_token()
        with tempfile.TemporaryDirectory() as tmp:
            for filename, data in files.items():
                fpath = os.path.join(tmp, filename)
                if isinstance(data, pd.DataFrame):
                    data.to_parquet(fpath)
                elif isinstance(data, dict):
                    with open(fpath, "w") as f:
                        json.dump(data, f, indent=2, default=str)
                else:
                    continue
                api.upload_file(
                    path_or_fileobj=fpath, path_in_repo=filename,
                    repo_id=HF_DATASET_REPO, repo_type="dataset",
                    token=token, commit_message=commit_message,
                )
                log.info(f"  Uploaded {filename}")
        return True
    except Exception as e:
        log.error(f"HF save failed: {e}")
        return False
