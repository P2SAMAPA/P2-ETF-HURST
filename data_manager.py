"""
data_manager.py — P2-ETF-HAWKES
=================================
HuggingFace read/write + incremental daily OHLCV update.
Provides clean load/save interface for all other modules.

Author: P2SAMAPA
"""

import os
import io
import json
import time
import random
import logging
import pandas as pd
import requests
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
import yfinance as yf

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/p2-etf-hawkes-data"
ETF_UNIVERSE    = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARKS      = ["SPY", "AGG"]
ALL_TICKERS     = ETF_UNIVERSE + BENCHMARKS
OHLCV_FIELDS    = ["Open", "High", "Low", "Close", "Volume"]

# NOTE: No custom requests.Session passed to yfinance.
# New yfinance versions manage their own curl_cffi session internally.
# Passing a requests.Session causes: "Yahoo API requires curl_cffi session" error.


# ── HuggingFace helpers ───────────────────────────────────────────────────────

def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")
    return token


def load_ohlcv_from_hf() -> pd.DataFrame | None:
    """Load ohlcv_data.parquet from HF dataset. Returns MultiIndex DataFrame."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="ohlcv_data.parquet",
            repo_type="dataset",
            token=_hf_token(),
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        log.info(f"Loaded OHLCV from HF: {df.shape}, "
                 f"{df.index[0].date()} → {df.index[-1].date()}")
        return df
    except Exception as e:
        log.warning(f"Could not load OHLCV from HF: {e}")
        return None


def load_metadata_from_hf() -> dict:
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="metadata.json",
            repo_type="dataset",
            token=_hf_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load metadata from HF: {e}")
        return {}


def load_signals_from_hf() -> pd.DataFrame | None:
    """Load latest signals parquet from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="signals_latest.parquet",
            repo_type="dataset",
            token=_hf_token(),
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.warning(f"Could not load signals from HF: {e}")
        return None


def load_intensity_history_from_hf() -> pd.DataFrame | None:
    """Load per-ETF intensity history from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="intensity_history.parquet",
            repo_type="dataset",
            token=_hf_token(),
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.warning(f"Could not load intensity history from HF: {e}")
        return None


def load_hurst_history_from_hf() -> pd.DataFrame | None:
    """Load rolling Hurst history from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="hurst_history.parquet",
            repo_type="dataset",
            token=_hf_token(),
        )
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        log.warning(f"Could not load Hurst history from HF: {e}")
        return None


def load_cross_excitation_from_hf() -> pd.DataFrame | None:
    """Load cross-ETF excitation matrix from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="cross_excitation.parquet",
            repo_type="dataset",
            token=_hf_token(),
        )
        df = pd.read_parquet(path)
        return df
    except Exception as e:
        log.warning(f"Could not load cross excitation matrix from HF: {e}")
        return None


def load_params_from_hf() -> dict | None:
    """Load fitted Hawkes parameters JSON from HF."""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="hawkes_params.json",
            repo_type="dataset",
            token=_hf_token(),
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"Could not load Hawkes params from HF: {e}")
        return None


def save_to_hf(files: dict, commit_message: str) -> bool:
    """
    Upload multiple files to HF dataset in a single commit.
    files = {repo_filename: local_bytes_or_path}
    """
    try:
        token = _hf_token()
        api   = HfApi(token=token)
        ops   = []
        for repo_file, content in files.items():
            if isinstance(content, (str, bytes)) and not isinstance(content, str):
                data = content
            elif isinstance(content, str) and os.path.exists(content):
                with open(content, "rb") as f:
                    data = f.read()
            elif isinstance(content, pd.DataFrame):
                buf = io.BytesIO()
                content.to_parquet(buf)
                data = buf.getvalue()
            elif isinstance(content, dict):
                data = json.dumps(content, indent=2).encode()
            else:
                data = content
            ops.append(CommitOperationAdd(
                path_in_repo=repo_file,
                path_or_fileobj=data,
            ))
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token,
            commit_message=commit_message,
            operations=ops,
        )
        log.info(f"HF commit: {commit_message} ({len(ops)} files)")
        return True
    except Exception as e:
        log.error(f"HF save failed: {e}")
        return False


# ── yfinance / Stooq fetch helpers ────────────────────────────────────────────

def _fetch_ohlcv_yf(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    for attempt in range(6):
        try:
            raw = yf.download(
                ticker, start=start, end=end,
                progress=False, auto_adjust=True,
                threads=False,
            )
            if raw.empty:
                raise ValueError(f"Empty YF response for {ticker}")
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]
            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = pd.MultiIndex.from_tuples([(ticker, f) for f in df.columns])
            return df
        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in ["rate limit", "too many", "429", "ratelimit"])
            if is_rate and attempt < 5:
                wait = 30 * (2 ** attempt) + random.randint(5, 15)
                log.warning(f"YF rate limited on {ticker} attempt {attempt+1}. Waiting {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"YF failed for {ticker}: {e}")
                return None
    return None


def _fetch_ohlcv_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
    for attempt in range(3):
        try:
            raw = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            if raw.empty:
                raise ValueError(f"Empty Stooq response for {ticker}")
            raw = raw.sort_index()
            mask = (raw.index >= start) & (raw.index <= end)
            raw  = raw.loc[mask]
            if raw.empty:
                raise ValueError(f"No Stooq data in range for {ticker}")
            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = pd.MultiIndex.from_tuples([(ticker, f) for f in df.columns])
            return df
        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                log.warning(f"Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying {wait}s")
                time.sleep(wait)
            else:
                log.warning(f"Stooq failed for {ticker}")
                return None
    return None


def fetch_ticker_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Try YF first, fall back to Stooq."""
    df = _fetch_ohlcv_yf(ticker, start, end)
    if df is None:
        log.info(f"Falling back to Stooq for {ticker}")
        df = _fetch_ohlcv_stooq(ticker, start, end)
    return df


# ── Incremental update ────────────────────────────────────────────────────────

def incremental_update(existing: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch only new rows since existing data's last date,
    concat, dedup, ffill and return updated DataFrame.
    """
    last_date = existing.index[-1]
    start_new = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
    end_new   = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = datetime.today().strftime("%Y-%m-%d")

    if pd.Timestamp(start_new) > pd.Timestamp(today_str):
        log.info("Data already up to date.")
        return existing

    log.info(f"Incremental update: {start_new} → {today_str}")
    frames = []
    for ticker in ALL_TICKERS:
        df = fetch_ticker_ohlcv(ticker, start_new, end_new)
        if df is not None:
            frames.append(df)
        time.sleep(random.uniform(0.5, 1.5))

    if not frames:
        log.warning("No new data fetched — returning existing.")
        return existing

    new_df   = pd.concat(frames, axis=1)
    updated  = pd.concat([existing, new_df])
    updated  = updated[~updated.index.duplicated(keep="last")]
    updated  = updated.sort_index()
    updated  = updated.ffill()
    log.info(f"Updated: {len(existing)} → {len(updated)} rows")
    return updated


def build_full_dataset(start: str = "2008-01-01") -> pd.DataFrame:
    """Full reseed — used when no existing data found."""
    end = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    frames = []
    for ticker in ALL_TICKERS:
        log.info(f"Fetching {ticker}...")
        df = fetch_ticker_ohlcv(ticker, start, end)
        if df is not None:
            frames.append(df)
        time.sleep(random.uniform(1.0, 2.5))
    if not frames:
        raise RuntimeError("No data fetched. Check network and API access.")
    ohlcv = pd.concat(frames, axis=1)
    ohlcv = ohlcv.sort_index()
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="last")]
    ohlcv = ohlcv.ffill()
    return ohlcv


def get_close_prices(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Extract Close price for all tickers as a flat DataFrame."""
    cols = [(t, "Close") for t in ALL_TICKERS if (t, "Close") in ohlcv.columns]
    df   = ohlcv[cols].copy()
    df.columns = [t for t, _ in cols]
    return df


def get_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from Close prices."""
    close = get_close_prices(ohlcv)
    return close.pct_change().dropna()


def get_volume(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Extract Volume for all tickers as a flat DataFrame."""
    cols = [(t, "Volume") for t in ALL_TICKERS if (t, "Volume") in ohlcv.columns]
    df   = ohlcv[cols].copy()
    df.columns = [t for t, _ in cols]
    return df
