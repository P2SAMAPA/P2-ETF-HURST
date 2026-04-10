"""
reseed.py — P2-ETF-HURST
=========================
ONE-TIME script to build complete OHLCV dataset from 2008.
Uses Yahoo Finance first, falls back to Stooq if YF fails.

ETFs (Option A) : TLT, LQD, HYG, VNQ, GLD, SLV
ETFs (Option B) : SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, XLB, XLRE
Benchmarks      : SPY, AGG  (SPY already in Option B, but we keep both)

Output: ohlcv_data.parquet  → HF dataset P2SAMAPA/p2-etf-hurst-data

Run manually: python reseed.py
"""

import os
import json
import time
import random
import pandas as pd
import yfinance as yf
from datetime import datetime
from huggingface_hub import HfApi, CommitOperationAdd

# Import config to get all tickers
from config import ALL_TICKERS

# ── Configuration ─────────────────────────────────────────────────────────────
HF_DATASET_REPO = "P2SAMAPA/p2-etf-hurst-data"   # updated
START_DATE      = "2008-01-01"
END_DATE        = datetime.today().strftime("%Y-%m-%d")
OHLCV_FIELDS    = ["Open", "High", "Low", "Close", "Volume"]


# ── Fetch helpers (unchanged) ─────────────────────────────────────────────────

def fetch_ohlcv_yf(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch full OHLCV from Yahoo Finance with exponential backoff."""
    for attempt in range(6):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if raw.empty:
                raise ValueError(f"Empty response for {ticker}")

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = [col[0] for col in raw.columns]

            # Keep only OHLCV columns
            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            if not available:
                raise ValueError(f"No OHLCV columns found for {ticker}")
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, f) for f in df.columns]
            )
            print(f"  ✅ {ticker} (YF): {len(df)} rows")
            return df

        except Exception as e:
            err = str(e).lower()
            is_rate = any(k in err for k in ["rate limit", "too many", "429", "ratelimit"])
            if is_rate and attempt < 5:
                wait = 30 * (2 ** attempt) + random.randint(5, 15)
                print(f"  ⚠️  YF rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ YF failed for {ticker} after {attempt+1} attempts: {e}")
                return None
    return None


def fetch_ohlcv_stooq(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Fetch full OHLCV from Stooq as fallback."""
    stooq_symbol = ticker.lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"

    for attempt in range(3):
        try:
            raw = pd.read_csv(url, parse_dates=["Date"], index_col="Date")
            if raw.empty:
                raise ValueError(f"Empty Stooq response for {ticker}")

            raw = raw.sort_index()
            mask = (raw.index >= start) & (raw.index <= end)
            raw = raw.loc[mask]
            if raw.empty:
                raise ValueError(f"No data in range for {ticker} from Stooq")

            # Stooq returns: Open, High, Low, Close, Volume
            available = [f for f in OHLCV_FIELDS if f in raw.columns]
            df = raw[available].copy()
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.columns = pd.MultiIndex.from_tuples(
                [(ticker, f) for f in df.columns]
            )
            print(f"  ✅ {ticker} (Stooq): {len(df)} rows")
            return df

        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                print(f"  ⚠️  Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stooq failed for {ticker} after 3 attempts.")
                return None
    return None


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame | None:
    """Try YF first, fall back to Stooq."""
    df = fetch_ohlcv_yf(ticker, start, end)
    if df is None:
        print(f"  🔄 Trying Stooq fallback for {ticker}...")
        df = fetch_ohlcv_stooq(ticker, start, end)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P2-ETF-HURST — Full OHLCV Reseed from 2008")
    print(f"Tickers : {ALL_TICKERS}")
    print(f"Range   : {START_DATE} → {END_DATE}")
    print("=" * 60)

    frames = []
    failed = []

    for ticker in ALL_TICKERS:
        print(f"\n--- {ticker} ---")
        df = fetch_ticker(ticker, START_DATE, END_DATE)
        if df is not None:
            frames.append(df)
        else:
            failed.append(ticker)
        time.sleep(random.uniform(1.0, 2.5))   # polite delay between tickers

    if not frames:
        raise RuntimeError("No data fetched from any source. Aborting.")

    if failed:
        print(f"\n⚠️  Failed tickers: {failed} — continuing with {len(frames)} tickers.")

    # Combine into single MultiIndex DataFrame: (ticker, field)
    ohlcv_df = pd.concat(frames, axis=1)
    ohlcv_df = ohlcv_df.sort_index()
    ohlcv_df = ohlcv_df[~ohlcv_df.index.duplicated(keep="last")]
    ohlcv_df = ohlcv_df.ffill()

    print(f"\n📊 OHLCV DataFrame shape : {ohlcv_df.shape}")
    print(f"   Columns (sample)      : {list(ohlcv_df.columns[:6])}")
    print(f"   Date range            : {ohlcv_df.index[0].date()} → {ohlcv_df.index[-1].date()}")

    # Save locally
    ohlcv_df.to_parquet("ohlcv_data.parquet")
    print(f"\n💾 Saved ohlcv_data.parquet ({os.path.getsize('ohlcv_data.parquet'):,} bytes)")

    # Metadata
    metadata = {
        "last_data_update":   str(ohlcv_df.index[-1].date()),
        "last_model_fit":     None,
        "dataset_version":    1,
        "seed_date":          str(datetime.today().date()),
        "rows":               len(ohlcv_df),
        "tickers":            ALL_TICKERS,
        "fields":             OHLCV_FIELDS,
    }
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("📝 Saved metadata.json")

    # Upload to HuggingFace
    print(f"\n📤 Uploading to HuggingFace: {HF_DATASET_REPO}")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    api = HfApi(token=token)
    for local_file, repo_file in [
        ("ohlcv_data.parquet", "ohlcv_data.parquet"),
        ("metadata.json",      "metadata.json"),
    ]:
        with open(local_file, "rb") as f:
            content = f.read()
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Reseed: {repo_file} — {metadata['last_data_update']}",
            operations=[CommitOperationAdd(
                path_in_repo=repo_file,
                path_or_fileobj=content,
            )],
        )
        print(f"  ✅ Uploaded {repo_file}")

    print("\n" + "=" * 60)
    print(f"🎉 RESEED COMPLETE — {len(ohlcv_df)} rows, {len(ALL_TICKERS)} tickers")
    print("=" * 60)


if __name__ == "__main__":
    main()
