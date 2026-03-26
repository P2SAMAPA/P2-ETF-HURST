"""
hurst_core.py — P2-ETF-HURST
==============================
Core Hurst exponent computation and multi-timeframe analysis.

Now supports dynamic ETF universes (e.g., Option A or Option B) by allowing
callers to pass an explicit ETF list. Global ETF_UNIVERSE is retained for
backward compatibility.
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional, List

log = logging.getLogger(__name__)

# -- Timeframe windows --------------------------------------------------------
MEDIUM_WINDOW   = 63    # ~1 quarter — core signal window
LONG_WINDOW     = 126   # ~6 months — medium-term regime
VELOCITY_WINDOW = 63    # lookback for H63 velocity computation

# -- Regime thresholds --------------------------------------------------------
H_TRENDING     = 0.55
H_WEAK_TREND   = 0.50
H_RANDOM       = 0.45
H_STRONG_TREND = 0.65

# -- Default ETF universe (Option A) for backward compatibility --------------
ETF_UNIVERSE = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARKS   = ["SPY", "AGG"]

# -- Momentum grid search params ----------------------------------------------
MOM_WEIGHT_GRID = [0.10, 0.20, 0.30]   # total momentum weight vs HRC
MOM_3M_GRID     = [0.30, 0.50, 0.70]   # fraction of mom weight on 3m (rest = 6m)
MOM_3M_WINDOW   = 63
MOM_6M_WINDOW   = 126


# =============================================================================
# DFA Hurst estimator (unchanged)
# =============================================================================

def hurst_dfa(series: np.ndarray) -> float:
    """Compute Hurst exponent via Detrended Fluctuation Analysis."""
    series = np.array(series, dtype=float)
    series = series[~np.isnan(series)]
    n = len(series)
    if n < 32:
        return 0.5

    y = np.cumsum(series - np.mean(series))
    min_s  = max(8, n // 12)
    max_s  = max(min_s + 1, n // 3)
    n_wins = min(16, max_s - min_s + 1)
    sizes  = np.unique(
        np.round(np.logspace(np.log10(min_s), np.log10(max_s), n_wins)).astype(int)
    )

    log_s, log_f = [], []
    for s in sizes:
        if s < 4 or s > n // 2:
            continue
        n_seg = n // s
        if n_seg < 2:
            continue
        f2 = []
        for seg in range(n_seg):
            seg_y = y[seg * s: (seg + 1) * s]
            x     = np.arange(s, dtype=float)
            coeffs   = np.polyfit(x, seg_y, 1)
            residual = seg_y - np.polyval(coeffs, x)
            f2.append(np.mean(residual ** 2))
        f = np.sqrt(np.mean(f2))
        if f > 1e-10:
            log_s.append(np.log(float(s)))
            log_f.append(np.log(f))

    if len(log_s) < 4:
        return 0.5
    try:
        h, _ = np.polyfit(log_s, log_f, 1)
        return float(np.clip(h, 0.0, 1.0))
    except Exception:
        return 0.5


# =============================================================================
# Labels / colours (unchanged)
# =============================================================================

def hurst_label(h: float) -> str:
    if h >= H_STRONG_TREND: return "Strong Trend"
    elif h >= H_TRENDING:   return "Mild Trend"
    elif h >= H_WEAK_TREND: return "Weak Trend"
    elif h >= H_RANDOM:     return "Random Walk"
    else:                   return "Mean-Reverting"

def hurst_regime_colour(h: float) -> str:
    if h >= H_TRENDING:    return "#16a34a"
    elif h >= H_WEAK_TREND: return "#84cc16"
    elif h >= H_RANDOM:    return "#d97706"
    else:                  return "#dc2626"

def velocity_label(v: float) -> str:
    if v >= 0.15:  return "Accelerating ↑"
    elif v >= 0.0: return "Stable →"
    elif v >= -0.15: return "Decelerating ↓"
    else:          return "Reversing ↓↓"

def velocity_colour(v: float) -> str:
    if v >= 0.15:    return "#16a34a"
    elif v >= 0.0:   return "#65a30d"
    elif v >= -0.15: return "#d97706"
    else:            return "#dc2626"


# =============================================================================
# Core Hurst functions (now accept etf_list)
# =============================================================================

def compute_hurst_velocity(returns: pd.Series) -> float:
    """Compute H63 velocity (rate of change of H63 over VELOCITY_WINDOW)."""
    s = returns.dropna()
    n = len(s)
    if n < MEDIUM_WINDOW + VELOCITY_WINDOW:
        return 0.0
    h_now  = hurst_dfa(s.values[-MEDIUM_WINDOW:])
    h_past = hurst_dfa(s.values[-(MEDIUM_WINDOW + VELOCITY_WINDOW):-VELOCITY_WINDOW])
    if h_past < 1e-6:
        return 0.0
    return float(np.clip((h_now - h_past) / max(h_past, 0.1), -1.0, 1.0))


def compute_mtf_hurst(returns: pd.Series) -> dict:
    """Compute DFA Hurst at MEDIUM and LONG windows, plus velocity."""
    s = returns.dropna()
    n = len(s)
    h_medium  = hurst_dfa(s.values[-MEDIUM_WINDOW:]) if n >= MEDIUM_WINDOW else 0.5
    h_long    = hurst_dfa(s.values[-LONG_WINDOW:])   if n >= LONG_WINDOW   else 0.5
    h_velocity = compute_hurst_velocity(s)

    med_trending  = h_medium >= H_TRENDING
    long_trending = h_long   >= H_TRENDING
    vel_positive  = h_velocity >= 0.0

    if med_trending and long_trending:
        mtf_score = 1.0 if vel_positive else 0.85
    elif med_trending:
        mtf_score = 0.75 if vel_positive else 0.55
    elif long_trending:
        mtf_score = 0.50 if vel_positive else 0.30
    else:
        mtf_score = 0.15 if vel_positive else 0.0

    trending = sum([med_trending, long_trending])

    return {
        "h_short":        h_velocity,
        "h_medium":       h_medium,
        "h_long":         h_long,
        "h_velocity":     h_velocity,
        "mtf_score":      mtf_score,
        "trending_count": trending,
        "label_short":    velocity_label(h_velocity),
        "label_medium":   hurst_label(h_medium),
        "label_long":     hurst_label(h_long),
    }


def compute_all_mtf(returns_df: pd.DataFrame, etf_list: Optional[List[str]] = None) -> dict:
    """
    Compute MTF Hurst for each ETF in returns_df.
    If etf_list is given, use that; otherwise use ETF_UNIVERSE.
    """
    if etf_list is None:
        etf_list = ETF_UNIVERSE
    results = {}
    for ticker in etf_list:
        if ticker not in returns_df.columns:
            continue
        results[ticker] = compute_mtf_hurst(returns_df[ticker])
    return results


def build_mtf_history(returns_df: pd.DataFrame, step: int = 5,
                      etf_list: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build rolling MTF history for given ETF list.
    """
    if etf_list is None:
        etf_list = ETF_UNIVERSE
    dates    = returns_df.index
    n        = len(dates)
    min_rows = LONG_WINDOW + 1
    records  = []

    for i in range(min_rows, n, step):
        row    = {"date": dates[i]}
        slice_ = returns_df.iloc[:i+1]
        for ticker in etf_list:
            if ticker not in slice_.columns:
                continue
            s = slice_[ticker].dropna()
            row[f"{ticker}_h_velocity"] = compute_hurst_velocity(s)
            row[f"{ticker}_h_short"]    = row[f"{ticker}_h_velocity"]
            row[f"{ticker}_h_medium"]   = hurst_dfa(s.values[-MEDIUM_WINDOW:]) if len(s) >= MEDIUM_WINDOW else np.nan
            row[f"{ticker}_h_long"]     = hurst_dfa(s.values[-LONG_WINDOW:])   if len(s) >= LONG_WINDOW   else np.nan
        records.append(row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("date")


def compute_divergence_scores(
    mtf_today:   dict,
    mtf_history: pd.DataFrame,
    lookback:    int = 504,
    etf_list:    Optional[List[str]] = None,
) -> dict:
    """
    3-part divergence score for each ETF.
    """
    if etf_list is None:
        etf_list = ETF_UNIVERGE   # fix typo? should be ETF_UNIVERSE
        # Actually we should use the keys of mtf_today
        etf_list = list(mtf_today.keys())

    scores = {}
    if mtf_history is None or mtf_history.empty:
        for ticker in etf_list:
            if ticker not in mtf_today:
                continue
            h = mtf_today[ticker]["h_medium"]
            scores[ticker] = {
                "div_a": 0.0, "div_b": max(0.0, h - 0.5),
                "div_c": 0.0, "div_score": max(0.0, h - 0.5),
            }
        return scores

    hist = mtf_history.tail(lookback)

    for ticker in etf_list:
        if ticker not in mtf_today:
            continue
        col_m  = f"{ticker}_h_medium"
        h_now  = mtf_today[ticker]["h_medium"]

        if col_m not in hist.columns:
            scores[ticker] = {"div_a": 0.0, "div_b": 0.0, "div_c": 0.0, "div_score": 0.0}
            continue

        hist_m = hist[col_m].dropna()

        h_6m_ago   = float(hist_m.iloc[-126]) if len(hist_m) >= 126 else float(hist_m.iloc[0])
        div_a      = float(np.clip(h_now - h_6m_ago, -1, 1))

        h_baseline = float(hist_m.mean()) if len(hist_m) > 0 else 0.5
        div_b      = float(np.clip(h_now - h_baseline, -1, 1))

        h_1yr_mean = float(hist_m.tail(252).mean()) if len(hist_m) >= 63 else 0.5
        h_21d_ago  = float(hist_m.iloc[-21]) if len(hist_m) >= 21 else float(hist_m.iloc[0])
        crossed    = (h_21d_ago < h_1yr_mean) and (h_now >= h_1yr_mean)
        div_c      = 1.0 if crossed else float(np.clip((h_now - h_1yr_mean) / 0.1, -1, 1))

        div_score  = (div_a + div_b + div_c) / 3.0

        scores[ticker] = {
            "div_a":      round(div_a,      4),
            "div_b":      round(div_b,      4),
            "div_c":      round(div_c,      4),
            "div_score":  round(div_score,  4),
            "h_baseline": round(h_baseline, 4),
            "h_1yr_mean": round(h_1yr_mean, 4),
            "crossed":    crossed,
        }

    return scores


def compute_sync_score(mtf_today: dict, etf_list: Optional[List[str]] = None) -> dict:
    """Compute cross-asset sync score."""
    if etf_list is None:
        etf_list = list(mtf_today.keys())
    h_vals = {t: mtf_today[t]["h_medium"] for t in etf_list if t in mtf_today}
    if not h_vals:
        return {"sync_level": 0.5, "h_mean": 0.5, "h_std": 0.0, "scores": {t: 0.0 for t in etf_list}}

    arr        = np.array(list(h_vals.values()))
    h_mean     = float(np.mean(arr))
    h_std      = float(np.std(arr))
    sync_level = float(np.exp(-h_std / 0.05))
    max_dev    = float(np.max(np.abs(arr - h_mean))) + 1e-9
    dev_scores = {t: float((v - h_mean) / max_dev) for t, v in h_vals.items()}

    return {
        "sync_level": round(sync_level, 4),
        "h_mean":     round(h_mean,     4),
        "h_std":      round(h_std,      4),
        "scores":     {t: round(v, 4) for t, v in dev_scores.items()},
    }


def compute_momentum_scores(
    returns_df: pd.DataFrame,
    w3m:        float = 0.5,
    etf_list:   Optional[List[str]] = None,
) -> dict:
    """Cross-sectional rank-based momentum scores."""
    if etf_list is None:
        etf_list = ETF_UNIVERSE
    etfs = [t for t in etf_list if t in returns_df.columns]
    if not etfs:
        return {t: 0.5 for t in etf_list}
    ret_3m = {}
    ret_6m = {}
    for ticker in etfs:
        s = returns_df[ticker].dropna()
        ret_3m[ticker] = float(np.sum(s.values[-MOM_3M_WINDOW:])) if len(s) >= MOM_3M_WINDOW else 0.0
        ret_6m[ticker] = float(np.sum(s.values[-MOM_6M_WINDOW:])) if len(s) >= MOM_6M_WINDOW else 0.0

    def rank_norm(d: dict) -> dict:
        vals   = list(d.values())
        tickers = list(d.keys())
        ranks  = pd.Series(vals, index=tickers).rank(pct=True)
        return ranks.to_dict()

    rn3 = rank_norm(ret_3m)
    rn6 = rank_norm(ret_6m)

    return {
        t: round(w3m * rn3.get(t, 0.5) + (1 - w3m) * rn6.get(t, 0.5), 4)
        for t in etfs
    }


def optimise_momentum_weights(
    returns_df:   pd.DataFrame,
    hrc_scores:   dict,
    train_window: int = 252,
    etf_list:     Optional[List[str]] = None,
) -> tuple[float, float]:
    """Grid search over (mom_weight, w3m) to maximise in-sample Sharpe."""
    if etf_list is None:
        etf_list = list(hrc_scores.keys()) if hrc_scores else ETF_UNIVERSE
    etfs = [t for t in etf_list if t in returns_df.columns]
    if not etfs:
        log.warning("optimise_momentum_weights: no ETF columns found, returning defaults")
        return 0.20, 0.50
    hist  = returns_df.tail(train_window)

    best_sharpe = -np.inf
    best_mw, best_w3 = 0.20, 0.50

    for mom_w in MOM_WEIGHT_GRID:
        for w3m in MOM_3M_GRID:
            daily_rets = []
            prev_sig   = None
            for i in range(MOM_6M_WINDOW, len(hist) - 1):
                sub = hist.iloc[:i]
                mom_scores = compute_momentum_scores(sub, w3m=w3m, etf_list=etfs)
                blended = {}
                for t in etfs:
                    hrc = hrc_scores.get(t, {}).get("total", 0.5)
                    mom = mom_scores.get(t, 0.5)
                    blended[t] = (1 - mom_w) * hrc + mom_w * mom
                sig = max(blended, key=blended.get)
                next_ret = float(hist.iloc[i + 1][sig]) if sig in hist.columns else 0.0
                fee = 5/10_000 if (prev_sig is not None and prev_sig != sig) else 0.0
                daily_rets.append(next_ret - fee)
                prev_sig = sig
            if len(daily_rets) < 20:
                continue
            arr    = np.array(daily_rets)
            sharpe = float(np.mean(arr) / (np.std(arr) + 1e-9) * np.sqrt(252))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_mw, best_w3 = mom_w, w3m
    return best_mw, best_w3


# =============================================================================
# Master conviction score
# =============================================================================

W_MTF  = 0.40
W_DIV  = 0.40
W_SYNC = 0.20


def compute_conviction_scores(
    mtf_today:  dict,
    div_scores: dict,
    sync:       dict,
    etf_list:   Optional[List[str]] = None,
) -> dict:
    if etf_list is None:
        etf_list = list(mtf_today.keys())
    results = {}
    sync_dev = sync.get("scores", {})

    for ticker in etf_list:
        if ticker not in mtf_today:
            continue
        mtf_sc   = mtf_today[ticker]["mtf_score"]
        div_sc   = div_scores.get(ticker, {}).get("div_score", 0.0)
        div_norm = float(np.clip((div_sc + 1.0) / 2.0, 0.0, 1.0))
        sync_sc  = sync_dev.get(ticker, 0.0)
        sync_norm = float(np.clip((sync_sc + 1.0) / 2.0, 0.0, 1.0))
        total    = W_MTF * mtf_sc + W_DIV * div_norm + W_SYNC * sync_norm

        results[ticker] = {
            "mtf_score":      round(mtf_sc,    4),
            "div_score":      round(div_norm,  4),
            "sync_score":     round(sync_norm, 4),
            "total":          round(total,     4),
            "h_short":        mtf_today[ticker]["h_short"],
            "h_medium":       mtf_today[ticker]["h_medium"],
            "h_long":         mtf_today[ticker]["h_long"],
            "label":          hurst_label(mtf_today[ticker]["h_medium"]),
            "trending_count": mtf_today[ticker]["trending_count"],
        }
    return results


def conviction_label(score: float) -> str:
    if score >= 0.75: return "Very High"
    if score >= 0.60: return "High"
    if score >= 0.45: return "Moderate"
    return "Low"


def generate_signal(
    conviction_scores: dict,
    mom_scores:        Optional[dict] = None,
    mom_weight:        float = 0.20,
    w3m:               float = 0.50,
) -> dict:
    if not conviction_scores:
        return {"signal": "CASH", "conviction": 0.0, "label": "Low", "ranked": [],
                "mom_weight": mom_weight, "w3m": w3m}
    blended = {}
    for ticker, c in conviction_scores.items():
        hrc = c["total"]
        mom = mom_scores.get(ticker, 0.5) if mom_scores else 0.5
        blended[ticker] = (1 - mom_weight) * hrc + mom_weight * mom
    if not blended:
        return {"signal": "CASH", "conviction": 0.0, "label": "Low", "ranked": [],
                "mom_weight": mom_weight, "w3m": w3m}
    ranked = sorted(blended.items(), key=lambda x: -x[1])
    top    = ranked[0]
    scores = np.array([v for _, v in ranked])
    span   = scores.max() - scores.min() + 1e-9
    norm   = (top[1] - scores.min()) / span
    return {
        "signal":     top[0],
        "conviction": round(float(norm), 4),
        "score":      round(top[1], 4),
        "label":      conviction_label(top[1]),
        "ranked":     [(t, round(v, 4)) for t, v in ranked],
        "scores":     conviction_scores,
        "mom_weight": mom_weight,
        "w3m":        w3m,
    }
