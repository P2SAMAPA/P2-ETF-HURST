"""
hurst_core.py — P2-ETF-HURST
==============================
Core Hurst exponent computation and multi-timeframe analysis.

Three timeframe windows:
  SHORT  = 21  trading days (~1 month)
  MEDIUM = 63  trading days (~1 quarter)
  LONG   = 252 trading days (~1 year)

Estimator: DFA (Detrended Fluctuation Analysis) — unbiased on short windows.
R/S was replaced because it systematically over-estimates H at n < 100,
producing spuriously high values (0.7-0.9) that wash out cross-ETF differences.

DFA regime thresholds:
  H > 0.55 → Trending / persistent
  H ≈ 0.50 → Random walk
  H < 0.45 → Anti-persistent / mean-reverting

Momentum overlay:
  Final score blends HRC conviction with 3m + 6m price momentum.
  Weights optimised in-fold during walk-forward (no look-ahead).
"""

import numpy as np
import pandas as pd
import logging
from typing import Optional

log = logging.getLogger(__name__)

# -- Timeframe windows --------------------------------------------------------
# SHORT_WINDOW removed — replaced by H_VELOCITY (rate of change of H63)
MEDIUM_WINDOW   = 63    # ~1 quarter — core signal window
LONG_WINDOW     = 126   # ~6 months — medium-term regime
VELOCITY_WINDOW = 63    # lookback for H63 velocity computation

# -- Regime thresholds --------------------------------------------------------
H_TRENDING     = 0.55
H_RANDOM       = 0.45
H_STRONG_TREND = 0.65

ETF_UNIVERSE = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
BENCHMARKS   = ["SPY", "AGG"]

# -- Momentum grid search params ----------------------------------------------
MOM_WEIGHT_GRID = [0.10, 0.20, 0.30]   # total momentum weight vs HRC
MOM_3M_GRID     = [0.30, 0.50, 0.70]   # fraction of mom weight on 3m (rest = 6m)
MOM_3M_WINDOW   = 63                    # ~3 months
MOM_6M_WINDOW   = 126                   # ~6 months (now same as LONG_WINDOW — intentional)


# =============================================================================
# DFA Hurst estimator
# =============================================================================

def hurst_dfa(series: np.ndarray) -> float:
    """
    Compute Hurst exponent via Detrended Fluctuation Analysis (DFA).

    DFA is unbiased down to ~20 sample points -- unlike R/S which systematically
    over-estimates H on short windows, making all ETFs look 'strongly trending'.

    Algorithm:
      1. Integrate the series (cumulative sum of mean-centred values)
      2. Divide into non-overlapping windows of size s
      3. Detrend each window with a linear fit
      4. Compute RMS of residuals F(s)
      5. F(s) ~ s^H  -->  H = slope of log(F) vs log(s)

    Returns H in [0, 1]. Returns 0.5 if too short or degenerate.
    """
    series = np.array(series, dtype=float)
    series = series[~np.isnan(series)]
    n = len(series)
    if n < 32:          # need at least 32 points for meaningful DFA
        return 0.5

    # Integrate
    y = np.cumsum(series - np.mean(series))

    # Window sizes: log-spaced
    # min_s = 8 (need ≥2 segments minimum, each ≥4 points)
    # max_s = n//3 (leave room for at least 3 segments)
    min_s  = max(8, n // 12)
    max_s  = max(min_s + 1, n // 3)
    n_wins = min(16, max_s - min_s + 1)
    sizes  = np.unique(
        np.round(np.logspace(np.log10(min_s), np.log10(max_s), n_wins)).astype(int)
    )

    log_s, log_f = [], []
    for s in sizes:
        s = int(s)
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


def hurst_rs(series: np.ndarray) -> float:
    """Alias -> DFA (R/S replaced due to short-window bias)."""
    return hurst_dfa(series)


# =============================================================================
# Labels / colours
# =============================================================================

def hurst_label(h: float) -> str:
    if h >= H_STRONG_TREND: return "Strong Trend"
    elif h >= H_TRENDING:   return "Mild Trend"
    elif h >= H_RANDOM:     return "Random Walk"
    else:                   return "Mean-Reverting"


def hurst_regime_colour(h: float) -> str:
    if h >= H_TRENDING: return "#16a34a"
    elif h >= H_RANDOM: return "#d97706"
    else:               return "#dc2626"


# =============================================================================
# Multi-timeframe Hurst
# =============================================================================

def compute_hurst_velocity(returns: pd.Series) -> float:
    """
    Compute H63 velocity — rate of change of the 63d Hurst exponent
    over the past VELOCITY_WINDOW days.

    Methodology:
      - Compute H63 at current point
      - Compute H63 at VELOCITY_WINDOW days ago
      - Velocity = (H_now - H_past) / H_past  (normalised rate of change)
      - Clipped to [-1, 1] and normalised to [0, 1] for scoring

    Positive velocity = regime accelerating into trend (buy signal)
    Negative velocity = regime decelerating (caution)
    """
    s = returns.dropna()
    n = len(s)

    min_needed = MEDIUM_WINDOW + VELOCITY_WINDOW
    if n < min_needed:
        return 0.0   # no velocity — insufficient history

    h_now  = hurst_dfa(s.values[-MEDIUM_WINDOW:])
    h_past = hurst_dfa(s.values[-(MEDIUM_WINDOW + VELOCITY_WINDOW):-VELOCITY_WINDOW])

    if h_past < 1e-6:
        return 0.0

    velocity = float(np.clip((h_now - h_past) / max(h_past, 0.1), -1.0, 1.0))
    return velocity


def velocity_label(v: float) -> str:
    if v >= 0.15:  return "Accelerating ↑"
    elif v >= 0.0: return "Stable →"
    elif v >= -0.15: return "Decelerating ↓"
    else:          return "Reversing ↓↓"


def velocity_colour(v: float) -> str:
    if v >= 0.15:    return "#16a34a"   # green
    elif v >= 0.0:   return "#65a30d"   # light green
    elif v >= -0.15: return "#d97706"   # amber
    else:            return "#dc2626"   # red


def compute_mtf_hurst(returns: pd.Series, date: Optional[pd.Timestamp] = None) -> dict:
    """
    Compute DFA Hurst at MEDIUM (63d) and LONG (126d) windows,
    plus H63 velocity (replaces unreliable SHORT/42d window).
    """
    s = returns.dropna()
    n = len(s)

    h_medium  = hurst_dfa(s.values[-MEDIUM_WINDOW:]) if n >= MEDIUM_WINDOW else 0.5
    h_long    = hurst_dfa(s.values[-LONG_WINDOW:])   if n >= LONG_WINDOW   else 0.5
    h_velocity = compute_hurst_velocity(s)

    # MTF score: based on medium + long, velocity as tiebreaker/boost
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
        "h_short":        h_velocity,    # repurposed slot: velocity in [-1,1]
        "h_medium":       h_medium,
        "h_long":         h_long,
        "h_velocity":     h_velocity,
        "mtf_score":      mtf_score,
        "mtf_strong":     med_trending,
        "trending_count": trending,
        "label_short":    velocity_label(h_velocity),   # velocity label
        "label_medium":   hurst_label(h_medium),
        "label_long":     hurst_label(h_long),
    }


def compute_all_mtf(returns_df: pd.DataFrame) -> dict:
    results = {}
    for ticker in ETF_UNIVERSE:
        if ticker not in returns_df.columns:
            continue
        results[ticker] = compute_mtf_hurst(returns_df[ticker])
    return results


# =============================================================================
# Rolling MTF history
# =============================================================================

def build_mtf_history(returns_df: pd.DataFrame, step: int = 5) -> pd.DataFrame:
    """Build rolling DFA Hurst history for all ETFs (every `step` days)."""
    dates    = returns_df.index
    n        = len(dates)
    min_rows = LONG_WINDOW + 1
    records  = []

    for i in range(min_rows, n, step):
        row    = {"date": dates[i]}
        slice_ = returns_df.iloc[:i+1]
        for ticker in ETF_UNIVERSE:
            if ticker not in slice_.columns:
                continue
            s = slice_[ticker].dropna()
            row[f"{ticker}_h_velocity"] = compute_hurst_velocity(s)
            row[f"{ticker}_h_short"]    = row[f"{ticker}_h_velocity"]  # alias for compat
            row[f"{ticker}_h_medium"]   = hurst_dfa(s.values[-MEDIUM_WINDOW:]) if len(s) >= MEDIUM_WINDOW else np.nan
            row[f"{ticker}_h_long"]     = hurst_dfa(s.values[-LONG_WINDOW:])   if len(s) >= LONG_WINDOW   else np.nan
        records.append(row)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records).set_index("date")


# =============================================================================
# Divergence scores
# =============================================================================

def compute_divergence_scores(
    mtf_today:   dict,
    mtf_history: pd.DataFrame,
    lookback:    int = 504,
) -> dict:
    """
    3-part divergence score per ETF:
      (a) H risen most vs own 6m ago  (momentum of regime)
      (b) H furthest above own 2yr baseline  (absolute persistence)
      (c) H recently crossed above 1yr mean  (fresh transition)
    """
    scores = {}

    if mtf_history is None or mtf_history.empty:
        for ticker in ETF_UNIVERSE:
            if ticker not in mtf_today:
                continue
            h = mtf_today[ticker]["h_medium"]
            scores[ticker] = {
                "div_a": 0.0, "div_b": max(0.0, h - 0.5),
                "div_c": 0.0, "div_score": max(0.0, h - 0.5),
            }
        return scores

    hist = mtf_history.tail(lookback)

    for ticker in ETF_UNIVERSE:
        if ticker not in mtf_today:
            continue
        col_m  = f"{ticker}_h_medium"
        col_l  = f"{ticker}_h_long"
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


# =============================================================================
# Cross-asset sync
# =============================================================================

def compute_sync_score(mtf_today: dict) -> dict:
    h_vals = {t: mtf_today[t]["h_medium"] for t in ETF_UNIVERSE if t in mtf_today}
    if not h_vals:
        return {"sync_level": 0.5, "h_mean": 0.5, "h_std": 0.0, "scores": {t: 0.0 for t in ETF_UNIVERSE}}

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


# =============================================================================
# Momentum overlay
# =============================================================================

def compute_momentum_scores(
    returns_df: pd.DataFrame,
    w3m:        float = 0.5,
) -> dict:
    """
    Cross-sectional rank-based momentum scores per ETF.
    w3m: weight on 3m momentum (1-w3m = weight on 6m momentum).
    Returns dict: ticker -> normalised momentum score in [0, 1].
    """
    etfs    = [t for t in ETF_UNIVERSE if t in returns_df.columns]
    if not etfs:
        return {t: 0.5 for t in ETF_UNIVERSE}
    n       = len(returns_df)

    ret_3m = {}
    ret_6m = {}
    for ticker in etfs:
        s = returns_df[ticker].dropna()
        ret_3m[ticker] = float(np.sum(s.values[-MOM_3M_WINDOW:])) if len(s) >= MOM_3M_WINDOW else 0.0
        ret_6m[ticker] = float(np.sum(s.values[-MOM_6M_WINDOW:])) if len(s) >= MOM_6M_WINDOW else 0.0

    # Cross-sectional rank normalisation -> [0, 1]
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
) -> tuple[float, float]:
    """
    Grid search over (mom_weight, w3m) to find the blend that maximised
    Sharpe on the trailing train_window in-sample.
    Returns (best_mom_weight, best_w3m).
    """
    etfs  = [t for t in ETF_UNIVERSE if t in returns_df.columns]
    if not etfs:
        log.warning("optimise_momentum_weights: no ETF columns found, returning defaults")
        return 0.20, 0.50
    hist  = returns_df.tail(train_window)

    best_sharpe = -np.inf
    best_mw, best_w3 = 0.20, 0.50   # defaults

    for mom_w in MOM_WEIGHT_GRID:
        for w3m in MOM_3M_GRID:
            daily_rets = []
            prev_sig   = None

            for i in range(MOM_6M_WINDOW, len(hist) - 1):
                sub = hist.iloc[:i]
                # Quick HRC scores (use pre-computed mtf_scores from hrc_scores dict)
                mom_scores = compute_momentum_scores(sub, w3m=w3m)
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
) -> dict:
    results  = {}
    sync_dev = sync.get("scores", {})

    for ticker in ETF_UNIVERSE:
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
    """
    Pick top ETF by blended HRC + momentum score.
    If mom_scores is None, uses pure HRC.
    """
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
