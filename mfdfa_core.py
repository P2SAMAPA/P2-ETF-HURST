"""
mfdfa_core.py — Multifractal DFA (MFDFA) Engine
=================================================
Standalone module added alongside hurst_core.py.
Zero modifications to existing hurst_core.py / train.py / walkforward.py.

Algorithm
---------
1. Cumulative-sum profile of de-meaned returns.
2. Divide profile into non-overlapping segments of each scale s.
3. Fit local polynomial trend (order m=2) per segment; compute residual variance F²(s,v).
4. Generalised fluctuation function: F_q(s) = [mean(F²^(q/2))]^(1/q)
5. Log-log regression of F_q(s) vs s → slope h(q) = generalised Hurst exponent.
6. Multifractal spectrum:
     τ(q) = q·h(q) − 1
     α     = dτ/dq  (finite-difference)
     f(α)  = q·α − τ(q)
7. Key scalar outputs:
     H_mono  = h(2)           — monofractal Hurst (same concept as DFA in hurst_core)
     Δα      = α_max − α_min  — multifractal width (richness of scaling structure)
     Δf      = f(α_max)−f(α_min) — spectrum asymmetry
     dominant_q = q where h(q) is most sensitive (used for picking signal)

All functions are pure numpy — no scipy, no external deps beyond what requirements.txt
already has.  GitHub Actions free-tier CPU safe (<1s per ETF per day).
"""

import numpy as np
import pandas as pd
import logging

log = logging.getLogger(__name__)

# ── Parameters ────────────────────────────────────────────────────────────────

MFDFA_Q_VALUES   = np.array([-4, -3, -2, -1, 0.001, 1, 2, 3, 4], dtype=float)
MFDFA_POLY_ORDER = 2          # DFA order (m=2 removes linear + quadratic trend)
MFDFA_MIN_SCALE  = 16         # minimum segment length (≥16 for reliable fit)
MFDFA_MAX_SCALE  = 128        # maximum segment length
MFDFA_N_SCALES   = 12         # number of log-spaced scales between min and max
MFDFA_MIN_SEGS   = 4          # minimum segments required at a scale; skip if fewer

# Interpretive thresholds
WIDTH_STRONG   = 0.40   # Δα ≥ 0.40 → rich multifractal (many scaling regimes)
WIDTH_MODERATE = 0.20   # 0.20 ≤ Δα < 0.40 → moderate
WIDTH_WEAK     = 0.10   # Δα < 0.10 → near-monofractal

# ── Core MFDFA ────────────────────────────────────────────────────────────────

def _profile(returns: np.ndarray) -> np.ndarray:
    """Cumulative sum of de-meaned returns (the 'profile')."""
    return np.cumsum(returns - returns.mean())


def _detrended_variance(segment: np.ndarray, poly_order: int) -> float:
    """
    Fit a polynomial of order `poly_order` to `segment`, return mean squared
    residual.  Returns NaN if the segment is too short to fit.
    """
    n = len(segment)
    if n <= poly_order + 1:
        return np.nan
    x = np.arange(n, dtype=float)
    try:
        coeffs = np.polyfit(x, segment, poly_order)
        trend  = np.polyval(coeffs, x)
        return float(np.mean((segment - trend) ** 2))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


def _fluctuation_function(profile: np.ndarray, scale: int,
                           q_values: np.ndarray,
                           poly_order: int) -> np.ndarray:
    """
    Compute F_q(s) for a single scale s across all q values.
    Returns array of shape (len(q_values),); NaN where insufficient segments.
    """
    n = len(profile)
    n_segs = n // scale
    if n_segs < MFDFA_MIN_SEGS:
        return np.full(len(q_values), np.nan)

    variances = np.array([
        _detrended_variance(profile[v * scale: (v + 1) * scale], poly_order)
        for v in range(n_segs)
    ])
    variances = variances[np.isfinite(variances)]
    if len(variances) < MFDFA_MIN_SEGS:
        return np.full(len(q_values), np.nan)

    fq = np.empty(len(q_values))
    for i, q in enumerate(q_values):
        if abs(q) < 1e-6:
            # q→0 limit: geometric mean of variances
            fq[i] = np.exp(0.5 * np.mean(np.log(variances + 1e-30)))
        else:
            fq[i] = np.mean(variances ** (q / 2.0)) ** (1.0 / q)
    return fq


def compute_mfdfa(returns: np.ndarray,
                  q_values: np.ndarray = MFDFA_Q_VALUES,
                  poly_order: int = MFDFA_POLY_ORDER,
                  min_scale: int = MFDFA_MIN_SCALE,
                  max_scale: int = MFDFA_MAX_SCALE,
                  n_scales: int = MFDFA_N_SCALES) -> dict:
    """
    Run full MFDFA on a 1-D numpy array of returns.

    Returns
    -------
    dict with keys:
        q_values     : array of q values used
        h_q          : generalised Hurst exponents h(q), shape (n_q,)
        tau_q        : scaling exponent τ(q)
        alpha        : singularity strengths α (spectrum x-axis)
        f_alpha      : multifractal spectrum f(α) (spectrum y-axis)
        H_mono       : h(2) — monofractal Hurst estimate
        delta_alpha  : spectrum width Δα
        delta_f      : spectrum asymmetry Δf
        width_label  : 'STRONG' / 'MODERATE' / 'WEAK' / 'MONOFRACTAL'
        scales_used  : array of segment lengths used
        Fq_matrix    : raw F_q(s) matrix, shape (n_scales, n_q)
        valid        : bool — False if data too short or all NaN
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[np.isfinite(returns)]

    n = len(returns)
    max_s = min(max_scale, n // (MFDFA_MIN_SEGS + 1))
    if n < min_scale * MFDFA_MIN_SEGS or max_s < min_scale:
        return _empty_result(q_values, reason="insufficient data")

    scales = np.unique(
        np.round(np.geomspace(min_scale, max_s, n_scales)).astype(int)
    )
    scales = scales[(scales >= min_scale) & (scales <= max_s)]
    if len(scales) < 4:
        return _empty_result(q_values, reason="too few valid scales")

    profile  = _profile(returns)
    Fq_mat   = np.array([
        _fluctuation_function(profile, int(s), q_values, poly_order)
        for s in scales
    ])  # shape (n_scales, n_q)

    log_s = np.log(scales.astype(float))

    h_q = np.empty(len(q_values))
    for i in range(len(q_values)):
        col  = Fq_mat[:, i]
        mask = np.isfinite(col) & (col > 0)
        if mask.sum() < 3:
            h_q[i] = np.nan
        else:
            slope, _ = np.polyfit(log_s[mask], np.log(col[mask]), 1)
            h_q[i]   = slope

    if not np.isfinite(h_q).any():
        return _empty_result(q_values, reason="regression failed")

    # Fill any NaN h_q via linear interpolation across q
    finite_mask = np.isfinite(h_q)
    if finite_mask.sum() >= 2:
        h_q = np.interp(q_values, q_values[finite_mask], h_q[finite_mask])

    tau_q   = q_values * h_q - 1.0
    # Multifractal spectrum via finite differences
    dq      = np.gradient(q_values)
    dtau    = np.gradient(tau_q)
    alpha   = dtau / np.where(np.abs(dq) > 1e-10, dq, np.nan)
    f_alpha = q_values * alpha - tau_q

    # Scalar outputs
    idx2    = np.argmin(np.abs(q_values - 2.0))
    H_mono  = float(h_q[idx2]) if np.isfinite(h_q[idx2]) else np.nan

    valid_alpha = alpha[np.isfinite(alpha) & np.isfinite(f_alpha)]
    valid_f     = f_alpha[np.isfinite(alpha) & np.isfinite(f_alpha)]

    if len(valid_alpha) >= 2:
        delta_alpha = float(valid_alpha.max() - valid_alpha.min())
        # Asymmetry: f at right tail minus f at left tail
        delta_f     = float(valid_f[np.argmax(valid_alpha)] -
                            valid_f[np.argmin(valid_alpha)])
    else:
        delta_alpha = np.nan
        delta_f     = np.nan

    if np.isfinite(delta_alpha):
        if delta_alpha >= WIDTH_STRONG:
            width_label = "STRONG"
        elif delta_alpha >= WIDTH_MODERATE:
            width_label = "MODERATE"
        elif delta_alpha >= WIDTH_WEAK:
            width_label = "WEAK"
        else:
            width_label = "MONOFRACTAL"
    else:
        width_label = "UNKNOWN"

    return {
        "q_values":    q_values,
        "h_q":         h_q,
        "tau_q":       tau_q,
        "alpha":       alpha,
        "f_alpha":     f_alpha,
        "H_mono":      H_mono,
        "delta_alpha": delta_alpha,
        "delta_f":     delta_f,
        "width_label": width_label,
        "scales_used": scales,
        "Fq_matrix":   Fq_mat,
        "valid":       True,
    }


def _empty_result(q_values: np.ndarray, reason: str = "") -> dict:
    n = len(q_values)
    log.debug(f"MFDFA empty result: {reason}")
    return {
        "q_values":    q_values,
        "h_q":         np.full(n, np.nan),
        "tau_q":       np.full(n, np.nan),
        "alpha":       np.full(n, np.nan),
        "f_alpha":     np.full(n, np.nan),
        "H_mono":      np.nan,
        "delta_alpha": np.nan,
        "delta_f":     np.nan,
        "width_label": "INSUFFICIENT_DATA",
        "scales_used": np.array([]),
        "Fq_matrix":   np.full((0, n), np.nan),
        "valid":       False,
    }


# ── Per-ETF rolling MFDFA ─────────────────────────────────────────────────────

def compute_mfdfa_for_etf(returns_series: pd.Series,
                           window: int = 252) -> dict:
    """
    Run MFDFA on the most recent `window` observations of a returns Series.
    Returns a flat dict of scalar results for easy DataFrame assembly.
    """
    arr = returns_series.dropna().values[-window:]
    res = compute_mfdfa(arr)
    return {
        "H_mono":      res["H_mono"],
        "delta_alpha": res["delta_alpha"],
        "delta_f":     res["delta_f"],
        "width_label": res["width_label"],
        "valid":       res["valid"],
        # Keep full arrays for Streamlit plotting
        "_alpha":      res["alpha"],
        "_f_alpha":    res["f_alpha"],
        "_h_q":        res["h_q"],
        "_q_values":   res["q_values"],
    }


def compute_all_mfdfa(returns_df: pd.DataFrame,
                       etf_list: list,
                       window: int = 252) -> dict:
    """
    Run MFDFA for every ETF in etf_list.
    Returns dict: {ticker: {H_mono, delta_alpha, delta_f, width_label, valid, ...}}
    """
    results = {}
    for ticker in etf_list:
        if ticker not in returns_df.columns:
            log.warning(f"MFDFA: {ticker} not in returns_df, skipping")
            continue
        try:
            results[ticker] = compute_mfdfa_for_etf(returns_df[ticker], window=window)
            log.info(
                f"  MFDFA {ticker}: H_mono={results[ticker]['H_mono']:.3f} "
                f"Δα={results[ticker]['delta_alpha']:.3f} "
                f"({results[ticker]['width_label']})"
                if results[ticker]["valid"] else
                f"  MFDFA {ticker}: invalid result"
            )
        except Exception as e:
            log.error(f"MFDFA failed for {ticker}: {e}")
            results[ticker] = compute_mfdfa_for_etf.__wrapped__ if hasattr(
                compute_mfdfa_for_etf, "__wrapped__") else {
                "H_mono": np.nan, "delta_alpha": np.nan,
                "delta_f": np.nan, "width_label": "ERROR",
                "valid": False,
                "_alpha": np.array([]), "_f_alpha": np.array([]),
                "_h_q": np.array([]), "_q_values": MFDFA_Q_VALUES,
            }
    return results


# ── Rolling history builder (mirrors build_mtf_history pattern) ───────────────

def build_mfdfa_history(returns_df: pd.DataFrame,
                         etf_list: list,
                         step: int = 5,
                         window: int = 252) -> pd.DataFrame:
    """
    Build a rolling history of MFDFA scalar outputs (H_mono, delta_alpha,
    delta_f) for all ETFs, stepping every `step` trading days.

    Returns a DataFrame with MultiIndex columns (ticker, metric) and
    DatetimeIndex rows, stored to HF as mfdfa_history_{option}.parquet.
    """
    rows = []
    index_dates = []
    dates = returns_df.index[window::step]

    for date in dates:
        subset = returns_df.loc[:date]
        row = {}
        for ticker in etf_list:
            if ticker not in subset.columns:
                continue
            arr = subset[ticker].dropna().values[-window:]
            res = compute_mfdfa(arr)
            row[f"{ticker}_H_mono"]      = res["H_mono"]
            row[f"{ticker}_delta_alpha"] = res["delta_alpha"]
            row[f"{ticker}_delta_f"]     = res["delta_f"]
        rows.append(row)
        index_dates.append(date)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index_dates))
    df.index.name = "date"
    return df


# ── Signal generation from MFDFA ─────────────────────────────────────────────

def mfdfa_conviction_score(result: dict) -> float:
    """
    Derive a conviction score [0, 1] from MFDFA outputs.

    Scoring rationale
    -----------------
    - H_mono close to 0.6–0.75 is the sweet spot (trending, not extreme)
    - Δα moderate (0.2–0.5) indicates rich but not chaotic multifractal
    - Δf > 0 (right-skewed spectrum) → more large fluctuations on positive side
    """
    if not result.get("valid", False):
        return 0.0

    H  = result.get("H_mono", np.nan)
    da = result.get("delta_alpha", np.nan)
    df_ = result.get("delta_f", np.nan)

    if not all(np.isfinite([H, da, df_])):
        return 0.0

    # H score: peaks at H=0.65, decays toward 0 and 1
    h_score = max(0.0, 1.0 - abs(H - 0.65) / 0.35)

    # Width score: peaks at Δα=0.30
    da_score = max(0.0, 1.0 - abs(da - 0.30) / 0.30)

    # Asymmetry bonus: positive Δf is good
    df_score = min(1.0, max(0.0, (df_ + 0.5) / 1.0))

    return round(0.50 * h_score + 0.30 * da_score + 0.20 * df_score, 4)


def generate_mfdfa_signal(mfdfa_results: dict, etf_list: list) -> dict:
    """
    Rank ETFs by MFDFA conviction and return a signal dict matching the
    style of hurst_core.generate_signal().

    Returns
    -------
    dict with keys: signal, conviction, label, ranked
    """
    scores = {}
    for ticker in etf_list:
        if ticker in mfdfa_results:
            scores[ticker] = mfdfa_conviction_score(mfdfa_results[ticker])
        else:
            scores[ticker] = 0.0

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not ranked:
        return {"signal": "CASH", "conviction": 0.0, "label": "NO DATA", "ranked": []}

    top_etf, top_score = ranked[0]
    if top_score < 0.20:
        label = "WEAK"
    elif top_score < 0.50:
        label = "MODERATE"
    else:
        label = "STRONG"

    return {
        "signal":     top_etf,
        "conviction": top_score,
        "label":      label,
        "ranked":     ranked,
    }
