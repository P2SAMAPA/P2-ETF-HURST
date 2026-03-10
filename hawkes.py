"""
hawkes.py — P2-ETF-HAWKES
===========================
Core Hawkes process implementation.

Supports:
  - Three event definitions (tested and compared by predictive accuracy)
  - Two kernels: Exponential (Markov) + Power Law (long memory), best by AIC
  - Univariate fit per ETF + multivariate cross-excitation matrix
  - Recursive intensity computation (O(n) via Markov property for exp kernel)
  - MLE parameter fitting via scipy.optimize

Mathematical reference:
  Hawkes (1971) — "Spectra of some self-exciting and mutually exciting
  point processes", Biometrika 58(1).

Author: P2SAMAPA
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal
from scipy.optimize import minimize
from scipy.special import gammaln

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
ETF_UNIVERSE = ["TLT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
KERNEL_TYPES = Literal["exponential", "powerlaw"]

EVENT_DEFINITIONS = {
    "return_only":    "│return│ > 1σ (rolling 63d)",
    "volume_only":    "Volume > 20d rolling mean",
    "combined":       "│return│ > 1σ AND Volume > 20d mean (strictest)",
}


# ── Event detection ───────────────────────────────────────────────────────────

def detect_events(
    returns: pd.Series,
    volume:  pd.Series | None = None,
    method:  str = "combined",
    ret_z_thresh: float = 1.0,
    vol_lookback: int   = 20,
    ret_lookback: int   = 63,
) -> pd.Series:
    """
    Detect significant market events for a single ETF.

    Parameters
    ----------
    returns      : daily return series (float)
    volume       : daily volume series (float), required for volume/combined methods
    method       : "return_only" | "volume_only" | "combined"
    ret_z_thresh : |return| Z-threshold (default 1.0 std dev)
    vol_lookback : rolling window for volume baseline (days)
    ret_lookback : rolling window for return std dev (days)

    Returns
    -------
    pd.Series of bool — True on event days
    """
    abs_ret = returns.abs()
    ret_std = abs_ret.rolling(ret_lookback, min_periods=10).std()
    ret_flag = abs_ret > (ret_z_thresh * ret_std)

    if method == "return_only":
        return ret_flag.fillna(False)

    if volume is None:
        log.warning("Volume not provided — falling back to return_only events")
        return ret_flag.fillna(False)

    vol_mean  = volume.rolling(vol_lookback, min_periods=5).mean()
    vol_flag  = volume > vol_mean

    if method == "volume_only":
        return vol_flag.fillna(False)

    # combined
    return (ret_flag & vol_flag).fillna(False)


def get_event_times(event_series: pd.Series) -> np.ndarray:
    """Convert boolean event series to array of event indices (day numbers)."""
    return np.where(event_series.values)[0].astype(float)


# ── Kernel functions ──────────────────────────────────────────────────────────

def exp_kernel(t: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """h(t) = alpha * exp(-beta * t)"""
    return alpha * np.exp(-beta * t)


def powerlaw_kernel(t: np.ndarray, k: float, c: float, theta: float) -> np.ndarray:
    """h(t) = k / (t + c)^(1 + theta)"""
    return k / np.power(t + c, 1.0 + theta)


# ── Log-likelihood ────────────────────────────────────────────────────────────

def _log_likelihood_exp(params: np.ndarray, event_times: np.ndarray, T: float) -> float:
    """
    Negative log-likelihood for univariate Hawkes with exponential kernel.
    Uses recursive O(n) computation via Markov property.

    params = [mu, alpha, beta]
    """
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha >= beta:
        return 1e10

    n = len(event_times)
    if n == 0:
        return mu * T   # no events → just baseline integral

    # Recursive intensity computation
    # R_i = sum_{j<i} alpha * exp(-beta*(t_i - t_j))
    # = alpha + exp(-beta*(t_i - t_{i-1})) * R_{i-1}  when beta > 0
    R = np.zeros(n)
    for i in range(1, n):
        dt   = event_times[i] - event_times[i - 1]
        R[i] = np.exp(-beta * dt) * (R[i - 1] + alpha)

    lam = mu + R   # intensity at each event time

    # Log-likelihood:
    # sum log(lambda*(t_i)) - integral_0^T lambda*(t) dt
    # integral = mu*T + sum_i alpha/beta * (1 - exp(-beta*(T - t_i)))
    ll_events  = np.sum(np.log(lam + 1e-12))
    integral   = mu * T + np.sum((alpha / beta) * (1.0 - np.exp(-beta * (T - event_times))))
    return -(ll_events - integral)


def _log_likelihood_pl(params: np.ndarray, event_times: np.ndarray, T: float) -> float:
    """
    Negative log-likelihood for univariate Hawkes with power-law kernel.
    O(n^2) — slower but captures longer memory.

    params = [mu, k, c, theta]
    """
    mu, k, c, theta = params
    if mu <= 0 or k <= 0 or c <= 0 or theta <= 0:
        return 1e10
    # Stability: ||h||_1 = k * c^(-theta) / theta < 1
    norm_h = k * (c ** (-theta)) / theta
    if norm_h >= 1.0:
        return 1e10

    n = len(event_times)
    if n == 0:
        return mu * T

    lam = np.zeros(n)
    for i in range(n):
        lam[i] = mu
        for j in range(i):
            dt = event_times[i] - event_times[j]
            lam[i] += powerlaw_kernel(np.array([dt]), k, c, theta)[0]

    ll_events = np.sum(np.log(lam + 1e-12))

    # Approximate integral via trapezoidal sum over event times
    integral = mu * T
    for j in range(n):
        # integral from t_j to T of h(t - t_j) dt
        # = k/theta * ((T - t_j + c)^(-theta) * (-1) ... closed form:
        # = k/theta * [c^(-theta) - (T - t_j + c)^(-theta)]
        dt_end    = T - event_times[j]
        integral += (k / theta) * (c ** (-theta) - (dt_end + c) ** (-theta))

    return -(ll_events - integral)


# ── Parameter fitting ─────────────────────────────────────────────────────────

@dataclass
class HawkesParams:
    ticker:     str
    kernel:     str
    mu:         float
    alpha:      float   = 0.0    # exponential only
    beta:       float   = 0.0    # exponential only
    k:          float   = 0.0    # power-law only
    c:          float   = 0.0    # power-law only
    theta:      float   = 0.0    # power-law only
    aic:        float   = 0.0
    n_events:   int     = 0
    branching:  float   = 0.0    # ||h||_1 — fraction of events that are offspring
    event_def:  str     = "combined"

    def to_dict(self) -> dict:
        return {
            "ticker":    self.ticker,
            "kernel":    self.kernel,
            "mu":        round(self.mu,    6),
            "alpha":     round(self.alpha, 6),
            "beta":      round(self.beta,  6),
            "k":         round(self.k,     6),
            "c":         round(self.c,     6),
            "theta":     round(self.theta, 6),
            "aic":       round(self.aic,   4),
            "n_events":  self.n_events,
            "branching": round(self.branching, 4),
            "event_def": self.event_def,
        }


def fit_exponential(
    event_times: np.ndarray,
    T: float,
    ticker: str = "",
    event_def: str = "combined",
    n_starts: int = 8,
) -> HawkesParams:
    """Fit exponential kernel Hawkes via MLE with multiple random starts."""
    best_nll = np.inf
    best_res = None

    for _ in range(n_starts):
        mu0    = np.random.uniform(0.001, 0.1)
        alpha0 = np.random.uniform(0.01,  0.5)
        beta0  = np.random.uniform(0.1,   2.0)
        x0     = [mu0, alpha0, beta0]
        bounds = [(1e-6, 10), (1e-6, 10), (1e-6, 50)]
        try:
            res = minimize(
                _log_likelihood_exp,
                x0=x0,
                args=(event_times, T),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_res = res
        except Exception:
            pass

    if best_res is None or not best_res.success:
        log.warning(f"Exp kernel fit did not converge for {ticker}")
        mu, alpha, beta = 0.01, 0.1, 1.0
    else:
        mu, alpha, beta = best_res.x

    n_params  = 3
    aic       = 2 * n_params + 2 * best_nll if best_res else 1e9
    branching = alpha / beta if beta > 0 else 0.0

    return HawkesParams(
        ticker=ticker, kernel="exponential",
        mu=mu, alpha=alpha, beta=beta,
        aic=aic, n_events=len(event_times),
        branching=branching, event_def=event_def,
    )


def fit_powerlaw(
    event_times: np.ndarray,
    T: float,
    ticker: str = "",
    event_def: str = "combined",
    n_starts: int = 6,
) -> HawkesParams:
    """Fit power-law kernel Hawkes via MLE."""
    best_nll = np.inf
    best_res = None

    for _ in range(n_starts):
        mu0    = np.random.uniform(0.001, 0.05)
        k0     = np.random.uniform(0.01,  0.3)
        c0     = np.random.uniform(0.1,   2.0)
        th0    = np.random.uniform(0.5,   2.0)
        x0     = [mu0, k0, c0, th0]
        bounds = [(1e-6, 5), (1e-6, 5), (1e-6, 10), (0.1, 5)]
        try:
            res = minimize(
                _log_likelihood_pl,
                x0=x0,
                args=(event_times, T),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if res.fun < best_nll:
                best_nll = res.fun
                best_res = res
        except Exception:
            pass

    if best_res is None or not best_res.success:
        log.warning(f"Power-law kernel fit did not converge for {ticker}")
        mu, k, c, theta = 0.01, 0.1, 1.0, 1.0
    else:
        mu, k, c, theta = best_res.x

    n_params  = 4
    aic       = 2 * n_params + 2 * best_nll if best_res else 1e9
    branching = k * (c ** (-theta)) / theta if theta > 0 else 0.0

    return HawkesParams(
        ticker=ticker, kernel="powerlaw",
        mu=mu, k=k, c=c, theta=theta,
        aic=aic, n_events=len(event_times),
        branching=branching, event_def=event_def,
    )


def fit_best_kernel(
    event_times: np.ndarray,
    T: float,
    ticker: str = "",
    event_def: str = "combined",
) -> HawkesParams:
    """Fit both kernels, return the one with lower AIC."""
    exp_params = fit_exponential(event_times, T, ticker, event_def)
    pl_params  = fit_powerlaw(event_times, T, ticker, event_def)
    log.info(f"{ticker} [{event_def}] — Exp AIC={exp_params.aic:.2f} | PL AIC={pl_params.aic:.2f}")
    return exp_params if exp_params.aic <= pl_params.aic else pl_params


# ── Intensity computation ─────────────────────────────────────────────────────

def compute_intensity_exp(
    params: HawkesParams,
    event_times: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """
    Compute λ*(t) at each eval_time using recursive Markov formula.
    O(n + m) where n = events, m = eval points.
    """
    mu, alpha, beta = params.mu, params.alpha, params.beta
    intensities = np.full(len(eval_times), mu)

    if len(event_times) == 0:
        return intensities

    for i, t in enumerate(eval_times):
        past = event_times[event_times < t]
        if len(past) == 0:
            continue
        intensities[i] = mu + np.sum(alpha * np.exp(-beta * (t - past)))

    return intensities


def compute_intensity_pl(
    params: HawkesParams,
    event_times: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Compute λ*(t) for power-law kernel."""
    mu, k, c, theta = params.mu, params.k, params.c, params.theta
    intensities = np.full(len(eval_times), mu)

    for i, t in enumerate(eval_times):
        past = event_times[event_times < t]
        if len(past) == 0:
            continue
        dts = t - past
        intensities[i] = mu + np.sum(powerlaw_kernel(dts, k, c, theta))

    return intensities


def compute_intensity(
    params: HawkesParams,
    event_times: np.ndarray,
    eval_times: np.ndarray,
) -> np.ndarray:
    """Dispatch to correct kernel."""
    if params.kernel == "exponential":
        return compute_intensity_exp(params, event_times, eval_times)
    else:
        return compute_intensity_pl(params, event_times, eval_times)


# ── Full pipeline per ETF ─────────────────────────────────────────────────────

def fit_etf(
    ticker:      str,
    returns:     pd.Series,
    volume:      pd.Series | None,
    event_def:   str = "combined",
) -> tuple[HawkesParams, np.ndarray, np.ndarray]:
    """
    Fit Hawkes model for a single ETF.

    Returns (params, event_times_array, intensity_array aligned to returns.index)
    """
    events      = detect_events(returns, volume, method=event_def)
    event_idx   = get_event_times(events)
    T           = float(len(returns))

    if len(event_idx) < 10:
        log.warning(f"{ticker}: only {len(event_idx)} events — results may be unreliable")

    params      = fit_best_kernel(event_idx, T, ticker, event_def)
    eval_times  = np.arange(len(returns), dtype=float)
    intensities = compute_intensity(params, event_idx, eval_times)

    return params, event_idx, intensities


def fit_all_etfs(
    returns_df: pd.DataFrame,
    volume_df:  pd.DataFrame | None,
    event_def:  str = "combined",
) -> dict:
    """
    Fit Hawkes for all ETFs in ETF_UNIVERSE.
    Returns dict: {ticker: {"params": HawkesParams, "intensity": np.ndarray, "events": np.ndarray}}
    """
    results = {}
    for ticker in ETF_UNIVERSE:
        if ticker not in returns_df.columns:
            log.warning(f"{ticker} not in returns_df — skipping")
            continue
        ret = returns_df[ticker].dropna()
        vol = volume_df[ticker].reindex(ret.index) if volume_df is not None and ticker in volume_df.columns else None
        try:
            params, event_idx, intensities = fit_etf(ticker, ret, vol, event_def)
            results[ticker] = {
                "params":    params,
                "intensity": intensities,
                "events":    event_idx,
            }
            log.info(f"{ticker} fitted — μ={params.mu:.4f} branching={params.branching:.3f} "
                     f"kernel={params.kernel} AIC={params.aic:.2f}")
        except Exception as e:
            log.error(f"Failed to fit {ticker}: {e}")
    return results


# ── Excitation ratio & signal ─────────────────────────────────────────────────

def compute_excitation_ratio(params: HawkesParams, current_intensity: float) -> float:
    """
    Excitation ratio = λ*(t) / μ
    > 1.0 means currently above baseline (self-exciting)
    = 1.0 means at baseline (no excitation)
    """
    if params.mu < 1e-9:
        return 1.0
    return current_intensity / params.mu


def get_signal(
    fit_results: dict,
    returns_df:  pd.DataFrame,
    volume_df:   pd.DataFrame | None = None,
    event_def:   str = "combined",
) -> dict:
    """
    Compute next-day signal from fitted Hawkes models.

    Returns dict with:
      - ranked ETFs by excitation ratio
      - top signal ETF
      - conviction score (0-1 normalised excitation ratio)
      - per-ETF excitation ratios
    """
    ratios = {}
    for ticker, res in fit_results.items():
        params    = res["params"]
        intensity = res["intensity"]
        current   = float(intensity[-1]) if len(intensity) > 0 else params.mu
        ratios[ticker] = compute_excitation_ratio(params, current)

    # Rank by excitation ratio descending
    ranked = sorted(ratios.items(), key=lambda x: -x[1])

    # Normalise to [0, 1] conviction score
    vals   = np.array([v for _, v in ratios.items()])
    min_v, max_v = vals.min(), vals.max()
    span   = max_v - min_v + 1e-9
    norm   = {t: (v - min_v) / span for t, v in ratios.items()}

    top_etf       = ranked[0][0]
    top_ratio     = ranked[0][1]
    conviction    = norm.get(top_etf, 0.0)

    return {
        "signal":        top_etf,
        "conviction":    round(conviction, 4),
        "excitation_ratios": {t: round(v, 4) for t, v in ratios.items()},
        "ranked":        [(t, round(v, 4)) for t, v in ranked],
        "top_ratio":     round(top_ratio, 4),
    }


# ── Cross-ETF excitation matrix ───────────────────────────────────────────────

def compute_cross_excitation_matrix(
    returns_df: pd.DataFrame,
    volume_df:  pd.DataFrame | None = None,
    event_def:  str = "combined",
) -> pd.DataFrame:
    """
    Approximate cross-ETF excitation: for each pair (i, j), compute the
    Pearson correlation between ETF i's event series and ETF j's lagged
    intensity increase.

    This is a computationally tractable proxy for the full multivariate
    Hawkes Γ matrix (which requires joint MLE over 36 parameters for 6 ETFs).

    Returns a 6x6 DataFrame where entry [i,j] = how much ETF j's activity
    predicts increased intensity in ETF i.
    """
    tickers = [t for t in ETF_UNIVERSE if t in returns_df.columns]
    n       = len(tickers)
    matrix  = pd.DataFrame(np.zeros((n, n)), index=tickers, columns=tickers)

    # Get event series per ETF
    event_series = {}
    for t in tickers:
        ret = returns_df[t].dropna()
        vol = volume_df[t].reindex(ret.index) if volume_df is not None and t in volume_df.columns else None
        ev  = detect_events(ret, vol, method=event_def)
        event_series[t] = ev.astype(float)

    # Correlate: does ETF j having an event today predict ETF i's return tomorrow?
    for i, ti in enumerate(tickers):
        for j, tj in enumerate(tickers):
            if ti == tj:
                matrix.loc[ti, tj] = 1.0
                continue
            try:
                # Lag ETF j events by 1 day, correlate with ETF i future |returns|
                ev_j   = event_series[tj].shift(1).dropna()
                abs_ri = returns_df[ti].abs().reindex(ev_j.index).dropna()
                common = ev_j.index.intersection(abs_ri.index)
                if len(common) < 30:
                    continue
                corr   = float(np.corrcoef(ev_j.loc[common], abs_ri.loc[common])[0, 1])
                matrix.loc[ti, tj] = round(corr, 4)
            except Exception:
                pass

    return matrix


# ── Intensity history (for plotting) ─────────────────────────────────────────

def build_intensity_history(
    fit_results: dict,
    index:       pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build a DataFrame of λ*(t) history for all fitted ETFs,
    indexed by date. Used for the Streamlit intensity chart.
    """
    cols = {}
    for ticker, res in fit_results.items():
        intensity = res["intensity"]
        n = min(len(intensity), len(index))
        cols[ticker] = intensity[:n]

    df = pd.DataFrame(cols, index=index[:max(len(v) for v in cols.values())])
    return df


# ── Best event definition selection ──────────────────────────────────────────

def select_best_event_def(
    returns_df: pd.DataFrame,
    volume_df:  pd.DataFrame | None,
    n_days_oos: int = 252,
) -> str:
    """
    Select best event definition by comparing average AIC across all ETFs.

    Fast approach: fit each event definition on the TRAINING set only
    (all data minus last n_days_oos). Lower average AIC = better model fit.
    This runs in ~1-2 minutes vs the previous rolling OOS loop (~25+ minutes).

    Returns the best event definition string.
    """
    cutoff    = len(returns_df) - n_days_oos
    train_ret = returns_df.iloc[:cutoff]
    train_vol = volume_df.iloc[:cutoff] if volume_df is not None else None

    results = {}
    for ev_def in EVENT_DEFINITIONS.keys():
        try:
            total_aic = 0.0
            n_fitted  = 0
            for ticker in ETF_UNIVERSE:
                if ticker not in train_ret.columns:
                    continue
                ret    = train_ret[ticker].dropna()
                vol    = train_vol[ticker].reindex(ret.index) if train_vol is not None and ticker in train_vol.columns else None
                events = detect_events(ret, vol, method=ev_def)
                ev_idx = get_event_times(events)
                T      = float(len(ret))
                if len(ev_idx) < 5:
                    continue
                # Fit exponential only for speed (power-law adds little info at selection stage)
                p = fit_exponential(ev_idx, T, ticker, ev_def, n_starts=3)
                total_aic += p.aic
                n_fitted  += 1
            avg_aic = total_aic / n_fitted if n_fitted > 0 else 1e9
            results[ev_def] = avg_aic
            log.info(f"Event def [{ev_def}]: avg AIC = {avg_aic:.2f} across {n_fitted} ETFs")
        except Exception as e:
            log.error(f"Event def [{ev_def}] selection failed: {e}")
            results[ev_def] = 1e9

    if not results:
        return "combined"

    # Lower AIC = better
    best = min(results, key=results.get)
    log.info(f"Best event definition: {best} (avg AIC={results[best]:.2f})")
    return best
