"""
app.py — P2-ETF-HAWKES
========================
Streamlit UI for the Hawkes Process ETF Rotation Model.

Tabs:
  📊 Option A — Hawkes Only
  📈 Option B — Hawkes + Hurst
  ℹ️  About

UI follows the same pattern as P2-ETF-RNN-LSTM and P2-ETF-REGIME-PREDICTOR:
  - White hero banner with coloured accent border
  - plotly_white throughout
  - Sidebar controls
  - Force Retrain + Refresh buttons

Author: P2SAMAPA
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF Hawkes",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from data_manager import (
        load_ohlcv_from_hf, load_signals_from_hf,
        load_intensity_history_from_hf, load_params_from_hf,
        load_hurst_history_from_hf, load_cross_excitation_from_hf,
        get_returns, get_volume, ETF_UNIVERSE, BENCHMARKS,
    )
    from hawkes import EVENT_DEFINITIONS
    from hurst import hurst_label, hurst_colour
    from strategy import (
        generate_signal_option_a, generate_signal_option_b,
        conviction_colour, etf_colour,
        next_trading_day_from_today, calculate_metrics,
        calculate_benchmark_metrics,
    )
except Exception as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    margin=dict(l=0, r=0, t=30, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#eeeeee"),
    hovermode="x unified",
)

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_ohlcv():
    return load_ohlcv_from_hf()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_signals():
    return load_signals_from_hf()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_intensity():
    return load_intensity_history_from_hf()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_params():
    return load_params_from_hf()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_hurst():
    return load_hurst_history_from_hf()

@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_cross_excitation():
    return load_cross_excitation_from_hf()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=60)
    st.title("P2-ETF Hawkes")
    st.caption("Self-Exciting Point Process ETF Rotation")

    now_est = datetime.now()
    st.write(f"🕒 **UTC:** {now_est.strftime('%a %b %d, %H:%M')}")
    st.divider()

    st.header("⚙️ Configuration")
    conviction_gate = st.slider(
        "Min Conviction Gate", min_value=0.1, max_value=0.8,
        value=0.3, step=0.05,
        help="Minimum conviction score to take a position (below = CASH)"
    )
    fee_bps = st.number_input(
        "Transaction Fee (bps)", min_value=0, max_value=50,
        value=5, step=1,
    )
    st.divider()

    st.subheader("📊 Display")
    benchmark = st.selectbox("Benchmark", ["SPY", "AGG", "None"])
    show_intensity = st.toggle("Show intensity history chart", value=True)
    st.divider()

    run_btn     = st.button("🚀 Run Model", type="primary", use_container_width=True)
    refresh_btn = st.button("🔄 Force Data Refresh", use_container_width=True)

    if refresh_btn:
        with st.spinner("🔄 Triggering pipeline via GitHub Actions..."):
            try:
                import requests as _req
                gh_token = os.environ.get("GH_PAT", "")
                gh_repo  = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-HAWKES")
                if not gh_token:
                    raise ValueError("GH_PAT secret not set in Streamlit")
                resp = _req.post(
                    f"https://api.github.com/repos/{gh_repo}/actions/workflows/daily.yml/dispatches",
                    headers={
                        "Authorization": f"Bearer {gh_token}",
                        "Accept": "application/vnd.github+json",
                    },
                    json={"ref": "main"},
                    timeout=15,
                )
                if resp.status_code == 204:
                    st.cache_data.clear()
                    st.success(
                        "✅ Pipeline triggered — check back in ~10 minutes, "
                        "then click **Run Model**."
                    )
                else:
                    st.error(f"❌ GitHub API returned {resp.status_code}")
            except Exception as ex:
                st.error(f"❌ Trigger failed: {ex}")

# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🔥 P2-ETF Hawkes Process Rotation Model")
st.caption(
    "Self-exciting point process · Exponential & Power-Law kernels · "
    "ETFs: TLT · LQD · HYG · VNQ · GLD · SLV"
)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_a, tab_b, tab_about = st.tabs([
    "📊 Option A — Hawkes Only",
    "📈 Option B — Hawkes + Hurst",
    "ℹ️ About",
])

# ══════════════════════════════════════════════════════════════════════════════
# ABOUT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    ## 🔥 P2-ETF Hawkes Process Rotation Model

    ### What is a Hawkes Process?
    A Hawkes process is a **self-exciting point process** — a mathematical model
    where past events increase the likelihood of future events. In financial markets,
    this captures the well-known phenomenon that bursts of trading activity tend to
    cluster: a large move in one ETF increases the probability of further large moves
    shortly after.

    Unlike the standard Poisson process (which assumes events arrive independently),
    the Hawkes process intensity is:

    **λ*(t) = μ + Σᵢ h(t − Tᵢ)**

    where μ is the baseline rate, and h(·) is the excitation kernel that decays
    over time after each event.

    ### Event Definitions (all three tested, best selected automatically)
    | Method | Definition |
    |--------|-----------|
    | Return only | \|return\| > 1σ (rolling 63d) |
    | Volume only | Volume > 20d rolling mean |
    | Combined | Both conditions simultaneously (strictest) |

    ### Kernels (both fitted, best by AIC)
    | Kernel | Formula | Property |
    |--------|---------|---------|
    | Exponential | h(t) = α·e^(−βt) | Markov, O(n) fast |
    | Power Law | h(t) = k/(t+c)^(1+θ) | Long memory, slower decay |

    ### Option A — Hawkes Only
    Signal = ETF with highest excitation ratio **λ*(t)/μ**
    Conviction = normalised excitation ratio across all 6 ETFs

    ### Option B — Hawkes + Hurst
    Combined conviction = **65% Hawkes excitation + 35% Hurst persistence**

    High λ*(t)/μ AND H > 0.5 = double confirmation of a trending, self-reinforcing move.

    ### Pipeline
    GitHub Actions runs daily at **9pm EST** on weekdays:
    fetch new OHLCV → fit Hawkes models → compute Hurst → generate signals → push to HuggingFace.

    ### Benchmarks
    SPY (US equities) and AGG (US bonds) for comparison.

    ---
    *Research tool only. Not investment advice.*
    """)

# ══════════════════════════════════════════════════════════════════════════════
# SHARED DATA LOADING (used by both Option A and B tabs)
# ══════════════════════════════════════════════════════════════════════════════
if not run_btn:
    with tab_a:
        c1, c2, c3 = st.columns(3)
        with c1: st.info("**Step 1** — Configure settings in the sidebar")
        with c2: st.info("**Step 2** — Click **🚀 Run Model**")
        with c3: st.info("**Step 3** — Review Hawkes signals and conviction")
    with tab_b:
        c1, c2, c3 = st.columns(3)
        with c1: st.info("**Step 1** — Configure settings in the sidebar")
        with c2: st.info("**Step 2** — Click **🚀 Run Model**")
        with c3: st.info("**Step 3** — Review Hawkes + Hurst combined signals")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("📥 Loading OHLCV data from HuggingFace..."):
    ohlcv = cached_load_ohlcv()
    if ohlcv is None:
        st.error("❌ No OHLCV data found. Run the reseed workflow first.")
        st.stop()

    returns_df = get_returns(ohlcv)
    volume_df  = get_volume(ohlcv)
    etf_ret    = returns_df[[t for t in ETF_UNIVERSE if t in returns_df.columns]]
    bm_ret     = returns_df[[t for t in BENCHMARKS  if t in returns_df.columns]]

    st.success(f"✅ OHLCV: {len(ohlcv):,} rows "
               f"({ohlcv.index[0].date()} → {ohlcv.index[-1].date()})")

# ── Check data staleness ──────────────────────────────────────────────────────
from datetime import date as _date
today_date  = _date.today()
data_date   = ohlcv.index[-1].date()
days_stale  = (today_date - data_date).days
if days_stale > 1:
    st.warning(
        f"⚠️ Data last updated **{data_date}** ({days_stale} days ago). "
        "Click **🔄 Force Data Refresh** to trigger a new pipeline run.",
        icon="📅"
    )

# ── Load pre-computed outputs ─────────────────────────────────────────────────
with st.spinner("🧠 Loading model outputs from HuggingFace..."):
    params_dict    = cached_load_params()
    intensity_hist = cached_load_intensity()
    signals_df     = cached_load_signals()
    hurst_df       = cached_load_hurst()
    cross_matrix   = cached_load_cross_excitation()

# ── Validate all outputs loaded ───────────────────────────────────────────────
if params_dict is None:
    st.error(
        "❌ No fitted model found on HuggingFace. "
        "Trigger the **Daily Training Pipeline** from GitHub Actions first, "
        "then click **🔄 Force Data Refresh** and try again."
    )
    st.stop()

event_def = params_dict.get("best_event_def", "combined")

if intensity_hist is None:
    st.warning("⚠️ Intensity history not found on HF — some charts will be unavailable.")

if hurst_df is None:
    st.warning("⚠️ Hurst history not found on HF — Option B will be unavailable.")
    st.stop()

if cross_matrix is None:
    st.warning("⚠️ Cross-excitation matrix not found on HF — heatmap will be unavailable.")

# ── Reconstruct fit_results from HF params + intensity ────────────────────────
# We rebuild HawkesParams objects from the stored JSON so signals can be
# generated without any MLE fitting in the app.
from hawkes import HawkesParams, compute_intensity, get_event_times, detect_events

fit_results = {}
for ticker in ETF_UNIVERSE:
    p = params_dict.get(ticker)
    if not p:
        continue
    params = HawkesParams(
        ticker    = ticker,
        kernel    = p.get("kernel", "exponential"),
        mu        = p.get("mu",     0.01),
        alpha     = p.get("alpha",  0.0),
        beta      = p.get("beta",   0.0),
        k         = p.get("k",      0.0),
        c         = p.get("c",      0.0),
        theta     = p.get("theta",  0.0),
        aic       = p.get("aic",    0.0),
        n_events  = p.get("n_events", 0),
        branching = p.get("branching", 0.0),
        event_def = p.get("event_def", event_def),
    )
    # Use pre-computed intensity from HF if available, else reconstruct
    if intensity_hist is not None and ticker in intensity_hist.columns:
        intensity_arr = intensity_hist[ticker].values
    else:
        ret    = etf_ret[ticker].dropna() if ticker in etf_ret.columns else pd.Series(dtype=float)
        vol    = volume_df[ticker].reindex(ret.index) if ticker in volume_df.columns else None
        events = detect_events(ret, vol, method=event_def)
        ev_idx = get_event_times(events)
        intensity_arr = compute_intensity(params, ev_idx,
                                          np.arange(len(ret), dtype=float))
    fit_results[ticker] = {
        "params":    params,
        "intensity": intensity_arr,
        "events":    np.array([]),   # not needed for signal generation
    }

# ── Generate signals from pre-fitted params (no MLE) ─────────────────────────
sig_a = generate_signal_option_a(fit_results, etf_ret, event_def)
sig_b = generate_signal_option_b(fit_results, etf_ret, hurst_df, event_def)

# True next trading day from today's clock
true_next_date = next_trading_day_from_today()


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: render a full signal tab (shared by A and B)
# ══════════════════════════════════════════════════════════════════════════════

def render_signal_tab(sig: dict, rf_rate: float = 0.045):
    option     = sig["option"]
    signal     = sig["signal"]
    conviction = sig["conviction"]
    conv_label = sig["label"]
    conv_col   = conviction_colour(conv_label)
    etf_col    = etf_colour(signal)

    accent_col = ("#16a34a" if conv_label in ("High", "Very High")
                  else "#d97706" if conv_label == "Moderate"
                  else "#dc2626")

    # ── Hero banner (white background, coloured left border) ─────────────────
    st.markdown(f"""
    <div style="background:#ffffff; border-radius:12px; padding:24px 32px;
                margin-bottom:16px; border:1px solid #e2e8f0;
                border-left:6px solid {accent_col};
                box-shadow:0 2px 8px rgba(0,0,0,0.06);">
      <div style="display:flex; justify-content:space-between; align-items:center;
                  flex-wrap:wrap; gap:16px;">
        <div>
          <div style="color:#6b7280; font-size:12px; letter-spacing:2px; margin-bottom:4px;">
            NEXT TRADING DAY SIGNAL — {true_next_date.strftime('%A %b %d, %Y')}
          </div>
          <div style="font-size:48px; font-weight:900; color:{etf_col};
                      letter-spacing:3px;">{signal}</div>
          <div style="color:#6b7280; font-size:13px; margin-top:6px;">
            Conviction: <span style="color:{conv_col}; font-weight:700;">
            {conv_label}</span>
            &nbsp;·&nbsp; Score: {conviction:.3f}
            &nbsp;·&nbsp; {sig['method']}
          </div>
        </div>
        <div style="text-align:right;">
          <div style="color:#6b7280; font-size:12px; margin-bottom:4px;">
            EVENT DEFINITION</div>
          <div style="background:#f8fafc; border:1px solid #e2e8f0;
                      border-radius:8px; padding:8px 16px; display:inline-block;">
            <span style="color:#374151; font-size:14px; font-weight:600;">
            {EVENT_DEFINITIONS.get(event_def, event_def)}</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── ETF excitation ratio cards ────────────────────────────────────────────
    st.subheader("🔥 Current Excitation Ratio λ*(t)/μ")
    st.caption("Ratio > 1.0 = currently above baseline (self-exciting)")

    excitation = sig["excitation"]
    cols       = st.columns(len(ETF_UNIVERSE))
    for col, ticker in zip(cols, ETF_UNIVERSE):
        ratio = excitation.get(ticker, 1.0)
        delta_pct = (ratio - 1.0) * 100
        col.metric(
            label=ticker,
            value=f"{ratio:.2f}×",
            delta=f"{delta_pct:+.1f}% vs baseline",
            delta_color="normal",
        )

    # ── Ranked bar chart ──────────────────────────────────────────────────────
    st.subheader("📊 ETF Ranking by Excitation")
    ranked  = sig["ranked"]
    r_ticks = [t for t, _ in ranked]
    r_vals  = [v for _, v in ranked]
    r_cols  = [etf_colour(t) for t in r_ticks]

    fig_rank = go.Figure(go.Bar(
        x=r_ticks, y=r_vals,
        marker_color=r_cols,
        text=[f"{v:.3f}" for v in r_vals],
        textposition="outside",
    ))
    fig_rank.update_layout(
        **CHART_LAYOUT, height=320,
        yaxis_title="Score",
        title=f"Option {option} Ranking",
    )
    st.plotly_chart(fig_rank, use_container_width=True, key=f"rank_{option}")

    # ── Option B extra: Hurst details ─────────────────────────────────────────
    if option == "B" and "details" in sig:
        if sig.get("hawkes_flat"):
            st.info(
                "ℹ️ **Hawkes scores are near-identical across ETFs** (spread < 0.05) — "
                "Hurst persistence is the primary differentiator today (weight boosted to 80%).",
                icon="📐"
            )
        st.subheader("📐 Hurst Exponent per ETF")
        h_cols = st.columns(len(ETF_UNIVERSE))
        for col, ticker in zip(h_cols, ETF_UNIVERSE):
            det = sig["details"].get(ticker, {})
            h   = det.get("hurst_H", 0.5)
            lbl = det.get("hurst_label", "—")
            col.metric(
                label=ticker,
                value=f"H = {h:.3f}",
                delta=lbl,
                delta_color="off",
            )
        # Combined score breakdown table
        st.subheader("🧮 Combined Score Breakdown")
        rows = []
        for t in ETF_UNIVERSE:
            det = sig["details"].get(t, {})
            rows.append({
                "ETF":           t,
                "Hawkes Score":  det.get("hawkes_score", 0),
                "Hurst H":       det.get("hurst_H", 0.5),
                "Hurst Score":   det.get("hurst_score", 0),
                "Combined":      det.get("combined", 0),
                "Hurst Regime":  det.get("hurst_label", "—"),
            })
        rows_df = pd.DataFrame(rows).set_index("ETF").sort_values("Combined", ascending=False)
        st.dataframe(rows_df.style.highlight_max(subset=["Combined"], color="#d1fae5"),
                     use_container_width=True)

    # ── Intensity history chart ───────────────────────────────────────────────
    if show_intensity and intensity_hist is not None:
        st.divider()
        st.subheader("📈 Intensity History λ*(t)")
        fig_int = go.Figure()
        for ticker in ETF_UNIVERSE:
            if ticker not in intensity_hist.columns:
                continue
            fig_int.add_trace(go.Scatter(
                x=intensity_hist.index,
                y=intensity_hist[ticker],
                name=ticker,
                line=dict(color=etf_colour(ticker), width=1.5),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>λ*={'{y:.4f}'}<extra>{ticker}</extra>",
            ))
        fig_int.update_layout(
            **CHART_LAYOUT, height=380,
            yaxis_title="Intensity λ*(t)",
            title="Self-Exciting Intensity Over Time",
        )
        st.plotly_chart(fig_int, use_container_width=True, key=f"intensity_{option}")

    # ── Cross-ETF excitation heatmap ──────────────────────────────────────────
    st.divider()
    st.subheader("🔗 Cross-ETF Excitation Heatmap")
    st.caption("Entry [i, j] = how much ETF j's activity predicts increased activity in ETF i")

    fig_heat = go.Figure(go.Heatmap(
        z=cross_matrix.values,
        x=cross_matrix.columns.tolist(),
        y=cross_matrix.index.tolist(),
        colorscale="RdYlGn",
        zmid=0,
        text=cross_matrix.round(3).values,
        texttemplate="%{text}",
        showscale=True,
        colorbar=dict(title="Correlation"),
    ))
    fig_heat.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Cross-ETF Excitation Matrix",
    )
    st.plotly_chart(fig_heat, use_container_width=True, key=f"heatmap_{option}")

    # ── Hawkes parameters table ───────────────────────────────────────────────
    st.divider()
    st.subheader("⚙️ Fitted Hawkes Parameters")
    if params_dict:
        param_rows = []
        for ticker in ETF_UNIVERSE:
            p = params_dict.get(ticker, {})
            if not p:
                continue
            param_rows.append({
                "ETF":       ticker,
                "Kernel":    p.get("kernel", "—"),
                "μ (base)":  f"{p.get('mu', 0):.5f}",
                "α":         f"{p.get('alpha', 0):.4f}" if p.get("kernel") == "exponential" else "—",
                "β":         f"{p.get('beta',  0):.4f}" if p.get("kernel") == "exponential" else "—",
                "k":         f"{p.get('k', 0):.4f}"     if p.get("kernel") == "powerlaw"    else "—",
                "Branching": f"{p.get('branching', 0):.3f}",
                "AIC":       f"{p.get('aic', 0):.1f}",
                "Events":    p.get("n_events", "—"),
            })
        if param_rows:
            st.dataframe(pd.DataFrame(param_rows).set_index("ETF"),
                         use_container_width=True)

    # ── Audit trail ───────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Signal History — Last 20 Entries")
    if signals_df is not None and not signals_df.empty:
        display_cols = [c for c in [
            f"signal_{option}", f"conviction_{option}", f"label_{option}",
            "event_def",
        ] + [f"{t}_excitation" for t in ETF_UNIVERSE]
            if c in signals_df.columns]
        if display_cols:
            st.dataframe(
                signals_df[display_cols].tail(20).sort_index(ascending=False),
                use_container_width=True,
            )
        else:
            st.dataframe(signals_df.tail(20).sort_index(ascending=False),
                         use_container_width=True)
    else:
        st.info("No signal history yet — pipeline hasn't run.")

    # ── Model info caption ────────────────────────────────────────────────────
    if params_dict:
        fit_date = params_dict.get("fit_date", "unknown")
        st.caption(
            f"Model fitted: **{fit_date}** · "
            f"Event def: **{event_def}** · "
            f"Conviction gate: **{conviction_gate:.2f}**"
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB A — Option A
# ══════════════════════════════════════════════════════════════════════════════
with tab_a:
    st.subheader("📊 Option A — Hawkes Process Only")
    st.caption(
        "Signal = ETF with highest current self-excitation ratio λ*(t)/μ. "
        "Pure Hawkes — no auxiliary indicators."
    )
    render_signal_tab(sig_a)

# ══════════════════════════════════════════════════════════════════════════════
# TAB B — Option B
# ══════════════════════════════════════════════════════════════════════════════
with tab_b:
    st.subheader("📈 Option B — Hawkes + Hurst Combined")
    st.caption(
        "Conviction = 65% Hawkes excitation + 35% Hurst persistence. "
        "Strong signal when both self-excitation AND long-run trend persistence align."
    )
    render_signal_tab(sig_b)
