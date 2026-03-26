"""
app.py — P2-ETF-HURST
======================
Streamlit dashboard for the Hurst Confluence ETF Rotation signal.
Reads all data from HuggingFace — no computation in the UI.

Now supports both Option A (FI/Commodities) and Option B (Equity Sectors).
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="P2-ETF-HURST",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    from data_manager import (
        load_ohlcv_from_hf, load_mtf_history_from_hf,
        load_signals_from_hf, load_walkforward_from_hf,
        load_metadata_from_hf, get_returns,
        ETF_UNIVERSE as FI_UNIVERSE, BENCHMARKS,
    )
    # New option-aware functions
    from data_manager import (
        load_signals, load_walkforward, load_mtf_history, load_metadata,
    )
    from hurst_core import (
        compute_all_mtf, compute_divergence_scores,
        compute_sync_score, compute_conviction_scores,
        compute_momentum_scores, optimise_momentum_weights,
        generate_signal, build_mtf_history,
        hurst_label, hurst_regime_colour,
        velocity_label, velocity_colour,
        conviction_label, W_MTF, W_DIV, W_SYNC,
        MEDIUM_WINDOW, LONG_WINDOW, VELOCITY_WINDOW,
        H_TRENDING, H_WEAK_TREND, H_RANDOM,
    )
    from walkforward import compute_wf_metrics
    # Import config for equity ETFs
    from config import OPTION_B_ETFS
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── Styling (unchanged) ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
.hero-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border-left: 5px solid #22d3ee;
    color: white;
}
.hero-ticker {
    font-family: 'DM Serif Display', serif;
    font-size: 5rem;
    line-height: 1;
    margin: 8px 0;
    color: #22d3ee;
}
.hero-label {
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #94a3b8;
    margin-bottom: 4px;
}
.hero-meta {
    font-size: 0.9rem;
    color: #cbd5e1;
    margin-top: 12px;
}
.regime-pill {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.stMetric label { font-size: 0.75rem !important; color: #64748b !important; }
.stMetric [data-testid="metric-value"] { font-family: 'DM Serif Display', serif; font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)

# ── Chart layout ──────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="DM Sans, sans-serif", size=12),
    margin=dict(l=0, r=0, t=40, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#ffffff",
)

# Colour maps for both universes
FI_COLOURS = {
    "TLT": "#3b82f6", "LQD": "#8b5cf6", "HYG": "#ef4444",
    "VNQ": "#14b8a6", "GLD": "#f59e0b", "SLV": "#6b7280",
}
EQ_COLOURS = {
    "SPY": "#3b82f6", "QQQ": "#14b8a6", "XLK": "#8b5cf6",
    "XLF": "#ef4444", "XLE": "#f59e0b", "XLV": "#10b981",
    "XLI": "#6b7280", "XLY": "#ec4899", "XLP": "#84cc16",
    "XLU": "#06b6d4", "GDX": "#f97316", "XME": "#a855f7",
}
def etf_colour(t, option='a'):
    if option == 'a':
        return FI_COLOURS.get(t, "#94a3b8")
    else:
        return EQ_COLOURS.get(t, "#94a3b8")

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_ohlcv():       return load_ohlcv_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_mtf(option='a'): return load_mtf_history(option) if option != 'a' else load_mtf_history_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_signals(option='a'): return load_signals(option) if option != 'a' else load_signals_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_wf(option='a'): return load_walkforward(option) if option != 'a' else load_walkforward_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_metadata(option='a'): return load_metadata(option) if option != 'a' else load_metadata_from_hf()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📐 P2-ETF-HURST")
    st.caption("Hurst Confluence ETF Rotation")
    now_est = datetime.utcnow()
    st.write(f"🕒 **UTC:** {now_est.strftime('%a %b %d, %H:%M')}")
    st.divider()

    st.subheader("📊 Display")
    benchmark    = st.selectbox("Benchmark", ["SPY", "AGG", "None"])
    show_mtf_history = st.toggle("Show MTF history chart", value=True)
    show_sync    = st.toggle("Show cross-asset sync", value=True)
    st.divider()

    run_btn     = None  # signal computed automatically on load
    refresh_btn = st.button("🔄 Force Data Refresh", use_container_width=True)

    if refresh_btn:
        with st.spinner("Triggering pipeline..."):
            try:
                import requests as _req
                gh_token = os.environ.get("GH_PAT", "")
                gh_repo  = os.environ.get("GITHUB_REPO", "P2SAMAPA/P2-ETF-HURST")
                if not gh_token:
                    raise ValueError("GH_PAT secret not set")
                resp = _req.post(
                    f"https://api.github.com/repos/{gh_repo}/actions/workflows/daily.yml/dispatches",
                    headers={"Authorization": f"Bearer {gh_token}",
                             "Accept": "application/vnd.github+json"},
                    json={"ref": "main"}, timeout=15,
                )
                if resp.status_code == 204:
                    st.cache_data.clear()
                    st.success("✅ Pipeline triggered — check back in ~10 min")
                else:
                    st.error(f"❌ GitHub API: {resp.status_code}")
            except Exception as ex:
                st.error(f"❌ {ex}")

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📐 P2-ETF-HURST")
st.caption(
    "Hurst Confluence · H63 Velocity · Medium (63d) · Long (126d) · "
    "Regime Divergence · Cross-Asset Synchronisation"
)

# Create two tabs: Option A (FI) and Option B (Equity)
tab_a, tab_b = st.tabs([
    "🌊 Option A — Fixed Income / Commodities",
    "📈 Option B — Equity Sectors",
])

# ------------------------------------------------------------------------------
# Helper function to render a complete option tab
def render_option_tab(option: str, etf_list: list, option_label: str):
    """Render all content for a given option (a or b)."""
    st.caption(f"ETFs: {' · '.join(etf_list)}")

    # Load option-specific data
    ohlcv      = cached_load_ohlcv()
    mtf_hist   = cached_load_mtf(option)
    signals_df = cached_load_signals(option)
    wf_df      = cached_load_wf(option)
    metadata   = cached_load_metadata(option) or {}

    if ohlcv is None:
        st.error("❌ No OHLCV data. Run the pipeline first.")
        return

    # Compute returns and select relevant columns
    returns_df = get_returns(ohlcv)
    etf_ret    = returns_df[[t for t in etf_list if t in returns_df.columns]]
    bm_ret     = returns_df[[t for t in BENCHMARKS if t in returns_df.columns]]

    # Compute today's signal from latest data
    mtf_today  = compute_all_mtf(etf_ret, etf_list=etf_list)
    div_scores = compute_divergence_scores(mtf_today, mtf_hist, etf_list=etf_list) if mtf_hist is not None else {}
    sync       = compute_sync_score(mtf_today, etf_list=etf_list)
    conviction = compute_conviction_scores(mtf_today, div_scores, sync, etf_list=etf_list)
    mom_w, w3m = optimise_momentum_weights(etf_ret, conviction, train_window=252, etf_list=etf_list)
    mom_scores = compute_momentum_scores(etf_ret, w3m=w3m, etf_list=etf_list)
    signal     = generate_signal(conviction, mom_scores, mom_weight=mom_w, w3m=w3m)

    data_date  = ohlcv.index[-1].date()
    days_stale = (datetime.utcnow().date() - data_date).days
    if days_stale > 3:
        st.warning(f"⚠️ Data is {days_stale} days old (last: {data_date}). Click **Force Data Refresh** to update.")

    # Hero banner
    top_etf    = signal["signal"]
    conv_label = signal["label"]
    conv_score = signal["score"]
    conv_colour = ("#22d3ee" if conv_label == "Very High" else
                   "#22c55e" if conv_label == "High" else
                   "#f59e0b" if conv_label == "Moderate" else "#ef4444")
    next_trade_day = (ohlcv.index[-1] + pd.offsets.BDay(1)).date()

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-label">Next Trading Day Signal — {next_trade_day}</div>
      <div class="hero-ticker" style="color:{conv_colour}">{top_etf}</div>
      <div class="hero-meta">
        Conviction: <strong style="color:{conv_colour}">{conv_label}</strong>
        &nbsp;·&nbsp; Score: {conv_score:.3f}
        &nbsp;·&nbsp; MTF {W_MTF:.0%} · Divergence {W_DIV:.0%} · Sync {W_SYNC:.0%}
        &nbsp;·&nbsp; Mom {mom_w:.0%} (3m:{w3m:.0%}/6m:{1-w3m:.0%})
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Conviction rankings
    st.subheader("🏆 ETF Conviction Rankings")
    ranked = signal["ranked"]
    cols   = st.columns(len(etf_list))
    for col, (ticker, score) in zip(cols, ranked):
        c = conviction.get(ticker, {})
        h_med = c.get("h_medium", 0.5)
        col.metric(
            label=ticker,
            value=f"{score:.3f}",
            delta=hurst_label(h_med),
            delta_color="off",
        )
    st.divider()

    # Score breakdown table
    st.subheader("🧮 Score Breakdown")
    rows = []
    for ticker, score in ranked:
        c = conviction.get(ticker, {})
        d = div_scores.get(ticker, {})
        rows.append({
            "ETF":          ticker,
            "Total Score":  c.get("total", 0),
            "MTF (40%)":    c.get("mtf_score", 0),
            "Divergence (40%)": c.get("div_score", 0),
            "Sync (20%)":   c.get("sync_score", 0),
            "H Velocity":   c.get("h_velocity", c.get("h_short", 0.0)),
            f"H ({MEDIUM_WINDOW}d)":  c.get("h_medium", 0.5),
            f"H ({LONG_WINDOW}d)": c.get("h_long", 0.5),
            "Trending TFs": c.get("trending_count", 0),
            "Regime":       c.get("label", "—"),
            "Crossed ↑":    "✅" if d.get("crossed") else "—",
        })
    rows_df = pd.DataFrame(rows).set_index("ETF")
    st.dataframe(
        rows_df.style.highlight_max(subset=["Total Score"], color="#d1fae5")
                     .format("{:.3f}", subset=["Total Score","MTF (40%)","Divergence (40%)","Sync (20%)","H Velocity",f"H ({MEDIUM_WINDOW}d)",f"H ({LONG_WINDOW}d)"]),
        use_container_width=True,
    )
    st.divider()

    # Cross-asset sync gauge
    if show_sync:
        st.subheader("🔗 Cross-Asset Synchronisation")
        sync_level = sync.get("sync_level", 0.5)
        sync_label = (
            "High Sync — Risk-Off Cluster" if sync_level > 0.7 else
            "Moderate Sync"                if sync_level > 0.4 else
            "Low Sync — Dispersion"
        )
        st.caption(
            f"Sync level: **{sync_level:.2f}** — {sync_label} · "
            f"H_mean={sync.get('h_mean', 0):.3f} · H_std={sync.get('h_std', 0):.3f}"
        )
        fig_sync = go.Figure(go.Bar(
            x=list(sync.get("scores", {}).keys()),
            y=list(sync.get("scores", {}).values()),
            marker_color=[
                "#22d3ee" if v > 0 else "#f87171"
                for v in sync.get("scores", {}).values()
            ],
            text=[f"{v:+.3f}" for v in sync.get("scores", {}).values()],
            textposition="outside",
        ))
        fig_sync.update_layout(
            **CHART_LAYOUT, height=280,
            yaxis_title="Deviation from Cluster Mean H",
            title="ETF Hurst Deviation from Cross-Asset Mean (positive = above cluster)",
        )
        st.plotly_chart(fig_sync, use_container_width=True, key=f"sync_chart_{option}")
        st.divider()

    # Signal history
    st.subheader("📋 Signal History")
    if signals_df is not None and not signals_df.empty:
        display_cols = ["signal", "conviction", "label"] + \
                       [f"{t}_total" for t in etf_list if f"{t}_total" in signals_df.columns]
        available = [c for c in display_cols if c in signals_df.columns]
        st.dataframe(
            signals_df[available].tail(20).sort_index(ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No signal history yet — pipeline hasn't run.")

    fit_date = metadata.get("last_model_fit", "unknown")
    st.caption(f"Last pipeline run: **{fit_date}** · Data through: **{data_date}**")

    st.divider()

    # MTF analysis (heatmap, regime timeline, per-ETF history)
    st.subheader("📊 Multi-Timeframe Hurst Analysis")
    st.caption(
        f"H63 Velocity ({VELOCITY_WINDOW}d lookback) · **{MEDIUM_WINDOW}d** (~1q) · "
        f"**{LONG_WINDOW}d** (~6m) · Trending threshold H > {H_TRENDING}"
    )

    # Heatmap: current values
    st.markdown("#### Current Hurst Heatmap — ETF × Timeframe")
    st.caption("H columns: Dark green = Strong Trend (>0.65) · Green = Mild Trend (>0.55) · Yellow-green = Weak Trend (0.50–0.55) · Amber = Random Walk · Red = Mean-Reverting")

    tickers_ord = [t for t in etf_list if t in conviction]

    vel_z, vel_t = [], []
    h_z,   h_t   = [], []
    h_windows = [(f"H {MEDIUM_WINDOW}d", "h_medium"), (f"H {LONG_WINDOW}d", "h_long")]

    for ticker in tickers_ord:
        c   = conviction[ticker]
        vel = c.get("h_velocity", c.get("h_short", 0.0))
        vel_z.append([vel])
        vel_t.append([f"{vel:+.3f}<br>{velocity_label(vel)}"])
        row_z, row_t = [], []
        for _, key in h_windows:
            h = c.get(key, 0.5)
            row_z.append(h)
            row_t.append(f"{h:.3f}<br>{hurst_label(h)}")
        h_z.append(row_z)
        h_t.append(row_t)

    from plotly.subplots import make_subplots
    fig_hm = make_subplots(
        rows=1, cols=2,
        column_widths=[0.33, 0.67],
        horizontal_spacing=0.02,
        shared_yaxes=True,
    )
    fig_hm.add_trace(go.Heatmap(
        z=vel_z, x=["H63 Velocity"], y=tickers_ord,
        text=vel_t, texttemplate="%{text}",
        textfont=dict(size=12, family="DM Sans"),
        colorscale=[
            [0.0,  "#dc2626"],   # strong negative — reversing
            [0.35, "#f97316"],   # negative — decelerating
            [0.50, "#d97706"],   # neutral
            [0.65, "#65a30d"],   # positive — stable
            [1.0,  "#16a34a"],   # strong positive — accelerating
        ],
        zmin=-0.5, zmax=0.5,
        showscale=False,
    ), row=1, col=1)
    fig_hm.add_trace(go.Heatmap(
        z=h_z, x=[w[0] for w in h_windows], y=tickers_ord,
        text=h_t, texttemplate="%{text}",
        textfont=dict(size=12, family="DM Sans"),
        colorscale=[
            [0.0,   "#dc2626"],   # 0.30 — mean-reverting
            [0.25,  "#f97316"],   # 0.45 — random walk
            [0.333, "#d97706"],   # 0.50 — random walk top
            [0.417, "#84cc16"],   # 0.55 — weak trend (0.50-0.55)
            [0.50,  "#16a34a"],   # 0.60 — mild trend
            [1.0,   "#052e16"],   # 0.90 — strong trend
        ],
        zmin=0.3, zmax=0.9,
        showscale=True,
        colorbar=dict(title="H", thickness=12, len=0.8, x=1.01),
    ), row=1, col=2)
    fig_hm.update_layout(
        **CHART_LAYOUT, height=280,
        title=f"Hurst Heatmap — {ohlcv.index[-1].date()} | Velocity · {MEDIUM_WINDOW}d · {LONG_WINDOW}d",
        xaxis=dict(side="top"),
        xaxis2=dict(side="top"),
        yaxis=dict(autorange="reversed"),
    )
    st.caption("Velocity column: green = H63 accelerating ↑ · red = decelerating ↓ · scale: −0.5 to +0.5")
    st.plotly_chart(fig_hm, use_container_width=True, key=f"heatmap_chart_{option}")
    st.divider()

    # Regime timeline (if history available)
    if show_mtf_history and mtf_hist is not None and not mtf_hist.empty:
        st.markdown("#### Regime Timeline — Which ETF Was Trending (63d)")
        st.caption(
            "Each band shows the fraction of days per month each ETF had H > 0.55 "
            "on the 63d window. Taller band = more consistently trending that month."
        )
        trend_cols = {t: f"{t}_h_medium" for t in etf_list
                      if f"{t}_h_medium" in mtf_hist.columns}
        if trend_cols:
            trend_df = (mtf_hist[[c for c in trend_cols.values()]] > H_TRENDING).astype(float)
            trend_df.columns = list(trend_cols.keys())
            monthly  = trend_df.resample("ME").mean()

            fig_rt = go.Figure()
            for ticker in etf_list:
                if ticker not in monthly.columns:
                    continue
                fig_rt.add_trace(go.Bar(
                    x=monthly.index,
                    y=monthly[ticker],
                    name=ticker,
                    marker_color=etf_colour(ticker, option),
                    hovertemplate="%{x|%b %Y}<br>" + ticker + ": %{y:.0%} trending days<extra></extra>",
                ))
            fig_rt.update_layout(
                **CHART_LAYOUT, height=360,
                barmode="stack",
                yaxis_title="Fraction of ETFs Trending",
                yaxis_tickformat=".0%",
                yaxis_range=[0, len(trend_cols)],
                title="Monthly Trending Regime Share — 63d Hurst > 0.55",
            )
            st.plotly_chart(fig_rt, use_container_width=True, key=f"regime_timeline_{option}")
            st.divider()

        # Per-ETF H history
        st.markdown("#### Per-ETF Hurst History")
        selected_etf = st.selectbox("Select ETF", etf_list, key=f"etf_selector_{option}")
        cols_etf = {
            "21d":  f"{selected_etf}_h_short",
            "63d":  f"{selected_etf}_h_medium",
            "252d": f"{selected_etf}_h_long",
        }
        fig_etf = go.Figure()
        line_styles = {"21d": dict(width=1, dash="dot"),
                       "63d": dict(width=2),
                       "252d": dict(width=1.5, dash="dash")}
        for label, col in cols_etf.items():
            if col not in mtf_hist.columns:
                continue
            fig_etf.add_trace(go.Scatter(
                x=mtf_hist.index, y=mtf_hist[col],
                name=f"H {label}",
                line=dict(color=etf_colour(selected_etf, option), **line_styles[label]),
                opacity=0.9 if label == "63d" else 0.5,
            ))
        fig_etf.add_hrect(y0=H_TRENDING, y1=0.9,
                          fillcolor="#16a34a", opacity=0.05, line_width=0,
                          annotation_text="Trending zone", annotation_position="top right")
        fig_etf.add_hrect(y0=0.2, y1=H_RANDOM,
                          fillcolor="#dc2626", opacity=0.05, line_width=0,
                          annotation_text="Mean-rev zone", annotation_position="bottom right")
        fig_etf.add_hline(y=H_TRENDING, line=dict(color="#16a34a", dash="dash", width=1))
        fig_etf.add_hline(y=H_RANDOM,   line=dict(color="#dc2626", dash="dash", width=1))
        fig_etf.update_layout(
            **CHART_LAYOUT, height=350,
            yaxis_title="Hurst Exponent H",
            yaxis_range=[0.2, 0.95],
            title=f"{selected_etf} — Hurst History (Velocity · {MEDIUM_WINDOW}d · {LONG_WINDOW}d)",
        )
        st.plotly_chart(fig_etf, use_container_width=True, key=f"per_etf_chart_{option}")
        st.divider()

    else:
        st.info("MTF history not yet available — pipeline must run first.")

    # Divergence detail
    st.markdown("#### Divergence Score Detail")
    st.caption(
        "**div_a** = H risen vs 6-month ago (momentum) · "
        "**div_b** = H above own 2yr baseline (persistence) · "
        "**div_c** = recently crossed 1yr mean (fresh transition)"
    )
    div_rows = []
    for ticker in etf_list:
        d = div_scores.get(ticker, {})
        c = conviction.get(ticker, {})
        div_rows.append({
            "ETF":        ticker,
            "div_a (momentum)":    d.get("div_a", 0),
            "div_b (persistence)": d.get("div_b", 0),
            "div_c (transition)":  d.get("div_c", 0),
            "Combined Div":        d.get("div_score", 0),
            "H Baseline (2yr)":   d.get("h_baseline", 0.5),
            "H 1yr Mean":         d.get("h_1yr_mean", 0.5),
            "Crossed ↑":          "✅" if d.get("crossed") else "—",
            "Regime":             c.get("label", "—"),
        })
    st.dataframe(
        pd.DataFrame(div_rows).set_index("ETF")
          .style.format("{:.3f}", subset=["div_a (momentum)", "div_b (persistence)",
                                          "div_c (transition)", "Combined Div",
                                          "H Baseline (2yr)", "H 1yr Mean"])
          .highlight_max(subset=["Combined Div"], color="#d1fae5"),
        use_container_width=True,
    )

    # Walk-forward backtest
    st.subheader("📈 Walk-Forward Backtest")
    st.caption(
        "Proper OOS backtest: 252-day rolling train window, 21-day step-forward, "
        "refit monthly. 5bps fee per rotation. No look-ahead bias."
    )

    if wf_df is None or wf_df.empty:
        st.info("Walk-forward results not available yet — trigger the pipeline.")
        return

    # Handle column naming (cum_strategy for new, cum_A for old)
    if "cum_strategy" in wf_df.columns:
        cum_col = "cum_strategy"
        ret_col = "ret"
    elif "cum_A" in wf_df.columns:
        cum_col = "cum_A"
        ret_col = "ret_A"
    else:
        st.warning(f"Walk-forward columns not recognised: {wf_df.columns.tolist()} — re-run the pipeline.")
        return

    metrics = compute_wf_metrics(wf_df)
    if not metrics:
        st.info("Walk-forward data format unrecognised — trigger a fresh pipeline run.")
        return

    # Benchmark metrics
    bm_ann, bm_sharpe, bm_dd = 0.0, 0.0, 0.0
    has_bm = False
    bench_choice = st.session_state.get("benchmark", "None")  # use global sidebar choice
    if bench_choice != "None" and f"cum_{bench_choice}" in wf_df.columns:
        bm_r   = wf_df[f"ret_{bench_choice}"].values
        bm_c   = wf_df[f"cum_{bench_choice}"].values
        n_bm   = len(bm_r)
        bm_ann = float(bm_c[-1] ** (252/n_bm) - 1) if n_bm > 0 else 0.0
        bm_sharpe = float(np.mean(bm_r - 0.045/252) / (np.std(bm_r)+1e-9) * np.sqrt(252))
        bm_cmax = np.maximum.accumulate(bm_c)
        bm_dd   = float(np.min((bm_c - bm_cmax) / (bm_cmax + 1e-9)))
        has_bm  = True

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return",  f"{metrics['ann_return']:+.1%}",
              delta=f"{metrics['ann_return']-bm_ann:+.1%} vs {bench_choice}" if has_bm else None)
    c2.metric("Sharpe",       f"{metrics['sharpe']:.2f}",
              delta=f"{metrics['sharpe']-bm_sharpe:+.2f} vs {bench_choice}" if has_bm else None)
    c3.metric("Max Drawdown", f"{metrics['max_dd']:.1%}",
              delta=f"{metrics['max_dd']-bm_dd:+.1%} vs {bench_choice}" if has_bm else None,
              delta_color="inverse")
    c4.metric("Hit Ratio",    f"{metrics['hit_ratio']:.1%}")
    c5.metric("OOS Days",     f"{metrics['n_days']:,}")

    # Cumulative return chart
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(
        x=wf_df.index, y=wf_df[cum_col],
        name="HRC Strategy (OOS)",
        line=dict(color="#22d3ee", width=2.5),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.05)",
    ))
    if has_bm:
        fig_wf.add_trace(go.Scatter(
            x=wf_df.index, y=wf_df[f"cum_{bench_choice}"],
            name=bench_choice,
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        ))
    fig_wf.add_hline(y=1.0, line=dict(color="#e2e8f0", width=1))
    fig_wf.update_layout(
        **CHART_LAYOUT, height=440,
        yaxis_title="Cumulative Return (1 = start)",
        title=f"HRC Strategy Walk-Forward OOS vs {bench_choice} | Train=252d · Step=21d · 5bps fee",
    )
    st.plotly_chart(fig_wf, use_container_width=True, key=f"wf_main_chart_{option}")

    # Signal distribution
    if "signal" in wf_df.columns:
        st.divider()
        st.markdown("#### Historical Signal Distribution")
        sig_counts = wf_df["signal"].value_counts()
        fig_dist = go.Figure(go.Bar(
            x=sig_counts.index,
            y=sig_counts.values,
            marker_color=[etf_colour(t, option) for t in sig_counts.index],
            text=sig_counts.values,
            textposition="outside",
        ))
        fig_dist.update_layout(
            **CHART_LAYOUT, height=280,
            yaxis_title="Days Held",
            title="Days Each ETF Was the Top Signal (Walk-Forward OOS)",
        )
        st.plotly_chart(fig_dist, use_container_width=True, key=f"sig_dist_chart_{option}")

        st.markdown("#### Rolling 252-Day Annualised Return")
        roll_ret = wf_df[ret_col].rolling(252).apply(
            lambda x: (np.prod(1 + x) ** (252/len(x)) - 1), raw=True
        )
        fig_roll = go.Figure()
        fig_roll.add_trace(go.Scatter(
            x=wf_df.index, y=roll_ret,
            name="HRC Rolling Ann. Return",
            line=dict(color="#22d3ee", width=2),
            fill="tozeroy",
            fillcolor="rgba(34,211,238,0.08)",
        ))
        if has_bm:
            bm_roll = wf_df[f"ret_{bench_choice}"].rolling(252).apply(
                lambda x: (np.prod(1 + x) ** (252/len(x)) - 1), raw=True
            )
            fig_roll.add_trace(go.Scatter(
                x=wf_df.index, y=bm_roll,
                name=bench_choice,
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            ))
        fig_roll.add_hline(y=0, line=dict(color="#e2e8f0", width=1))
        fig_roll.update_layout(
            **CHART_LAYOUT, height=300,
            yaxis_title="Rolling 252d Ann. Return",
            yaxis_tickformat=".0%",
            title="Rolling 1-Year Annualised Return",
        )
        st.plotly_chart(fig_roll, use_container_width=True, key=f"roll_ret_chart_{option}")


# ── Render both tabs ──────────────────────────────────────────────────────────
with tab_a:
    render_option_tab('a', FI_UNIVERSE, "Fixed Income / Commodities")

with tab_b:
    render_option_tab('b', OPTION_B_ETFS, "Equity Sectors")
