"""
app.py — P2-ETF-HURST
======================
Streamlit dashboard for the Hurst Confluence ETF Rotation signal.
Reads all data from HuggingFace — no computation in the UI.
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
        ETF_UNIVERSE, BENCHMARKS,
    )
    from hurst_core import (
        compute_all_mtf, compute_divergence_scores,
        compute_sync_score, compute_conviction_scores,
        generate_signal, hurst_label, hurst_regime_colour,
        conviction_label, W_MTF, W_DIV, W_SYNC,
        SHORT_WINDOW, MEDIUM_WINDOW, LONG_WINDOW,
        H_TRENDING, H_RANDOM,
    )
    from walkforward import compute_wf_metrics
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# ── Styling ───────────────────────────────────────────────────────────────────
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

ETF_COLOURS = {
    "TLT": "#3b82f6", "LQD": "#8b5cf6", "HYG": "#ef4444",
    "VNQ": "#14b8a6", "GLD": "#f59e0b", "SLV": "#6b7280",
}

def etf_colour(t): return ETF_COLOURS.get(t, "#94a3b8")

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_ohlcv():       return load_ohlcv_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_mtf():         return load_mtf_history_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_signals():     return load_signals_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_wf():          return load_walkforward_from_hf()
@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_metadata():    return load_metadata_from_hf()

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
    "Hurst Confluence · Multi-Timeframe (21d/63d/252d) · "
    "Regime Divergence · Cross-Asset Synchronisation · "
    f"ETFs: {' · '.join(ETF_UNIVERSE)}"
)

tab_signal, tab_mtf, tab_wf, tab_about = st.tabs([
    "🎯 Signal & Scores",
    "📊 MTF Analysis",
    "📈 Walk-Forward Backtest",
    "ℹ️ About",
])

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("📥 Loading data from HuggingFace..."):
    ohlcv      = cached_load_ohlcv()
    mtf_hist   = cached_load_mtf()
    signals_df = cached_load_signals()
    wf_df      = cached_load_wf()
    metadata   = cached_load_metadata() or {}

if ohlcv is None:
    st.error("❌ No OHLCV data. Run the pipeline first.")
    st.stop()

# Debug: show column sample if returns fail
try:
    returns_df = get_returns(ohlcv)
except Exception as e:
    st.error(f"❌ Failed to parse OHLCV columns: {e}")
    st.info(f"Column sample: {ohlcv.columns[:10].tolist()}")
    st.stop()
etf_ret     = returns_df[[t for t in ETF_UNIVERSE if t in returns_df.columns]]
bm_ret      = returns_df[[t for t in BENCHMARKS  if t in returns_df.columns]]

# ── Compute today's signal live from latest data ──────────────────────────────
# Always compute live — Hurst is fast (~2s), no pipeline needed for signal
with st.spinner("⚙️ Computing Hurst Confluence scores..."):
    mtf_today  = compute_all_mtf(etf_ret)
    div_scores = compute_divergence_scores(mtf_today, mtf_hist)
    sync       = compute_sync_score(mtf_today)
    conviction = compute_conviction_scores(mtf_today, div_scores, sync)
    signal     = generate_signal(conviction)

# ── Staleness warning ─────────────────────────────────────────────────────────
from datetime import date as _date
data_date  = ohlcv.index[-1].date()
days_stale = (_date.today() - data_date).days
if days_stale > 3:
    st.warning(f"⚠️ Data is {days_stale} days old (last: {data_date}). "
               "Click **Force Data Refresh** to update.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SIGNAL & SCORES
# ══════════════════════════════════════════════════════════════════════════════
with tab_signal:

    # ── Hero banner ───────────────────────────────────────────────────────────
    top_etf    = signal["signal"]
    conv_label = signal["label"]
    conv_score = signal["score"]
    top_data   = conviction.get(top_etf, {})

    conv_colour = ("#22d3ee" if conv_label == "Very High" else
                   "#22c55e" if conv_label == "High" else
                   "#f59e0b" if conv_label == "Moderate" else "#ef4444")

    # Next trading day from last data date
    next_trade_day = pd.offsets.BDay(1).apply(ohlcv.index[-1]).date()

    st.markdown(f"""
    <div class="hero-card">
      <div class="hero-label">Next Trading Day Signal — {next_trade_day}</div>
      <div class="hero-ticker" style="color:{conv_colour}">{top_etf}</div>
      <div class="hero-meta">
        Conviction: <strong style="color:{conv_colour}">{conv_label}</strong>
        &nbsp;·&nbsp; Score: {conv_score:.3f}
        &nbsp;·&nbsp; MTF {W_MTF:.0%} · Divergence {W_DIV:.0%} · Sync {W_SYNC:.0%}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Conviction score breakdown ────────────────────────────────────────────
    st.subheader("🏆 ETF Conviction Rankings")

    ranked = signal["ranked"]
    cols   = st.columns(len(ETF_UNIVERSE))
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

    # ── Score breakdown table ─────────────────────────────────────────────────
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
            "H (21d)":      c.get("h_short", 0.5),
            "H (63d)":      c.get("h_medium", 0.5),
            "H (252d)":     c.get("h_long", 0.5),
            "Trending TFs": c.get("trending_count", 0),
            "Regime":       c.get("label", "—"),
            "Crossed ↑":    "✅" if d.get("crossed") else "—",
        })
    rows_df = pd.DataFrame(rows).set_index("ETF")
    st.dataframe(
        rows_df.style.highlight_max(subset=["Total Score"], color="#d1fae5")
                     .format("{:.3f}", subset=["Total Score","MTF (40%)","Divergence (40%)","Sync (20%)","H (21d)","H (63d)","H (252d)"]),
        use_container_width=True,
    )

    st.divider()

    # ── Cross-asset sync gauge ────────────────────────────────────────────────
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
        st.plotly_chart(fig_sync, use_container_width=True, key="sync_chart")

    st.divider()

    # ── Signal history ────────────────────────────────────────────────────────
    st.subheader("📋 Signal History")
    if signals_df is not None and not signals_df.empty:
        display_cols = ["signal", "conviction", "label"] + \
                       [f"{t}_total" for t in ETF_UNIVERSE if f"{t}_total" in signals_df.columns]
        available = [c for c in display_cols if c in signals_df.columns]
        st.dataframe(
            signals_df[available].tail(20).sort_index(ascending=False),
            use_container_width=True,
        )
    else:
        st.info("No signal history yet — pipeline hasn't run.")

    fit_date = metadata.get("last_model_fit", "unknown")
    st.caption(f"Last pipeline run: **{fit_date}** · Data through: **{data_date}**")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MTF ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_mtf:
    st.subheader("📊 Multi-Timeframe Hurst Analysis")
    st.caption(
        f"Three windows: **{SHORT_WINDOW}d** (short) · **{MEDIUM_WINDOW}d** (medium) · "
        f"**{LONG_WINDOW}d** (long) · Trending threshold H > {H_TRENDING}"
    )

    # ── Current MTF heatmap ───────────────────────────────────────────────────
    st.markdown("#### Current Hurst Values by Timeframe")
    windows = [("21d", "h_short"), ("63d", "h_medium"), ("252d", "h_long")]
    header_cols = st.columns([1] + [1]*len(windows))
    header_cols[0].markdown("**ETF**")
    for label, _ in windows:
        header_cols[windows.index((label, _)) + 1].markdown(f"**H {label}**")

    for ticker in ETF_UNIVERSE:
        if ticker not in conviction:
            continue
        c    = conviction[ticker]
        row  = st.columns([1] + [1]*len(windows))
        row[0].markdown(f"**{ticker}**")
        for i, (label, key) in enumerate(windows):
            h   = c.get(key, 0.5)
            col = hurst_regime_colour(h)
            row[i+1].markdown(
                f"<span style='color:{col};font-weight:600'>{h:.3f}</span> "
                f"<small style='color:#94a3b8'>{hurst_label(h)}</small>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── MTF history chart ─────────────────────────────────────────────────────
    if show_mtf_history and mtf_hist is not None and not mtf_hist.empty:
        st.markdown("#### Rolling Hurst History — Medium Window (63d)")
        fig_h = go.Figure()
        for ticker in ETF_UNIVERSE:
            col = f"{ticker}_h_medium"
            if col not in mtf_hist.columns:
                continue
            fig_h.add_trace(go.Scatter(
                x=mtf_hist.index,
                y=mtf_hist[col],
                name=ticker,
                line=dict(color=etf_colour(ticker), width=1.5),
                hovertemplate=f"%{{x|%Y-%m-%d}}<br>H={'{y:.3f}'}<extra>{ticker}</extra>",
            ))
        # Regime zones
        fig_h.add_hline(y=H_TRENDING, line=dict(color="#22c55e", dash="dash", width=1),
                        annotation_text="Trending (0.55)", annotation_position="right")
        fig_h.add_hline(y=H_RANDOM, line=dict(color="#ef4444", dash="dash", width=1),
                        annotation_text="Mean-Rev (0.45)", annotation_position="right")
        fig_h.add_hline(y=0.5, line=dict(color="#94a3b8", dash="dot", width=1))
        fig_h.update_layout(
            **CHART_LAYOUT, height=400,
            yaxis_title="Hurst Exponent H",
            yaxis_range=[0.2, 0.9],
            title="Rolling 63d Hurst Exponent — All ETFs",
        )
        st.plotly_chart(fig_h, use_container_width=True, key="mtf_hist_chart")

        # ── MTF alignment history ─────────────────────────────────────────────
        st.markdown("#### Multi-Timeframe Alignment Over Time")
        st.caption("Number of timeframes trending (H > 0.55) per ETF per day")
        fig_align = go.Figure()
        for ticker in ETF_UNIVERSE:
            cs = [f"{ticker}_h_short", f"{ticker}_h_medium", f"{ticker}_h_long"]
            available_cols = [c for c in cs if c in mtf_hist.columns]
            if len(available_cols) < 2:
                continue
            align_series = (mtf_hist[available_cols] > H_TRENDING).sum(axis=1)
            fig_align.add_trace(go.Scatter(
                x=mtf_hist.index, y=align_series,
                name=ticker,
                line=dict(color=etf_colour(ticker), width=1.5),
                opacity=0.8,
            ))
        fig_align.update_layout(
            **CHART_LAYOUT, height=300,
            yaxis_title="Trending Timeframes (0-3)",
            yaxis=dict(tickvals=[0, 1, 2, 3]),
            title="MTF Alignment History — Trending Timeframe Count",
        )
        st.plotly_chart(fig_align, use_container_width=True, key="mtf_align_chart")
    elif not show_mtf_history:
        st.info("Enable 'Show MTF history chart' in the sidebar to view this section.")
    else:
        st.info("MTF history not yet available — pipeline must run first.")

    st.divider()

    # ── Divergence detail ─────────────────────────────────────────────────────
    st.markdown("#### Divergence Score Detail")
    st.caption(
        "**div_a** = H risen vs 6-month ago (momentum) · "
        "**div_b** = H above own 2yr baseline (persistence) · "
        "**div_c** = recently crossed 1yr mean (fresh transition)"
    )
    div_rows = []
    for ticker in ETF_UNIVERSE:
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


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WALK-FORWARD BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_wf:
    st.subheader("📈 Walk-Forward Backtest")
    st.caption(
        "Proper OOS backtest: 252-day rolling train window, 21-day step-forward, "
        "refit monthly. 5bps fee per rotation. No look-ahead bias."
    )

    if wf_df is None or wf_df.empty:
        st.info("Walk-forward results not available yet — trigger the pipeline.")
        st.stop()

    # Handle both new format (cum_strategy) and old Hawkes format (cum_A)
    if "cum_strategy" in wf_df.columns:
        cum_col = "cum_strategy"
        ret_col = "ret"
    elif "cum_A" in wf_df.columns:
        cum_col = "cum_A"
        ret_col = "ret_A"
    else:
        st.warning(f"Walk-forward columns not recognised: {wf_df.columns.tolist()} — re-run the pipeline.")
        st.stop()

    metrics = compute_wf_metrics(wf_df)
    if not metrics:
        st.info("Walk-forward data format unrecognised — trigger a fresh pipeline run.")
        st.stop()

    # ── Benchmark metrics ─────────────────────────────────────────────────────
    bm_ann, bm_sharpe, bm_dd = 0.0, 0.0, 0.0
    has_bm = False
    if benchmark != "None" and f"cum_{benchmark}" in wf_df.columns:
        bm_r   = wf_df[f"ret_{benchmark}"].values
        bm_c   = wf_df[f"cum_{benchmark}"].values
        n_bm   = len(bm_r)
        bm_ann = float(bm_c[-1] ** (252/n_bm) - 1) if n_bm > 0 else 0.0
        bm_sharpe = float(np.mean(bm_r - 0.045/252) / (np.std(bm_r)+1e-9) * np.sqrt(252))
        bm_cmax = np.maximum.accumulate(bm_c)
        bm_dd   = float(np.min((bm_c - bm_cmax) / (bm_cmax + 1e-9)))
        has_bm  = True

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Ann. Return",  f"{metrics['ann_return']:+.1%}",
              delta=f"{metrics['ann_return']-bm_ann:+.1%} vs {benchmark}" if has_bm else None)
    c2.metric("Sharpe",       f"{metrics['sharpe']:.2f}",
              delta=f"{metrics['sharpe']-bm_sharpe:+.2f} vs {benchmark}" if has_bm else None)
    c3.metric("Max Drawdown", f"{metrics['max_dd']:.1%}",
              delta=f"{metrics['max_dd']-bm_dd:+.1%} vs {benchmark}" if has_bm else None,
              delta_color="inverse")
    c4.metric("Hit Ratio",    f"{metrics['hit_ratio']:.1%}")
    c5.metric("OOS Days",     f"{metrics['n_days']:,}")

    # ── Cumulative return chart ────────────────────────────────────────────────
    fig_wf = go.Figure()
    fig_wf.add_trace(go.Scatter(
        x=wf_df.index, y=wf_df[cum_col],
        name="HRC Strategy (OOS)",
        line=dict(color="#22d3ee", width=2.5),
        fill="tozeroy", fillcolor="rgba(34,211,238,0.05)",
    ))
    if has_bm:
        fig_wf.add_trace(go.Scatter(
            x=wf_df.index, y=wf_df[f"cum_{benchmark}"],
            name=benchmark,
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
        ))
    fig_wf.add_hline(y=1.0, line=dict(color="#e2e8f0", width=1))
    fig_wf.update_layout(
        **CHART_LAYOUT, height=440,
        yaxis_title="Cumulative Return (1 = start)",
        title=f"HRC Strategy Walk-Forward OOS vs {benchmark} | Train=252d · Step=21d · 5bps fee",
    )
    st.plotly_chart(fig_wf, use_container_width=True, key="wf_main_chart")

    # ── Signal distribution ───────────────────────────────────────────────────
    st.divider()
    if "signal" in wf_df.columns:
        st.markdown("#### Historical Signal Distribution")
        sig_counts = wf_df["signal"].value_counts()
        fig_dist = go.Figure(go.Bar(
            x=sig_counts.index,
            y=sig_counts.values,
            marker_color=[etf_colour(t) for t in sig_counts.index],
            text=sig_counts.values,
            textposition="outside",
        ))
        fig_dist.update_layout(
            **CHART_LAYOUT, height=280,
            yaxis_title="Days Held",
            title="Days Each ETF Was the Top Signal (Walk-Forward OOS)",
        )
        st.plotly_chart(fig_dist, use_container_width=True, key="sig_dist_chart")

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
            bm_roll = wf_df[f"ret_{benchmark}"].rolling(252).apply(
                lambda x: (np.prod(1 + x) ** (252/len(x)) - 1), raw=True
            )
            fig_roll.add_trace(go.Scatter(
                x=wf_df.index, y=bm_roll,
                name=benchmark,
                line=dict(color="#94a3b8", width=1.5, dash="dot"),
            ))
        fig_roll.add_hline(y=0, line=dict(color="#e2e8f0", width=1))
        fig_roll.update_layout(
            **CHART_LAYOUT, height=300,
            yaxis_title="Rolling 252d Ann. Return",
            yaxis_tickformat=".0%",
            title="Rolling 1-Year Annualised Return",
        )
        st.plotly_chart(fig_roll, use_container_width=True, key="roll_ret_chart")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("ℹ️ About P2-ETF-HURST")
    st.markdown(f"""
### Hurst Confluence (HRC) Signal Architecture

The signal conviction score per ETF is a weighted combination of three components:

| Component | Weight | Logic |
|-----------|--------|-------|
| **Multi-Timeframe Alignment** | {W_MTF:.0%} | H computed at 21d, 63d, 252d. Score = 1.0 if all 3 trending, 0.75 if short+medium align, 0.5 if 2-of-3 |
| **Hurst Divergence** | {W_DIV:.0%} | Blend of: (a) H risen vs 6m ago, (b) H above own 2yr baseline, (c) recently crossed 1yr mean |
| **Cross-Asset Sync** | {W_SYNC:.0%} | Reward ETFs whose H diverges positively from the cluster mean |

### Regime Thresholds

| H Range | Regime |
|---------|--------|
| H ≥ 0.65 | Strong Trend |
| 0.55 ≤ H < 0.65 | Mild Trend |
| 0.45 ≤ H < 0.55 | Random Walk |
| H < 0.45 | Mean-Reverting |

### Walk-Forward Backtest Methodology

- **Train window**: 252 trading days (~1 year)
- **Step size**: 21 days (monthly rebalance)
- **Signal**: Top ETF by HRC conviction score computed on train window
- **Fee**: 5bps per rotation
- **No look-ahead**: Each day's signal uses only data available up to that point

### ETF Universe

| Ticker | Name | Asset Class |
|--------|------|-------------|
| TLT | iShares 20yr Treasury | Long-duration Treasuries |
| LQD | iShares IG Corp Bond | Investment Grade Bonds |
| HYG | iShares High Yield Bond | High Yield Bonds |
| VNQ | Vanguard Real Estate | US REITs |
| GLD | SPDR Gold Shares | Gold |
| SLV | iShares Silver Trust | Silver |

### Data Sources
- OHLCV: Yahoo Finance via `yfinance`
- Stored on HuggingFace: `P2SAMAPA/p2-etf-hurst-data`
- Pipeline runs daily after US market close (Mon–Fri)

---
*For educational and research purposes only. Not financial advice.*
    """)
