"""app.py — Streamlit dashboard for P2-ETF-MOMENTUM-REVERSAL engine."""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
from huggingface_hub import HfApi

st.set_page_config(
    page_title="MOMENTUM-REVERSAL · P2Quant",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-momentum-reversal-results"
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"

FI_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQ_TICKERS = [
    "SPY",
    "QQQ",
    "XLK",
    "XLF",
    "XLE",
    "XLV",
    "XLI",
    "XLY",
    "XLP",
    "XLU",
    "GDX",
    "XME",
    "IWM",
    "IWF",
    "XSD",
    "XBI",
    "XLB",
    "XLRE",
]

UNIVERSE_COLOURS = {
    "TLT": "#1B4F8A",
    "VCIT": "#2E86C1",
    "LQD": "#148F77",
    "HYG": "#B7950B",
    "VNQ": "#6C3483",
    "GLD": "#CA6F1E",
    "SLV": "#717D7E",
    "SPY": "#C0392B",
    "QQQ": "#922B21",
    "XLK": "#1A5276",
    "XLF": "#117A65",
    "XLE": "#784212",
    "XLV": "#1D8348",
    "XLI": "#2471A3",
    "XLY": "#7D6608",
    "XLP": "#6E2F83",
    "XLU": "#17202A",
    "GDX": "#B7950B",
    "XME": "#5D6D7E",
    "IWM": "#E74C3C",
    "IWF": "#1ABC9C",
    "XSD": "#8E44AD",
    "XBI": "#E67E22",
    "XLB": "#2ECC71",
    "XLRE": "#F39C12",
}


@st.cache_data(ttl=3600, show_spinner="Loading results from HuggingFace…")
def load_results() -> tuple[pd.DataFrame, bool]:
    """Load scores via direct parquet — no datasets/pyarrow build dependency."""
    try:
        hf_token = os.environ.get("HF_TOKEN")
        api = HfApi()
        files = list(
            api.list_repo_files(
                HF_RESULTS_REPO,
                repo_type="dataset",
                token=hf_token if hf_token else None,
            )
        )
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            raise ValueError("No parquet files found in HF dataset repo.")

        headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else {}
        dfs = []
        for fname in parquet_files:
            url = f"https://huggingface.co/datasets/{HF_RESULTS_REPO}/resolve/main/{fname}"
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            dfs.append(pd.read_parquet(io.BytesIO(resp.content)))

        df = pd.concat(dfs, ignore_index=True)
        if df.empty:
            raise ValueError("Dataset loaded but contains no rows.")

        df["date"] = pd.to_datetime(df["date"])
        if "universe" not in df.columns:
            df["universe"] = df["ticker"].apply(lambda t: "fi" if t in FI_TICKERS else "equity")

        dedup_cols = [c for c in ["date", "ticker", "universe"] if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols, keep="last")

        return df.sort_values("date"), False

    except Exception as e:
        st.warning(f"Could not load HF dataset ({e}). Showing synthetic demo data.")
        return _synthetic_demo(), True


def _synthetic_demo() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=120)
    tickers = FI_TICKERS + EQ_TICKERS
    rows = []
    for date in dates:
        scores = rng.normal(0, 1, len(tickers))
        for i, ticker in enumerate(tickers):
            universe = "fi" if ticker in FI_TICKERS else "equity"
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "score_raw": float(scores[i]),
                    "score_adj": float(scores[i]),
                    "ci_lower": float(scores[i] - rng.uniform(0.2, 0.5)),
                    "ci_upper": float(scores[i] + rng.uniform(0.2, 0.5)),
                    "vix": float(rng.uniform(12, 35)),
                    "dispersion_confidence": float(rng.uniform(0.4, 1.0)),
                    "alpha_w": 0.4,
                    "beta_w": 0.3,
                    "gamma_w": 0.2,
                    "delta_w": 0.1,
                    "universe": universe,
                }
            )
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("date")["score_adj"].rank(ascending=False, method="min").astype(int)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔄 MOMENTUM-REVERSAL")
    st.markdown(
        "**P2Quant Engine #93**  \n"
        "Jegadeesh-Titman Multi-Horizon  \n"
        "Momentum ↔ Reversal Transition"
    )
    st.divider()

    universe_opt = st.selectbox("Universe", ["combined", "fi", "equity"], index=0)
    lookback = st.slider("Lookback (trading days)", 21, 252, 126, step=21)
    top_n = st.slider("Top N ETFs", 1, 10, 5)
    show_ci = st.toggle("Show 95% confidence intervals", value=True)

    st.divider()
    st.markdown(
        f"**Results**  \n[{HF_RESULTS_REPO}](https://huggingface.co/datasets/{HF_RESULTS_REPO})\n\n"
        "**Repo**  \n[P2SAMAPA/P2-ETF-MOMENTUM-REVERSAL](https://github.com/P2SAMAPA/P2-ETF-MOMENTUM-REVERSAL)"
    )
    if st.button("🔄 Refresh data"):
        st.cache_data.clear()
        st.rerun()

# ── Load data ─────────────────────────────────────────────────────────────────
df_all, is_demo = load_results()
if is_demo:
    st.info("📊 Displaying **synthetic demo data**. Live results appear after the next training run.")

# Always filter by universe — "combined" shows the combined universe run, not all universes merged
df_all = df_all[df_all["universe"] == universe_opt]

max_date = df_all["date"].max()
cutoff = max_date - pd.Timedelta(days=lookback * 1.5)
df = df_all[df_all["date"] >= cutoff].copy()
latest_date = df["date"].max()

# Re-rank within this universe for the display (clean 1..N)
df["rank"] = df.groupby("date")["score_adj"].rank(ascending=False, method="min").astype(int)
df_today = df[df["date"] == latest_date].sort_values("rank")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🔄 Momentum-Reversal · ETF Rankings")
st.caption(
    f"Engine: **MOMENTUM-REVERSAL** · Universe: **{universe_opt.upper()}** · "
    f"Latest: **{latest_date.date()}** · {len(df_today)} ETFs scored"
    + (" · ⚠️ DEMO" if is_demo else " · ✅ Live")
)

# ── KPI row ───────────────────────────────────────────────────────────────────
avg_conf = df_today["dispersion_confidence"].mean() if "dispersion_confidence" in df_today.columns else 0
avg_vix = df_today["vix"].mean() if "vix" in df_today.columns else 0
alpha_w = df_today["alpha_w"].mean() if "alpha_w" in df_today.columns else 0.4
gamma_w = df_today["gamma_w"].mean() if "gamma_w" in df_today.columns else 0.2

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("ETFs Ranked", len(df_today))
c2.metric("VIX Level", f"{avg_vix:.1f}")
c3.metric("Signal Confidence", f"{avg_conf*100:.1f}%")
c4.metric("Momentum Weight (α)", f"{alpha_w:.2f}")
c5.metric("Reversal Weight (γ)", f"{gamma_w:.2f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Today's Rankings",
        "📈 Score History",
        "⚖️ Weight Dynamics",
        "ℹ️ Engine Info",
    ]
)

with tab1:
    st.subheader(f"Rankings for {latest_date.date()}")
    try:
        import plotly.graph_objects as go

        df_plot = df_today.sort_values("score_adj", ascending=False).reset_index(drop=True)
        colours = [UNIVERSE_COLOURS.get(t, "#888") for t in df_plot["ticker"]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=df_plot["ticker"],
                y=df_plot["score_adj"],
                marker_color=colours,
                marker_line_width=0,
                text=df_plot.apply(lambda r: f"#{int(r['rank'])}  {r['score_adj']:.2f}", axis=1),
                textposition="outside",
                textfont=dict(size=10),
                name="Score (z)",
            )
        )
        if show_ci and "ci_lower" in df_plot.columns:
            fig.update_traces(
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=(df_plot["ci_upper"] - df_plot["score_adj"]).clip(lower=0).tolist(),
                    arrayminus=(df_plot["score_adj"] - df_plot["ci_lower"]).clip(lower=0).tolist(),
                    color="rgba(80,80,80,0.45)",
                    thickness=1.5,
                    width=4,
                )
            )
        fig.add_hline(
            y=0,
            line_dash="dot",
            line_color="rgba(128,128,128,0.5)",
            line_width=1,
        )
        fig.update_layout(
            title=dict(
                text=f"Momentum-Reversal Rankings — {latest_date.date()} · sorted best → worst",
                font=dict(size=14),
            ),
            xaxis=dict(title="ETF", tickangle=-35, tickfont=dict(size=11)),
            yaxis=dict(title="Score (z-score)", zeroline=False),
            showlegend=False,
            height=540,
            margin=dict(t=70, b=90, l=60, r=20),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.3,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        st.bar_chart(df_today.set_index("ticker")["score_adj"])

    display_cols = [
        c
        for c in [
            "rank",
            "ticker",
            "score_adj",
            "score_raw",
            "ci_lower",
            "ci_upper",
            "vix",
            "dispersion_confidence",
        ]
        if c in df_today.columns
    ]
    st.dataframe(
        df_today[display_cols]
        .rename(
            columns={
                "score_adj": "Score (z)",
                "score_raw": "Score (raw)",
                "ci_lower": "CI Lower",
                "ci_upper": "CI Upper",
                "dispersion_confidence": "Confidence",
            }
        )
        .style.format(
            {
                "Score (z)": "{:.3f}",
                "Score (raw)": "{:.4f}",
                "CI Lower": "{:.3f}",
                "CI Upper": "{:.3f}",
                "vix": "{:.1f}",
                "Confidence": "{:.1%}",
            }
        ),
        use_container_width=True,
        height=350,
    )

with tab2:
    st.subheader("Score History — Top ETFs")
    pivot = df.pivot_table(index="date", columns="ticker", values="score_adj").sort_index()
    top_tickers = pivot.abs().mean().nlargest(top_n).index.tolist()
    try:
        import plotly.graph_objects as go

        fig2 = go.Figure()
        for ticker in top_tickers:
            if ticker not in pivot.columns:
                continue
            s = pivot[ticker].dropna()
            fig2.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s.values,
                    mode="lines",
                    name=ticker,
                    line=dict(width=2, color=UNIVERSE_COLOURS.get(ticker, "#888")),
                )
            )
        fig2.add_hline(y=0, line_dash="dot", line_color="gray", line_width=1)
        fig2.update_layout(
            title=f"Top {top_n} ETFs by Mean |Score|",
            xaxis_title="Date",
            yaxis_title="Score (z-score)",
            height=420,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        st.line_chart(pivot[top_tickers])

with tab3:
    st.subheader("OLS Weight Dynamics — How the Engine Adapts")
    st.caption(
        "α = momentum weight · β = core momentum · γ = reversal penalty · δ = long-run reversal drag  \n"
        "Weights are refitted every 63 days via rolling OLS on forward 21-day returns."
    )
    if all(c in df.columns for c in ["alpha_w", "beta_w", "gamma_w", "delta_w"]):
        weight_df = (
            df.groupby("date")[["alpha_w", "beta_w", "gamma_w", "delta_w"]]
            .mean()
            .reset_index()
            .sort_values("date")
            .tail(lookback)
        )
        try:
            import plotly.graph_objects as go

            fig3 = go.Figure()
            colours_w = {
                "alpha_w": "#1B4F8A",
                "beta_w": "#27AE60",
                "gamma_w": "#E74C3C",
                "delta_w": "#F39C12",
            }
            labels = {
                "alpha_w": "α (momentum)",
                "beta_w": "β (core mom)",
                "gamma_w": "γ (reversal)",
                "delta_w": "δ (long-run)",
            }
            for col in ["alpha_w", "beta_w", "gamma_w", "delta_w"]:
                fig3.add_trace(
                    go.Scatter(
                        x=weight_df["date"],
                        y=weight_df[col],
                        mode="lines",
                        name=labels[col],
                        line=dict(width=2, color=colours_w[col]),
                    )
                )
            fig3.update_layout(
                title="Rolling OLS Weights Over Time",
                xaxis_title="Date",
                yaxis_title="Weight",
                height=380,
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig3, use_container_width=True)
        except ImportError:
            st.line_chart(weight_df.set_index("date"))

        # Dispersion confidence
        if "dispersion_confidence" in df.columns:
            conf_df = df.groupby("date")["dispersion_confidence"].mean().reset_index().tail(lookback)
            st.subheader("Cross-Sectional Dispersion Confidence")
            st.caption("Below 50% = low signal environment; engine scores are dampened.")
            try:
                fig4 = go.Figure()
                conf_colours = [
                    "#E74C3C" if v < 0.5 else "#27AE60" for v in conf_df["dispersion_confidence"]
                ]
                fig4.add_trace(
                    go.Bar(
                        x=conf_df["date"],
                        y=conf_df["dispersion_confidence"],
                        marker_color=conf_colours,
                        name="Confidence",
                    )
                )
                fig4.add_hline(
                    y=0.5,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="50% threshold",
                )
                fig4.update_layout(
                    height=280,
                    yaxis=dict(tickformat=".0%"),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig4, use_container_width=True)
            except ImportError:
                st.line_chart(conf_df.set_index("date"))
    else:
        st.info("Weight dynamics not available in current dataset.")

with tab4:
    st.subheader("Engine Specification")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**Engine ID:** MOMENTUM-REVERSAL
**Category:** Cross-Sectional Momentum Decay & Reversal Detection
**Suite Version:** P2Quant v9 · April 2026

**Core Algorithm**
- Jegadeesh-Titman multi-horizon return decomposition
- 5 windows: 1m (reversal) · 3m · 6m · 12m-skip · 36m (long-run)
- Rolling OLS weight estimation (63-day refit, 504-day window)
- VIX regime conditioning
- Cross-sectional dispersion filter

**Scoring Formula**
```
Score = α·z(r_12m - r_1m) + β·z(r_6m)
      - γ·z(r_1m) - δ·z(r_36m)
```
- **r_12m**: skip-month adjusted (excludes last 21 days)
- **α, β, γ, δ**: fitted by rolling OLS on forward 21d returns

**Regime Conditioning**
- VIX > 25: boost γ (reversal penalty), reduce α (momentum crashes)
- VIX < 15: boost α (momentum persists), reduce γ
        """)
    with col_b:
        st.markdown("""
**Key Innovation**

No existing engine explicitly models the momentum-to-reversal transition.
The skip-month adjustment is critical — including the most recent month
biases toward short-term reversal, contaminating the momentum signal.

The dispersion filter prevents trading when all ETFs move together
(crisis periods) — low cross-sectional spread means no ranking signal.

**Horizon Interpretation**
- `r_1m < 0`: recent loser → reversal candidate (buy signal)
- `r_12m_skip > 0`: 12-month winner → momentum (buy signal)
- `r_36m > 0`: long-run winner → mean-reversion risk (sell signal)

High-conviction long = recent loser + long-term winner + low VIX

**References**
- Jegadeesh & Titman (1993) — *Returns to Buying Winners and Selling Losers*
- Jegadeesh & Titman (2001) — *Profitability of Momentum Strategies*
- De Bondt & Thaler (1985) — *Does the Stock Market Overreact?*
        """)
        st.link_button(
            "📦 GitHub Repo",
            "https://github.com/P2SAMAPA/P2-ETF-MOMENTUM-REVERSAL",
        )
        st.link_button(
            "🤗 Results Dataset",
            f"https://huggingface.co/datasets/{HF_RESULTS_REPO}",
        )
        st.link_button(
            "🤗 Input Data",
            f"https://huggingface.co/datasets/{HF_DATA_REPO}",
        )

st.divider()
st.caption(
    "P2Quant Engine Master Map v9 · April 2026 · "
    "[P2SAMAPA](https://github.com/P2SAMAPA) · "
    f"Last refreshed: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
)
