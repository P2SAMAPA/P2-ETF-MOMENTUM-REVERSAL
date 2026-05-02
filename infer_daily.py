"""infer_daily.py — Run inference for today's date and push scores."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

from logging_utils import get_logger
from main import COMBINED_TICKERS, EQ_TICKERS, FI_TICKERS
from momentum_reversal import (
    DISPERSION_WINDOW,
    MIN_HISTORY,
    compute_multi_horizon_returns,
    compute_raw_scores,
    cross_sectional_zscore,
    dispersion_filter,
    fit_ols_weights,
    vix_regime_adjust,
)
from publisher import push_results

log = get_logger("momentum_reversal.infer_daily")


def run_daily_inference(universe: str = "combined") -> None:
    hf_token = os.environ.get("HF_TOKEN")

    path = hf_hub_download(
        repo_id="P2SAMAPA/fi-etf-macro-signal-master-data",
        filename="master_data.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)

    if universe == "fi":
        tickers = [t for t in FI_TICKERS if t in df.columns]
    elif universe == "equity":
        tickers = [t for t in EQ_TICKERS if t in df.columns]
    else:
        tickers = [t for t in COMBINED_TICKERS if t in df.columns]

    # Data is ETF closing prices — use directly, forward-fill any gaps
    prices = df[tickers].copy().ffill()
    vix = df["VIX"] if "VIX" in df.columns else pd.Series(20.0, index=df.index)

    if len(prices) < MIN_HISTORY:
        log.warning("Insufficient history for inference (%d rows)", len(prices))
        return

    multi_ret = compute_multi_horizon_returns(prices)
    log_p = prices.apply(lambda c: c.apply(lambda x: float(np.log(max(x, 1e-8)))))
    forward_ret = (log_p.shift(-21) - log_p).dropna(how="all")

    # Fit weights on all available history
    date_idx = len(multi_ret) - 1
    weights = fit_ols_weights(multi_ret, forward_ret, date_idx)

    # Today's date
    today = pd.Timestamp.today().normalize()
    if today.weekday() >= 5:
        today = today + pd.offsets.BDay(1)

    # Use last available date in data
    last_date = multi_ret.index[-1]
    vix_level = float(vix.get(last_date, 20.0))
    adj_weights = vix_regime_adjust(weights, vix_level)

    # Dispersion confidence
    # multi_ret has MultiIndex columns: (ticker, horizon) — use xs to slice by horizon
    r12m_all = multi_ret.xs("r_12m_skip", axis=1, level=1)
    r12m_history = r12m_all.std(axis=1).dropna()
    disp_series = r12m_history.tail(DISPERSION_WINDOW)
    r12m_today = r12m_all.iloc[-1]
    confidence = dispersion_filter(r12m_today, disp_series)

    mr_row = multi_ret.iloc[[-1]]
    raw_scores = compute_raw_scores(mr_row, adj_weights) * confidence
    final_z = cross_sectional_zscore(raw_scores)

    ci_half = 1.96 * raw_scores.std() / max(len(tickers) ** 0.5, 1)

    rows = []
    for ticker in tickers:
        score = float(final_z.get(ticker, 0.0))
        raw = float(raw_scores.get(ticker, 0.0))
        rows.append(
            {
                "date": today.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "score_raw": raw if np.isfinite(raw) else 0.0,
                "score_adj": score if np.isfinite(score) else 0.0,
                "ci_lower": (score - ci_half) if np.isfinite(score) else 0.0,
                "ci_upper": (score + ci_half) if np.isfinite(score) else 0.0,
                "vix": vix_level,
                "dispersion_confidence": round(confidence, 4),
                "alpha_w": round(adj_weights["alpha"], 4),
                "beta_w": round(adj_weights["beta"], 4),
                "gamma_w": round(adj_weights["gamma"], 4),
                "delta_w": round(adj_weights["delta"], 4),
                "universe": universe,
            }
        )

    daily_df = pd.DataFrame(rows)
    daily_df["rank"] = (
        daily_df["score_adj"].rank(ascending=False, method="min").astype(int)
    )

    log.info(
        "Daily inference: %s | %d ETFs | VIX=%.1f | confidence=%.2f",
        today.date(),
        len(rows),
        vix_level,
        confidence,
    )
    push_results(daily_df, token=hf_token)
    log.info("Today's scores pushed ✅")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--universe", default="combined", choices=["fi", "equity", "combined"]
    )
    args = parser.parse_args()
    run_daily_inference(args.universe)
