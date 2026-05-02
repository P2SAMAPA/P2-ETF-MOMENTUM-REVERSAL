"""main.py — CLI orchestrator for P2-ETF-MOMENTUM-REVERSAL engine."""

from __future__ import annotations

import argparse
import os

import pandas as pd
from huggingface_hub import hf_hub_download

from logging_utils import get_logger
from momentum_reversal import run_engine
from publisher import push_results

log = get_logger("momentum_reversal.main")

HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_RESULTS_REPO = "P2SAMAPA/p2-etf-momentum-reversal-results"

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
COMBINED_TICKERS = FI_TICKERS + EQ_TICKERS


def load_master_data(hf_token: str | None = None) -> pd.DataFrame:
    path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        filename="master_data.parquet",
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    log.info("Loaded master data: %d rows x %d cols", len(df), len(df.columns))
    return df


def get_universe(df: pd.DataFrame, universe: str) -> tuple[pd.DataFrame, pd.Series]:
    if universe == "fi":
        tickers = [t for t in FI_TICKERS if t in df.columns]
    elif universe == "equity":
        tickers = [t for t in EQ_TICKERS if t in df.columns]
    else:
        tickers = [t for t in COMBINED_TICKERS if t in df.columns]

    # Build price series from log-returns
    log_ret = df[tickers].copy()
    # Reconstruct price index (base 100)
    prices = (1 + log_ret.fillna(0)).cumprod() * 100
    vix = df["VIX"] if "VIX" in df.columns else pd.Series(20.0, index=df.index)
    log.info("Universe '%s': %d tickers", universe, len(tickers))
    return prices, vix


def cmd_run(args: argparse.Namespace) -> None:
    hf_token = os.environ.get("HF_TOKEN")
    df = load_master_data(hf_token)
    prices, vix = get_universe(df, args.universe)
    scores_df = run_engine(
        prices=prices,
        vix=vix,
        universe=args.universe,
        output_dir=args.output_dir,
    )
    log.info("Pushing %d rows to HuggingFace...", len(scores_df))
    push_results(scores_df, token=hf_token)
    log.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="P2-ETF-MOMENTUM-REVERSAL Engine")
    parser.add_argument(
        "--universe", default="combined", choices=["fi", "equity", "combined"]
    )
    parser.add_argument("--output_dir", default="results")
    cmd_run(parser.parse_args())
