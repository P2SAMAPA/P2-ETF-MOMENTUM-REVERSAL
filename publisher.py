"""publisher.py — Push results to HuggingFace Dataset."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

from logging_utils import get_logger

log = get_logger(__name__)

HF_RESULTS_REPO = "P2SAMAPA/p2-etf-momentum-reversal-results"


def push_results(
    results_df: pd.DataFrame,
    hf_repo: str = HF_RESULTS_REPO,
    token: str | None = None,
) -> None:
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise EnvironmentError("Set HF_TOKEN environment variable or pass token=")

    log.info("Pushing %d rows to %s", len(results_df), hf_repo)
    df = results_df.copy()
    if pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    try:
        existing = load_dataset(hf_repo, split="train", token=hf_token)
        df_existing = existing.to_pandas()
        df_existing["date"] = pd.to_datetime(df_existing["date"]).dt.strftime("%Y-%m-%d")
        dedup_cols = [
            c for c in ["date", "ticker", "universe"] if c in df_existing.columns and c in df.columns
        ]
        if dedup_cols:
            new_keys = set(zip(*[df[c] for c in dedup_cols]))
            mask = ~pd.Series(list(zip(*[df_existing[c] for c in dedup_cols]))).isin(new_keys)
            df_existing = df_existing[mask.values]
        combined = pd.concat([df_existing, df], ignore_index=True)
        dedup_cols2 = [c for c in ["date", "ticker", "universe"] if c in combined.columns]
        if dedup_cols2:
            combined = combined.drop_duplicates(subset=dedup_cols2, keep="last")
        log.info(
            "Merged: %d existing + %d new = %d total rows",
            len(df_existing),
            len(df),
            len(combined),
        )
    except Exception as exc:
        log.warning("No existing dataset (%s) — creating fresh.", exc)
        combined = df

    combined = combined.sort_values(
        [c for c in ["date", "universe", "ticker"] if c in combined.columns]
    ).reset_index(drop=True)

    dataset = Dataset.from_pandas(combined, preserve_index=False)
    dataset.push_to_hub(hf_repo, split="train", token=hf_token)
    log.info("Pushed successfully → %s (%d rows)", hf_repo, len(combined))


def save_locally(results_df: pd.DataFrame, output_dir: str = "results") -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    min_date = results_df["date"].astype(str).min()
    max_date = results_df["date"].astype(str).max()
    path = out / f"scores_{min_date}_{max_date}.parquet"
    results_df.to_parquet(path, index=False)
    log.info("Saved locally → %s", path)
    return path
