"""momentum_reversal.py — Cross-Sectional Momentum-Reversal Transition Engine.

Implements the Jegadeesh-Titman multi-horizon framework:
    - 1-month  : short-term reversal zone
    - 3-month  : early momentum
    - 6-month  : core momentum
    - 12-month : full momentum (skip-month adjusted, excludes last 21 days)
    - 36-month : long-run reversal zone

Score = α·z(r_12m - r_1m) + β·z(r_6m) - γ·z(r_1m) - δ·z(r_36m)
Weights fitted by rolling OLS on forward 21-day returns, re-estimated every 63 days.
VIX regime conditioning adjusts weights in high/low volatility environments.
Dispersion filter suppresses signal when cross-sectional spread is low.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
LOOKBACKS = {
    "r_1m": 21,
    "r_3m": 63,
    "r_6m": 126,
    "r_12m_skip": (22, 252),  # skip-month: days 22 to 252
    "r_36m": 756,
}

VIX_HIGH = 25.0
VIX_LOW = 15.0
DISPERSION_WINDOW = 126  # 6-month rolling dispersion baseline
OLS_REFIT_FREQ = 63  # refit weights every quarter
OLS_TRAIN_WINDOW = 504  # 2-year rolling OLS training window
FORWARD_RETURN_DAYS = 21  # predict 1-month forward return
MIN_HISTORY = 756 + 21  # need 36m + 1m forward


def compute_multi_horizon_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute multi-horizon returns for each ETF.

    Args:
        prices: (T, N) DataFrame of adjusted close prices.

    Returns:
        DataFrame with MultiIndex columns: (ticker, horizon).
    """
    tickers = prices.columns.tolist()
    log_prices = np.log(prices)
    
    # Dictionary to collect data for each ticker
    all_data = {}
    
    for ticker in tickers:
        lp = log_prices[ticker]
        
        # FIX: Create a temporary DataFrame with all horizons as columns
        # Use .values to ensure we get 1D arrays
        temp_df = pd.DataFrame(index=prices.index)
        
        # Simple horizon returns
        temp_df["r_1m"] = (lp - lp.shift(21)).values
        temp_df["r_3m"] = (lp - lp.shift(63)).values
        temp_df["r_6m"] = (lp - lp.shift(126)).values
        temp_df["r_36m"] = (lp - lp.shift(756)).values
        
        # Skip-month momentum: 12m return excluding most recent month
        temp_df["r_12m_skip"] = (lp.shift(21) - lp.shift(252)).values
        
        # Store as a DataFrame with multi-level columns
        all_data[ticker] = temp_df
    
    # Concatenate along axis=1 with outer level as ticker
    result = pd.concat(all_data, axis=1)
    
    # Rename columns to have (ticker, horizon) structure
    result.columns = result.columns.swaplevel(0, 1)
    result.columns = result.columns.set_names(["horizon", "ticker"])
    
    return result


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score across ETFs for a single date."""
    mu = series.mean()
    sigma = series.std()
    if sigma < 1e-8:
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def compute_raw_scores(
    multi_returns: pd.DataFrame,
    weights: dict[str, float],
) -> pd.Series:
    """Compute the momentum-reversal composite score for one date.

    Args:
        multi_returns: Single-row slice with columns (ticker, horizon).
        weights: Dict with keys alpha, beta, gamma, delta.

    Returns:
        Series of scores indexed by ticker.
    """
    # MultiIndex columns are (ticker, horizon) — use xs to slice by horizon (level=1)
    tickers = multi_returns.columns.get_level_values(0).unique()

    def get_horizon(name: str) -> pd.Series:
        try:
            return multi_returns.xs(name, axis=1, level=1).iloc[0]
        except (KeyError, IndexError):
            return pd.Series(np.nan, index=tickers)

    r_1m = get_horizon("r_1m")
    r_6m = get_horizon("r_6m")
    r_12m = get_horizon("r_12m_skip")
    r_36m = get_horizon("r_36m")

    # Cross-sectional z-scores
    z_momentum = cross_sectional_zscore(r_12m - r_1m)  # momentum minus reversal
    z_core = cross_sectional_zscore(r_6m)
    z_reversal = cross_sectional_zscore(r_1m)
    z_longrun = cross_sectional_zscore(r_36m)

    score = (
        weights["alpha"] * z_momentum
        + weights["beta"] * z_core
        - weights["gamma"] * z_reversal
        - weights["delta"] * z_longrun
    )

    return score


def fit_ols_weights(
    multi_returns: pd.DataFrame,
    forward_returns: pd.DataFrame,
    train_end_idx: int,
    train_window: int = OLS_TRAIN_WINDOW,
) -> dict[str, float]:
    """Fit OLS weights on historical data.

    Regresses forward 21-day return on the 4 z-score factors.

    Returns:
        Dict with alpha, beta, gamma, delta. Falls back to equal weights on failure.
    """
    default = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}

    start = max(0, train_end_idx - train_window)
    dates = multi_returns.index[start:train_end_idx]

    if len(dates) < 126:
        return default

    # FIX: Get tickers from the new column structure
    tickers = multi_returns.columns.get_level_values(0).unique().tolist()
    x_rows, y_rows = [], []

    for date in dates:
        if date not in forward_returns.index:
            continue
        fwd = forward_returns.loc[date]

        try:
            mr_row = multi_returns.loc[[date]]
        except KeyError:
            continue

        for ticker in tickers:
            if ticker not in fwd.index:
                continue
            y_val = fwd[ticker]
            if not np.isfinite(y_val):
                continue

            # Build feature vector for this ticker on this date
            # MultiIndex columns are (ticker, horizon) — xs by horizon level=1
            def gz(name: str) -> float:
                try:
                    val = mr_row.xs(name, axis=1, level=1)[ticker].iloc[0]
                    return float(val) if np.isfinite(val) else 0.0
                except Exception:
                    return 0.0

            r_1m_v = gz("r_1m")
            r_6m_v = gz("r_6m")
            r_12m_v = gz("r_12m_skip")
            r_36m_v = gz("r_36m")

            x_rows.append([r_12m_v - r_1m_v, r_6m_v, -r_1m_v, -r_36m_v])
            y_rows.append(y_val)

    if len(x_rows) < 50:
        return default

    x_mat = np.array(x_rows)
    y = np.array(y_rows)

    # Winsorise to prevent outlier contamination
    y = np.clip(y, np.percentile(y, 1), np.percentile(y, 99))

    try:
        # Full OLS via least squares
        coeffs, _, _, _ = np.linalg.lstsq(np.column_stack([np.ones(len(x_mat)), x_mat]), y, rcond=None)
        alpha, beta, gamma, delta = (
            coeffs[1],
            coeffs[2],
            abs(coeffs[3]),
            abs(coeffs[4]),
        )
        # Normalise so they sum to 1
        total = abs(alpha) + abs(beta) + abs(gamma) + abs(delta) + 1e-8
        return {
            "alpha": max(0, alpha) / total,
            "beta": max(0, beta) / total,
            "gamma": max(0, gamma) / total,
            "delta": max(0, delta) / total,
        }
    except Exception:
        return default


def vix_regime_adjust(
    weights: dict[str, float],
    vix_level: float,
) -> dict[str, float]:
    """Adjust weights based on VIX regime.

    High VIX: strengthen reversal penalty (momentum crashes in high vol).
    Low VIX: strengthen momentum component (trends persist in calm markets).
    """
    w = weights.copy()
    if vix_level > VIX_HIGH:
        # High vol: boost reversal penalty, reduce momentum
        scale = min((vix_level - VIX_HIGH) / 15.0, 1.0)
        w["gamma"] = w["gamma"] * (1 + scale)
        w["alpha"] = w["alpha"] * (1 - 0.3 * scale)
    elif vix_level < VIX_LOW:
        # Low vol: boost momentum, reduce reversal penalty
        scale = min((VIX_LOW - vix_level) / 10.0, 1.0)
        w["alpha"] = w["alpha"] * (1 + 0.3 * scale)
        w["gamma"] = w["gamma"] * (1 - 0.2 * scale)
    # Re-normalise
    total = sum(abs(v) for v in w.values()) + 1e-8
    return {k: v / total for k, v in w.items()}


def dispersion_filter(
    cross_section_r12m: pd.Series,
    dispersion_history: pd.Series,
) -> float:
    """Return signal confidence [0, 1] based on cross-sectional dispersion.

    Low dispersion = no meaningful momentum signal → confidence near 0.
    High dispersion = strong cross-sectional spread → confidence near 1.
    """
    current_disp = cross_section_r12m.std()
    if len(dispersion_history) < 21:
        return 1.0
    median_disp = dispersion_history.median()
    if median_disp < 1e-8:
        return 1.0
    # Linear scaling — confidence = ratio clipped to [0.2, 1.0]
    # Avoids sigmoid over-suppression; minimum 20% confidence always passed through
    ratio = current_disp / median_disp
    confidence = float(np.clip(ratio, 0.2, 1.0))
    return confidence


def run_engine(
    prices: pd.DataFrame,  # (T, N) adjusted close prices
    vix: pd.Series,  # (T,) VIX levels
    universe: str = "combined",
    output_dir: str = "results",
) -> pd.DataFrame:
    """Run the full momentum-reversal engine.

    Args:
        prices:     Adjusted close prices for all ETFs.
        vix:        VIX index series aligned with prices.
        universe:   'fi', 'equity', or 'combined'.
        output_dir: Directory to save results CSV.

    Returns:
        DataFrame with daily scores for all ETFs.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tickers = prices.columns.tolist()
    print(f"Running MOMENTUM-REVERSAL engine: {len(tickers)} ETFs, {len(prices)} days")

    # Compute multi-horizon returns
    multi_ret = compute_multi_horizon_returns(prices)

    # Forward returns for OLS fitting
    log_prices = np.log(prices)
    forward_ret = (log_prices.shift(-FORWARD_RETURN_DAYS) - log_prices).dropna(how="all")

    # Dispersion history tracker
    disp_history: list[float] = []

    # OLS weights — refit every OLS_REFIT_FREQ days
    current_weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}
    last_fit_idx = -1

    rows = []
    valid_dates = multi_ret.index[multi_ret.index >= prices.index[MIN_HISTORY]]

    for i, date in enumerate(valid_dates):
        date_idx = multi_ret.index.get_loc(date)

        # Refit OLS weights periodically
        if date_idx - last_fit_idx >= OLS_REFIT_FREQ:
            current_weights = fit_ols_weights(multi_ret, forward_ret, date_idx)
            last_fit_idx = date_idx

        # VIX regime adjustment
        vix_level = float(vix.get(date, 20.0))
        adj_weights = vix_regime_adjust(current_weights, vix_level)

        # FIX: Get r_12m_skip data for today
        r12m_today = (
            multi_ret.xs("r_12m_skip", axis=1, level=1).loc[date]
            if "r_12m_skip" in multi_ret.columns.get_level_values(1)
            else pd.Series()
        )
        disp_series = pd.Series(disp_history[-DISPERSION_WINDOW:])
        confidence = dispersion_filter(r12m_today, disp_series)
        if len(r12m_today) > 0:
            disp_history.append(r12m_today.std())

        # Compute scores
        mr_row = multi_ret.loc[[date]]
        raw_scores = compute_raw_scores(mr_row, adj_weights)
        raw_scores = raw_scores * confidence  # apply dispersion scaling

        # Final cross-sectional z-score
        final_z = cross_sectional_zscore(raw_scores)

        # CI via bootstrap (simple ±1 std of component z-scores)
        component_std = raw_scores.std()
        ci_half = 1.96 * component_std / max(len(tickers) ** 0.5, 1)

        for ticker in tickers:
            score = float(final_z.get(ticker, 0.0))
            raw = float(raw_scores.get(ticker, 0.0))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
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

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(valid_dates)} dates...")

    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("date")["score_adj"].rank(ascending=False, method="min").astype(int)

    scores_path = out / "scores.csv"
    df.to_csv(scores_path, index=False)
    print(f"Scores saved → {scores_path} ({len(df)} rows)")

    return df
