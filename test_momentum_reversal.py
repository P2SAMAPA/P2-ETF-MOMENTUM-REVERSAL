"""test_momentum_reversal.py — Unit tests for the momentum-reversal engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from momentum_reversal import (
    compute_multi_horizon_returns,
    compute_raw_scores,
    cross_sectional_zscore,
    dispersion_filter,
    vix_regime_adjust,
)


@pytest.fixture()
def sample_prices() -> pd.DataFrame:
    """Generate synthetic price data for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-01", periods=1000)
    tickers = ["SPY", "QQQ", "TLT", "GLD", "HYG"]
    returns = rng.normal(0.0003, 0.012, (1000, 5))
    prices = pd.DataFrame(
        (1 + returns).cumprod(axis=0) * 100,
        index=dates,
        columns=tickers,
    )
    return prices


def test_multi_horizon_returns_shape(sample_prices):
    result = compute_multi_horizon_returns(sample_prices)
    assert isinstance(result, pd.DataFrame)
    assert "r_1m" in result.columns.get_level_values(0)
    assert "r_12m_skip" in result.columns.get_level_values(0)
    assert result.shape[0] == len(sample_prices)


def test_multi_horizon_skip_month(sample_prices):
    """Skip-month r_12m should differ from straight 252-day return."""
    multi = compute_multi_horizon_returns(sample_prices)
    ticker = sample_prices.columns[0]
    lp = np.log(sample_prices[ticker])
    straight_12m = lp - lp.shift(252)
    skip_12m = multi["r_12m_skip"][ticker]
    # They should differ (skip excludes last month)
    diff = (straight_12m - skip_12m).dropna().abs().mean()
    assert diff > 0


def test_cross_sectional_zscore():
    s = pd.Series({"A": 1.0, "B": 2.0, "C": 3.0, "D": 0.0})
    z = cross_sectional_zscore(s)
    assert abs(z.mean()) < 1e-10
    assert abs(z.std() - 1.0) < 0.1


def test_cross_sectional_zscore_all_same():
    s = pd.Series({"A": 5.0, "B": 5.0, "C": 5.0})
    z = cross_sectional_zscore(s)
    assert (z == 0.0).all()


def test_vix_regime_adjust_high_vix():
    base = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}
    adj = vix_regime_adjust(base, vix_level=35.0)
    # High VIX: gamma should increase relative to alpha
    assert adj["gamma"] / adj["alpha"] > base["gamma"] / base["alpha"]


def test_vix_regime_adjust_low_vix():
    base = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}
    adj = vix_regime_adjust(base, vix_level=10.0)
    # Low VIX: alpha should increase relative to gamma
    assert adj["alpha"] / adj["gamma"] > base["alpha"] / base["gamma"]


def test_vix_regime_adjust_weights_positive():
    base = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}
    for vix in [10, 20, 30, 40]:
        adj = vix_regime_adjust(base, vix_level=float(vix))
        assert all(v >= 0 for v in adj.values())


def test_dispersion_filter_high_dispersion():
    """High dispersion relative to history → confidence near 1."""
    current = pd.Series(np.random.randn(25) * 0.5)
    history = pd.Series([0.05] * 100)  # low historical dispersion
    conf = dispersion_filter(current, history)
    assert conf > 0.5


def test_dispersion_filter_low_dispersion():
    """Low dispersion relative to history → confidence near 0."""
    current = pd.Series([0.001] * 25)  # very low current dispersion
    history = pd.Series([0.5] * 100)  # high historical dispersion
    conf = dispersion_filter(current, history)
    assert conf < 0.5


def test_dispersion_filter_empty_history():
    current = pd.Series(np.random.randn(10))
    conf = dispersion_filter(current, pd.Series([], dtype=float))
    assert conf == 1.0


def test_compute_raw_scores_finite(sample_prices):
    multi = compute_multi_horizon_returns(sample_prices)
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}
    # Use a date with sufficient history
    date = multi.index[800]
    row = multi.loc[[date]]
    scores = compute_raw_scores(row, weights)
    finite_scores = scores.dropna()
    assert len(finite_scores) > 0
    assert all(np.isfinite(v) for v in finite_scores)


def test_compute_raw_scores_returns_series(sample_prices):
    multi = compute_multi_horizon_returns(sample_prices)
    weights = {"alpha": 0.4, "beta": 0.3, "gamma": 0.2, "delta": 0.1}
    date = multi.index[800]
    scores = compute_raw_scores(multi.loc[[date]], weights)
    assert isinstance(scores, pd.Series)
    assert len(scores) == len(sample_prices.columns)
