"""Demo market data generator for the Virtual ETF Portfolio Optimizer.

The app is designed to use live Yahoo Finance data through yfinance, but a
public web demo should never break just because a data vendor is temporarily
unavailable. This module creates realistic synthetic ETF-like price series for
portfolio research demonstrations.

The generated data is NOT real market data and should be labeled as demo data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AssetProfile:
    """Synthetic return profile for one ETF-like asset."""

    annual_return: float
    annual_volatility: float
    market_beta: float
    rate_beta: float
    inflation_beta: float
    defensive_beta: float = 0.0


DEFAULT_PROFILES: dict[str, AssetProfile] = {
    "SPY": AssetProfile(0.085, 0.170, 1.00, -0.10, 0.05),
    "QQQ": AssetProfile(0.105, 0.230, 1.25, -0.15, 0.03),
    "IWM": AssetProfile(0.080, 0.245, 1.15, -0.05, 0.08),
    "EFA": AssetProfile(0.070, 0.190, 0.85, -0.05, 0.15),
    "EEM": AssetProfile(0.078, 0.250, 0.95, -0.02, 0.20),
    "TLT": AssetProfile(0.038, 0.145, -0.25, 0.95, -0.10, 0.25),
    "IEF": AssetProfile(0.032, 0.070, -0.12, 0.55, -0.05, 0.15),
    "GLD": AssetProfile(0.055, 0.165, 0.05, 0.10, 0.65, 0.20),
    "VNQ": AssetProfile(0.075, 0.220, 0.80, -0.35, 0.20),
    "DBC": AssetProfile(0.050, 0.210, 0.25, -0.05, 0.85),
    "VTI": AssetProfile(0.084, 0.165, 1.00, -0.10, 0.05),
    "VXUS": AssetProfile(0.071, 0.190, 0.82, -0.05, 0.16),
    "BND": AssetProfile(0.030, 0.060, -0.10, 0.40, -0.05, 0.15),
}

FALLBACK_PROFILE = AssetProfile(0.065, 0.180, 0.70, 0.00, 0.10)
TRADING_DAYS = 252


def _parse_date(value: Optional[str | date | datetime], fallback: str) -> pd.Timestamp:
    """Convert a date-like value to pandas Timestamp."""
    if value is None:
        return pd.Timestamp(fallback)
    return pd.Timestamp(value)


def generate_demo_prices(
    tickers: Iterable[str],
    start_date: str | date | datetime = "2015-01-01",
    end_date: Optional[str | date | datetime] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic ETF-like adjusted close prices.

    Parameters
    ----------
    tickers:
        Asset symbols to generate.
    start_date, end_date:
        Business-date range. If ``end_date`` is omitted, today's date is used.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Synthetic adjusted close price data indexed by business date.
    """
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]
    if len(clean_tickers) < 2:
        raise ValueError("At least two tickers are required to generate demo prices.")

    start = _parse_date(start_date, "2015-01-01")
    end = _parse_date(end_date, datetime.today().strftime("%Y-%m-%d"))
    if end <= start:
        raise ValueError("end_date must be later than start_date.")

    dates = pd.bdate_range(start=start, end=end)
    if len(dates) < 252:
        raise ValueError("Demo data requires at least one year of business dates.")

    rng = np.random.default_rng(seed)
    n = len(dates)

    # Common factors: equity market, rates, inflation/commodities, defensive flight-to-quality.
    factor_vols = np.array([0.0105, 0.0045, 0.0065, 0.0040])
    factors = rng.normal(0.0, factor_vols, size=(n, 4))

    # Add a few controlled stress episodes to make drawdown/risk charts interesting.
    stress_windows = [
        (int(n * 0.33), int(n * 0.36), -0.009, 0.003, -0.001, 0.004),
        (int(n * 0.62), int(n * 0.64), -0.013, -0.001, 0.004, 0.005),
        (int(n * 0.82), int(n * 0.84), -0.008, 0.002, 0.002, 0.003),
    ]
    for left, right, mkt, rates, infl, defensive in stress_windows:
        if 0 <= left < right <= n:
            factors[left:right, 0] += mkt
            factors[left:right, 1] += rates
            factors[left:right, 2] += infl
            factors[left:right, 3] += defensive

    prices: dict[str, pd.Series] = {}
    for i, ticker in enumerate(clean_tickers):
        profile = DEFAULT_PROFILES.get(ticker, FALLBACK_PROFILE)
        beta_vector = np.array(
            [profile.market_beta, profile.rate_beta, profile.inflation_beta, profile.defensive_beta]
        )
        factor_component = factors @ beta_vector
        target_daily_vol = profile.annual_volatility / np.sqrt(TRADING_DAYS)
        idiosyncratic_vol = max(0.001, target_daily_vol * 0.45)
        noise = rng.normal(0.0, idiosyncratic_vol, size=n)
        daily_drift = profile.annual_return / TRADING_DAYS
        log_returns = daily_drift - 0.5 * target_daily_vol**2 + factor_component + noise
        # Slightly stagger starting prices so charts do not overlap perfectly.
        start_price = 100.0 + 5.0 * i
        prices[ticker] = pd.Series(start_price * np.exp(np.cumsum(log_returns)), index=dates)

    price_frame = pd.DataFrame(prices).round(4)
    return price_frame
