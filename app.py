# Virtual ETF Portfolio Optimizer Streamlit app
# Public demo app with editable sidebar controls and interactive charts.

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

# ======================== PORTFOLIO ENGINE ========================


from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

try:
    from sklearn.covariance import LedoitWolf
except Exception:  # pragma: no cover - optional fallback
    LedoitWolf = None  # type: ignore


TRADING_DAYS = 252
EPSILON = 1e-12

ObjectiveName = Literal[
    "equal_weight",
    "minimum_volatility",
    "maximum_sharpe",
    "risk_parity",
    "maximum_diversification",
]

DISPLAY_NAMES: dict[str, str] = {
    "equal_weight": "Equal Weight",
    "minimum_volatility": "Minimum Volatility",
    "maximum_sharpe": "Maximum Sharpe",
    "risk_parity": "Risk Parity",
    "maximum_diversification": "Maximum Diversification",
}


class MarketDataError(RuntimeError):
    """Raised when market data cannot be downloaded or parsed."""


class OptimizationError(RuntimeError):
    """Raised when a constrained optimization problem is infeasible."""


@dataclass
class OptimizerSettings:
    """Configuration for one virtual portfolio optimization study."""

    tickers: list[str] = field(
        default_factory=lambda: ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "VNQ", "DBC"]
    )
    benchmark: str = "SPY"
    start_date: str = "2015-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 100_000.0
    min_weight: float = 0.00
    max_weight: float = 0.35
    risk_free_rate: float = 0.03
    lookback_days: int = 756
    rebalance_frequency: str = "ME"
    transaction_cost_bps: float = 5.0
    min_history_ratio: float = 0.90
    winsorize_returns: bool = False
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    covariance_method: str = "ledoit_wolf"
    random_seed: int = 42

    def normalized_tickers(self) -> list[str]:
        """Return unique uppercase tickers while preserving order."""
        return normalize_tickers(self.tickers)


def normalize_tickers(tickers: Iterable[str]) -> list[str]:
    """Normalize user-provided ticker symbols."""
    clean: list[str] = []
    for ticker in tickers:
        symbol = str(ticker).strip().upper()
        if symbol and symbol not in clean:
            clean.append(symbol)
    return clean


def validate_settings(settings: OptimizerSettings) -> None:
    """Validate portfolio settings before data or optimization work begins."""
    tickers = settings.normalized_tickers()
    if len(tickers) < 2:
        raise ValueError("At least two ETF tickers are required.")
    if settings.initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if not 0.0 <= settings.min_weight <= settings.max_weight <= 1.0:
        raise ValueError("Weight bounds must satisfy 0 <= min_weight <= max_weight <= 1.")
    n_assets = len(tickers)
    if n_assets * settings.min_weight > 1.0 + 1e-12:
        raise ValueError("Minimum weight constraint is infeasible for this number of assets.")
    if n_assets * settings.max_weight < 1.0 - 1e-12:
        raise ValueError("Maximum weight constraint is infeasible for this number of assets.")
    if settings.lookback_days < 126:
        raise ValueError("lookback_days should be at least 126 trading days.")
    if settings.transaction_cost_bps < 0:
        raise ValueError("transaction_cost_bps cannot be negative.")
    if not 0 <= settings.winsorize_lower < settings.winsorize_upper <= 1:
        raise ValueError("Invalid winsorization quantile bounds.")


def download_yahoo_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Download adjusted ETF close prices from Yahoo Finance via yfinance.

    yfinance is imported lazily so this project can still run in demo mode if
    the package is unavailable.
    """
    symbols = normalize_tickers(tickers)
    if len(symbols) < 2:
        raise ValueError("download_yahoo_prices requires at least two symbols.")

    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise MarketDataError("yfinance is not installed. Install requirements or use demo mode.") from exc

    try:
        raw = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        raise MarketDataError(f"Yahoo Finance download failed: {exc}") from exc

    if raw is None or raw.empty:
        raise MarketDataError("Yahoo Finance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(raw.columns.get_level_values(0))
        if "Close" in level0:
            prices = raw["Close"].copy()
        elif "Adj Close" in level0:
            prices = raw["Adj Close"].copy()
        else:
            raise MarketDataError("Could not find Close or Adj Close columns in Yahoo response.")
    else:
        close_col = "Close" if "Close" in raw.columns else "Adj Close" if "Adj Close" in raw.columns else None
        if close_col is None:
            raise MarketDataError("Could not find Close or Adj Close in Yahoo response.")
        prices = raw[[close_col]].copy()
        prices.columns = symbols[:1]

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()].copy()
    return prices


def load_market_prices(settings: OptimizerSettings, data_mode: str = "demo", allow_fallback: bool = True) -> tuple[pd.DataFrame, str]:
    """Load market prices from live Yahoo Finance data or generated demo data.

    Returns a tuple of ``(prices, data_label)``.
    """
    validate_settings(settings)
    tickers = settings.normalized_tickers()
    mode = str(data_mode).strip().lower()

    if mode in {"live", "yahoo", "yfinance", "live yahoo finance"}:
        try:
            return download_yahoo_prices(tickers, settings.start_date, settings.end_date), "Live Yahoo Finance data"
        except Exception as exc:
            if not allow_fallback:
                raise
            print(f"Live data failed; falling back to demo data. Reason: {exc}")

    return (
        generate_demo_prices(
            tickers=tickers,
            start_date=settings.start_date,
            end_date=settings.end_date or datetime.today().strftime("%Y-%m-%d"),
            seed=settings.random_seed,
        ),
        "Synthetic demo data",
    )


def clean_prices(prices: pd.DataFrame, settings: OptimizerSettings) -> pd.DataFrame:
    """Clean adjusted close prices and remove unreliable assets."""
    if prices is None or prices.empty:
        raise MarketDataError("Price frame is empty.")

    clean = prices.copy()
    clean.index = pd.to_datetime(clean.index)
    clean = clean.sort_index()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    clean = clean.apply(pd.to_numeric, errors="coerce")

    valid_ratio = clean.notna().mean()
    keep_columns = valid_ratio[valid_ratio >= settings.min_history_ratio].index.tolist()
    if len(keep_columns) < 2:
        raise MarketDataError("Fewer than two assets passed the minimum history filter.")
    clean = clean[keep_columns]

    clean = clean.ffill().bfill()
    clean = clean.dropna(how="any")
    positive_columns = [col for col in clean.columns if (clean[col] > 0).all()]
    clean = clean[positive_columns]
    if clean.shape[1] < 2:
        raise MarketDataError("Fewer than two assets have valid positive prices after cleaning.")

    return clean


def compute_daily_returns(prices: pd.DataFrame, settings: OptimizerSettings) -> pd.DataFrame:
    """Compute daily simple returns from adjusted close prices."""
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if settings.winsorize_returns:
        lower = returns.quantile(settings.winsorize_lower)
        upper = returns.quantile(settings.winsorize_upper)
        returns = returns.clip(lower=lower, upper=upper, axis=1)
    if returns.empty:
        raise MarketDataError("Return frame is empty after cleaning.")
    return returns


def estimate_expected_returns(returns: pd.DataFrame) -> pd.Series:
    """Estimate annualized arithmetic expected returns."""
    return returns.mean() * TRADING_DAYS


def estimate_covariance(returns: pd.DataFrame, method: str = "ledoit_wolf") -> pd.DataFrame:
    """Estimate annualized covariance matrix."""
    method_key = str(method).strip().lower()
    if method_key == "ledoit_wolf" and LedoitWolf is not None:
        model = LedoitWolf().fit(returns.values)
        cov = pd.DataFrame(model.covariance_, index=returns.columns, columns=returns.columns)
    else:
        cov = returns.cov()
    cov = cov * TRADING_DAYS
    # Symmetrize to protect numerical optimizers from tiny floating point asymmetry.
    cov = (cov + cov.T) / 2.0
    return cov


def portfolio_return(weights: np.ndarray, expected_returns: pd.Series) -> float:
    """Annualized expected portfolio return."""
    return float(np.dot(weights, expected_returns.values))


def portfolio_volatility(weights: np.ndarray, covariance: pd.DataFrame) -> float:
    """Annualized expected portfolio volatility."""
    variance = float(weights.T @ covariance.values @ weights)
    return float(np.sqrt(max(variance, 0.0)))


def portfolio_sharpe(weights: np.ndarray, expected_returns: pd.Series, covariance: pd.DataFrame, risk_free_rate: float) -> float:
    """Expected Sharpe ratio."""
    vol = portfolio_volatility(weights, covariance)
    if vol <= EPSILON:
        return np.nan
    return (portfolio_return(weights, expected_returns) - risk_free_rate) / vol


def _feasible_equal_weight(n_assets: int, min_weight: float, max_weight: float) -> np.ndarray:
    """Create a feasible equal-weight starting point."""
    weight = 1.0 / n_assets
    if min_weight - EPSILON <= weight <= max_weight + EPSILON:
        return np.repeat(weight, n_assets)
    # If pure equal weight is not feasible, start from the lower bound and
    # distribute remaining capital while respecting max weights.
    weights = np.repeat(min_weight, n_assets)
    remaining = 1.0 - weights.sum()
    for i in range(n_assets):
        add = min(max_weight - weights[i], remaining)
        weights[i] += add
        remaining -= add
        if remaining <= EPSILON:
            break
    if abs(weights.sum() - 1.0) > 1e-8:
        raise OptimizationError("Could not build a feasible starting portfolio.")
    return weights


def _optimization_bounds_and_constraints(settings: OptimizerSettings, n_assets: int):
    bounds = tuple((settings.min_weight, settings.max_weight) for _ in range(n_assets))
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    return bounds, constraints


def _solve_constrained_objective(
    objective_fn,
    x0: np.ndarray,
    bounds,
    constraints,
    maxiter: int = 1_000,
) -> np.ndarray:
    """Run SLSQP and return a clean weight vector."""
    result = minimize(
        objective_fn,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": maxiter, "ftol": 1e-10, "disp": False},
    )
    if not result.success:
        raise OptimizationError(result.message)
    weights = np.asarray(result.x, dtype=float)
    weights[np.abs(weights) < 1e-10] = 0.0
    weights = weights / weights.sum()
    return weights


def optimize_portfolio(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    settings: OptimizerSettings,
    objective: ObjectiveName,
) -> pd.Series:
    """Optimize one portfolio objective under long-only fully invested constraints."""
    assets = list(expected_returns.index)
    n_assets = len(assets)
    x0 = _feasible_equal_weight(n_assets, settings.min_weight, settings.max_weight)
    bounds, constraints = _optimization_bounds_and_constraints(settings, n_assets)

    if objective == "equal_weight":
        return pd.Series(x0, index=assets, name=DISPLAY_NAMES[objective])

    cov_values = covariance.loc[assets, assets].values
    mu_values = expected_returns.loc[assets].values
    asset_vols = np.sqrt(np.clip(np.diag(cov_values), 0.0, np.inf))

    def min_vol_obj(w: np.ndarray) -> float:
        return float(w.T @ cov_values @ w)

    def neg_sharpe_obj(w: np.ndarray) -> float:
        variance = float(w.T @ cov_values @ w)
        vol = np.sqrt(max(variance, 0.0))
        if vol <= EPSILON:
            return 1e6
        return -float((w @ mu_values - settings.risk_free_rate) / vol)

    def risk_parity_obj(w: np.ndarray) -> float:
        portfolio_variance = float(w.T @ cov_values @ w)
        if portfolio_variance <= EPSILON:
            return 1e6
        marginal_risk = cov_values @ w
        risk_contribution = w * marginal_risk / portfolio_variance
        target = np.repeat(1.0 / n_assets, n_assets)
        return float(np.sum((risk_contribution - target) ** 2))

    def max_diversification_obj(w: np.ndarray) -> float:
        portfolio_vol = np.sqrt(max(float(w.T @ cov_values @ w), 0.0))
        if portfolio_vol <= EPSILON:
            return 1e6
        weighted_asset_vol = float(w @ asset_vols)
        return -weighted_asset_vol / portfolio_vol

    objective_map = {
        "minimum_volatility": min_vol_obj,
        "maximum_sharpe": neg_sharpe_obj,
        "risk_parity": risk_parity_obj,
        "maximum_diversification": max_diversification_obj,
    }
    if objective not in objective_map:
        raise ValueError(f"Unknown objective: {objective}")

    try:
        weights = _solve_constrained_objective(objective_map[objective], x0, bounds, constraints)
    except OptimizationError:
        # Fallback to minimum volatility if Sharpe or diversification objective is unstable.
        if objective != "minimum_volatility":
            weights = _solve_constrained_objective(min_vol_obj, x0, bounds, constraints)
        else:
            raise
    return pd.Series(weights, index=assets, name=DISPLAY_NAMES[objective])


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Compute percentage contribution of each asset to portfolio variance."""
    aligned_cov = covariance.loc[weights.index, weights.index]
    w = weights.values
    variance = float(w.T @ aligned_cov.values @ w)
    if variance <= EPSILON:
        return pd.Series(np.nan, index=weights.index, name="Risk Contribution")
    marginal = aligned_cov.values @ w
    contribution = w * marginal / variance
    return pd.Series(contribution, index=weights.index, name="Risk Contribution")


def optimize_all_static_portfolios(
    returns: pd.DataFrame,
    settings: OptimizerSettings,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Estimate inputs and optimize all supported static portfolios."""
    mu = estimate_expected_returns(returns)
    cov = estimate_covariance(returns, settings.covariance_method)

    objectives: list[ObjectiveName] = [
        "equal_weight",
        "minimum_volatility",
        "maximum_sharpe",
        "risk_parity",
        "maximum_diversification",
    ]
    weights = []
    rows = []
    for objective in objectives:
        w = optimize_portfolio(mu, cov, settings, objective)
        weights.append(w)
        weights_array = w.values
        rows.append(
            {
                "Strategy": DISPLAY_NAMES[objective],
                "Expected Return": portfolio_return(weights_array, mu),
                "Expected Volatility": portfolio_volatility(weights_array, cov),
                "Expected Sharpe": portfolio_sharpe(weights_array, mu, cov, settings.risk_free_rate),
                "Maximum Weight": float(w.max()),
                "Number of Holdings": int((w > 1e-6).sum()),
            }
        )
    weights_df = pd.DataFrame(weights)
    summary = pd.DataFrame(rows).set_index("Strategy")
    return weights_df, summary, mu, cov


def sample_feasible_weights(settings: OptimizerSettings, n_assets: int, n_samples: int = 5_000) -> np.ndarray:
    """Generate random feasible long-only portfolios under min/max constraints."""
    rng = np.random.default_rng(settings.random_seed)
    accepted: list[np.ndarray] = []
    remaining = 1.0 - n_assets * settings.min_weight
    if remaining < -EPSILON:
        raise OptimizationError("Minimum weight constraint is infeasible.")

    attempts = 0
    max_attempts = max(25_000, n_samples * 20)
    while len(accepted) < n_samples and attempts < max_attempts:
        attempts += 1
        candidate = np.repeat(settings.min_weight, n_assets) + remaining * rng.dirichlet(np.ones(n_assets))
        if np.all(candidate <= settings.max_weight + 1e-12):
            accepted.append(candidate)

    if not accepted:
        accepted.append(_feasible_equal_weight(n_assets, settings.min_weight, settings.max_weight))

    return np.vstack(accepted)


def build_efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    settings: OptimizerSettings,
    n_samples: int = 5_000,
) -> pd.DataFrame:
    """Approximate an efficient frontier using random feasible portfolios."""
    assets = expected_returns.index.tolist()
    random_weights = sample_feasible_weights(settings, len(assets), n_samples=n_samples)
    mu = expected_returns.loc[assets].values
    cov = covariance.loc[assets, assets].values

    portfolio_returns = random_weights @ mu
    portfolio_variances = np.einsum("ij,jk,ik->i", random_weights, cov, random_weights)
    portfolio_vols = np.sqrt(np.clip(portfolio_variances, 0.0, np.inf))
    sharpes = (portfolio_returns - settings.risk_free_rate) / np.where(portfolio_vols > EPSILON, portfolio_vols, np.nan)

    return pd.DataFrame(
        {
            "Expected Return": portfolio_returns,
            "Expected Volatility": portfolio_vols,
            "Expected Sharpe": sharpes,
        }
    ).dropna()


def portfolio_returns_from_weights(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute daily portfolio returns from aligned return data and weights."""
    aligned = returns.loc[:, weights.index]
    return aligned @ weights


def make_sixty_forty_returns(returns: pd.DataFrame, equity_symbol: str = "SPY", bond_symbol: str = "IEF") -> Optional[pd.Series]:
    """Create a simple 60/40 benchmark if the required assets are available."""
    if equity_symbol not in returns.columns:
        return None
    if bond_symbol not in returns.columns:
        bond_candidates = [col for col in ["BND", "TLT"] if col in returns.columns]
        if not bond_candidates:
            return None
        bond_symbol = bond_candidates[0]
    return 0.60 * returns[equity_symbol] + 0.40 * returns[bond_symbol]


def _rebalance_start_dates(returns: pd.DataFrame, lookback_days: int, frequency: str) -> list[pd.Timestamp]:
    """Get first available trading day in each rebalance period after lookback."""
    eligible = returns.iloc[lookback_days:].copy()
    dates: list[pd.Timestamp] = []
    for _, block in eligible.groupby(pd.Grouper(freq=frequency)):
        if not block.empty:
            dates.append(pd.Timestamp(block.index[0]))
    return dates


def walk_forward_backtest(
    returns: pd.DataFrame,
    settings: OptimizerSettings,
    objectives: Optional[list[ObjectiveName]] = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Run monthly walk-forward out-of-sample portfolio optimization.

    At each rebalance date, the optimizer estimates inputs using only the
    historical lookback window ending before the rebalance date. The resulting
    weights are applied to the following period. This avoids look-ahead bias.
    """
    if objectives is None:
        objectives = [
            "equal_weight",
            "minimum_volatility",
            "maximum_sharpe",
            "risk_parity",
            "maximum_diversification",
        ]

    if len(returns) <= settings.lookback_days + 21:
        raise ValueError("Not enough return history for the selected lookback window.")

    rebalance_dates = _rebalance_start_dates(returns, settings.lookback_days, settings.rebalance_frequency)
    if len(rebalance_dates) < 3:
        raise ValueError("Not enough rebalance dates for walk-forward backtest.")

    strategy_returns: dict[str, list[pd.Series]] = {DISPLAY_NAMES[obj]: [] for obj in objectives}
    weight_history: dict[str, list[pd.Series]] = {DISPLAY_NAMES[obj]: [] for obj in objectives}
    previous_weights: dict[str, Optional[pd.Series]] = {DISPLAY_NAMES[obj]: None for obj in objectives}
    turnover_records: list[dict[str, object]] = []
    cost_rate = settings.transaction_cost_bps / 10_000.0

    for idx, rebalance_date in enumerate(rebalance_dates):
        next_rebalance = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else returns.index[-1] + pd.Timedelta(days=1)
        period_returns = returns.loc[(returns.index >= rebalance_date) & (returns.index < next_rebalance)].copy()
        if period_returns.empty:
            continue

        train_end_position = returns.index.get_indexer([rebalance_date], method="nearest")[0]
        train = returns.iloc[max(0, train_end_position - settings.lookback_days):train_end_position].copy()
        if len(train) < max(126, int(settings.lookback_days * 0.50)):
            continue

        mu = estimate_expected_returns(train)
        cov = estimate_covariance(train, settings.covariance_method)

        for objective in objectives:
            strategy_name = DISPLAY_NAMES[objective]
            try:
                weights = optimize_portfolio(mu, cov, settings, objective)
            except Exception:
                weights = optimize_portfolio(mu, cov, settings, "equal_weight")
                weights.name = strategy_name

            raw_returns = portfolio_returns_from_weights(period_returns, weights).copy()
            previous = previous_weights[strategy_name]
            turnover = float(weights.abs().sum()) if previous is None else float((weights - previous).abs().sum())
            if not raw_returns.empty:
                raw_returns.iloc[0] = raw_returns.iloc[0] - turnover * cost_rate
            strategy_returns[strategy_name].append(raw_returns)
            weight_history[strategy_name].append(weights.rename(rebalance_date))
            previous_weights[strategy_name] = weights
            turnover_records.append(
                {
                    "Date": rebalance_date,
                    "Strategy": strategy_name,
                    "Turnover": turnover,
                    "Transaction Cost": turnover * cost_rate,
                }
            )

    returns_df = pd.DataFrame({name: pd.concat(parts).sort_index() for name, parts in strategy_returns.items() if parts})

    # Add simple benchmarks for context.
    if settings.benchmark in returns.columns:
        benchmark = returns.loc[returns_df.index, settings.benchmark].copy()
        returns_df[f"Benchmark {settings.benchmark}"] = benchmark
    sixty_forty = make_sixty_forty_returns(returns)
    if sixty_forty is not None:
        returns_df["60/40 Benchmark"] = sixty_forty.loc[returns_df.index]

    weights_dict = {
        name: pd.DataFrame(parts).sort_index() for name, parts in weight_history.items() if parts
    }
    turnover_df = pd.DataFrame(turnover_records)
    if not turnover_df.empty:
        turnover_df = turnover_df.sort_values(["Date", "Strategy"]).reset_index(drop=True)
    return returns_df.dropna(how="all"), weights_dict, turnover_df


def wealth_index(returns: pd.Series | pd.DataFrame, initial_capital: float = 100_000.0) -> pd.Series | pd.DataFrame:
    """Convert returns to portfolio value path."""
    return initial_capital * (1.0 + returns.fillna(0.0)).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute percentage drawdown from a return series."""
    wealth = (1.0 + returns.dropna()).cumprod()
    if wealth.empty:
        return pd.Series(dtype=float)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def drawdown_frame(returns: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown for every strategy column."""
    return pd.DataFrame({col: drawdown_series(returns[col]) for col in returns.columns})


def annualized_return(returns: pd.Series) -> float:
    """Compound annual growth rate from daily returns."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan
    total = float((1.0 + clean).prod() - 1.0)
    years = len(clean) / TRADING_DAYS
    if years <= 0:
        return np.nan
    return float((1.0 + total) ** (1.0 / years) - 1.0)


def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    clean = returns.dropna()
    if len(clean) < 2:
        return np.nan
    return float(clean.std(ddof=1) * np.sqrt(TRADING_DAYS))


def downside_volatility(returns: pd.Series) -> float:
    """Annualized downside deviation."""
    clean = returns.dropna()
    downside = clean[clean < 0]
    if len(downside) < 2:
        return np.nan
    return float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough drawdown."""
    dd = drawdown_series(returns)
    return float(dd.min()) if not dd.empty else np.nan


def var_cvar(returns: pd.Series, level: float = 0.95) -> tuple[float, float]:
    """Historical daily VaR and CVaR at the requested confidence level."""
    clean = returns.dropna()
    if clean.empty:
        return np.nan, np.nan
    threshold = float(clean.quantile(1.0 - level))
    tail = clean[clean <= threshold]
    cvar = float(tail.mean()) if not tail.empty else np.nan
    return threshold, cvar


def alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float) -> tuple[float, float, float]:
    """Annualized alpha, beta, and correlation versus a benchmark."""
    frame = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
    if frame.shape[0] < 30:
        return np.nan, np.nan, np.nan
    y = frame.iloc[:, 0]
    x = frame.iloc[:, 1]
    variance = float(np.var(x, ddof=1))
    if variance <= EPSILON:
        return np.nan, np.nan, np.nan
    beta = float(np.cov(y, x, ddof=1)[0, 1] / variance)
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / TRADING_DAYS) - 1.0
    alpha_daily = float((y.mean() - rf_daily) - beta * (x.mean() - rf_daily))
    alpha = alpha_daily * TRADING_DAYS
    corr = float(y.corr(x))
    return alpha, beta, corr


def performance_summary(
    returns: pd.DataFrame,
    settings: OptimizerSettings,
    benchmark_column: Optional[str] = None,
    turnover: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create an institutional-style performance summary table."""
    if returns.empty:
        raise ValueError("returns cannot be empty.")

    if benchmark_column is None:
        candidates = [col for col in returns.columns if col.startswith("Benchmark")]
        benchmark_column = candidates[0] if candidates else None

    rows: list[dict[str, float | str]] = []
    for column in returns.columns:
        series = returns[column].dropna()
        cagr = annualized_return(series)
        vol = annualized_volatility(series)
        down_vol = downside_volatility(series)
        sharpe = (cagr - settings.risk_free_rate) / vol if vol and vol > EPSILON else np.nan
        sortino = (cagr - settings.risk_free_rate) / down_vol if down_vol and down_vol > EPSILON else np.nan
        mdd = max_drawdown(series)
        calmar = cagr / abs(mdd) if mdd and abs(mdd) > EPSILON else np.nan
        var95, cvar95 = var_cvar(series, level=0.95)
        total = float((1.0 + series).prod() - 1.0)
        hit_rate = float((series > 0).mean()) if len(series) else np.nan
        skew = float(series.skew()) if len(series) > 2 else np.nan
        kurt = float(series.kurtosis()) if len(series) > 3 else np.nan

        alpha = beta = corr = np.nan
        if benchmark_column and benchmark_column in returns.columns and column != benchmark_column:
            alpha, beta, corr = alpha_beta(series, returns[benchmark_column], settings.risk_free_rate)

        avg_turnover = np.nan
        if turnover is not None and not turnover.empty and "Strategy" in turnover.columns:
            subset = turnover.loc[turnover["Strategy"] == column, "Turnover"]
            if not subset.empty:
                avg_turnover = float(subset.mean())

        rows.append(
            {
                "Strategy": column,
                "Total Return": total,
                "CAGR": cagr,
                "Volatility": vol,
                "Sharpe": sharpe,
                "Sortino": sortino,
                "Max Drawdown": mdd,
                "Calmar": calmar,
                "Daily VaR 95%": var95,
                "Daily CVaR 95%": cvar95,
                "Hit Rate": hit_rate,
                "Skew": skew,
                "Excess Kurtosis": kurt,
                "Alpha vs Benchmark": alpha,
                "Beta vs Benchmark": beta,
                "Correlation vs Benchmark": corr,
                "Average Turnover": avg_turnover,
            }
        )
    summary = pd.DataFrame(rows).set_index("Strategy")
    return summary.sort_values("Sharpe", ascending=False)


def asset_summary(prices: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Summarize asset-level return, risk, and drawdown characteristics."""
    rows = []
    for col in returns.columns:
        rows.append(
            {
                "Ticker": col,
                "Start Price": float(prices[col].iloc[0]),
                "End Price": float(prices[col].iloc[-1]),
                "Total Return": float(prices[col].iloc[-1] / prices[col].iloc[0] - 1.0),
                "CAGR": annualized_return(returns[col]),
                "Volatility": annualized_volatility(returns[col]),
                "Max Drawdown": max_drawdown(returns[col]),
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


def run_full_analysis(
    settings: OptimizerSettings,
    data_mode: str = "demo",
    n_frontier_samples: int = 4_000,
) -> dict[str, object]:
    """Run the full virtual ETF optimization workflow."""
    validate_settings(settings)
    prices_raw, data_label = load_market_prices(settings, data_mode=data_mode, allow_fallback=True)
    prices = clean_prices(prices_raw, settings)
    returns = compute_daily_returns(prices, settings)

    # Align settings tickers to assets that survived cleaning.
    effective_settings = OptimizerSettings(**{**settings.__dict__, "tickers": list(returns.columns)})
    static_weights, static_summary, expected_returns, covariance = optimize_all_static_portfolios(returns, effective_settings)
    frontier = build_efficient_frontier(expected_returns, covariance, effective_settings, n_samples=n_frontier_samples)
    backtest_returns, weights_history, turnover = walk_forward_backtest(returns, effective_settings)
    perf = performance_summary(backtest_returns, effective_settings, turnover=turnover)
    assets = asset_summary(prices, returns)

    return {
        "settings": effective_settings,
        "data_label": data_label,
        "prices": prices,
        "returns": returns,
        "asset_summary": assets,
        "expected_returns": expected_returns,
        "covariance": covariance,
        "static_weights": static_weights,
        "static_summary": static_summary,
        "frontier": frontier,
        "backtest_returns": backtest_returns,
        "weights_history": weights_history,
        "turnover": turnover,
        "performance": perf,
    }

# ======================== REPORT BUILDER ========================


import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOTLY_TEMPLATE = "plotly_dark"


def pct(value: float | int | None) -> str:
    """Format a float as a percentage."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.2%}"


def num(value: float | int | None, decimals: int = 2) -> str:
    """Format a number for report display."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.{decimals}f}"


def money(value: float | int | None) -> str:
    """Format a number as dollars."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):,.0f}"


def style_figure(
    fig: go.Figure,
    title: str,
    height: int = 560,
    show_legend: bool = True,
) -> go.Figure:
    """Apply consistent visual styling to a Plotly figure.

    The Streamlit app is often viewed on narrow laptop screens. A default
    Plotly legend placed above the chart can overlap the title and make the
    dashboard hard to read. This helper pushes legends below the plotting
    area, removes noisy legend titles such as "variable", and gives every
    chart enough bottom margin for wrapped legend rows.
    """
    bottom_margin = 125 if show_legend else 55
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "y": 0.98, "yanchor": "top"},
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=55, r=35, t=95, b=bottom_margin),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
            title_text="",
        ),
        legend_title_text="",
        hovermode="x unified",
        font=dict(size=12),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    if not show_legend:
        fig.update_layout(showlegend=False)
    return fig


def create_figures(results: dict[str, Any]) -> dict[str, go.Figure]:
    """Create all interactive dashboard figures."""
    settings: OptimizerSettings = results["settings"]
    prices: pd.DataFrame = results["prices"]
    returns: pd.DataFrame = results["returns"]
    static_weights: pd.DataFrame = results["static_weights"]
    static_summary: pd.DataFrame = results["static_summary"]
    frontier: pd.DataFrame = results["frontier"]
    backtest_returns: pd.DataFrame = results["backtest_returns"]
    covariance: pd.DataFrame = results["covariance"]
    weights_history: dict[str, pd.DataFrame] = results["weights_history"]

    normalized = prices.divide(prices.iloc[0]).multiply(100.0)
    fig_prices = px.line(normalized, x=normalized.index, y=normalized.columns)
    fig_prices.update_yaxes(title="Indexed Value, Start = 100")
    fig_prices.update_xaxes(title="Date")
    style_figure(fig_prices, "ETF Universe: Normalized Historical Price Paths", height=620)

    corr = returns.corr()
    fig_corr = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.2f}<extra></extra>",
        )
    )
    style_figure(fig_corr, "ETF Return Correlation Matrix", height=560, show_legend=False)

    weights_long = static_weights.reset_index(names="Strategy").melt(
        id_vars="Strategy", var_name="ETF", value_name="Weight"
    )
    fig_weights = px.bar(weights_long, x="Strategy", y="Weight", color="ETF", text_auto=".1%")
    fig_weights.update_yaxes(tickformat=".0%", title="Portfolio Weight")
    fig_weights.update_xaxes(title="", tickangle=-15)
    style_figure(fig_weights, "Static Optimized ETF Allocations", height=660)

    fig_frontier = px.scatter(
        frontier,
        x="Expected Volatility",
        y="Expected Return",
        color="Expected Sharpe",
        opacity=0.55,
        labels={
            "Expected Volatility": "Expected Volatility",
            "Expected Return": "Expected Return",
            "Expected Sharpe": "Expected Sharpe",
        },
    )
    fig_frontier.update_xaxes(tickformat=".0%")
    fig_frontier.update_yaxes(tickformat=".0%")
    for strategy, row in static_summary.iterrows():
        fig_frontier.add_trace(
            go.Scatter(
                x=[row["Expected Volatility"]],
                y=[row["Expected Return"]],
                mode="markers+text",
                name=strategy,
                text=[strategy],
                textposition="top center",
                marker=dict(size=13, symbol="diamond", line=dict(width=1, color="white")),
                showlegend=False,
                hovertemplate=(
                    f"{html.escape(strategy)}<br>Expected Return: %{{y:.2%}}"
                    "<br>Expected Volatility: %{x:.2%}<extra></extra>"
                ),
            )
        )
    style_figure(fig_frontier, "Efficient Frontier with Optimized Portfolios", height=660, show_legend=False)

    wealth = wealth_index(backtest_returns, settings.initial_capital)
    fig_wealth = px.line(wealth, x=wealth.index, y=wealth.columns)
    fig_wealth.update_yaxes(title="Portfolio Value", tickprefix="$", separatethousands=True)
    fig_wealth.update_xaxes(title="Date")
    style_figure(fig_wealth, f"Growth of {money(settings.initial_capital)} Virtual Portfolio", height=640)

    dd = drawdown_frame(backtest_returns)
    fig_drawdown = px.line(dd, x=dd.index, y=dd.columns)
    fig_drawdown.update_yaxes(title="Drawdown", tickformat=".0%")
    fig_drawdown.update_xaxes(title="Date")
    style_figure(fig_drawdown, "Portfolio Drawdowns by Strategy", height=620)

    rolling_vol = backtest_returns.rolling(63).std() * np.sqrt(252)
    fig_rolling_vol = px.line(rolling_vol.dropna(how="all"), x=rolling_vol.dropna(how="all").index, y=rolling_vol.columns)
    fig_rolling_vol.update_yaxes(title="Annualized Volatility", tickformat=".0%")
    fig_rolling_vol.update_xaxes(title="Date")
    style_figure(fig_rolling_vol, "Rolling 63-Day Annualized Volatility", height=560)

    # Use the current Maximum Sharpe weights if available; otherwise use the first static strategy.
    selected_strategy = "Maximum Sharpe" if "Maximum Sharpe" in static_weights.index else static_weights.index[0]
    risk_contrib = risk_contributions(static_weights.loc[selected_strategy], covariance).reset_index()
    risk_contrib.columns = ["ETF", "Risk Contribution"]
    fig_risk_contrib = px.bar(risk_contrib, x="ETF", y="Risk Contribution", text_auto=".1%")
    fig_risk_contrib.update_yaxes(title="Risk Contribution", tickformat=".0%")
    style_figure(fig_risk_contrib, f"Risk Contribution: {selected_strategy}", height=500, show_legend=False)

    if selected_strategy in weights_history:
        weight_path = weights_history[selected_strategy].copy()
        weight_long = weight_path.reset_index(names="Date").melt(id_vars="Date", var_name="ETF", value_name="Weight")
        fig_weight_history = px.area(weight_long, x="Date", y="Weight", color="ETF")
        fig_weight_history.update_yaxes(title="Weight", tickformat=".0%")
        style_figure(fig_weight_history, f"Walk-Forward Allocation History: {selected_strategy}", height=560)
    else:
        fig_weight_history = go.Figure()
        style_figure(fig_weight_history, "Walk-Forward Allocation History", height=560)

    return {
        "prices": fig_prices,
        "correlation": fig_corr,
        "weights": fig_weights,
        "frontier": fig_frontier,
        "wealth": fig_wealth,
        "drawdown": fig_drawdown,
        "rolling_volatility": fig_rolling_vol,
        "risk_contribution": fig_risk_contrib,
        "weight_history": fig_weight_history,
    }


def performance_table_html(performance: pd.DataFrame) -> str:
    """Render performance table as HTML."""
    table = performance.copy()
    percent_columns = [
        "Total Return",
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Daily VaR 95%",
        "Daily CVaR 95%",
        "Hit Rate",
        "Alpha vs Benchmark",
        "Average Turnover",
    ]
    number_columns = ["Sharpe", "Sortino", "Calmar", "Skew", "Excess Kurtosis", "Beta vs Benchmark", "Correlation vs Benchmark"]
    for col in percent_columns:
        if col in table.columns:
            table[col] = table[col].map(pct)
    for col in number_columns:
        if col in table.columns:
            table[col] = table[col].map(lambda x: num(x, 2))
    return table.reset_index().to_html(index=False, classes="data-table", border=0, escape=False)


def weights_table_html(static_weights: pd.DataFrame) -> str:
    """Render static weights table as HTML."""
    table = static_weights.copy()
    for col in table.columns:
        table[col] = table[col].map(pct)
    return table.reset_index(names="Strategy").to_html(index=False, classes="data-table", border=0, escape=False)


def _chart_div(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    """Convert a Plotly figure to an embeddable div."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn" if include_plotlyjs else False, config={"displaylogo": False})


def build_html_report(results: dict[str, Any], output_path: str | Path) -> Path:
    """Build a standalone LinkedIn-shareable HTML report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    settings: OptimizerSettings = results["settings"]
    performance: pd.DataFrame = results["performance"]
    static_weights: pd.DataFrame = results["static_weights"]
    data_label = str(results.get("data_label", "Data source not specified"))
    figures = create_figures(results)

    top_strategy = performance.index[0]
    top_row = performance.loc[top_strategy]
    end_value = wealth_index(results["backtest_returns"], settings.initial_capital).iloc[-1].max()

    chart_html = []
    for i, key in enumerate(["wealth", "drawdown", "frontier", "weights", "correlation", "rolling_volatility", "risk_contribution", "weight_history", "prices"]):
        chart_html.append(f'<section class="chart-card">{_chart_div(figures[key], include_plotlyjs=(i == 0))}</section>')

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Virtual ETF Portfolio Optimization & Risk Management Engine</title>
  <meta name="description" content="Interactive ETF allocation dashboard with optimization, walk-forward backtesting, transaction costs, and institutional risk metrics.">
  <meta property="og:title" content="Virtual ETF Portfolio Optimization & Risk Management Engine">
  <meta property="og:description" content="ETF optimizer with efficient frontier, risk dashboard, walk-forward backtest, and virtual portfolio reporting.">
  <meta property="og:type" content="website">
  <meta property="og:image" content="assets/preview.png">
  <style>{DEFAULT_CSS}</style>
</head>
<body>
  <header class="hero">
    <div class="hero-content">
      <p class="eyebrow">Quant Finance Project</p>
      <h1>Virtual ETF Portfolio Optimization & Risk Management Engine</h1>
      <p class="subtitle">An editable Python project that simulates institutional ETF allocation research using constrained optimization, walk-forward backtesting, transaction costs, and professional risk analytics.</p>
      <div class="hero-actions">
        <a href="#dashboard">View Dashboard</a>
        <a href="#methodology" class="secondary">Methodology</a>
      </div>
    </div>
    <div class="hero-card">
      <span>Best Strategy by Sharpe</span>
      <strong>{html.escape(str(top_strategy))}</strong>
      <p>CAGR {pct(top_row.get('CAGR'))} | Sharpe {num(top_row.get('Sharpe'), 2)} | Max Drawdown {pct(top_row.get('Max Drawdown'))}</p>
    </div>
  </header>

  <main>
    <section class="kpi-grid">
      <div class="kpi"><span>Initial Virtual Capital</span><strong>{money(settings.initial_capital)}</strong></div>
      <div class="kpi"><span>Best Ending Value</span><strong>{money(end_value)}</strong></div>
      <div class="kpi"><span>ETF Universe</span><strong>{len(settings.tickers)} ETFs</strong></div>
      <div class="kpi"><span>Data Mode</span><strong>{html.escape(data_label)}</strong></div>
    </section>

    <section id="dashboard" class="section-heading">
      <p class="eyebrow">Interactive Dashboard</p>
      <h2>Portfolio Results</h2>
      <p>Charts are interactive. Hover, zoom, isolate strategies, and export images from the Plotly toolbar.</p>
    </section>

    {''.join(chart_html)}

    <section class="section-heading">
      <p class="eyebrow">Institutional Metrics</p>
      <h2>Performance Summary</h2>
    </section>
    <section class="table-card">{performance_table_html(performance)}</section>

    <section class="section-heading">
      <p class="eyebrow">Allocation Output</p>
      <h2>Static Optimized Weights</h2>
    </section>
    <section class="table-card">{weights_table_html(static_weights)}</section>

    <section id="methodology" class="methodology">
      <p class="eyebrow">Methodology</p>
      <h2>How the Engine Works</h2>
      <div class="method-grid">
        <div><h3>1. Data Pipeline</h3><p>Collects ETF adjusted prices, validates history, removes unreliable assets, and computes daily returns.</p></div>
        <div><h3>2. Risk Model</h3><p>Estimates expected returns and annualized covariance, with Ledoit-Wolf shrinkage when available.</p></div>
        <div><h3>3. Optimization</h3><p>Builds Equal Weight, Minimum Volatility, Maximum Sharpe, Risk Parity, and Maximum Diversification portfolios.</p></div>
        <div><h3>4. Backtest</h3><p>Uses only historical data available at each rebalance date, then tests the next period out of sample.</p></div>
        <div><h3>5. Costs</h3><p>Applies configurable transaction costs based on portfolio turnover at each rebalance.</p></div>
        <div><h3>6. Reporting</h3><p>Exports interactive charts, allocation tables, risk metrics, and a LinkedIn-shareable web report.</p></div>
      </div>
    </section>

    <section class="disclaimer">
      <strong>Educational use only.</strong> This is a virtual research project, not investment advice. Results depend on assumptions, data quality, model design, and historical periods. Historical or synthetic demo performance does not guarantee future results.
    </section>
  </main>
</body>
</html>
"""
    output.write_text(html_doc, encoding="utf-8")
    return output


DEFAULT_CSS = """
:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --ink: #0f172a;
  --muted: #64748b;
  --accent: #2563eb;
  --accent-dark: #1e40af;
  --border: #e2e8f0;
  --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
  background: radial-gradient(circle at top left, #dbeafe 0, transparent 35%), var(--bg);
}
a { color: inherit; }
.hero {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 32px;
  padding: 72px 8vw 56px;
  align-items: center;
}
.hero-content h1 {
  font-size: clamp(38px, 6vw, 72px);
  line-height: 0.95;
  margin: 10px 0 20px;
  letter-spacing: -0.06em;
}
.subtitle { font-size: 19px; line-height: 1.7; color: var(--muted); max-width: 880px; }
.eyebrow { text-transform: uppercase; letter-spacing: 0.14em; font-size: 12px; font-weight: 800; color: var(--accent); margin: 0; }
.hero-actions { display: flex; gap: 14px; margin-top: 28px; flex-wrap: wrap; }
.hero-actions a {
  text-decoration: none;
  background: var(--accent);
  color: white;
  padding: 13px 18px;
  border-radius: 999px;
  font-weight: 800;
  box-shadow: var(--shadow);
}
.hero-actions a.secondary { background: white; color: var(--accent-dark); border: 1px solid var(--border); }
.hero-card {
  background: rgba(255, 255, 255, 0.88);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: 28px;
  padding: 28px;
}
.hero-card span, .kpi span { display: block; color: var(--muted); font-size: 13px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.08em; }
.hero-card strong { display: block; font-size: 34px; line-height: 1.1; margin: 12px 0; }
.hero-card p { color: var(--muted); line-height: 1.6; margin-bottom: 0; }
main { padding: 0 8vw 80px; }
.kpi-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 18px; margin-bottom: 42px; }
.kpi, .chart-card, .table-card, .methodology, .disclaimer {
  background: var(--card);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
  border-radius: 24px;
}
.kpi { padding: 22px; }
.kpi strong { display: block; font-size: 27px; margin-top: 8px; }
.section-heading { margin: 50px 0 20px; }
.section-heading h2, .methodology h2 { font-size: 36px; letter-spacing: -0.04em; margin: 8px 0; }
.section-heading p:not(.eyebrow) { color: var(--muted); }
.chart-card { padding: 14px; margin: 20px 0; overflow: hidden; }
.table-card { padding: 22px; margin: 20px 0 38px; overflow-x: auto; }
.data-table { border-collapse: collapse; width: 100%; font-size: 14px; }
.data-table th { text-align: left; background: #f8fafc; color: #334155; }
.data-table th, .data-table td { border-bottom: 1px solid var(--border); padding: 12px 10px; white-space: nowrap; }
.data-table tr:hover td { background: #f8fafc; }
.methodology { padding: 28px; margin-top: 44px; }
.method-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px; margin-top: 22px; }
.method-grid div { padding: 18px; background: #f8fafc; border-radius: 18px; border: 1px solid var(--border); }
.method-grid h3 { margin: 0 0 8px; }
.method-grid p { color: var(--muted); line-height: 1.6; margin: 0; }
.disclaimer { margin-top: 26px; padding: 20px 24px; color: #475569; line-height: 1.7; }
@media (max-width: 980px) {
  .hero { grid-template-columns: 1fr; padding-top: 48px; }
  .kpi-grid, .method-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 640px) {
  main, .hero { padding-left: 18px; padding-right: 18px; }
  .kpi-grid, .method-grid { grid-template-columns: 1fr; }
}
"""


def generate_demo_report(output_path: str | Path) -> Path:
    """Run the default demo analysis and write an HTML report."""
    settings = OptimizerSettings(end_date="2026-07-14")
    results = run_full_analysis(settings=settings, data_mode="demo", n_frontier_samples=3_500)
    return build_html_report(results, output_path)

# ======================== STREAMLIT APP ========================


import tempfile
from pathlib import Path

# Ensure local project modules are importable on Streamlit Cloud and local runs.
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Virtual ETF Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
.block-container { padding-top: 2.5rem; padding-bottom: 4rem; }
.metric-card {
    padding: 1.1rem 1.2rem;
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 1.1rem;
    background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(248,250,252,0.92));
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
}
.small-muted { color: #64748b; font-size: 0.92rem; }
.big-title { font-size: 3.0rem; line-height: 1.0; font-weight: 850; letter-spacing: -0.055em; margin-bottom: 0.4rem; }
.section-note { color: #64748b; margin-top: -0.4rem; }
.custom-kpi-card {
    padding: 1.0rem 1.05rem;
    border: 1px solid rgba(148, 163, 184, 0.28);
    border-radius: 1.0rem;
    background: rgba(15, 23, 42, 0.42);
    min-height: 118px;
    overflow: hidden;
}
.custom-kpi-label {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    color: rgba(248, 250, 252, 0.86);
    font-size: 0.95rem;
    font-weight: 700;
    margin-bottom: 0.55rem;
}
.custom-kpi-value {
    color: #ffffff;
    font-size: clamp(1.45rem, 2.25vw, 2.35rem);
    line-height: 1.08;
    font-weight: 850;
    letter-spacing: -0.035em;
    white-space: normal;
    overflow-wrap: anywhere;
}
.custom-kpi-value.strategy {
    font-size: clamp(1.05rem, 1.65vw, 1.75rem);
    letter-spacing: -0.025em;
}
.help-dot {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    width: 1.05rem;
    height: 1.05rem;
    border: 1px solid rgba(248, 250, 252, 0.48);
    border-radius: 999px;
    color: rgba(248, 250, 252, 0.88);
    font-size: 0.72rem;
    font-weight: 800;
    cursor: help;
}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


def parse_tickers(text: str) -> list[str]:
    """Parse comma- or whitespace-separated ETF tickers."""
    raw = text.replace("\n", ",").replace(";", ",").split(",")
    tickers = [item.strip().upper() for item in raw if item.strip()]
    unique: list[str] = []
    for ticker in tickers:
        if ticker not in unique:
            unique.append(ticker)
    return unique


@st.cache_data(show_spinner=False, ttl=60 * 30)
def cached_analysis(settings_dict: dict[str, Any], data_mode: str, n_frontier_samples: int) -> dict[str, Any]:
    """Run and cache the expensive analysis step."""
    settings = OptimizerSettings(**settings_dict)
    return run_full_analysis(settings=settings, data_mode=data_mode, n_frontier_samples=n_frontier_samples)


def format_performance_table(performance: pd.DataFrame) -> pd.DataFrame:
    """Format metrics for Streamlit display while keeping raw calculations separate."""
    table = performance.copy()
    pct_cols = [
        "Total Return",
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Daily VaR 95%",
        "Daily CVaR 95%",
        "Hit Rate",
        "Alpha vs Benchmark",
        "Average Turnover",
    ]
    num_cols = ["Sharpe", "Sortino", "Calmar", "Skew", "Excess Kurtosis", "Beta vs Benchmark", "Correlation vs Benchmark"]
    for col in pct_cols:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    for col in num_cols:
        if col in table:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    return table


def format_weight_list(weights: pd.Series, limit: int = 4) -> str:
    """Create a compact sentence-friendly list of top portfolio weights."""
    clean = weights.dropna().sort_values(ascending=False)
    clean = clean[clean > 1e-4].head(limit)
    if clean.empty:
        return "no material positions"
    return ", ".join(f"{ticker} ({value:.1%})" for ticker, value in clean.items())


def _metric_value(row: pd.Series, column: str) -> float:
    """Safely extract a numeric metric from a performance row."""
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def strategy_universe(performance: pd.DataFrame) -> list[str]:
    """Return only strategy rows, excluding benchmark rows when possible."""
    preferred = [
        "Equal Weight",
        "Minimum Volatility",
        "Maximum Sharpe",
        "Risk Parity",
        "Maximum Diversification",
    ]
    available = [name for name in preferred if name in performance.index]
    return available if available else list(performance.index)


def generate_result_explanation(results: dict[str, Any], best_strategy: str, best_ending_value: float) -> str:
    """Generate a written interpretation of the current optimization result.

    The goal is to make the dashboard portfolio-profile friendly: a viewer can
    understand what the best result means without reading the source code.
    The text is generated from the currently selected sidebar assumptions.
    """
    settings: OptimizerSettings = results["settings"]
    performance: pd.DataFrame = results["performance"]
    static_weights: pd.DataFrame = results["static_weights"]
    turnover: pd.DataFrame = results["turnover"]

    if best_strategy not in performance.index:
        return "The optimizer completed, but there was not enough information to generate a detailed written interpretation."

    best = performance.loc[best_strategy]
    candidate_names = strategy_universe(performance)
    candidates = performance.loc[candidate_names]

    highest_cagr_strategy = candidates["CAGR"].dropna().idxmax() if "CAGR" in candidates else best_strategy
    lowest_drawdown_strategy = candidates["Max Drawdown"].dropna().idxmax() if "Max Drawdown" in candidates else best_strategy

    top_weights = "not available for this strategy"
    if best_strategy in static_weights.index:
        top_weights = format_weight_list(static_weights.loc[best_strategy])
    elif "Maximum Sharpe" in static_weights.index:
        top_weights = format_weight_list(static_weights.loc["Maximum Sharpe"])

    benchmark_name = f"Benchmark {settings.benchmark}"
    benchmark_comment = ""
    if benchmark_name in performance.index:
        benchmark = performance.loc[benchmark_name]
        cagr_delta = _metric_value(best, "CAGR") - _metric_value(benchmark, "CAGR")
        drawdown_delta = _metric_value(best, "Max Drawdown") - _metric_value(benchmark, "Max Drawdown")
        cagr_word = "higher" if cagr_delta >= 0 else "lower"
        dd_word = "less severe" if drawdown_delta >= 0 else "more severe"
        benchmark_comment = (
            f" Compared with {benchmark_name}, the selected strategy produced a CAGR that was "
            f"{abs(cagr_delta):.2%} {cagr_word} and a maximum drawdown that was "
            f"{abs(drawdown_delta):.2%} {dd_word}."
        )

    turnover_text = "not material or not available"
    if turnover is not None and not turnover.empty and "Strategy" in turnover.columns:
        strat_turnover = turnover.loc[turnover["Strategy"] == best_strategy, "Turnover"]
        if not strat_turnover.empty:
            turnover_text = f"{float(strat_turnover.mean()):.1%} average turnover per rebalance"

    return f"""
**What this result means:** Under the current assumptions, **{best_strategy}** has the strongest risk-adjusted profile among the optimized ETF strategies shown in the dashboard. The walk-forward backtest simulates investing **{money(settings.initial_capital)}** and re-optimizing on a **{settings.lookback_days}-trading-day lookback window** with **{settings.transaction_cost_bps:.1f} bps** of simulated transaction cost per turnover.

The selected strategy ended at approximately **{money(best_ending_value)}**, with **{pct(_metric_value(best, 'CAGR'))} CAGR**, **{pct(_metric_value(best, 'Volatility'))} annualized volatility**, a **{num(_metric_value(best, 'Sharpe'), 2)} Sharpe ratio**, and a **{pct(_metric_value(best, 'Max Drawdown'))} maximum drawdown**. The strongest static allocation exposure is concentrated in **{top_weights}**.

The dashboard should be read as a trade-off, not as a return guarantee. **{highest_cagr_strategy}** produced the highest CAGR among the optimized strategies, while **{lowest_drawdown_strategy}** had the least severe maximum drawdown. This helps separate return generation from downside risk control.{benchmark_comment}

Average simulated rebalancing intensity for **{best_strategy}** was **{turnover_text}**. Higher turnover can improve adaptation to new market conditions, but it also increases implementation costs and makes the strategy less practical if trading costs, taxes, or liquidity constraints are high.
"""


def metric_card(label: str, value: str, help_text: str, *, compact_value: bool = False) -> None:
    """Render a readable KPI card with a visible question-mark explanation icon."""
    value_class = "custom-kpi-value strategy" if compact_value else "custom-kpi-value"
    st.markdown(
        f"""
        <div class="custom-kpi-card">
            <div class="custom-kpi-label">
                <span>{html.escape(label)}</span>
                <span class="help-dot" title="{html.escape(help_text)}">?</span>
            </div>
            <div class="{value_class}">{html.escape(str(value))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def settings_to_dict(settings: OptimizerSettings) -> dict[str, Any]:
    """Convert settings dataclass to a cache-friendly dictionary."""
    return {
        "tickers": list(settings.tickers),
        "benchmark": settings.benchmark,
        "start_date": settings.start_date,
        "end_date": settings.end_date,
        "initial_capital": settings.initial_capital,
        "min_weight": settings.min_weight,
        "max_weight": settings.max_weight,
        "risk_free_rate": settings.risk_free_rate,
        "lookback_days": settings.lookback_days,
        "rebalance_frequency": settings.rebalance_frequency,
        "transaction_cost_bps": settings.transaction_cost_bps,
        "min_history_ratio": settings.min_history_ratio,
        "winsorize_returns": settings.winsorize_returns,
        "winsorize_lower": settings.winsorize_lower,
        "winsorize_upper": settings.winsorize_upper,
        "covariance_method": settings.covariance_method,
        "random_seed": settings.random_seed,
    }


st.sidebar.title("Editable Controls")
st.sidebar.caption("Change the assumptions, rerun the engine, and view the updated portfolio dashboard.")

with st.sidebar:
    data_mode_label = st.radio(
        "Data mode",
        options=["Demo data", "Live Yahoo Finance"],
        index=0,
        help="Choose demo data for a stable public web demo, or Live Yahoo Finance to download current historical ETF prices through yfinance.",
    )
    data_mode = "live" if data_mode_label == "Live Yahoo Finance" else "demo"

    tickers_text = st.text_area(
        "ETF universe",
        value="SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, VNQ, DBC",
        height=95,
        help="Comma-separated ETF tickers included in the optimization universe. The optimizer allocates only across these assets.",
    )
    tickers = parse_tickers(tickers_text)

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input(
            "Start date",
            value=pd.Timestamp("2015-01-01").date(),
            help="First date used in the analysis. A longer history gives the optimizer more data, but may include older market regimes.",
        )
    with col_b:
        end_date = st.date_input(
            "End date",
            value=pd.Timestamp.today().date(),
            help="Last date used in the analysis. In live mode, this should be today or a recent trading date.",
        )

    initial_capital = st.number_input(
        "Virtual starting capital",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Starting dollar amount used to convert backtested returns into a virtual portfolio value. It does not affect percentage returns.",
    )
    max_weight = st.slider(
        "Max ETF weight",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05,
        help="Largest allocation any single ETF is allowed to receive. Lower values force more diversification; higher values allow more concentration.",
    )
    min_weight = st.slider(
        "Min ETF weight",
        min_value=0.00,
        max_value=0.20,
        value=0.00,
        step=0.01,
        help="Smallest allocation each ETF must receive. Use 0% to let the optimizer exclude ETFs; higher values force every ETF to remain in the portfolio.",
    )
    risk_free_rate = st.slider(
        "Annual risk-free rate",
        min_value=0.00,
        max_value=0.10,
        value=0.03,
        step=0.005,
        help="Annual cash-like return assumption used to calculate excess return metrics such as Sharpe ratio.",
    )
    lookback_days = st.slider(
        "Optimization lookback days",
        min_value=252,
        max_value=1260,
        value=756,
        step=63,
        help="Number of past trading days used at each rebalance to estimate returns, volatility, and correlations. 252 trading days is roughly one year.",
    )
    transaction_cost_bps = st.slider(
        "Transaction cost per turnover",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0,
        help="Simulated trading cost in basis points applied to portfolio turnover at each rebalance. 5 bps equals 0.05% of traded value.",
    )
    rebalance_frequency = st.selectbox(
        "Rebalance frequency",
        options=["ME", "QE"],
        format_func=lambda x: "Monthly" if x == "ME" else "Quarterly",
        help="How often the virtual portfolio is re-optimized and rebalanced. Monthly adapts faster; quarterly usually lowers turnover.",
    )
    frontier_samples = st.slider(
        "Efficient frontier samples",
        min_value=1_000,
        max_value=10_000,
        value=4_000,
        step=1_000,
        help="Number of random feasible portfolios used to visualize the risk-return opportunity set. More samples create a denser frontier but run slower.",
    )

    run_button = st.button("Run optimization", type="primary", use_container_width=True)

st.markdown('<div class="big-title">Virtual ETF Portfolio Optimization & Risk Management Engine</div>', unsafe_allow_html=True)
st.markdown(
    "An interactive quantitative finance dashboard for ETF allocation research, walk-forward backtesting, transaction-cost simulation, and institutional risk analytics."
)

if len(tickers) < 2:
    st.error("Please enter at least two ETF tickers.")
    st.stop()

settings = OptimizerSettings(
    tickers=tickers,
    benchmark="SPY" if "SPY" in tickers else tickers[0],
    start_date=str(start_date),
    end_date=str(end_date),
    initial_capital=float(initial_capital),
    min_weight=float(min_weight),
    max_weight=float(max_weight),
    risk_free_rate=float(risk_free_rate),
    lookback_days=int(lookback_days),
    rebalance_frequency=rebalance_frequency,
    transaction_cost_bps=float(transaction_cost_bps),
)

try:
    with st.spinner("Running ETF optimization, walk-forward backtest, and risk analytics..."):
        results = cached_analysis(settings_to_dict(settings), data_mode, int(frontier_samples))
except Exception as exc:
    st.error("The optimizer could not complete with the current settings.")
    st.exception(exc)
    st.stop()

performance = results["performance"]
static_weights = results["static_weights"]
backtest_returns = results["backtest_returns"]
turnover = results["turnover"]
figures = create_figures(results)

# Highlight the best optimized strategy by Sharpe ratio, excluding simple benchmarks
# from the headline KPI. Benchmarks remain in the performance table and charts.
optimized_names = strategy_universe(performance)
optimized_performance = performance.loc[optimized_names]
sharpe_candidates = optimized_performance["Sharpe"].dropna() if "Sharpe" in optimized_performance.columns else pd.Series(dtype=float)
best_strategy = sharpe_candidates.idxmax() if not sharpe_candidates.empty else optimized_performance.index[0]
best_row = performance.loc[best_strategy]
ending_values = (1.0 + backtest_returns.fillna(0.0)).cumprod().iloc[-1] * settings.initial_capital
best_ending_value = float(ending_values.get(best_strategy, ending_values.max()))
result_explanation = generate_result_explanation(results, best_strategy, best_ending_value)

st.info(f"Data source used: {results['data_label']}. This project is virtual research and not investment advice.")

kpi_cols = st.columns([1.8, 1.2, 1.0, 1.0])
with kpi_cols[0]:
    metric_card(
        "Best Strategy",
        str(best_strategy),
        "Optimized strategy with the highest Sharpe ratio under the current assumptions. Benchmarks are excluded from this headline selection.",
        compact_value=True,
    )
with kpi_cols[1]:
    metric_card(
        "Best Ending Value",
        money(best_ending_value),
        "Ending virtual portfolio value for the best strategy after the walk-forward backtest period.",
    )
with kpi_cols[2]:
    metric_card(
        "Best CAGR",
        pct(best_row.get("CAGR")),
        "Compound Annual Growth Rate: the annualized return that would compound the starting value into the ending value over the backtest period.",
    )
with kpi_cols[3]:
    metric_card(
        "Best Sharpe",
        f"{best_row.get('Sharpe'):.2f}" if pd.notna(best_row.get("Sharpe")) else "n/a",
        "Risk-adjusted return metric calculated as annualized excess return divided by annualized volatility. Higher is generally better.",
    )

main_tabs = st.tabs(["Dashboard", "Allocations", "Performance Table"])

with main_tabs[0]:
    st.subheader("Interactive Dashboard")
    st.markdown("### Result Explanation")
    st.markdown(result_explanation)
    st.divider()
    st.plotly_chart(figures["wealth"], use_container_width=True, theme=None, config={"displaylogo": False})
    st.plotly_chart(figures["drawdown"], use_container_width=True, theme=None, config={"displaylogo": False})
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figures["frontier"], use_container_width=True, theme=None, config={"displaylogo": False})
        st.plotly_chart(figures["risk_contribution"], use_container_width=True, theme=None, config={"displaylogo": False})
    with col2:
        st.plotly_chart(figures["correlation"], use_container_width=True, theme=None, config={"displaylogo": False})
        st.plotly_chart(figures["rolling_volatility"], use_container_width=True, theme=None, config={"displaylogo": False})
    st.plotly_chart(figures["prices"], use_container_width=True, theme=None, config={"displaylogo": False})

with main_tabs[1]:
    st.subheader("Portfolio Allocations")
    st.plotly_chart(figures["weights"], use_container_width=True, theme=None, config={"displaylogo": False})
    st.plotly_chart(figures["weight_history"], use_container_width=True, theme=None, config={"displaylogo": False})
    st.dataframe(static_weights.style.format("{:.2%}"), use_container_width=True)

with main_tabs[2]:
    st.subheader("Performance Metrics")
    st.dataframe(format_performance_table(performance), use_container_width=True)
    if not turnover.empty:
        st.subheader("Turnover and Simulated Transaction Costs")
        st.dataframe(turnover.tail(30), use_container_width=True)


st.download_button(
    "Download performance table as CSV",
    data=performance.to_csv().encode("utf-8"),
    file_name="performance_summary.csv",
    mime="text/csv",
)
