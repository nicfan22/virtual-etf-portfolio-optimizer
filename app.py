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
    """Apply institutional terminal-style visual formatting to Plotly figures."""
    bottom_margin = 128 if show_legend else 62
    fig.update_layout(
        title={
            "text": f"<b>{title}</b>",
            "x": 0.02,
            "xanchor": "left",
            "y": 0.97,
            "yanchor": "top",
            "font": {"size": 18, "color": "#E6EDF3"},
        },
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="#07111F",
        plot_bgcolor="#07111F",
        height=height,
        margin=dict(l=62, r=38, t=88, b=bottom_margin),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(148,163,184,0.18)",
            borderwidth=0,
            font=dict(size=11, color="#C9D3DF"),
            title_text="",
        ),
        legend_title_text="",
        hovermode="x unified",
        font=dict(size=12, color="#C9D3DF"),
        hoverlabel=dict(bgcolor="#0B1220", font_size=12, font_color="#E6EDF3", bordercolor="#1F2A3A"),
    )
    fig.update_xaxes(
        automargin=True,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        linecolor="rgba(148,163,184,0.22)",
        zerolinecolor="rgba(250,204,21,0.26)",
        title_font=dict(color="#9AA7B5"),
        tickfont=dict(color="#C9D3DF"),
    )
    fig.update_yaxes(
        automargin=True,
        showgrid=True,
        gridcolor="rgba(148,163,184,0.12)",
        linecolor="rgba(148,163,184,0.22)",
        zerolinecolor="rgba(250,204,21,0.26)",
        title_font=dict(color="#9AA7B5"),
        tickfont=dict(color="#C9D3DF"),
    )
    try:
        fig.update_traces(selector=dict(type="scatter"), line=dict(width=2.2))
    except Exception:
        pass
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
    """Build a standalone web report."""
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
      <p class="subtitle">Institutional ETF allocation research system using constrained optimization, walk-forward backtesting, transaction costs, and professional risk analytics.</p>
      <div class="hero-actions">
        <a href="#dashboard">View Dashboard</a>
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


from datetime import datetime
from typing import Any
import html

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="ETF Allocation Terminal | Nicola Fanelli",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

LINKEDIN_URL = "https://www.linkedin.com/in/nicola-fanelli-8ab498360/"

TERMINAL_CSS = """
<style>
:root {
    --terminal-bg: #040A12;
    --terminal-bg-2: #07111F;
    --terminal-panel: rgba(7, 17, 31, 0.94);
    --terminal-panel-2: rgba(11, 18, 32, 0.96);
    --terminal-border: rgba(148, 163, 184, 0.18);
    --terminal-border-strong: rgba(56, 189, 248, 0.26);
    --terminal-text: #E6EDF3;
    --terminal-muted: #93A4B8;
    --terminal-subtle: #667085;
    --terminal-cyan: #38BDF8;
    --terminal-teal: #2DD4BF;
    --terminal-amber: #FACC15;
    --terminal-green: #22C55E;
    --terminal-red: #F87171;
    --terminal-purple: #A78BFA;
}

.stApp {
    background:
        radial-gradient(circle at 8% -8%, rgba(56, 189, 248, 0.22), transparent 34%),
        radial-gradient(circle at 88% 2%, rgba(250, 204, 21, 0.10), transparent 30%),
        linear-gradient(180deg, #040A12 0%, #06101D 52%, #030712 100%);
    color: var(--terminal-text);
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 3.5rem;
    max-width: 1540px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07111F 0%, #0B1220 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.16);
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: #DCE7F3;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F8FAFC;
}

[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #FACC15 0%, #F59E0B 100%);
    color: #111827;
    border: 0;
    font-weight: 850;
    letter-spacing: 0.01em;
}

[data-testid="stSidebar"] .stButton > button:hover {
    filter: brightness(1.06);
    color: #111827;
}

hr {
    border-color: rgba(148, 163, 184, 0.14) !important;
}

.terminal-shell {
    border: 1px solid var(--terminal-border);
    border-radius: 24px;
    padding: 1.2rem 1.25rem 1.15rem;
    background:
        linear-gradient(135deg, rgba(7, 17, 31, 0.95), rgba(15, 23, 42, 0.76)),
        radial-gradient(circle at top right, rgba(56, 189, 248, 0.14), transparent 36%);
    box-shadow: 0 22px 65px rgba(0, 0, 0, 0.36);
    margin-bottom: 1.15rem;
}

.terminal-topline {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    flex-wrap: wrap;
}

.terminal-eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--terminal-amber);
    font-weight: 850;
    margin-bottom: 0.4rem;
}

.terminal-title {
    font-size: clamp(2.0rem, 4.4vw, 4.15rem);
    line-height: 0.93;
    letter-spacing: -0.065em;
    font-weight: 900;
    color: #F8FAFC;
    margin: 0;
}

.terminal-subtitle {
    color: #B8C5D3;
    font-size: 1.03rem;
    line-height: 1.58;
    max-width: 970px;
    margin: 0.95rem 0 0;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    flex-shrink: 0;
}

.creator-name {
    font-size: 0.82rem;
    color: #9AA7B5;
    font-weight: 750;
    white-space: nowrap;
}

.signature-square {
    width: 54px;
    height: 54px;
    border-radius: 15px;
    background: linear-gradient(135deg, #0A66C2, #1D4ED8);
    color: #FFFFFF !important;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    text-decoration: none !important;
    font-weight: 950;
    letter-spacing: -0.04em;
    box-shadow: 0 16px 38px rgba(10, 102, 194, 0.32);
    border: 1px solid rgba(255, 255, 255, 0.18);
    position: relative;
}
.signature-square .nf-mark { font-size: 1.13rem; line-height: 1; }
.signature-square .in-mark {
    position: absolute;
    right: 6px;
    bottom: 5px;
    font-size: 0.56rem;
    font-weight: 900;
    background: rgba(255,255,255,0.17);
    padding: 2px 3px;
    border-radius: 4px;
}
.signature-square:hover { transform: translateY(-1px); filter: brightness(1.08); }

.status-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 1.15rem;
}
.status-tile {
    background: rgba(2, 6, 23, 0.42);
    border: 1px solid rgba(148, 163, 184, 0.13);
    border-radius: 16px;
    padding: 0.75rem 0.85rem;
}
.status-label {
    color: #7C8CA1;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 0.66rem;
    font-weight: 850;
    margin-bottom: 0.23rem;
}
.status-value {
    color: #F8FAFC;
    font-size: 0.92rem;
    font-weight: 800;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.control-title {
    padding: 0.95rem 0 0.25rem;
    color: #F8FAFC;
    font-size: 1.05rem;
    font-weight: 900;
    letter-spacing: -0.02em;
}
.control-caption {
    color: #93A4B8;
    font-size: 0.86rem;
    line-height: 1.45;
    margin-bottom: 0.9rem;
}
.control-section {
    margin: 1.0rem 0 0.55rem;
    color: #FACC15;
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    font-weight: 900;
}

.kpi-card {
    min-height: 142px;
    padding: 1.05rem 1.08rem;
    border: 1px solid var(--terminal-border);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(7, 17, 31, 0.94));
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.28);
    overflow: visible;
    position: relative;
}
.kpi-card:before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 18px 18px 0 0;
    background: linear-gradient(90deg, var(--terminal-cyan), var(--terminal-amber));
    opacity: 0.85;
}
.kpi-label {
    display: flex;
    align-items: center;
    gap: 0.38rem;
    margin-bottom: 0.55rem;
    color: #9AA7B5;
    font-size: 0.77rem;
    font-weight: 900;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    line-height: 1.25;
}
.kpi-help {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.02rem;
    height: 1.02rem;
    border: 1px solid rgba(148, 163, 184, 0.72);
    border-radius: 999px;
    color: rgba(203, 213, 225, 0.92);
    font-size: 0.70rem;
    font-weight: 900;
    cursor: help;
}
.kpi-value {
    color: #F8FAFC;
    font-size: clamp(1.35rem, 2.1vw, 2.45rem);
    line-height: 1.06;
    font-weight: 900;
    letter-spacing: -0.04em;
    white-space: normal;
    overflow-wrap: anywhere;
}
.kpi-value.strategy { font-size: clamp(1.18rem, 1.55vw, 1.92rem); letter-spacing: -0.025em; }
.kpi-footnote { margin-top: 0.52rem; color: #6F8092; font-size: 0.76rem; line-height: 1.35; }

.panel {
    border: 1px solid var(--terminal-border);
    border-radius: 22px;
    background: var(--terminal-panel);
    padding: 1.1rem 1.15rem;
    box-shadow: 0 18px 50px rgba(0, 0, 0, 0.24);
    margin-bottom: 1rem;
}
.panel-title {
    color: #F8FAFC;
    font-size: 1.12rem;
    font-weight: 900;
    letter-spacing: -0.025em;
    margin-bottom: 0.22rem;
}
.panel-subtitle {
    color: #93A4B8;
    font-size: 0.91rem;
    line-height: 1.45;
    margin-bottom: 0.82rem;
}
.terminal-brief {
    border: 1px solid rgba(56, 189, 248, 0.22);
    border-radius: 22px;
    background:
        linear-gradient(135deg, rgba(8, 47, 73, 0.36), rgba(7, 17, 31, 0.82)),
        radial-gradient(circle at top right, rgba(45, 212, 191, 0.11), transparent 42%);
    padding: 1.1rem 1.2rem;
    margin-bottom: 1rem;
    box-shadow: 0 20px 52px rgba(0, 0, 0, 0.25);
}
.terminal-brief h3 {
    margin: 0 0 0.5rem;
    font-size: 1.22rem;
    color: #F8FAFC;
    letter-spacing: -0.03em;
}
.terminal-brief p, .terminal-brief li { color: #C8D4E3; line-height: 1.62; }

.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.8rem; }
.terminal-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.42rem 0.62rem;
    border-radius: 999px;
    background: rgba(15, 23, 42, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.17);
    color: #DCE7F3;
    font-size: 0.78rem;
    font-weight: 780;
}
.terminal-badge strong { color: #FACC15; }

[data-testid="stTabs"] button { color: #C8D4E3; font-weight: 780; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #FACC15; }
[data-testid="stDataFrame"] { border: 1px solid rgba(148, 163, 184, 0.15); border-radius: 16px; overflow: hidden; }

.footer-signature {
    margin-top: 1.7rem;
    color: #7C8CA1;
    font-size: 0.84rem;
    text-align: center;
}

@media (max-width: 980px) {
    .status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .terminal-title { font-size: 2.25rem; }
}
@media (max-width: 620px) {
    .status-grid { grid-template-columns: 1fr; }
    .terminal-shell { padding: 1rem; }
    .header-actions { width: 100%; justify-content: flex-start; }
}
</style>
"""

st.markdown(TERMINAL_CSS, unsafe_allow_html=True)


def parse_tickers(text: str) -> list[str]:
    """Parse comma-, semicolon-, or newline-separated ETF tickers."""
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


def format_weights_table(weights: pd.DataFrame) -> pd.DataFrame:
    """Format portfolio weights for display."""
    table = weights.copy()
    for col in table.columns:
        table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return table


def strategy_universe(performance: pd.DataFrame) -> list[str]:
    """Return only model strategy rows, excluding benchmark rows when possible."""
    preferred = [
        "Equal Weight",
        "Minimum Volatility",
        "Maximum Sharpe",
        "Risk Parity",
        "Maximum Diversification",
    ]
    available = [name for name in preferred if name in performance.index]
    return available if available else list(performance.index)


def metric_value(row: pd.Series, column: str) -> float:
    """Safely extract a numeric metric from a performance row."""
    value = row.get(column, np.nan)
    return float(value) if pd.notna(value) else np.nan


def format_weight_list(weights: pd.Series, limit: int = 5) -> str:
    """Create a compact sentence-friendly list of top portfolio weights."""
    clean = weights.dropna().sort_values(ascending=False)
    clean = clean[clean > 1e-4].head(limit)
    if clean.empty:
        return "no material positions"
    return ", ".join(f"{ticker} ({value:.1%})" for ticker, value in clean.items())


def risk_label(sharpe: float, max_drawdown: float) -> str:
    """Convert core risk metrics into a plain-English research label."""
    if pd.isna(sharpe) or pd.isna(max_drawdown):
        return "Review"
    if sharpe >= 1.0 and max_drawdown > -0.20:
        return "High quality"
    if sharpe >= 0.60 and max_drawdown > -0.35:
        return "Balanced"
    if max_drawdown <= -0.45:
        return "High drawdown"
    return "Moderate"


def render_header(settings: OptimizerSettings | None = None, data_label: str | None = None) -> None:
    """Render a terminal-style app header with signature and status badges."""
    linkedin_url = html.escape(LINKEDIN_URL, quote=True)
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M")
    tickers_count = f"{len(settings.tickers)} ETFs" if settings else "ETF Universe"
    rebalance_text = "Monthly" if settings and settings.rebalance_frequency == "ME" else "Quarterly" if settings else "Rebalance"
    lookback_text = f"{settings.lookback_days}d" if settings else "Lookback"
    source_text = html.escape(str(data_label or "Ready"))

    st.markdown(
        f"""
        <div class="terminal-shell">
            <div class="terminal-topline">
                <div>
                    <div class="terminal-eyebrow">Institutional Quant Research Dashboard</div>
                    <h1 class="terminal-title">ETF Allocation Terminal</h1>
                    <p class="terminal-subtitle">
                        Portfolio optimization, walk-forward backtesting, transaction-cost simulation,
                        and risk analytics for a diversified ETF universe.
                    </p>
                    <div class="badge-row">
                        <span class="terminal-badge">Objective <strong>Risk-adjusted allocation</strong></span>
                        <span class="terminal-badge">Engine <strong>Python / SciPy / Plotly</strong></span>
                        <span class="terminal-badge">Use case <strong>Asset allocation research</strong></span>
                    </div>
                </div>
                <div class="header-actions">
                    <span class="creator-name">Nicola Fanelli</span>
                    <a class="signature-square" href="{linkedin_url}" target="_blank" rel="noopener noreferrer" title="Open Nicola Fanelli's LinkedIn profile">
                        <span class="nf-mark">NF</span><span class="in-mark">in</span>
                    </a>
                </div>
            </div>
            <div class="status-grid">
                <div class="status-tile"><div class="status-label">Data Source</div><div class="status-value">{source_text}</div></div>
                <div class="status-tile"><div class="status-label">Universe</div><div class="status-value">{tickers_count}</div></div>
                <div class="status-tile"><div class="status-label">Lookback</div><div class="status-value">{lookback_text}</div></div>
                <div class="status-tile"><div class="status-label">Rebalance</div><div class="status-value">{rebalance_text}</div></div>
                <div class="status-tile"><div class="status-label">Last Run</div><div class="status-value">{now_text}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, help_text: str, footnote: str = "", *, value_class: str = "") -> None:
    """Render a readable KPI card with a hoverable question-mark hint."""
    safe_label = html.escape(label)
    safe_value = html.escape(str(value))
    safe_help = html.escape(help_text, quote=True)
    safe_note = html.escape(footnote)
    cls = f"kpi-value {value_class}".strip()
    note_html = f'<div class="kpi-footnote">{safe_note}</div>' if footnote else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label"><span>{safe_label}</span><span class="kpi-help" title="{safe_help}">?</span></div>
            <div class="{cls}">{safe_value}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel(title: str, subtitle: str = "") -> None:
    """Open a styled panel title. Use before Streamlit-native content."""
    st.markdown(
        f"""
        <div class="panel-title">{html.escape(title)}</div>
        <div class="panel-subtitle">{html.escape(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def generate_investment_committee_brief(results: dict[str, Any], best_strategy: str, best_ending_value: float) -> str:
    """Generate a concise written interpretation of the currently displayed result."""
    settings: OptimizerSettings = results["settings"]
    performance: pd.DataFrame = results["performance"]
    static_weights: pd.DataFrame = results["static_weights"]
    turnover: pd.DataFrame = results["turnover"]

    if best_strategy not in performance.index:
        return "The optimizer completed, but there was not enough information to generate a written interpretation."

    best = performance.loc[best_strategy]
    candidate_names = strategy_universe(performance)
    candidates = performance.loc[candidate_names]

    highest_cagr_strategy = candidates["CAGR"].dropna().idxmax() if "CAGR" in candidates else best_strategy
    lowest_drawdown_strategy = candidates["Max Drawdown"].dropna().idxmax() if "Max Drawdown" in candidates else best_strategy

    top_weights = "not available"
    if best_strategy in static_weights.index:
        top_weights = format_weight_list(static_weights.loc[best_strategy])
    elif "Maximum Sharpe" in static_weights.index:
        top_weights = format_weight_list(static_weights.loc["Maximum Sharpe"])

    benchmark_name = f"Benchmark {settings.benchmark}"
    benchmark_sentence = ""
    if benchmark_name in performance.index:
        benchmark = performance.loc[benchmark_name]
        cagr_delta = metric_value(best, "CAGR") - metric_value(benchmark, "CAGR")
        drawdown_delta = metric_value(best, "Max Drawdown") - metric_value(benchmark, "Max Drawdown")
        cagr_word = "higher" if cagr_delta >= 0 else "lower"
        dd_word = "less severe" if drawdown_delta >= 0 else "more severe"
        benchmark_sentence = (
            f"Relative to {benchmark_name}, the strategy delivered a CAGR that was **{abs(cagr_delta):.2%} {cagr_word}** "
            f"and a maximum drawdown that was **{abs(drawdown_delta):.2%} {dd_word}**."
        )

    turnover_text = "not available"
    if turnover is not None and not turnover.empty and "Strategy" in turnover.columns:
        strat_turnover = turnover.loc[turnover["Strategy"] == best_strategy, "Turnover"]
        if not strat_turnover.empty:
            turnover_text = f"{float(strat_turnover.mean()):.1%} average turnover per rebalance"

    return f"""
### Investment Committee Brief

Under the current assumptions, **{best_strategy}** ranks as the strongest optimized strategy by Sharpe ratio. The simulation starts with **{money(settings.initial_capital)}**, uses a **{settings.lookback_days}-trading-day estimation window**, rebalances **{'monthly' if settings.rebalance_frequency == 'ME' else 'quarterly'}**, and deducts **{settings.transaction_cost_bps:.1f} bps** of simulated transaction cost per unit of turnover.

The selected strategy finished at approximately **{money(best_ending_value)}**, with **{pct(metric_value(best, 'CAGR'))} CAGR**, **{pct(metric_value(best, 'Volatility'))} annualized volatility**, **{num(metric_value(best, 'Sharpe'), 2)} Sharpe**, and **{pct(metric_value(best, 'Max Drawdown'))} maximum drawdown**. The largest current allocation exposures are **{top_weights}**.

The result should be interpreted as an allocation trade-off rather than a prediction. **{highest_cagr_strategy}** generated the highest CAGR among the optimized strategies, while **{lowest_drawdown_strategy}** controlled drawdown best. {benchmark_sentence}

Implementation note: **{best_strategy}** had **{turnover_text}**. Higher turnover can improve adaptation, but it can also reduce real-world investability once taxes, bid-ask spreads, liquidity, and market impact are considered.
"""


# --------------------------- Sidebar Controls ---------------------------

st.sidebar.markdown('<div class="control-title">ETF Allocation Terminal</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="control-caption">Adjust assumptions, rerun the optimizer, and review the impact on allocation, risk, and performance.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown('<div class="control-section">Data</div>', unsafe_allow_html=True)
    data_mode_label = st.radio(
        "Data mode",
        options=["Demo data", "Live Yahoo Finance"],
        index=0,
        help="Choose the data source. Demo data keeps the public app stable; Live Yahoo Finance attempts to download real ETF history with yfinance.",
    )
    data_mode = "live" if data_mode_label == "Live Yahoo Finance" else "demo"

    tickers_text = st.text_area(
        "ETF universe",
        value="SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, VNQ, DBC",
        height=96,
        help="Comma-separated ETF tickers included in the optimization universe. These are the investable assets the optimizer can allocate to.",
    )
    tickers = parse_tickers(tickers_text)

    st.markdown('<div class="control-section">Backtest Window</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input(
            "Start date",
            value=pd.Timestamp("2015-01-01").date(),
            help="First date of the historical sample used for returns, covariance estimation, and walk-forward backtesting.",
        )
    with col_b:
        end_date = st.date_input(
            "End date",
            value=pd.Timestamp.today().date(),
            help="Last date of the historical sample. Use today for the most recent available data.",
        )

    initial_capital = st.number_input(
        "Virtual starting capital",
        min_value=10_000,
        max_value=10_000_000,
        value=100_000,
        step=10_000,
        help="Hypothetical starting amount used to convert strategy returns into a virtual portfolio value curve.",
    )

    st.markdown('<div class="control-section">Portfolio Constraints</div>', unsafe_allow_html=True)
    max_weight = st.slider(
        "Max ETF weight",
        min_value=0.10,
        max_value=1.00,
        value=0.35,
        step=0.05,
        help="Upper allocation cap for any single ETF. Lower caps force diversification; higher caps allow concentrated portfolios.",
    )
    min_weight = st.slider(
        "Min ETF weight",
        min_value=0.00,
        max_value=0.20,
        value=0.00,
        step=0.01,
        help="Minimum allocation required for each ETF. A zero value allows the optimizer to exclude an ETF from the final allocation.",
    )

    st.markdown('<div class="control-section">Optimization Assumptions</div>', unsafe_allow_html=True)
    risk_free_rate = st.slider(
        "Annual risk-free rate",
        min_value=0.00,
        max_value=0.10,
        value=0.03,
        step=0.005,
        help="Annual risk-free return assumption used for Sharpe ratio, Sortino ratio, and alpha calculations.",
    )
    lookback_days = st.slider(
        "Optimization lookback days",
        min_value=252,
        max_value=1260,
        value=756,
        step=63,
        help="Number of historical trading days used at each rebalance to estimate returns and the covariance matrix. 252 is roughly one trading year.",
    )
    transaction_cost_bps = st.slider(
        "Transaction cost per turnover",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0,
        help="Simulated trading cost in basis points. For example, 5 bps means 0.05% cost on 100% portfolio turnover.",
    )
    rebalance_frequency = st.selectbox(
        "Rebalance frequency",
        options=["ME", "QE"],
        format_func=lambda x: "Monthly" if x == "ME" else "Quarterly",
        help="How often the portfolio is re-optimized and rebalanced in the walk-forward backtest. Monthly adapts faster; quarterly usually trades less.",
    )
    frontier_samples = st.slider(
        "Efficient frontier samples",
        min_value=1_000,
        max_value=10_000,
        value=4_000,
        step=1_000,
        help="Number of random feasible portfolios plotted for the efficient frontier. Higher values create a smoother chart but run slower.",
    )

    run_button = st.button("Run terminal analysis", type="primary", use_container_width=True)


if len(tickers) < 2:
    render_header()
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
    with st.spinner("Running allocation engine, walk-forward backtest, and risk analytics..."):
        results = cached_analysis(settings_to_dict(settings), data_mode, int(frontier_samples))
except Exception as exc:
    render_header(settings)
    st.error("The optimizer could not complete with the current settings.")
    st.exception(exc)
    st.stop()

settings = results["settings"]
performance: pd.DataFrame = results["performance"]
static_weights: pd.DataFrame = results["static_weights"]
backtest_returns: pd.DataFrame = results["backtest_returns"]
turnover: pd.DataFrame = results["turnover"]
figures = create_figures(results)

optimized_names = strategy_universe(performance)
optimized_performance = performance.loc[optimized_names]
sharpe_candidates = optimized_performance["Sharpe"].dropna() if "Sharpe" in optimized_performance.columns else pd.Series(dtype=float)
best_strategy = sharpe_candidates.idxmax() if not sharpe_candidates.empty else optimized_performance.index[0]
best_row = performance.loc[best_strategy]
ending_values = (1.0 + backtest_returns.fillna(0.0)).cumprod().iloc[-1] * settings.initial_capital
best_ending_value = float(ending_values.get(best_strategy, ending_values.max()))
committee_brief = generate_investment_committee_brief(results, best_strategy, best_ending_value)
current_allocation_strategy = best_strategy if best_strategy in static_weights.index else ("Maximum Sharpe" if "Maximum Sharpe" in static_weights.index else static_weights.index[0])
current_allocation = static_weights.loc[current_allocation_strategy].sort_values(ascending=False)
current_allocation = current_allocation[current_allocation > 1e-4]
risk_grade = risk_label(metric_value(best_row, "Sharpe"), metric_value(best_row, "Max Drawdown"))

render_header(settings, str(results.get("data_label", "Data source not specified")))

kpi_cols = st.columns(6)
with kpi_cols[0]:
    render_kpi_card(
        "Best Strategy",
        best_strategy,
        "Optimized strategy with the highest Sharpe ratio among model-generated portfolios. Benchmarks are excluded from this headline ranking.",
        footnote="Ranked by Sharpe",
        value_class="strategy",
    )
with kpi_cols[1]:
    render_kpi_card(
        "Ending Value",
        money(best_ending_value),
        "Final value of the virtual portfolio at the end of the walk-forward backtest after simulated rebalancing and transaction costs.",
        footnote=f"Start: {money(settings.initial_capital)}",
    )
with kpi_cols[2]:
    render_kpi_card(
        "CAGR",
        pct(best_row.get("CAGR")),
        "Compound Annual Growth Rate. It converts the full-period portfolio return into an annualized compounded return.",
        footnote="Annualized return",
    )
with kpi_cols[3]:
    render_kpi_card(
        "Sharpe",
        f"{best_row.get('Sharpe'):.2f}" if pd.notna(best_row.get("Sharpe")) else "n/a",
        "Risk-adjusted return metric comparing annualized excess return against annualized volatility. Higher is generally better.",
        footnote="Excess return / vol",
    )
with kpi_cols[4]:
    render_kpi_card(
        "Max Drawdown",
        pct(best_row.get("Max Drawdown")),
        "Largest peak-to-trough portfolio loss during the walk-forward backtest.",
        footnote="Downside stress",
    )
with kpi_cols[5]:
    render_kpi_card(
        "Risk Grade",
        risk_grade,
        "Plain-English summary based on Sharpe ratio and maximum drawdown. It is a dashboard label, not an investment rating.",
        footnote="Research label",
        value_class="strategy",
    )

st.markdown("<br>", unsafe_allow_html=True)
main_tabs = st.tabs(["Executive Dashboard", "Risk Monitor", "Allocation Book", "Performance Ledger"])

with main_tabs[0]:
    st.markdown('<div class="terminal-brief">', unsafe_allow_html=True)
    st.markdown(committee_brief)
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.65, 1.0])
    with left:
        render_panel("Portfolio Value Path", "Growth of the virtual account across optimized strategies and benchmarks.")
        st.plotly_chart(figures["wealth"], use_container_width=True, theme=None, config={"displaylogo": False})
    with right:
        render_panel("Current Recommended Allocation", f"Static output for {current_allocation_strategy} under the current constraints.")
        allocation_display = pd.DataFrame({"ETF": current_allocation.index, "Weight": current_allocation.values})
        st.dataframe(
            allocation_display.style.format({"Weight": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Download current allocation",
            data=allocation_display.to_csv(index=False).encode("utf-8"),
            file_name="current_recommended_allocation.csv",
            mime="text/csv",
            use_container_width=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        render_panel("Efficient Frontier", "Feasible risk-return combinations with optimized portfolios overlaid.")
        st.plotly_chart(figures["frontier"], use_container_width=True, theme=None, config={"displaylogo": False})
    with col2:
        render_panel("Correlation Structure", "Daily return correlation across the ETF universe.")
        st.plotly_chart(figures["correlation"], use_container_width=True, theme=None, config={"displaylogo": False})

with main_tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        render_panel("Drawdown Monitor", "Peak-to-trough losses by strategy. Lower magnitude drawdowns are generally preferred.")
        st.plotly_chart(figures["drawdown"], use_container_width=True, theme=None, config={"displaylogo": False})
    with col2:
        render_panel("Rolling Volatility", "Realized 63-day annualized volatility after walk-forward portfolio construction.")
        st.plotly_chart(figures["rolling_volatility"], use_container_width=True, theme=None, config={"displaylogo": False})

    col3, col4 = st.columns(2)
    with col3:
        render_panel("Risk Contribution", "Approximate contribution of each ETF to portfolio variance for the selected allocation.")
        st.plotly_chart(figures["risk_contribution"], use_container_width=True, theme=None, config={"displaylogo": False})
    with col4:
        render_panel("Underlying ETF Price Paths", "Normalized price history for the investable ETF universe.")
        st.plotly_chart(figures["prices"], use_container_width=True, theme=None, config={"displaylogo": False})

with main_tabs[2]:
    render_panel("Static Optimized Allocations", "Portfolio weights for each optimization method under current constraints.")
    st.plotly_chart(figures["weights"], use_container_width=True, theme=None, config={"displaylogo": False})
    st.dataframe(format_weights_table(static_weights), use_container_width=True)

    render_panel("Walk-Forward Allocation History", "How the selected strategy changed its allocation through time.")
    st.plotly_chart(figures["weight_history"], use_container_width=True, theme=None, config={"displaylogo": False})

with main_tabs[3]:
    render_panel("Institutional Performance Ledger", "Return, volatility, drawdown, tail-risk, benchmark, and turnover metrics.")
    st.dataframe(format_performance_table(performance), use_container_width=True)

    if not turnover.empty:
        render_panel("Turnover and Simulated Transaction Costs", "Recent rebalancing intensity and estimated implementation drag.")
        st.dataframe(turnover.tail(30), use_container_width=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "Download performance table",
            data=performance.to_csv().encode("utf-8"),
            file_name="performance_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl2:
        st.download_button(
            "Download optimized weights",
            data=static_weights.to_csv().encode("utf-8"),
            file_name="optimized_weights.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with dl3:
        st.download_button(
            "Download committee brief",
            data=committee_brief.encode("utf-8"),
            file_name="investment_committee_brief.md",
            mime="text/markdown",
            use_container_width=True,
        )

st.markdown(
    f"""
    <div class="footer-signature">
        Built by Nicola Fanelli · Virtual research project · Educational use only, not investment advice ·
        <a href="{html.escape(LINKEDIN_URL, quote=True)}" target="_blank" rel="noopener noreferrer">LinkedIn</a>
    </div>
    """,
    unsafe_allow_html=True,
)
