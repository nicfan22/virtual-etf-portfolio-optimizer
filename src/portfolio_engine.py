"""Portfolio analytics, optimization, and backtesting engine.

This module is intentionally framework-agnostic. It can be imported by the
Streamlit app, a Colab notebook, a scheduled report script, or tests.
"""

from __future__ import annotations

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

from .demo_data import generate_demo_prices

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
