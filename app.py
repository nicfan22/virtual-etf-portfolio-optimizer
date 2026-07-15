"""Streamlit web app for the Virtual ETF Portfolio Optimizer.

Deploy this file to Streamlit Community Cloud to get a public URL for LinkedIn.
The app is editable: change sidebar inputs, modify this Python file, or edit
src/portfolio_engine.py to extend the quantitative logic.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.portfolio_engine import OptimizerSettings, run_full_analysis
from src.report_builder import build_html_report, create_figures, money, pct

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
st.sidebar.caption("Change the assumptions, rerun the engine, and deploy this app as a shareable URL.")

with st.sidebar:
    data_mode_label = st.radio(
        "Data mode",
        options=["Demo data", "Live Yahoo Finance"],
        index=0,
        help="Demo mode always works. Live mode requires internet access and the yfinance package.",
    )
    data_mode = "live" if data_mode_label == "Live Yahoo Finance" else "demo"

    tickers_text = st.text_area(
        "ETF universe",
        value="SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, VNQ, DBC",
        height=95,
        help="Use comma-separated ETF tickers.",
    )
    tickers = parse_tickers(tickers_text)

    col_a, col_b = st.columns(2)
    with col_a:
        start_date = st.date_input("Start date", value=pd.Timestamp("2015-01-01").date())
    with col_b:
        end_date = st.date_input("End date", value=pd.Timestamp.today().date())

    initial_capital = st.number_input("Virtual starting capital", min_value=10_000, max_value=10_000_000, value=100_000, step=10_000)
    max_weight = st.slider("Max ETF weight", min_value=0.10, max_value=1.00, value=0.35, step=0.05)
    min_weight = st.slider("Min ETF weight", min_value=0.00, max_value=0.20, value=0.00, step=0.01)
    risk_free_rate = st.slider("Annual risk-free rate", min_value=0.00, max_value=0.10, value=0.03, step=0.005)
    lookback_days = st.slider("Optimization lookback days", min_value=252, max_value=1260, value=756, step=63)
    transaction_cost_bps = st.slider("Transaction cost per turnover", min_value=0.0, max_value=50.0, value=5.0, step=1.0)
    rebalance_frequency = st.selectbox("Rebalance frequency", options=["ME", "QE"], format_func=lambda x: "Monthly" if x == "ME" else "Quarterly")
    frontier_samples = st.slider("Efficient frontier samples", min_value=1_000, max_value=10_000, value=4_000, step=1_000)

    run_button = st.button("Run optimization", type="primary", use_container_width=True)

st.markdown('<div class="big-title">Virtual ETF Portfolio Optimization & Risk Management Engine</div>', unsafe_allow_html=True)
st.markdown(
    "A LinkedIn-ready quantitative finance web app for ETF allocation research, walk-forward backtesting, transaction-cost simulation, and institutional risk analytics."
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

best_strategy = performance.index[0]
best_row = performance.loc[best_strategy]
ending_values = (1.0 + backtest_returns.fillna(0.0)).cumprod().iloc[-1] * settings.initial_capital
best_ending_value = float(ending_values.max())

st.info(f"Data source used: {results['data_label']}. This project is virtual research and not investment advice.")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Best Strategy", best_strategy)
with kpi_cols[1]:
    st.metric("Best Ending Value", money(best_ending_value))
with kpi_cols[2]:
    st.metric("Best CAGR", pct(best_row.get("CAGR")))
with kpi_cols[3]:
    st.metric("Best Sharpe", f"{best_row.get('Sharpe'):.2f}" if pd.notna(best_row.get("Sharpe")) else "n/a")

main_tabs = st.tabs(["Dashboard", "Allocations", "Performance Table", "Methodology", "LinkedIn Export"])

with main_tabs[0]:
    st.subheader("Interactive Dashboard")
    st.plotly_chart(figures["wealth"], use_container_width=True)
    st.plotly_chart(figures["drawdown"], use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(figures["frontier"], use_container_width=True)
        st.plotly_chart(figures["risk_contribution"], use_container_width=True)
    with col2:
        st.plotly_chart(figures["correlation"], use_container_width=True)
        st.plotly_chart(figures["rolling_volatility"], use_container_width=True)
    st.plotly_chart(figures["prices"], use_container_width=True)

with main_tabs[1]:
    st.subheader("Portfolio Allocations")
    st.plotly_chart(figures["weights"], use_container_width=True)
    st.plotly_chart(figures["weight_history"], use_container_width=True)
    st.dataframe(static_weights.style.format("{:.2%}"), use_container_width=True)

with main_tabs[2]:
    st.subheader("Performance Metrics")
    st.dataframe(format_performance_table(performance), use_container_width=True)
    if not turnover.empty:
        st.subheader("Turnover and Simulated Transaction Costs")
        st.dataframe(turnover.tail(30), use_container_width=True)

with main_tabs[3]:
    st.subheader("Methodology")
    st.markdown(
        """
        This engine follows a professional research workflow:

        1. Download or generate ETF price data.
        2. Clean and validate price histories.
        3. Compute daily returns, expected returns, covariance, and correlation.
        4. Build Equal Weight, Minimum Volatility, Maximum Sharpe, Risk Parity, and Maximum Diversification portfolios.
        5. Run a walk-forward backtest using only data available at each rebalance date.
        6. Apply simulated transaction costs from portfolio turnover.
        7. Compare strategies using CAGR, volatility, Sharpe, Sortino, drawdown, VaR, CVaR, alpha, beta, and turnover.

        The project is intentionally virtual. It is designed to demonstrate quantitative finance engineering, not to provide investment advice.
        """
    )

with main_tabs[4]:
    st.subheader("Export a LinkedIn-Shareable Report")
    st.write("Generate a standalone HTML dashboard that can be hosted through GitHub Pages or attached to your portfolio website.")

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = Path(tmpdir) / "virtual_etf_optimizer_report.html"
        build_html_report(results, report_path)
        html_bytes = report_path.read_bytes()

    st.download_button(
        label="Download standalone HTML report",
        data=html_bytes,
        file_name="virtual_etf_optimizer_report.html",
        mime="text/html",
        use_container_width=True,
    )

    st.markdown(
        """
        **Recommended LinkedIn workflow**

        1. Push this project to GitHub.
        2. Deploy `app.py` to Streamlit Community Cloud for an interactive URL.
        3. Or publish the `/docs` folder with GitHub Pages for a static URL.
        4. Add the URL to your LinkedIn post and your LinkedIn Featured section.
        """
    )

st.download_button(
    "Download performance table as CSV",
    data=performance.to_csv().encode("utf-8"),
    file_name="performance_summary.csv",
    mime="text/csv",
)
