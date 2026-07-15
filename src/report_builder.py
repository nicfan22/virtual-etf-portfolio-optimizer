"""HTML report and visualization builder for the ETF optimizer."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .portfolio_engine import (
    OptimizerSettings,
    drawdown_frame,
    risk_contributions,
    run_full_analysis,
    wealth_index,
)

PLOTLY_TEMPLATE = "plotly_white"


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


def style_figure(fig: go.Figure, title: str, height: int = 520) -> go.Figure:
    """Apply consistent visual styling to a Plotly figure."""
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left"},
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=40, r=30, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
    )
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
    style_figure(fig_prices, "ETF Universe: Normalized Historical Price Paths")

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
    style_figure(fig_corr, "ETF Return Correlation Matrix", height=560)

    weights_long = static_weights.reset_index(names="Strategy").melt(
        id_vars="Strategy", var_name="ETF", value_name="Weight"
    )
    fig_weights = px.bar(weights_long, x="Strategy", y="Weight", color="ETF", text_auto=".1%")
    fig_weights.update_yaxes(tickformat=".0%", title="Portfolio Weight")
    fig_weights.update_xaxes(title="")
    style_figure(fig_weights, "Static Optimized ETF Allocations", height=600)

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
                marker=dict(size=12, symbol="diamond"),
                hovertemplate=(
                    f"{html.escape(strategy)}<br>Expected Return: %{{y:.2%}}"
                    "<br>Expected Volatility: %{x:.2%}<extra></extra>"
                ),
            )
        )
    style_figure(fig_frontier, "Efficient Frontier with Optimized Portfolios", height=620)

    wealth = wealth_index(backtest_returns, settings.initial_capital)
    fig_wealth = px.line(wealth, x=wealth.index, y=wealth.columns)
    fig_wealth.update_yaxes(title="Portfolio Value", tickprefix="$", separatethousands=True)
    fig_wealth.update_xaxes(title="Date")
    style_figure(fig_wealth, f"Growth of {money(settings.initial_capital)} Virtual Portfolio")

    dd = drawdown_frame(backtest_returns)
    fig_drawdown = px.line(dd, x=dd.index, y=dd.columns)
    fig_drawdown.update_yaxes(title="Drawdown", tickformat=".0%")
    fig_drawdown.update_xaxes(title="Date")
    style_figure(fig_drawdown, "Portfolio Drawdowns by Strategy")

    rolling_vol = backtest_returns.rolling(63).std() * np.sqrt(252)
    fig_rolling_vol = px.line(rolling_vol.dropna(how="all"), x=rolling_vol.dropna(how="all").index, y=rolling_vol.columns)
    fig_rolling_vol.update_yaxes(title="Annualized Volatility", tickformat=".0%")
    fig_rolling_vol.update_xaxes(title="Date")
    style_figure(fig_rolling_vol, "Rolling 63-Day Annualized Volatility")

    # Use the current Maximum Sharpe weights if available; otherwise use the first static strategy.
    selected_strategy = "Maximum Sharpe" if "Maximum Sharpe" in static_weights.index else static_weights.index[0]
    risk_contrib = risk_contributions(static_weights.loc[selected_strategy], covariance).reset_index()
    risk_contrib.columns = ["ETF", "Risk Contribution"]
    fig_risk_contrib = px.bar(risk_contrib, x="ETF", y="Risk Contribution", text_auto=".1%")
    fig_risk_contrib.update_yaxes(title="Risk Contribution", tickformat=".0%")
    style_figure(fig_risk_contrib, f"Risk Contribution: {selected_strategy}", height=460)

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
