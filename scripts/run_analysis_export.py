"""Run the ETF optimizer and export tables plus a static HTML report."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.portfolio_engine import OptimizerSettings, run_full_analysis  # noqa: E402
from src.report_builder import build_html_report  # noqa: E402


if __name__ == "__main__":
    settings = OptimizerSettings(end_date="2026-07-14")
    results = run_full_analysis(settings, data_mode="demo")

    outputs = ROOT / "outputs"
    outputs.mkdir(exist_ok=True)

    results["prices"].to_csv(outputs / "clean_prices.csv")
    results["returns"].to_csv(outputs / "daily_returns.csv")
    results["asset_summary"].to_csv(outputs / "asset_summary.csv")
    results["static_weights"].to_csv(outputs / "optimized_weights.csv")
    results["static_summary"].to_csv(outputs / "static_optimizer_summary.csv")
    results["frontier"].to_csv(outputs / "efficient_frontier_samples.csv", index=False)
    results["backtest_returns"].to_csv(outputs / "backtest_strategy_returns.csv")
    results["performance"].to_csv(outputs / "performance_summary.csv")
    results["turnover"].to_csv(outputs / "turnover_report.csv", index=False)

    build_html_report(results, ROOT / "docs" / "index.html")
    build_html_report(results, outputs / "virtual_etf_optimizer_report.html")
    print(f"Outputs written to: {outputs}")
