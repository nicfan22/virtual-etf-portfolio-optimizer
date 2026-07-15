# Virtual ETF Portfolio Optimization & Risk Management Engine

An editable, LinkedIn-ready quantitative finance project that simulates how an asset manager, robo-advisor, or quant research team could build, optimize, backtest, and report on multi-asset ETF portfolios.

This repository includes two publishable outputs:

1. **Interactive Streamlit app**: deploy `app.py` and share the live URL.
2. **Static GitHub Pages report**: publish the `/docs` folder and share the project URL.

> Educational project only. This is virtual research, not investment advice.

---

## What the Project Does

The engine analyzes a diversified ETF universe across equities, bonds, international markets, real estate, gold, and commodities. It builds multiple portfolio strategies, compares them out of sample, applies simulated transaction costs, and generates professional risk analytics.

Default ETF universe:

```text
SPY, QQQ, IWM, EFA, EEM, TLT, IEF, GLD, VNQ, DBC
```

Portfolio methods:

- Equal Weight
- Minimum Volatility
- Maximum Sharpe
- Risk Parity
- Maximum Diversification
- SPY benchmark
- 60/40 benchmark when bond data is available

---

## Key Features

- Editable Streamlit dashboard
- LinkedIn-shareable URL deployment path
- GitHub Pages-ready static HTML report
- Yahoo Finance live-data mode through `yfinance`
- Synthetic demo-data mode so the app always works in public demos
- Data cleaning and validation pipeline
- Return, volatility, covariance, and correlation analysis
- Ledoit-Wolf shrinkage covariance estimation
- Constrained long-only portfolio optimization
- Efficient frontier visualization
- Walk-forward out-of-sample backtesting
- Simulated transaction costs from portfolio turnover
- Institutional metrics: CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar, VaR, CVaR, alpha, beta, correlation, turnover
- Interactive Plotly charts
- Exportable standalone HTML report

---

## Repository Structure

```text
virtual-etf-portfolio-optimizer/
|
|-- app.py                         # Streamlit app for public URL deployment
|-- requirements.txt               # Python dependencies
|-- runtime.txt                    # Python runtime for Streamlit Cloud
|-- config.yaml                    # Editable project assumptions
|
|-- src/
|   |-- portfolio_engine.py         # Optimization, backtesting, metrics
|   |-- report_builder.py           # Plotly figures and static HTML report
|   |-- demo_data.py                # Synthetic demo market data generator
|
|-- scripts/
|   |-- generate_static_report.py   # Creates docs/index.html
|   |-- run_analysis_export.py      # Exports CSV tables and HTML report
|
|-- docs/
|   |-- index.html                  # GitHub Pages static report
|
|-- linkedin/
|   |-- linkedin_post.md            # Ready-to-edit LinkedIn post
|   |-- resume_bullets.md           # Resume bullets
|   |-- deployment_guide.md         # Step-by-step URL deployment guide
|
|-- .streamlit/
|   |-- config.toml                 # Streamlit theme
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL that Streamlit prints in your terminal.

---

## Deploy as a Public Streamlit App URL

1. Create a GitHub repository named `virtual-etf-portfolio-optimizer`.
2. Upload all project files.
3. Go to Streamlit Community Cloud.
4. Connect your GitHub account.
5. Select your repository.
6. Set the main file path to:

```text
app.py
```

7. Deploy.
8. Copy the live app URL and add it to LinkedIn.

Recommended LinkedIn title:

```text
Virtual ETF Portfolio Optimization & Risk Management Engine
```

---

## Deploy as a GitHub Pages Static Report URL

This repository already includes:

```text
docs/index.html
```

To publish it:

1. Push the project to GitHub.
2. Open your repository settings.
3. Go to **Pages**.
4. Set source to **Deploy from a branch**.
5. Choose branch `main` and folder `/docs`.
6. Save.
7. Your project page will be available at:

```text
https://YOUR_USERNAME.github.io/virtual-etf-portfolio-optimizer/
```

---

## Generate the Static Report Again

```bash
python scripts/generate_static_report.py
```

Or export all outputs:

```bash
python scripts/run_analysis_export.py
```

---

## How to Customize

Edit the portfolio universe in the Streamlit sidebar or modify `config.yaml`.

Common changes:

```yaml
portfolio:
  tickers:
    - VTI
    - VXUS
    - BND
    - QQQ
    - GLD
  max_weight: 0.40
  lookback_days: 756
  transaction_cost_bps: 5.0
```

You can also edit:

- `src/portfolio_engine.py` to change quant models
- `src/report_builder.py` to change dashboard layout
- `docs/index.html` to edit the public static page
- `linkedin/linkedin_post.md` to customize your LinkedIn announcement

---

## Resume Description

**Virtual ETF Portfolio Optimization & Risk Management Engine**  
Python, Streamlit, Pandas, NumPy, SciPy, Scikit-learn, Plotly, yFinance

Built an editable ETF allocation and risk analytics web app that optimizes multi-asset portfolios using constrained optimization, Ledoit-Wolf covariance estimation, risk parity, maximum Sharpe, minimum volatility, maximum diversification, and walk-forward backtesting. Simulated transaction costs, benchmarked strategies against SPY and 60/40 portfolios, and generated institutional performance metrics including CAGR, volatility, Sharpe, Sortino, max drawdown, VaR, CVaR, alpha, beta, and turnover. Deployed the project as a LinkedIn-shareable dashboard URL.

---

## Limitations

- Historical and synthetic demo results do not guarantee future performance.
- The project does not model taxes, ETF expense ratios, real bid-ask spreads, market impact, liquidity limits, or broker execution.
- Live-data quality depends on external data-provider availability.
- This is an educational portfolio project, not a financial advisory tool.
