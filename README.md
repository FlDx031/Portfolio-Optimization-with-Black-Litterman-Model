# Portfolio allocation with Black–Litterman

This master's-level educational project uses macroeconomic return forecasts as views in a Black–Litterman portfolio. The notebook presents the data, methodology, and results; four small Python modules contain the calculations. The backtest uses a manual monthly loop.

**Research question: do macroeconomic factors add value beyond a simple momentum signal?** Outperformance is not assumed. A finding of no improvement is also a valid result.

## Getting started

Use **Python 3.12**, the version tested for this project. First clone the repository and enter its root:

```bash
git clone https://github.com/FlDx031/Portfolio-Optimization-with-Black-Litterman-Model.git
cd Portfolio-Optimization-with-Black-Litterman-Model
```

Then create an environment and run the offline tests.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

macOS or Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The offline tests require no API key. The main notebook run requires a free FRED API key to retrieve historical ALFRED snapshots. Set it as described below before opening the notebook. In Jupyter, select **Kernel → Restart Kernel and Run All Cells**. The first run downloads Yahoo Finance prices and ALFRED snapshots; later runs reuse `data/cache/`. Network failures are reported and never replaced with artificial data. The dates are fixed. Incomplete prices, including a missing NYSE session at the beginning, middle, or end of the requested range, cause an explicit failure.

`requirements.txt` pins the tested direct dependencies. Each run's manifest records calculation-library versions. To preserve indirect dependencies as well, run `python -m pip freeze > requirements-local.txt` and archive that file with the cache and results.

### Run the main macro experiment

Get a free [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) and set it **before starting Jupyter**:

```powershell
# PowerShell: replace this value locally; do not put the key in the notebook.
$env:FRED_API_KEY = "your-key"
$env:RUN_MODE = "macro"
python -m notebook Portfolio_BlackLitterman.ipynb
```

On macOS/Linux, use `export FRED_API_KEY="your-key"` before launching Jupyter. The default `RUN_MODE=macro` requires the key and stops with a clear error if it is missing. The project never silently substitutes today's revised macro series.

The earlier version used `pandas_datareader` to download current FRED series without a key. Those series could contain revisions made after a historical backtest decision. The current experiment instead requests ALFRED snapshots as they existed at each decision date; [the FRED API requires a key](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) for these requests.

For a **price-only diagnostic** without a key, set `$env:RUN_MODE = "price_only"` in PowerShell before starting Jupyter (or `export RUN_MODE=price_only` on macOS/Linux). This checks the price and backtest pipeline but excludes BL Ridge and cannot answer the macro research question. Set `RUN_MODE` back to `macro` for the main experiment.

The initial macro download makes four requests per signal month and may take several minutes. Snapshots are cached as they arrive, without storing the key. If interrupted, a later run resumes the missing downloads.

## Repository layout

```text
Portfolio_BlackLitterman.ipynb   # Entry point: data, explanations, tables, charts
src/
    data.py                       # Adjusted prices, trading calendar, ALFRED snapshots
    model.py                      # Covariance, Ridge, momentum, Black–Litterman, optimization
    backtest.py                   # Monthly loop, costs, and performance statistics
    plots.py                      # Four simple charts
tests/test_project.py             # Deterministic offline tests
requirements.txt
.github/workflows/tests.yml       # Automated tests
LICENSE                           # Apache 2.0
```

`data/cache/` and `results/` are created at runtime and ignored by Git. The former `Main/QPM_BlackLitterman.ipynb` notebook and `BacktestBL.py` have been replaced; their previous versions remain in Git history.

## Data and strategies

The fixed universe contains five USD-listed ETFs: SPY (US equities), EFA (developed equities outside the US), EEM (emerging markets), IEF (US Treasury bonds), and GLD (gold). The download starts in January 2010 and ends on January 15, 2026. The first sessions of January 2026 close the final holding period. The effective performance period begins after the initial training period and is displayed by the notebook.

This choice avoids reconstructing a historical portfolio from today's winning stocks and market capitalizations. The ETF universe was still selected retrospectively; it is not a historical database of every instrument available on each date.

| Strategy | Construction |
|---|---|
| Equal weight | Equal weights, rebalanced monthly |
| Markowitz | Historical monthly means and covariance |
| BL Momentum | Reference prior combined with a momentum signal |
| BL Ridge (main strategy) | Same prior combined with macro-based Ridge predictions |
| SPY | Buy and hold, without rebalancing |

Equal weight is the benchmark drawn from the same asset universe. SPY provides an equity reference with a different risk profile from a portfolio that also holds bonds and gold. The optimized strategies use the same return–risk objective, are fully invested, and allow neither leverage nor short selling.

## Backtest chronology

1. At the month's final trading close, estimate parameters using prices available through that close.
2. Calculate target weights and apply them at the next session's close, at the beginning of the following month.
3. Hold positions until the next month's first session. Weights drift as prices change.
4. Measure the buying and selling needed to return to the new target weights, pay the costs, and repeat.

For example, a late-January signal buys at the first February close and targets the return through the first March close. That return is **not yet known at the late-February signal**, so it is excluded from training at that point. The exported timeline makes this ordering auditable.

All strategies begin after the same warm-up period, including in the price-only diagnostic. An incomplete final month is never annualized as a full month. Returns before the first trade are excluded from the metrics. An optimization failure stops the calculation instead of silently changing strategies.

### Macro data available at the time

For each signal date, [FRED/ALFRED observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html) are requested with `realtime_start = realtime_end = the day before the signal`. This [real-time period](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) selects information known then, rather than revisions available today. Using the previous day avoids ambiguity about release times during the signal session.

Each factor uses the latest two completed observation months available in that snapshot. No future value fills a gap, and historical factor rows remain frozen for training.

| Series | Factor | Transformation |
|---|---|---|
| GS10 | 10-year Treasury yield | Difference, in percentage points |
| VIXCLS | VIX | Difference between the latest available monthly values |
| CPIAUCSL | Inflation | Relative change in CPI |
| INDPRO | Industrial activity | Relative change in production |

Publication delays differ by series. Inflation is not called a “surprise” because no market consensus is used. Industrial activity is measured by industrial production, not a PMI.

## Core formulas

### 1. One time unit: the month

Returns, covariances, views, and uncertainties are monthly. Only the final performance measures are annualized. The constant assumed risk-free rate is converted as follows:

```math
r_f=(1+r_f^{annual})^{1/12}-1.
```

The covariance is estimated from 36 month-end returns and stabilized with a fixed adjustment:

```math
\Sigma=0.8\,\widehat\Sigma+0.2\,\mathrm{diag}(\widehat\Sigma).
```

This is fixed shrinkage toward the diagonal, **not** the Ledoit–Wolf estimator. A very small diagonal term provides numerical stability.

### 2. Allocation prior

For an equal-weight reference portfolio $w_0$ and risk aversion $\delta=3$:

```math
\pi=\delta\Sigma w_0,\qquad w_{0,i}=1/N.
```

$\pi$ represents monthly excess returns above the risk-free rate. This prior describes a reference allocation; it does not claim to reproduce market equilibrium or estimate market risk aversion from SPY.

### 3. Momentum views

At the end of month $t$, the latest month is excluded:

```math
m_{i,t}=\frac{p_{i,t-1}}{p_{i,t-12}}-1,\qquad z_i=\frac{m_i-\bar m}{\sigma_m}.
```

```math
Q_i=\pi_i+kz_i,\qquad k=0.005,\qquad \Omega=\mathrm{diag}(\tau\Sigma).
```

$k$ is a fixed monthly view strength, not an annual return. If all signals are identical, their standardized scores are zero. The rule measures eleven months of performance and skips the most recent month.

### 4. Ridge views and uncertainty

Factors known at the signal predict the next holding period's return. The regression minimizes:

```math
\sum_t(R_{i,t}-a_i-X_t^Tb_i)^2+\alpha\lVert b_i\rVert_2^2,\qquad\alpha=1.
```

Factors are centered and scaled using training observations only. The intercept is not penalized. At most 60 fully observed factor–return pairs are retained.

The latest 12 known observations provide chronological validation. Each prediction is refitted using only earlier observations, starting with at least 24 training observations. Thus:

```math
\Omega_{ii}=\mathrm{MSE}_{i,\mathrm{validation}},\qquad Q_i=\widehat R_i-r_f.
```

After validation, Ridge is refitted on all currently known pairs to produce the next view. This absolute prediction does not include the prior, and training error is not substituted for validation MSE.

### 5. Black–Litterman

With absolute views $P=I$, $A=\tau\Sigma$, and $\tau=0.05$:

```math
\mu_{BL}=\pi+A(A+\Omega)^{-1}(Q-\pi),
```

```math
\Sigma_{BL}=\Sigma+A-A(A+\Omega)^{-1}A.
```

As uncertainty $\Omega$ increases, a view moves the prior less. The code solves linear systems instead of explicitly inverting matrices.

### 6. Optimal weights

The optimizer maximizes a simple mean–variance utility:

```math
\max_w\;w^T\mu-\frac{\delta}{2}w^T\Sigma w,\qquad\sum_iw_i=1,\quad0\leq w_i\leq1.
```

BL uses posterior moments; Markowitz uses the historical mean excess return and estimated covariance. This objective avoids difficulties with maximizing a Sharpe ratio when all expected returns are negative. Sharpe remains a performance measure, not the optimization objective.

### 7. Costs and performance

Let $w_t^-$ be the pre-trade weights after drift, and let $c=0.001$ (10 basis points per amount bought or sold):

```math
T_t=\sum_i|w_{i,t}-w_{i,t}^-|,\qquad R_{p,t}^{net}=(1-cT_t)(1+w_t^TR_t)-1.
```

The initial purchase from cash has $T=1$. Completely replacing one asset with another has $T=2$. SPY pays the same entry cost. Positions are valued at the end of the test without terminal liquidation. This weight-based approximation does not solve for the tiny share adjustment needed to fund transaction costs exactly.

```math
V_0=1,\quad V_t=V_{t-1}(1+R_{p,t}^{net}),\quad CAGR=V_T^{12/T}-1,
```

```math
\sigma_{ann}=\sigma(R^{net})\sqrt{12},\quad Sharpe=\frac{12\,\overline{(R^{net}-r_f)}}{\sigma_{ann}},
```

```math
DD_t=\frac{V_t}{\max_{0\leq s\leq t}V_s}-1.
```

Initial wealth is included in drawdown, including when the first month loses money. Annual turnover is the monthly mean of $T_t$ multiplied by 12; it includes the initial entry.

## Results to examine in the notebook

- Adjusted prices indexed to 100 to put the assets in context.
- Cumulative wealth after costs and monthly drawdowns.
- Total return, CAGR, volatility, Sharpe, maximum drawdown, and turnover.
- Portfolio composition over time to inspect concentration.
- Return–risk comparison and sensitivity to 0, 10, and 25 basis points of costs.
- Timeline and CSV/PNG exports, with parameters, versions, and data fingerprints in `results/run.json`.

These are the **expected analysis outputs**, not a promised level of performance. Interpretation should compare BL Ridge with BL Momentum and equal weight, then check whether any gain survives costs at comparable risk. Testing other periods is a robustness check, not an invitation to select the best-performing one.

### Empirical status

No BL Ridge result from real historical macro data has been produced in this repository yet. The notebook's saved outputs have been cleared so a previous price-only diagnostic is not mistaken for the main experiment. After a run with a FRED key, `results/run.json` records `run_mode: "macro"` and `macro_alfred: true` alongside data fingerprints and the results. Figures are not forecasts and may change if the data provider revises historical prices.

## Limitations

Adjusted Yahoo prices are a snapshot downloaded today, not vintage price data. They approximate returns with reinvested dividends, but the exact dividend payment calendar is not simulated. The provider may correct prices later. Preserve the cache and its fingerprints to reproduce a run.

The universe was selected retrospectively. Costs are constant and do not represent an order book: there is no market impact, taxation, variable spread, liquidity constraint, or fractional-share constraint. The risk-free rate is constant. Monthly drawdown can understate losses within a month. Hyperparameters are educational choices rather than calibrated estimates, and one period cannot establish predictive ability. ALFRED addresses macro data availability at the time, but not every limitation of a historical experiment.

## Verification

```bash
python -m unittest discover -s tests -v
```

The tests use explicitly labeled synthetic prices to check cost accounting, weight drift, label availability, the invariance of past weights when future data change, BL confidence limits, Ridge validation, initial drawdown, NYSE session coverage, and mocked ALFRED requests and caching. They do not provide empirical validation of the strategy or a live FRED test.

When changing the project, edit functions in `src/`, run the tests, and then rerun the notebook from top to bottom. Never publish an API key. Notebook outputs can be cleared before sharing a change if they make the diff too large.

## References

- Black, F. and Litterman, R. (1991), *Combining Investor Views with Market Equilibrium*, Journal of Fixed Income.
- Idzorek, T. (2007), *A Step-by-Step Guide to the Black–Litterman Model*.
- Jegadeesh, N. and Titman, S. (1993), *Returns to Buying Winners and Selling Losers*, Journal of Finance. The multi-asset signal here is an educational adaptation, not a replication of that paper.
- [FRED documentation: real-time periods](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html).

Educational project under the Apache 2.0 license; no investment recommendation.
