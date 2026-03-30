# Quantitative Portfolio Optimization
## Black–Litterman Model & Macro-Factor Return Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-blue?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research-lightgrey?style=flat-square)

---

## Overview

This research project explores **quantitative portfolio construction** within the Black–Litterman framework, combining:

- **Bayesian portfolio allocation** via the Black–Litterman model
- **Macro-factor return forecasting** using Ridge Regression as investor views
- **Momentum-based views** as a simple benchmark strategy
- **Mean–variance optimization** within the Markowitz framework
- **Backtesting** using vectorbt to evaluate out-of-sample performance

---

## Research Question

> *Does macroeconomic information contain predictive signal on asset returns **beyond** what prices themselves already convey?
> If a rule as simple as momentum can match or outperform a macro regression model, is the added complexity justified?*

To answer this, the project compares two approaches for generating Black–Litterman views:

| Approach | Description | Complexity |
|----------|-------------|------------|
| **Ridge Regression** | Views derived from macro factors (rates, inflation, VIX, industrial production) | High |
| **Momentum** | Views derived from 12-month price momentum (Jegadeesh & Titman, 1993) | Low |

Both are then fed into the same Black–Litterman + Markowitz pipeline and evaluated against a **SPY buy-and-hold benchmark**.

---

## Theoretical Background

### 1. Mean–Variance Optimization (Markowitz)

The classical portfolio optimization problem:

```
minimize:   wᵀΣw
subject to: wᵀμ = μₚ,  ∑ᵢ wᵢ = 1,  wᵢ ≥ 0
```

where **μ** is the vector of expected returns, **Σ** the covariance matrix, and **w** the weight vector. This framework is highly sensitive to return estimation errors and prone to extreme weights.

---

### 2. Black–Litterman Model

The Black–Litterman model addresses these limitations by blending market equilibrium returns with investor views in a Bayesian setting.

**Implied equilibrium returns (prior):**
```
π = δ · Σ · wₘ
```

**Posterior expected returns:**
```
μ_BL = [ (τΣ)⁻¹ + Pᵀ Ω⁻¹ P ]⁻¹ [ (τΣ)⁻¹ π + Pᵀ Ω⁻¹ Q ]
```

| Symbol | Description |
|--------|-------------|
| π | Implied equilibrium returns (CAPM) |
| P | View matrix |
| Q | View returns |
| Ω | View uncertainty matrix |
| τ | Scaling parameter |

---

### 3. View Generation — Two Methods

#### Method A — Ridge Regression (ML approach)

For each asset *i*, expected returns are modelled as:

```
Rᵢ,ₜ₊₁ = αᵢ + βᵢ,₁F¹ₜ + βᵢ,₂F²ₜ + ... + βᵢ,ₖFᵏₜ + εᵢ,ₜ₊₁
```

Four macroeconomic factors are used:

| Factor | Source (FRED) | Transformation |
|--------|--------------|----------------|
| Δ 10Y Treasury Yield | `GS10` | First difference |
| CPI Inflation | `CPIAUCSL` | Monthly return |
| Δ VIX | `VIXCLS` | First difference |
| Industrial Production | `INDPRO` | Monthly return |

The model is estimated via **Ridge Regression** (L2 regularisation) to handle multicollinearity. The view uncertainty matrix **Ω** is calibrated using the out-of-sample **MSE** of each regression — a model with higher prediction error contributes a less confident view.

#### Method B — Momentum (benchmark)

Based on the well-documented momentum factor (Jegadeesh & Titman, 1993):

```
Momᵢ = P(t-21) / P(t-252) - 1       # 12-month return, skipping last month
zᵢ   = (Momᵢ - mean(Mom)) / std(Mom) # cross-sectional normalisation
Qᵢ   = πᵢ + zᵢ × k                  # view = prior adjusted by momentum signal
```

The last month is excluded to avoid short-term reversal. This approach requires only price data and zero modelling effort — making it a strong and honest benchmark.

---

## Methodology

1. **Data Collection** — Asset prices (yfinance), market cap weights, macroeconomic indicators (FRED)
2. **Covariance Estimation** — Ledoit-Wolf shrinkage (`constant_correlation`) for robust Σ
3. **View Generation** — Either Ridge Regression or Momentum (see above)
4. **Black–Litterman** — Compute posterior returns μ_BL and Σ_BL
5. **Portfolio Optimization** — Maximize Sharpe Ratio using μ_BL and Σ_BL
6. **Backtesting** — Monthly rebalancing simulation via vectorbt, benchmarked against SPY

---

## Project Structure

```
├── Main/
│   └── QPM_BlackLitterman.ipynb   # Main notebook
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Technologies

| Library | Usage |
|---------|-------|
| NumPy / pandas | Data manipulation |
| scikit-learn | Ridge Regression, StandardScaler |
| PyPortfolioOpt | Black–Litterman, Efficient Frontier |
| yfinance | Market price data |
| pandas_datareader | Macroeconomic data (FRED) |
| vectorbt | Portfolio backtesting |
| matplotlib / seaborn | Visualization |

---

## Getting Started

```bash
git clone https://github.com/FlDx031/Portfolio-Optimization-with-Black-Litterman-Model.git
cd Portfolio-Optimization-with-Black-Litterman-Model
pip install -r requirements.txt
jupyter notebook
```

---

## References

[1] Black, F., & Litterman, R. (1991). *Combining investor views with market equilibrium*. The Journal of Fixed Income.

[2] Idzorek, T. (2007). *A step-by-step guide to the Black-Litterman model*. In: Forecasting Expected Returns in the Financial Markets. Elsevier.

[3] Jegadeesh, N., & Titman, S. (1993). *Returns to buying winners and selling losers: Implications for stock market efficiency*. The Journal of Finance, 48(1), 65–91.

[4] Ledoit, O., & Wolf, M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices*. Journal of Multivariate Analysis, 88(2), 365–411.

---

> ⚠️ **Disclaimer** — This project is for educational and research purposes only. It does not constitute financial advice.
