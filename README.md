# Quantitative Portfolio Optimization 
## Black–Litterman Model & Return Prediction via Multiple Linear Regression

## Overview

This repository explores **quantitative portfolio construction** by combining:

- **Return prediction using Multiple Linear Regression**
- **Bayesian portfolio allocation using the Black–Litterman model**
- **Mean–variance optimization within the Markowitz framework**

The project illustrates how **macroeconomic-driven return forecasts** can be integrated into **modern portfolio theory** to obtain more stable and economically interpretable portfolio allocations.

---

## 🎯 Objectives

- Predict asset returns using **macroeconomic factors**
- Incorporate regression-based views into portfolio allocation via **Black–Litterman**
- Compare classical mean–variance optimization with Bayesian allocation
- Highlight the role of **confidence in investor views**

---

## Theoretical Background

### 1. Mean–Variance Optimization (Markowitz)

The classical portfolio optimization problem is defined as:

\[
\min_w \; w^\top \Sigma w
\quad \text{s.t.} \quad
w^\top \mu = \mu_p, \quad \sum_{i=1}^N w_i = 1
\]

where:
- \( \mu \) is the vector of expected returns
- \( \Sigma \) is the covariance matrix of asset returns

While foundational, this framework is known to be:
- Highly sensitive to estimation errors
- Prone to producing extreme portfolio weights

---

### 2. Black–Litterman Model

The Black–Litterman model addresses these limitations by combining:

- **Market equilibrium returns** derived from Capital Asset Pricing Model (CAPM)
- **Investor views** expressed with uncertainty

The posterior expected returns are given by:

\[
\mu_{BL} =
\left[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
\left[(\tau \Sigma)^{-1} \pi + P^\top \Omega^{-1} q\right]
\]

where:
- \( \pi \) are implied equilibrium returns
- \( P \) encodes the structure of investor views
- \( q \) represents expected excess returns
- \( \Omega \) captures confidence in those views
- \( \tau \) scales prior uncertainty

This Bayesian formulation yields smoother and more intuitive portfolio allocations.

---

### 3. Return Prediction via Multiple Linear Regression

Expected asset returns are estimated using **macroeconomic variables** such as:
- Inflation
- GDP growth
- Interest rates
- Yield curve indicators
- Other relevant macro factors

The regression model is:

\[
R_{i,t} = \alpha_i + \beta_i^\top X_t + \varepsilon_{i,t}
\]

where:
- \( X_t \) is the vector of macroeconomic factors
- Regression forecasts are used as **views** in the Black–Litterman framework

---

## Methodology

1. **Data Collection**
   - Asset prices and returns
   - Market capitalization weights
   - Macroeconomic indicators (inflation, GDP, interest rates, etc.)

2. **Return Prediction**
   - Estimate multiple linear regression models
   - Analyze economic interpretability of coefficients
   - Generate expected return views

3. **Black–Litterman Construction**
   - Compute implied equilibrium returns
   - Encode regression outputs as investor views
   - Calibrate uncertainty through the \( \Omega \) matrix

4. **Portfolio Optimization**
   - Solve mean–variance optimization problems
---

## Technologies Used

- **Python**
- NumPy
- pandas
- scikit-learn
- SciPy
- PyPortfolioOpt
- matplotlib / seaborn
- Jupyter Notebook

---

## How to Run

```bash
git clone https://github.com/yourusername/yourrepo.git
cd yourrepo
pip install -r requirements.txt
jupyter notebook
```
## 📚 References

- Black, F; Litterman, R. Combining investor views with market equilibrium. The Journal of Fixed Income, 1991.
- Idzorek T. A step-by-step guide to the Black-Litterman model: Incorporating user-specified confidence levels. In: Forecasting Expected Returns in the Financial Markets. Elsevier Ltd; 2007. p. 17–38.

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice.
