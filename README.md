# Quantitative Portfolio Optimization 
## Macro-Factor Return Forecasting and Portfolio Optimization using the Black-Litterman Model

## Overview

This repository explores **quantitative portfolio construction** by combining:

- **Return prediction using Multiple Linear Regression**
- **Bayesian portfolio allocation using the Black–Litterman model**
- **Mean–variance optimization within the Markowitz framework**

The project aims to illustrate how **macroeconomic-driven return forecasts** can be integrated into **modern portfolio theory** to obtain more stable and economically interpretable portfolio allocations.

---

## Objectives

- Predict asset returns using **macroeconomic factors**
- Incorporate regression-based views into portfolio allocation via **Black–Litterman**
- Compare classical mean–variance optimization with Bayesian allocation
- Highlight the role of **confidence in investor views**

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
   - Calibrate uncertainty through the Ω matrix

4. **Portfolio Optimization**
   - Solve mean–variance optimization problems
---

## Technologies Used

- **Python**
- NumPy
- pandas
- scikit-learn
- SciPy
- yfinance 
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
## References

- Black, F; Litterman, R. Combining investor views with market equilibrium. The Journal of Fixed Income, 1991.
- Idzorek T. A step-by-step guide to the Black-Litterman model: Incorporating user-specified confidence levels. In: Forecasting Expected Returns in the Financial Markets. Elsevier Ltd; 2007. p. 17–38.

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice.

## License

This project is licensed under the Apache License - see the LICENSE file for details.
