"""Small charts with consistent labels; plotting never changes the calculations."""

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def plot_prices(prices):
    ax = (100 * prices / prices.iloc[0]).plot(figsize=(11, 4), linewidth=1.5)
    ax.set(title="Adjusted prices — 100 at the start of the sample", xlabel="", ylabel="Index (start = 100)")
    ax.grid(alpha=0.2)
    return ax.figure


def plot_results(result):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    (100 * result["wealth"]).plot(ax=axes[0], linewidth=1.8)
    axes[0].set(title="Growth of 100 invested — after costs", ylabel="Portfolio value", xlabel="")
    result["drawdown"].plot(ax=axes[1], legend=False, linewidth=1.4)
    axes[1].set(title="Drawdown from the previous peak — monthly observations", ylabel="Drawdown", xlabel="")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


def plot_weights(result, strategy="BL Momentum"):
    weights = result["weights"].xs(strategy)
    ax = weights.plot.area(figsize=(11, 4), linewidth=0, alpha=0.85)
    ax.set(title=f"Portfolio weights at each rebalance — {strategy}", ylabel="Weight", xlabel="", ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.figure.tight_layout()
    return ax.figure


def plot_risk_return(result):
    metrics = result["metrics"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, row in metrics.iterrows():
        ax.scatter(row["Volatility"], row["CAGR"], s=70)
        ax.annotate(name, (row["Volatility"], row["CAGR"]), xytext=(5, 6), textcoords="offset points")
    ax.set(title="Realized return and risk — after costs", xlabel="Annualized volatility", ylabel="Annualized return (CAGR)")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.margins(0.25)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig
