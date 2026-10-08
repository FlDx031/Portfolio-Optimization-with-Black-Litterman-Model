"""Offline tests of accounting and chronology, using explicitly synthetic data."""

import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch, Mock

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from src.backtest import performance, run_backtest, trading_cost
from src.data import fred_snapshot, load_macro_features, load_prices, monthly_schedule, validate_prices
from src.model import black_litterman, momentum_views, optimal_weights, reference_prior, ridge_predict, ridge_views


def synthetic_prices():
    """Deterministic test fixture, never presented as market performance."""
    rng = np.random.default_rng(42)
    dates = mcal.get_calendar("NYSE").valid_days("2012-01-02", "2023-02-28").tz_localize(None)
    returns = rng.normal([0.0003, 0.0001, 0.0002], [0.01, 0.005, 0.009], (len(dates), 3))
    return pd.DataFrame(100 * np.exp(returns.cumsum(axis=0)), index=dates, columns=["A", "B", "SPY"])


class ProjectTests(unittest.TestCase):
    def test_notebook_cells_compile(self):
        notebook = json.loads((Path(__file__).resolve().parents[1] / "Portfolio_BlackLitterman.ipynb").read_text(encoding="utf-8"))
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), f"notebook cell {i}", "exec")

    def test_notebook_main_mode_requires_macro_key(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "Portfolio_BlackLitterman.ipynb").read_text(encoding="utf-8"))
        configuration = "".join(notebook["cells"][3]["source"])
        with patch.dict(os.environ, {"RUN_MODE": "macro"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "requires FRED_API_KEY"):
                exec(configuration, {"ROOT": root, "os": os})

    def test_first_loss_and_arithmetic_sharpe(self):
        result = performance([-0.1, 0.0], rf_annual=0)
        self.assertAlmostEqual(result["Max drawdown"], -0.1)
        self.assertAlmostEqual(result["Total return"], -0.1)
        self.assertAlmostEqual(result["Sharpe"], np.mean([-0.1, 0]) / np.std([-0.1, 0], ddof=1) * np.sqrt(12))

    def test_costs_include_entry_and_both_sides(self):
        self.assertEqual(trading_cost([0.5, 0.5], [0, 0], 0.001), (1.0, 0.001))
        turnover, fee = trading_cost([0, 1], [1, 0], 0.001)
        self.assertEqual((turnover, fee), (2.0, 0.002))

    def test_prior_recovers_reference_weights(self):
        sigma = np.array([[0.003, 0.0005], [0.0005, 0.001]])
        prior = reference_prior(sigma)
        np.testing.assert_allclose(optimal_weights(prior, sigma), [0.5, 0.5], atol=1e-6)

    def test_black_litterman_confidence_limits(self):
        sigma = np.diag([0.01, 0.02])
        prior, q = np.array([0.01, 0.02]), np.array([0.03, -0.01])
        confident, cov = black_litterman(prior, sigma, q, np.eye(2) * 1e-12)
        uncertain, _ = black_litterman(prior, sigma, q, np.eye(2) * 1e9)
        np.testing.assert_allclose(confident, q, atol=1e-8)
        np.testing.assert_allclose(uncertain, prior, atol=1e-8)
        self.assertGreater(np.linalg.eigvalsh(cov).min(), 0)

    def test_identical_momentum_is_finite(self):
        prices = np.ones((13, 2)) * 100
        q, omega = momentum_views(prices, np.array([0.01, 0.02]), np.eye(2))
        np.testing.assert_allclose(q, [0.01, 0.02])
        self.assertTrue(np.isfinite(omega).all())

    def test_ridge_validation_uses_prior_rows(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 4))
        y = rng.normal(0, 0.03, (40, 2))
        q, omega = ridge_views(x, y, x[-1], 0.002)
        errors = [y[t] - ridge_predict(x[:t], y[:t], x[t]) for t in range(28, 40)]
        np.testing.assert_allclose(np.diag(omega), np.mean(np.square(errors), axis=0))
        np.testing.assert_allclose(q, ridge_predict(x, y, x[-1]) - 0.002)

    def test_manual_accounting_and_timing(self):
        prices = synthetic_prices()
        result = run_backtest(prices, ["A", "B"])
        row = result["timeline"].iloc[0]
        gross = (prices.loc[row.exit, ["A", "B"]] / prices.loc[row.execution, ["A", "B"]] - 1).mean()
        self.assertAlmostEqual(result["returns"]["Equal weight"].iloc[0], 0.999 * (1 + gross) - 1)
        drifted = 0.5 * prices.loc[row.exit, ["A", "B"]] / prices.loc[row.execution, ["A", "B"]] / (1 + gross)
        expected_turnover = np.abs(0.5 - drifted).sum()
        self.assertAlmostEqual(result["turnover"]["Equal weight"].iloc[1], expected_turnover)
        self.assertTrue((result["timeline"].signal < result["timeline"].execution).all())
        self.assertTrue((result["timeline"].last_known_target_end <= result["timeline"].signal).all())
        np.testing.assert_allclose(result["weights"].sum(axis=1), 1, atol=1e-8)
        self.assertTrue((result["weights"] >= 0).all().all())
        self.assertEqual(result["turnover"]["SPY"].sum(), 1.0)

    def test_cost_scenarios_share_weights_and_dates(self):
        prices = synthetic_prices()
        free = run_backtest(prices, ["A", "B"], cost_bps=0)
        costly = run_backtest(prices, ["A", "B"], cost_bps=25)
        pd.testing.assert_frame_equal(free["weights"], costly["weights"])
        self.assertTrue((costly["wealth"].iloc[-1] < free["wealth"].iloc[-1]).all())

    def test_incomplete_final_month_is_only_an_exit(self):
        prices = synthetic_prices().loc[:"2023-02-10"]
        result = run_backtest(prices, ["A", "B"])
        self.assertEqual(result["timeline"].exit.iloc[-1], pd.Timestamp("2023-02-01"))
        self.assertEqual(result["timeline"].execution.iloc[-1], pd.Timestamp("2023-01-03"))

    def test_future_data_do_not_change_past_weights(self):
        prices = synthetic_prices()
        schedule = monthly_schedule(prices)
        rng = np.random.default_rng(12)
        macro = pd.DataFrame(rng.normal(size=(len(schedule), 4)), index=pd.DatetimeIndex(schedule.signal))
        original = run_backtest(prices, ["A", "B"], macro=macro)
        cutoff = pd.Timestamp("2020-01-01")
        changed_prices, changed_macro = prices.copy(), macro.copy()
        changed_prices.loc[changed_prices.index >= cutoff, "A"] *= 3
        changed_macro.loc[changed_macro.index >= cutoff] += 100
        changed = run_backtest(changed_prices, ["A", "B"], macro=changed_macro)
        mask = original["weights"].index.get_level_values("date") < cutoff
        pd.testing.assert_frame_equal(original["weights"].loc[mask], changed["weights"].loc[mask])

    def test_missing_prices_fail(self):
        prices = synthetic_prices()
        prices.iloc[10, 0] = np.nan
        with self.assertRaises(ValueError):
            validate_prices(prices)

    def test_missing_nyse_session_fails(self):
        prices = synthetic_prices()
        missing = prices.index[20]
        with self.assertRaisesRegex(ValueError, str(missing.date())):
            monthly_schedule(prices.drop(index=missing))
        with self.assertRaisesRegex(ValueError, str(missing.date())):
            run_backtest(prices.drop(index=missing), ["A", "B"])

    @patch("src.data.yf.download")
    def test_truncated_download_fails(self, download):
        dates = mcal.get_calendar("NYSE").valid_days("2020-01-02", "2020-01-06").tz_localize(None)
        download.return_value = pd.DataFrame(
            [[1.0, 2.0], [1.1, 2.1]], index=dates[:-1],
            columns=pd.MultiIndex.from_tuples([("Close", "A"), ("Close", "B")]))
        with tempfile.TemporaryDirectory() as cache:
            with self.assertRaisesRegex(ValueError, str(dates[-1].date())):
                load_prices(["A", "B"], "2020-01-02", "2020-01-06", cache)

    @patch("src.data.time.sleep")
    @patch("src.data.requests.get")
    def test_vintage_api_and_cache(self, get, sleep):
        get.return_value = Mock(status_code=200)
        get.return_value.json.return_value = {"observations": [
            {"date": "2019-11-01", "value": "100"}, {"date": "2019-12-01", "value": "101"}]}
        with tempfile.TemporaryDirectory() as cache:
            first = fred_snapshot("CPIAUCSL", "2020-01-30", "test-key", cache)
            second = fred_snapshot("CPIAUCSL", "2020-01-30", "", cache)
        self.assertEqual(get.call_count, 1)
        params = get.call_args.kwargs["params"]
        self.assertEqual(params["realtime_start"], "2020-01-30")
        self.assertEqual(params["realtime_end"], "2020-01-30")
        np.testing.assert_array_equal(first, second)

    @patch("src.data.fred_snapshot")
    def test_macro_excludes_signal_day_and_current_month(self, snapshot):
        snapshot.return_value = pd.Series([100.0, 110.0, 999.0],
                                          index=pd.to_datetime(["2019-11-01", "2019-12-01", "2020-01-01"]))
        features = load_macro_features(pd.to_datetime(["2020-01-31"]), "test-key")
        self.assertAlmostEqual(features.inflation.iloc[0], 0.1)
        self.assertEqual(features.yield_change.iloc[0], 10.0)
        self.assertEqual(snapshot.call_args.args[1], pd.Timestamp("2020-01-30"))

    @patch("src.data.yf.download")
    def test_price_order_adjustment_and_cache(self, download):
        dates = mcal.get_calendar("NYSE").valid_days("2020-01-02", "2020-01-06").tz_localize(None)
        download.return_value = pd.DataFrame([[1.1234567890123457, 2], [2, 3.2345678901234567], [3, 4]], index=dates,
                                             columns=pd.MultiIndex.from_tuples([("Close", "B"), ("Close", "A")]))
        with tempfile.TemporaryDirectory() as cache:
            prices = load_prices(["A", "B"], "2020-01-02", "2020-01-06", cache)
            again = load_prices(["A", "B"], "2020-01-02", "2020-01-06", cache)
        self.assertEqual(list(prices.columns), ["A", "B"])
        self.assertTrue(download.call_args.kwargs["auto_adjust"])
        self.assertEqual(download.call_count, 1)
        np.testing.assert_array_equal(prices, again)


if __name__ == "__main__":
    unittest.main()
