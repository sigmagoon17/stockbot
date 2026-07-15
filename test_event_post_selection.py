import ast
import os
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from alpaca_client import submit_scored_multileg_orders
from event_analysis import client, configured_openai_timeout_seconds
from streamlit.testing.v1 import AppTest
from scanner_post_selection import (
    BEAR_PUT_DEBIT_SPREADS,
    BULL_CALL_DEBIT_SPREADS,
    CALL_CREDIT_SPREADS,
    IRON_CONDORS,
    PUT_CREDIT_SPREADS,
    STRATEGY_BUILD_ORDER,
    STRATEGY_OPTIONS,
    analyze_top_candidates,
    run_selected_strategy_builders,
)
from stock2dupe import ranking_test_scored


def candidate(ticker, score, long_strike, short_strike):
    scored = ranking_test_scored(
        ticker,
        "bull call debit spread",
        score,
        short_strike,
        long_strike,
        expiration="2026-08-21",
    )
    return replace(
        scored,
        trade=replace(
            scored.trade,
            entry_type="debit",
            max_risk=1.5,
            max_profit=3.5,
        ),
    )


def event_analysis(ticker, adjustment=0, available=True):
    return SimpleNamespace(
        ticker=ticker,
        adjustment=adjustment,
        confidence="high" if available else "low",
        label="supportive" if adjustment > 0 else "neutral",
        summary=f"Event context for {ticker}",
        headlines_used=[],
        available=available,
    )


def candidate_review(ticker, available=True):
    return SimpleNamespace(
        verdict="good" if available else "watch",
        confidence="high" if available else "low",
        summary=f"Candidate context for {ticker}",
        strengths=[],
        risks=[],
        action="Review the quantitative setup.",
        available=available,
    )


class PostSelectionEventTests(unittest.TestCase):
    def setUp(self):
        self.execution_candidates = [
            candidate("SPY", 92, 600, 605),
            candidate("SPY", 90, 601, 606),
            candidate("NVDA", 88, 180, 185),
        ]

    def run_analysis(self, event_loader=None, reviewer=None):
        return analyze_top_candidates(
            self.execution_candidates,
            {"SPY": {"1D Move %": 1.0}, "NVDA": {"1D Move %": -2.0}},
            load_ticker_event=event_loader
            or (lambda ticker: (event_analysis(ticker, 7), "miss")),
            review_candidate=reviewer
            or (
                lambda scored, ticker_event, price_move: (
                    candidate_review(scored.trade.ticker),
                    "miss",
                )
            ),
            candidate_key=lambda scored: (
                scored.trade.ticker,
                scored.trade.long_strike,
            ),
            unavailable_event=lambda ticker, error: event_analysis(
                ticker, available=False
            ),
            unavailable_review=lambda scored, error: candidate_review(
                scored.trade.ticker, available=False
            ),
        )

    def test_ten_watchlist_tickers_only_analyze_two_top_candidate_tickers(self):
        watchlist = [
            "SPY",
            "NVDA",
            "AAPL",
            "MSFT",
            "QQQ",
            "META",
            "AMD",
            "TSLA",
            "AMZN",
            "GOOGL",
        ]
        calls = []
        result = self.run_analysis(
            event_loader=lambda ticker: (
                calls.append(ticker) or event_analysis(ticker),
                "miss",
            )
        )
        self.assertEqual(10, len(watchlist))
        self.assertEqual(["SPY", "NVDA"], calls)
        self.assertEqual(2, result.diagnostics["ticker_event_calls"])

    def test_exactly_three_candidate_reviews_run(self):
        calls = []

        def reviewer(scored, ticker_event, price_move):
            calls.append((scored.trade.ticker, ticker_event.ticker, price_move))
            return candidate_review(scored.trade.ticker), "miss"

        result = self.run_analysis(reviewer=reviewer)
        self.assertEqual(3, len(calls))
        self.assertEqual(3, result.diagnostics["candidate_review_calls"])

    def test_event_results_do_not_change_selected_candidate_identity(self):
        before = list(self.execution_candidates)
        result = self.run_analysis()
        self.assertEqual(before, result.execution_candidates)
        self.assertTrue(
            all(
                original is returned
                for original, returned in zip(
                    self.execution_candidates, result.execution_candidates
                )
            )
        )
        self.assertTrue(
            all(scored.event_adjustment == 0 for scored in result.execution_candidates)
        )

    def test_event_timeout_returns_unavailable_and_keeps_candidates(self):
        def timeout(ticker):
            raise TimeoutError(f"timed out for {ticker}")

        result = self.run_analysis(event_loader=timeout)
        self.assertEqual(self.execution_candidates, result.execution_candidates)
        self.assertEqual(3, result.diagnostics["candidate_review_calls"])
        self.assertTrue(
            all(not analysis.available for analysis in result.event_analyses.values())
        )

    def test_unavailable_event_analysis_does_not_block_paper_preflight(self):
        result = self.run_analysis(
            event_loader=lambda ticker: (
                event_analysis(ticker, available=False),
                "miss",
            )
        )
        submitted = []

        def submit(legs, quantity, limit_price, client_order_id=None):
            submitted.append(legs)
            return {"id": "paper-order", "status": "accepted"}, []

        with (
            patch("alpaca_client.get_alpaca_positions", return_value=([], [])),
            patch("alpaca_client.get_open_alpaca_orders", return_value=([], [])),
            patch("alpaca_client.get_recent_alpaca_orders", return_value=([], [])),
            patch("alpaca_client.submit_multileg_order", side_effect=submit),
        ):
            paper_results = submit_scored_multileg_orders(
                result.execution_candidates[:1],
                quantity=1,
                limit=1,
                scan_run_id="offline-event-test",
            )
        self.assertEqual(1, len(submitted))
        self.assertEqual("accepted", paper_results[0]["Status"].lower())

    def test_scan_watchlist_contains_no_event_analysis_call(self):
        source = Path("app.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        scan_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "scan_watchlist"
        )
        called_names = {
            node.func.id
            for node in ast.walk(scan_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("get_event_analysis", called_names)
        self.assertNotIn("get_deep_event_analysis", called_names)
        self.assertNotIn("get_cached_deep_event_analysis", called_names)
        self.assertNotIn("analyze_candidate_setup", called_names)
        self.assertNotIn("select_execution_candidates", called_names)
        self.assertLess(
            source.index(
                "execution_candidates = select_execution_candidates(scored_trades, limit=3)"
            ),
            source.index(
                "post_selection_analysis = analyze_execution_candidates("
            ),
        )


class StrategySelectionTests(unittest.TestCase):
    def builders(self, calls):
        return {
            strategy: lambda strategy=strategy: calls.append(strategy)
            or [f"{strategy} output"]
            for strategy in STRATEGY_OPTIONS
        }

    def test_all_five_strategies_preserve_existing_builder_output_order(self):
        calls = []
        results = run_selected_strategy_builders(
            STRATEGY_OPTIONS,
            self.builders(calls),
        )
        self.assertEqual(list(STRATEGY_BUILD_ORDER), calls)
        self.assertEqual(list(STRATEGY_BUILD_ORDER), list(results))
        self.assertEqual(
            [f"{strategy} output" for strategy in STRATEGY_BUILD_ORDER],
            [item for rows in results.values() for item in rows],
        )

    def test_each_stable_strategy_id_calls_only_its_builder(self):
        for strategy_id in STRATEGY_OPTIONS:
            with self.subTest(strategy_id=strategy_id):
                calls = []
                results = run_selected_strategy_builders(
                    [strategy_id],
                    self.builders(calls),
                )
                self.assertEqual([strategy_id], calls)
                self.assertEqual([strategy_id], list(results))

    def test_empty_selection_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "at least one strategy"):
            run_selected_strategy_builders([], self.builders([]))

    def test_unknown_strategy_id_fails_clearly(self):
        unknown = "calendar_spread"
        with self.assertRaisesRegex(ValueError, unknown):
            run_selected_strategy_builders([unknown], self.builders([]))

    def test_disabled_strategy_builder_is_not_called(self):
        calls = []
        selected = [
            PUT_CREDIT_SPREADS,
            CALL_CREDIT_SPREADS,
            BULL_CALL_DEBIT_SPREADS,
            BEAR_PUT_DEBIT_SPREADS,
        ]
        results = run_selected_strategy_builders(selected, self.builders(calls))
        self.assertNotIn(IRON_CONDORS, calls)
        self.assertNotIn(IRON_CONDORS, results)


class StrategySelectionUIBoundaryTests(unittest.TestCase):
    def test_empty_ui_selection_stops_before_every_side_effect(self):
        with (
            patch("history_tracker.update_expired_history") as update_history,
            patch("stock_universe.prefilter_tickers") as prefilter,
            patch("stock2dupe.get_option_chain_result") as option_chain,
            patch("event_analysis.get_deep_event_analysis") as event_analysis,
            patch("event_analysis.analyze_candidate_setup") as candidate_review,
            patch("alpaca_client.get_alpaca_positions") as alpaca_positions,
        ):
            app = AppTest.from_file("app.py", default_timeout=30)
            app.run()
            next(
                widget
                for widget in app.multiselect
                if widget.label == "Strategies"
            ).set_value([])
            next(
                widget
                for widget in app.checkbox
                if widget.label == "Prefilter before options scan"
            ).set_value(True)
            next(
                widget
                for widget in app.button
                if widget.label == "Scan Watchlist"
            ).click()
            app.run()

        self.assertTrue(
            any("Select at least one strategy." in error.value for error in app.error)
        )
        update_history.assert_not_called()
        prefilter.assert_not_called()
        option_chain.assert_not_called()
        event_analysis.assert_not_called()
        candidate_review.assert_not_called()
        alpaca_positions.assert_not_called()


class OpenAITimeoutConfigurationTests(unittest.TestCase):
    def test_timeout_defaults_to_twenty_seconds(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(20.0, configured_openai_timeout_seconds())

    def test_timeout_is_configurable_and_invalid_values_are_safe(self):
        self.assertEqual(7.5, configured_openai_timeout_seconds("7.5"))
        self.assertEqual(20.0, configured_openai_timeout_seconds("invalid"))
        self.assertEqual(20.0, configured_openai_timeout_seconds("0"))

    def test_openai_client_has_no_automatic_retries(self):
        self.assertEqual(0, client.max_retries)


if __name__ == "__main__":
    unittest.main()
