import ast
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scheduled_scan import analyze_execution_candidates
from stock2dupe import ScanPreferences, ranking_test_scored


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


def event(ticker, adjustment=0):
    return SimpleNamespace(
        adjustment=adjustment,
        confidence="medium",
        label="supportive" if adjustment > 0 else "neutral",
        summary=f"Event context for {ticker}",
        headlines_used=[],
        available=True,
    )


def review(ticker):
    return SimpleNamespace(
        verdict="good",
        confidence="medium",
        summary=f"Candidate context for {ticker}",
        strengths=[],
        risks=[],
        action="Review the quantitative setup.",
        available=True,
    )


class ScheduledPostSelectionTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            candidate("SPY", 92, 600, 605),
            candidate("SPY", 90, 601, 606),
            candidate("NVDA", 88, 180, 185),
        ]
        self.preferences = ScanPreferences(500, "neutral", "moderate")
        self.price_moves = {"SPY": {}, "NVDA": {}}

    def test_unique_top_tickers_and_each_candidate_are_analyzed_once(self):
        event_calls = []
        review_calls = []

        with (
            patch(
                "scheduled_scan.get_deep_event_analysis",
                side_effect=lambda ticker, outlook: (
                    event_calls.append(ticker) or event(ticker, 8)
                ),
            ),
            patch(
                "scheduled_scan.analyze_candidate_setup",
                side_effect=lambda scored, ticker_event, move: (
                    review_calls.append(scored.trade.ticker)
                    or review(scored.trade.ticker)
                ),
            ),
        ):
            result = analyze_execution_candidates(
                self.candidates, self.preferences, self.price_moves
            )

        self.assertEqual(["SPY", "NVDA"], event_calls)
        self.assertEqual(["SPY", "SPY", "NVDA"], review_calls)
        self.assertEqual(2, result.diagnostics["ticker_event_calls"])
        self.assertEqual(3, result.diagnostics["candidate_review_calls"])

    def test_event_results_do_not_replace_or_rescore_candidates(self):
        with (
            patch(
                "scheduled_scan.get_deep_event_analysis",
                side_effect=lambda ticker, outlook: event(ticker, 10),
            ),
            patch(
                "scheduled_scan.analyze_candidate_setup",
                side_effect=lambda scored, ticker_event, move: review(
                    scored.trade.ticker
                ),
            ),
        ):
            result = analyze_execution_candidates(
                self.candidates, self.preferences, self.price_moves
            )

        self.assertEqual(self.candidates, result.execution_candidates)
        self.assertTrue(
            all(
                original is returned
                for original, returned in zip(
                    self.candidates, result.execution_candidates
                )
            )
        )
        self.assertTrue(
            all(candidate.event_adjustment == 0 for candidate in self.candidates)
        )

    def test_scheduled_main_selects_before_analysis_and_never_calls_basic_events(self):
        source = Path("scheduled_scan.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        main_function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "main"
        )
        called_names = {
            node.func.id
            for node in ast.walk(main_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("get_event_analysis", called_names)
        self.assertNotIn("get_deep_event_analysis", called_names)
        self.assertLess(
            source.index(
                "execution_candidates = select_execution_candidates(scored_trades, limit=3)"
            ),
            source.index("post_selection_analysis = analyze_execution_candidates("),
        )


if __name__ == "__main__":
    unittest.main()
