import os
import unittest
from dataclasses import replace
from datetime import date


os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SECRET_KEY", "test-key")

from history_tracker import expiration_failure_values, expiration_result_values
from paper_exit import evaluate_paper_exit, take_profit_target_per_share
from scanner_tracking import (
    new_scan_run_id,
    normalize_history_row,
    setup_key_for_trade,
)
from stock2dupe import (
    ScanPreferences,
    apply_price_move_mode,
    diversify_scored_trades,
    ranking_test_scored,
    ranking_test_trade,
    score_trade,
    select_execution_candidates,
    strategy_direction,
)


class ExecutionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.displayed = diversify_scored_trades(
            [
                ranking_test_scored("SPY", "bull call debit spread", 91, 605, 600),
                ranking_test_scored(
                    "SPY", "bear put debit spread", 89, 590, 595, option_type="put"
                ),
                ranking_test_scored("NVDA", "bull call debit spread", 88, 210, 205),
                ranking_test_scored(
                    "QQQ", "put credit spread", 84, 480, 475, option_type="put"
                ),
            ]
        )

    def test_full_ranking_keeps_several_strategies_for_ticker(self):
        self.assertEqual(2, sum(item.trade.ticker == "SPY" for item in self.displayed))

    def test_execution_candidates_have_no_repeated_ticker(self):
        selected = select_execution_candidates(self.displayed)
        self.assertEqual(len(selected), len({item.trade.ticker for item in selected}))

    def test_execution_returns_fewer_when_distinct_tickers_missing(self):
        selected = select_execution_candidates(self.displayed[:2], limit=3)
        self.assertEqual(1, len(selected))

    def test_highest_ranked_survives_ticker_cap(self):
        selected = select_execution_candidates(self.displayed)
        self.assertEqual(91, selected[0].total_score)
        self.assertEqual("bull call debit spread", selected[0].trade.strategy)

    def test_stable_score_tie_breaking(self):
        tied = [
            replace(
                ranking_test_scored("AAPL", "bull call debit spread", 80, 105, 100),
                normalized_ticker_score=80,
                quant_score=75,
            ),
            replace(
                ranking_test_scored("MSFT", "bull call debit spread", 80, 505, 500),
                normalized_ticker_score=90,
                quant_score=70,
            ),
            replace(
                ranking_test_scored("NVDA", "bull call debit spread", 80, 205, 200),
                normalized_ticker_score=90,
                quant_score=78,
            ),
        ]
        self.assertEqual(
            ["NVDA", "MSFT", "AAPL"],
            [item.trade.ticker for item in select_execution_candidates(tied)],
        )

    def test_directional_strategies(self):
        self.assertEqual("bullish", strategy_direction("put credit spread"))
        self.assertEqual("bullish", strategy_direction("bull call debit spread"))
        self.assertEqual("bearish", strategy_direction("call credit spread"))
        self.assertEqual("bearish", strategy_direction("bear put debit spread"))

    def test_iron_condor_is_neutral(self):
        self.assertEqual("neutral", strategy_direction("iron condor"))


class SetupHistoryTests(unittest.TestCase):
    def setUp(self):
        self.trade = ranking_test_trade("SPY", "bull call debit spread", 605, 600)

    def test_setup_key_is_deterministic(self):
        self.assertEqual(setup_key_for_trade(self.trade), setup_key_for_trade(self.trade))

    def test_different_strikes_change_setup_key(self):
        changed = replace(self.trade, short_strike=606)
        self.assertNotEqual(setup_key_for_trade(self.trade), setup_key_for_trade(changed))

    def test_recommendations_share_setup_but_not_run_id(self):
        self.assertEqual(setup_key_for_trade(self.trade), setup_key_for_trade(self.trade))
        self.assertNotEqual(new_scan_run_id(), new_scan_run_id())

    def test_old_rows_remain_readable(self):
        normalized = normalize_history_row(
            {
                "id": 12,
                "scan_time": "2026-07-01T12:00:00+00:00",
                "ticker": "SPY",
                "strategy": "bull call debit spread",
                "expiration": "2026-07-31",
                "long_strike": 600,
                "short_strike": 605,
            }
        )
        self.assertTrue(normalized["scan_run_id"].startswith("legacy-run-"))
        self.assertTrue(normalized["setup_key"])
        self.assertEqual(1, normalized["times_recommended"])


class PaperExitTests(unittest.TestCase):
    def test_tp50_debit(self):
        self.assertEqual(3.33, take_profit_target_per_share("debit", 1.66, 3.34, "tp50"))

    def test_tp50_credit(self):
        self.assertEqual(0.60, take_profit_target_per_share("credit", 1.20, 1.20, "tp50"))

    def test_close_order_is_not_submitted_twice(self):
        decision = evaluate_paper_exit(
            entry_type="debit",
            entry_price_per_share=1.66,
            max_profit_per_share=3.34,
            current_value_per_share=3.40,
            policy="tp50",
            close_order_status="accepted",
        )
        self.assertFalse(decision.should_close)


class PriceMoveModeTests(unittest.TestCase):
    def test_all_price_move_modes(self):
        self.assertEqual((8, 8), apply_price_move_mode(8, "Full"))
        self.assertEqual((8, 3), apply_price_move_mode(8, "Conservative"))
        self.assertEqual((-6, -6), apply_price_move_mode(-6, "Conservative"))
        self.assertEqual((8, 0), apply_price_move_mode(8, "Shadow"))
        self.assertEqual((8, 0), apply_price_move_mode(8, "Off"))

    def test_raw_and_effective_adjustments_are_separate(self):
        trade = ranking_test_trade("AAPL", "bull call debit spread", 105, 100)
        scored = score_trade(
            trade,
            ScanPreferences(500, "bullish", "moderate", price_move_mode="Shadow"),
            price_move={"1D Move %": 4, "5D Move %": 5, "Move vs 20D Vol": 2},
        )
        self.assertGreater(scored.raw_price_move_adjustment, 0)
        self.assertEqual(0, scored.effective_price_move_adjustment)
        self.assertEqual(scored.base_score_without_price_move, scored.total_score)


class ExpirationTrackingTests(unittest.TestCase):
    def test_expired_position_is_marked_closed(self):
        values = expiration_result_values(
            {
                "entry_type": "debit",
                "entry_price": 200,
                "max_risk": 200,
                "scan_time": "2026-07-01T12:00:00+00:00",
            },
            date(2026, 7, 31),
            150,
            100,
        )
        self.assertEqual("expired", values["expiration_status"])
        self.assertEqual("closed", values["starting_status"])
        self.assertEqual("expiration", values["exit_reason"])

    def test_failed_expiration_is_visible_and_retryable(self):
        values = expiration_failure_values("closing data unavailable")
        self.assertEqual("expiration_update_failed", values["expiration_status"])
        self.assertTrue(values["update_retryable"])
        self.assertEqual("closing data unavailable", values["last_update_error"])


if __name__ == "__main__":
    unittest.main()
