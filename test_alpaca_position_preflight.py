import unittest
from dataclasses import replace
from unittest.mock import patch

from alpaca_client import (
    option_symbol,
    submit_scored_multileg_orders,
    trade_multileg_order_details,
)
from stock2dupe import ranking_test_scored


def debit_candidate(
    ticker="SPY",
    score=90,
    long_strike=600,
    short_strike=605,
):
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


def condor_candidate(ticker="SPY", score=88):
    scored = ranking_test_scored(
        ticker,
        "iron condor",
        score,
        610,
        590,
        expiration="2026-08-21",
        option_type="mixed",
    )
    return replace(
        scored,
        trade=replace(
            scored.trade,
            option_type="mixed",
            put_long_strike=580,
            put_short_strike=585,
            call_short_strike=615,
            call_long_strike=620,
        ),
    )


class AlpacaPositionPreflightTests(unittest.TestCase):
    def run_batch(self, candidates, positions=None, position_errors=None):
        submitted = []

        def submit(legs, quantity, limit_price, client_order_id=None):
            submitted.append(legs)
            return (
                {
                    "id": f"order-{len(submitted)}",
                    "client_order_id": client_order_id,
                    "status": "accepted",
                    "order_class": "mleg",
                },
                [],
            )

        with (
            patch(
                "alpaca_client.get_alpaca_positions",
                return_value=(positions or [], position_errors or []),
            ) as get_positions,
            patch("alpaca_client.get_recent_alpaca_orders", return_value=([], [])),
            patch("alpaca_client.submit_multileg_order", side_effect=submit),
        ):
            results = submit_scored_multileg_orders(
                candidates,
                quantity=1,
                limit=len(candidates),
                scan_run_id="offline-test-run",
            )
        get_positions.assert_called_once_with()
        return results, submitted

    def test_no_live_positions_permits_submission(self):
        results, submitted = self.run_batch([debit_candidate()])
        self.assertEqual(1, len(submitted))
        self.assertEqual("accepted", results[0]["Status"])

    def test_live_long_overlapping_sell_to_open_skips_candidate(self):
        candidate = debit_candidate()
        short_symbol = option_symbol("SPY", "2026-08-21", "call", 605)
        results, submitted = self.run_batch(
            [candidate], [{"symbol": short_symbol, "qty": "1"}]
        )
        self.assertEqual([], submitted)
        self.assertEqual("Skipped", results[0]["Status"])
        self.assertIn(short_symbol, results[0]["Message"])
        self.assertIn("qty 1", results[0]["Message"])
        self.assertIn("sell_to_close", results[0]["Message"])

    def test_live_short_overlapping_buy_to_open_skips_candidate(self):
        candidate = debit_candidate()
        long_symbol = option_symbol("SPY", "2026-08-21", "call", 600)
        results, submitted = self.run_batch(
            [candidate], [{"symbol": long_symbol, "qty": "-2"}]
        )
        self.assertEqual([], submitted)
        self.assertEqual("Skipped", results[0]["Status"])
        self.assertIn("qty -2", results[0]["Message"])
        self.assertIn("buy_to_close", results[0]["Message"])

    def test_overlap_on_either_spread_leg_skips_entire_spread(self):
        candidate = debit_candidate()
        for strike in (600, 605):
            with self.subTest(strike=strike):
                symbol = option_symbol("SPY", "2026-08-21", "call", strike)
                results, submitted = self.run_batch(
                    [candidate], [{"symbol": symbol, "qty": "1"}]
                )
                self.assertEqual([], submitted)
                self.assertEqual("Skipped", results[0]["Status"])

    def test_overlap_on_any_condor_leg_skips_entire_condor(self):
        candidate = condor_candidate()
        legs, _, _ = trade_multileg_order_details(candidate)
        for leg in legs:
            with self.subTest(symbol=leg["symbol"]):
                results, submitted = self.run_batch(
                    [candidate], [{"symbol": leg["symbol"], "qty": "1"}]
                )
                self.assertEqual([], submitted)
                self.assertEqual("Skipped", results[0]["Status"])

    def test_unrelated_position_does_not_block_submission(self):
        unrelated = option_symbol("QQQ", "2026-08-21", "put", 500)
        results, submitted = self.run_batch(
            [debit_candidate()], [{"symbol": unrelated, "qty": "3"}]
        )
        self.assertEqual(1, len(submitted))
        self.assertEqual("accepted", results[0]["Status"])

    def test_shared_symbol_allows_only_higher_ranked_candidate(self):
        higher = debit_candidate(score=95, long_strike=600, short_strike=605)
        lower = debit_candidate(score=90, long_strike=600, short_strike=610)
        results, submitted = self.run_batch([higher, lower])
        self.assertEqual(1, len(submitted))
        self.assertEqual("accepted", results[0]["Status"])
        self.assertEqual("Skipped", results[1]["Status"])
        self.assertIn(
            option_symbol("SPY", "2026-08-21", "call", 600),
            results[1]["Message"],
        )
        self.assertIn("higher-ranked", results[1]["Message"])

    def test_position_lookup_failure_submits_nothing(self):
        results, submitted = self.run_batch(
            [debit_candidate()], position_errors=["service unavailable"]
        )
        self.assertEqual([], submitted)
        self.assertEqual("Error", results[0]["Status"])
        self.assertIn("no paper orders were submitted", results[0]["Message"])

    def test_positive_and_negative_string_quantities_are_handled(self):
        candidate = debit_candidate()
        symbols = [leg["symbol"] for leg in trade_multileg_order_details(candidate)[0]]
        for symbol, quantity in ((symbols[0], "-1.5"), (symbols[1], "+2")):
            with self.subTest(quantity=quantity):
                results, submitted = self.run_batch(
                    [candidate], [{"symbol": symbol, "qty": quantity}]
                )
                self.assertEqual([], submitted)
                self.assertEqual("Skipped", results[0]["Status"])

    def test_zero_quantity_position_does_not_block(self):
        symbol = option_symbol("SPY", "2026-08-21", "call", 600)
        results, submitted = self.run_batch(
            [debit_candidate()], [{"symbol": symbol, "qty": "0"}]
        )
        self.assertEqual(1, len(submitted))
        self.assertEqual("accepted", results[0]["Status"])

    def test_opening_intents_and_condor_shape_are_unchanged(self):
        debit_legs, _, _ = trade_multileg_order_details(debit_candidate())
        self.assertEqual(
            ["buy_to_open", "sell_to_open"],
            [leg["position_intent"] for leg in debit_legs],
        )
        condor_legs, _, _ = trade_multileg_order_details(condor_candidate())
        self.assertEqual(4, len(condor_legs))
        self.assertEqual(
            ["buy_to_open", "sell_to_open", "sell_to_open", "buy_to_open"],
            [leg["position_intent"] for leg in condor_legs],
        )


if __name__ == "__main__":
    unittest.main()
