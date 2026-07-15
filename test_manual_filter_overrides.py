import hashlib
import inspect
import ast
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from alpaca_client import (
    option_symbol,
    submit_manual_multileg_order,
    submit_scored_multileg_orders,
    trade_multileg_order_details,
    validate_manual_limit_price,
)
from stock2dupe import (
    ScanPreferences,
    Trade,
    evaluate_filter_failures,
    manual_order_safety_failures,
    manual_override_decision,
    passes_filters,
    ranking_test_scored,
    ranking_test_trade,
    scan_trades,
    select_execution_candidates,
)


def preferences(**updates):
    values = {
        "max_risk": 500,
        "outlook": "neutral",
        "risk_tolerance": "moderate",
    }
    values.update(updates)
    return ScanPreferences(**values)


def credit_trade(**updates):
    trade = ranking_test_trade(
        "SPY", "call credit spread", 105, 100, expiration="2026-08-21"
    )
    return replace(trade, **updates)


def debit_trade(**updates):
    trade = ranking_test_trade(
        "SPY", "bull call debit spread", 105, 100, expiration="2026-08-21"
    )
    return replace(
        trade,
        delta=0.5,
        entry_type="debit",
        credit=0.0,
        max_risk=2.0,
        max_profit=3.0,
        bid=1.8,
        ask=2.0,
        **updates,
    )


def condor_trade(**updates):
    trade = Trade(
        ticker="SPY",
        strategy="iron condor",
        expiration="2026-08-21",
        option_type="mixed",
        delta=0.2,
        volatility_rank=55,
        earnings_before_exp=False,
        open_interest=1000,
        volume=500,
        dte=35,
        bid=1.1,
        ask=1.5,
        credit=1.5,
        max_risk=3.5,
        underlying_price=100,
        expected_move=4,
        short_strike=95,
        long_strike=105,
        short_bid=1.0,
        short_ask=1.1,
        long_bid=0.4,
        long_ask=0.5,
        short_delta=-0.2,
        long_delta=0.2,
        put_long_strike=90,
        put_short_strike=95,
        call_short_strike=105,
        call_long_strike=110,
        put_expected_move_cushion=1,
        call_expected_move_cushion=1,
        minimum_expected_move_cushion=1,
    )
    return replace(trade, **updates)


class ConfigurableFilterTests(unittest.TestCase):
    def test_default_vertical_width_matches_existing_boundary(self):
        self.assertTrue(passes_filters(credit_trade(ask=1.19), preferences())[0])
        passed, reasons = passes_filters(credit_trade(ask=1.20), preferences())
        self.assertFalse(passed)
        self.assertIn("bid/ask spread is wider than 0.20", reasons)

    def test_configurable_vertical_width_allows_twenty_cents(self):
        passed, reasons = passes_filters(
            credit_trade(ask=1.20),
            preferences(vertical_max_bid_ask_width=0.25),
        )
        self.assertTrue(passed, reasons)

    def test_condor_default_width_remains_forty_cents(self):
        self.assertTrue(passes_filters(condor_trade(), preferences())[0])
        passed, reasons = passes_filters(
            condor_trade(ask=1.51), preferences()
        )
        self.assertFalse(passed)
        self.assertIn(
            "condor four-leg bid/ask spread is wider than configured maximum",
            reasons,
        )

    def test_relative_bid_ask_ratio_for_credit_and_debit(self):
        credit_failures = evaluate_filter_failures(
            credit_trade(ask=1.20),
            preferences(max_bid_ask_to_trade_value_ratio=0.19),
        )
        debit_failures = evaluate_filter_failures(
            debit_trade(),
            preferences(max_bid_ask_to_trade_value_ratio=0.09),
        )
        self.assertIn("bid_ask_ratio", {item.filter_id for item in credit_failures})
        self.assertIn("bid_ask_ratio", {item.filter_id for item in debit_failures})


class ManualOverrideDecisionTests(unittest.TestCase):
    def test_rejected_trade_retains_every_original_reason(self):
        trade = credit_trade(ask=1.20, volume=1)
        failures = evaluate_filter_failures(trade, preferences())
        _, reasons = passes_filters(trade, preferences())
        self.assertEqual([item.description for item in failures], reasons)
        self.assertEqual(2, len(reasons))

    def test_overriding_bid_ask_does_not_override_low_volume(self):
        failures = evaluate_filter_failures(
            credit_trade(ask=1.20, volume=1), preferences()
        )
        decision = manual_override_decision(failures, ["bid_ask_width"])
        self.assertFalse(decision.allowed)
        self.assertEqual(
            ["volume"],
            [failure.filter_id for failure in decision.non_overridden_failures],
        )

    def test_non_overridable_failure_always_blocks(self):
        trade = credit_trade(quote_source="last price estimate", ask=1.20)
        failures = evaluate_filter_failures(trade, preferences())
        selected = [item.filter_id for item in failures if item.overridable]
        decision = manual_override_decision(failures, selected)
        self.assertFalse(decision.allowed)
        self.assertIn(
            "quote_source",
            {item.filter_id for item in decision.non_overridable_failures},
        )

    def test_invalid_economics_are_non_overridable(self):
        failures = manual_order_safety_failures(credit_trade(max_risk=2.0))
        self.assertIn(
            "impossible_spread_economics",
            {item.filter_id for item in failures if not item.overridable},
        )

    def test_overridden_candidate_never_enters_automatic_selection(self):
        trade = credit_trade(ask=1.20)
        failures = evaluate_filter_failures(trade, preferences())
        self.assertTrue(
            manual_override_decision(failures, ["bid_ask_width"]).allowed
        )
        passing, rejected = scan_trades([trade], preferences())
        self.assertEqual([], passing)
        self.assertEqual(1, len(rejected))
        self.assertEqual([], select_execution_candidates(passing))


class ManualOverridePaperSafetyTests(unittest.TestCase):
    def candidate(self):
        scored = ranking_test_scored(
            "SPY",
            "bull call debit spread",
            70,
            105,
            100,
            expiration="2026-08-21",
        )
        return replace(scored, trade=debit_trade())

    def test_override_uses_broker_preflight_and_owned_leg_blocks(self):
        legs, limit_price, _ = trade_multileg_order_details(self.candidate())
        owned_symbol = option_symbol("SPY", "2026-08-21", "call", 105)
        with (
            patch(
                "alpaca_client.get_alpaca_positions",
                return_value=([{"symbol": owned_symbol, "qty": "1"}], []),
            ) as positions,
            patch("alpaca_client.get_open_alpaca_orders", return_value=([], [])),
            patch("alpaca_client.submit_multileg_order") as submit,
        ):
            order, errors, message = submit_manual_multileg_order(
                legs, 1, limit_price, "override-test"
            )
        positions.assert_called_once_with()
        submit.assert_not_called()
        self.assertIsNone(order)
        self.assertEqual([], errors)
        self.assertIn(owned_symbol, message)

    def test_clean_override_reaches_mocked_paper_endpoint(self):
        legs, limit_price, _ = trade_multileg_order_details(self.candidate())
        with (
            patch("alpaca_client.get_alpaca_positions", return_value=([], [])),
            patch("alpaca_client.get_open_alpaca_orders", return_value=([], [])),
            patch(
                "alpaca_client.submit_multileg_order",
                return_value=({"id": "paper-only", "status": "accepted"}, []),
            ) as submit,
        ):
            order, errors, message = submit_manual_multileg_order(
                legs, 1, limit_price, "override-test"
            )
        submit.assert_called_once()
        self.assertEqual("paper-only", order["id"])
        self.assertEqual([], errors)
        self.assertIsNone(message)

    def test_live_endpoint_blocks_before_any_request(self):
        legs, limit_price, _ = trade_multileg_order_details(self.candidate())
        with (
            patch(
                "alpaca_client.alpaca_config_status",
                return_value={"is_paper": False},
            ),
            patch("alpaca_client.get_alpaca_positions") as positions,
            patch("alpaca_client.submit_multileg_order") as submit,
        ):
            order, errors, message = submit_manual_multileg_order(
                legs, 1, limit_price, "override-test"
            )
        positions.assert_not_called()
        submit.assert_not_called()
        self.assertIsNone(order)
        self.assertIsNone(message)
        self.assertIn("not using the paper endpoint", errors[0])

    def test_manual_limit_price_boundaries(self):
        accepted = (
            ("debit", 1.50, 5.0),
            ("debit", 4.99, 5.0),
            ("credit", 1.00, 5.0),
        )
        rejected = (
            ("debit", 5.00, 5.0),
            ("debit", 5.01, 5.0),
            ("credit", 5.00, 5.0),
        )
        for entry_type, limit_price, width in accepted:
            with self.subTest(entry_type=entry_type, limit_price=limit_price):
                self.assertEqual(
                    [], validate_manual_limit_price(entry_type, limit_price, width)
                )
        for entry_type, limit_price, width in rejected:
            with self.subTest(entry_type=entry_type, limit_price=limit_price):
                errors = validate_manual_limit_price(
                    entry_type, limit_price, width
                )
                self.assertEqual(1, len(errors))
                self.assertIn("less than the $5.00 spread width", errors[0])

    def test_invalid_manual_limit_values_are_rejected(self):
        for invalid in (None, "bad", 0, -1, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=invalid):
                self.assertTrue(validate_manual_limit_price("debit", invalid, 5.0))
        self.assertTrue(validate_manual_limit_price("unsupported", 1.0, 5.0))
        for invalid_width in (None, "bad", 0, -1, float("nan"), float("inf")):
            with self.subTest(width=invalid_width):
                self.assertTrue(
                    validate_manual_limit_price("credit", 1.0, invalid_width)
                )

    def test_invalid_limit_stops_before_broker_and_submit(self):
        legs, _, _ = trade_multileg_order_details(self.candidate())
        with (
            patch("alpaca_client.get_alpaca_positions") as positions,
            patch("alpaca_client.get_open_alpaca_orders") as open_orders,
            patch("alpaca_client.submit_multileg_order") as submit,
        ):
            order, errors, message = submit_manual_multileg_order(
                legs,
                1,
                5.0,
                "override-test",
                expected_entry_type="debit",
                expected_spread_width=5.0,
            )
        positions.assert_not_called()
        open_orders.assert_not_called()
        submit.assert_not_called()
        self.assertIsNone(order)
        self.assertIsNone(message)
        self.assertIn("less than the $5.00 spread width", errors[0])

    def test_valid_limit_reaches_existing_manual_preflight(self):
        legs, _, _ = trade_multileg_order_details(self.candidate())
        with (
            patch("alpaca_client.get_alpaca_positions", return_value=([], [])) as positions,
            patch("alpaca_client.get_open_alpaca_orders", return_value=([], [])),
            patch(
                "alpaca_client.submit_multileg_order",
                return_value=({"id": "paper-only", "status": "accepted"}, []),
            ) as submit,
        ):
            order, errors, message = submit_manual_multileg_order(
                legs,
                1,
                1.5,
                "override-test",
                expected_entry_type="debit",
                expected_spread_width=5.0,
            )
        positions.assert_called_once_with()
        submit.assert_called_once()
        self.assertEqual("paper-only", order["id"])
        self.assertEqual([], errors)
        self.assertIsNone(message)


class IsolationTests(unittest.TestCase):
    def test_automatic_submitter_has_no_override_parameter(self):
        from alpaca_client import submit_scored_multileg_orders

        self.assertNotIn(
            "override", inspect.signature(submit_scored_multileg_orders).parameters
        )

    def test_scheduled_scanner_is_unchanged(self):
        source = Path("scheduled_scan.py").read_text(encoding="utf-8")
        digest = hashlib.sha256(source.replace("\r\n", "\n").encode()).hexdigest()
        self.assertEqual(
            "ed3a5fcaa302fa80d10776c753bef3fbf816330d3081c80ae1d4a6f4fb13b64e",
            digest,
        )

    def test_automatic_submission_does_not_use_manual_limit_validation(self):
        source = inspect.getsource(submit_scored_multileg_orders)
        self.assertNotIn("validate_manual_limit_price", source)

    def test_ui_limit_rejection_returns_before_submit_and_history(self):
        tree = ast.parse(Path("app.py").read_text(encoding="utf-8"))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "render_manual_override_paper_form"
        )
        calls = {
            node.func.id: node.lineno
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {
                "validate_manual_limit_price",
                "submit_manual_multileg_order",
                "append_alpaca_paper_orders",
            }
        }
        validation_guard = next(
            node
            for node in ast.walk(function)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "limit_errors"
        )
        self.assertTrue(
            any(isinstance(node, ast.Return) for node in validation_guard.body)
        )
        self.assertLess(
            calls["validate_manual_limit_price"],
            calls["submit_manual_multileg_order"],
        )
        self.assertLess(
            calls["submit_manual_multileg_order"],
            calls["append_alpaca_paper_orders"],
        )


class OverrideHistoryCompatibilityTests(unittest.TestCase):
    class FakeSupabase:
        def __init__(self, missing_override_columns=False):
            self.missing_override_columns = missing_override_columns
            self.mode = None
            self.pending_rows = None
            self.insert_attempts = []

        def table(self, _name):
            return self

        def select(self, _columns):
            self.mode = "select"
            return self

        def insert(self, rows):
            self.mode = "insert"
            self.pending_rows = rows
            return self

        def execute(self):
            if self.mode == "select":
                return SimpleNamespace(data=[])
            rows = [dict(row) for row in self.pending_rows]
            self.insert_attempts.append(rows)
            if (
                self.missing_override_columns
                and len(self.insert_attempts) == 1
                and rows[0].get("manual_override")
            ):
                raise RuntimeError("manual_override missing from schema cache")
            return SimpleNamespace(data=rows)

    @staticmethod
    def order_result(manual_override=False):
        result = {
            "Status": "accepted",
            "Order ID": "order-1",
            "Client Order ID": "client-1",
            "Ticker": "SPY",
            "Strategy": "call credit spread",
            "Expiration": "2026-08-21",
            "Selection Method": "manual_override" if manual_override else "manual",
            "Entry Type": "credit",
            "Limit Price": 1.0,
            "Max Profit": 1.0,
            "Max Risk": 4.0,
            "Quantity": 1,
            "Order Class": "mleg",
            "Leg Key": "leg-key",
            "Message": "paper order",
        }
        if manual_override:
            result.update(
                {
                    "Manual Override": True,
                    "Overridden Filters": ["bid_ask_width"],
                    "Original Rejection Reasons": ["wide quote"],
                    "Override Timestamp": "2026-07-14T20:00:00+00:00",
                    "Original Quantitative Score": 61,
                }
            )
        return result

    def test_normal_order_does_not_require_override_migration(self):
        from history_tracker import append_alpaca_paper_orders

        client = self.FakeSupabase()
        with patch("history_tracker.supabase", client):
            errors = append_alpaca_paper_orders([self.order_result()])
        self.assertEqual([], errors)
        self.assertNotIn("manual_override", client.insert_attempts[0][0])

    def test_missing_migration_falls_back_to_message_metadata(self):
        from history_tracker import append_alpaca_paper_orders

        client = self.FakeSupabase(missing_override_columns=True)
        with patch("history_tracker.supabase", client):
            warnings = append_alpaca_paper_orders(
                [self.order_result(manual_override=True)]
            )
        self.assertEqual(2, len(client.insert_attempts))
        fallback = client.insert_attempts[1][0]
        self.assertNotIn("manual_override", fallback)
        self.assertIn("manual_override_metadata=", fallback["message"])
        self.assertIn("supabase_manual_filter_overrides.sql", warnings[0])


if __name__ == "__main__":
    unittest.main()
