import os
import unittest
from dataclasses import replace
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch


os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SECRET_KEY", "test-key")

from alpaca_client import scan_client_order_id
from backfill_scan_history import apply_backfill, fetch_scan_history
from history_tracker import (
    _refresh_paper_close_order,
    active_symbol_order_counts,
    expiration_is_ready,
    opening_order_update_values,
    paper_exit_decision_for_order,
    paper_order_is_active,
)
from paper_exit import submit_claimed_paper_exit
from scanner_tracking import build_history_backfill_updates, normalize_history_row
from stock2dupe import (
    ranking_test_scored,
    select_history_candidates,
)


def filled_order(**overrides):
    order = {
        "id": 1,
        "order_id": "open-1",
        "setup_key": "a" * 32,
        "status": "filled",
        "opening_order_status": "filled",
        "opening_filled_at": "2026-07-14T14:00:00+00:00",
        "opening_filled_avg_price": 2.0,
        "position_status": "open",
        "close_order_status": None,
        "entry_type": "debit",
        "limit_price": 1.0,
        "max_profit": 4.0,
        "max_risk": 3.0,
        "quantity": 1,
        "exit_policy": "tp50",
        "leg_key": (
            "NVDA260717C00200000:buy:buy_to_open:1|"
            "NVDA260717C00205000:sell:sell_to_open:1"
        ),
    }
    order.update(overrides)
    return order


class LegacyNormalizationAndBackfillTests(unittest.TestCase):
    def test_null_migration_columns_receive_fallbacks(self):
        row = {
            "id": 9,
            "scan_time": "2026-07-01T12:00:00+00:00",
            "ticker": "AAPL",
            "strategy": "bull call debit spread",
            "expiration": "2026-07-31",
            "long_strike": 200,
            "short_strike": 205,
            "setup_key": None,
            "scan_run_id": None,
            "scanner_version": None,
            "selection_method": None,
            "first_seen_at": None,
            "last_seen_at": None,
            "times_recommended": None,
            "entry_timestamp": None,
            "realized_pnl": None,
            "actual_realized_pnl": 12.5,
            "maximum_favorable_excursion": None,
            "highest_unrealized_pnl": 31.0,
            "maximum_adverse_excursion": None,
            "lowest_unrealized_pnl": -14.0,
        }
        normalized = normalize_history_row(row)
        self.assertEqual("legacy-9", normalized["scan_run_id"])
        self.assertEqual("legacy", normalized["scanner_version"])
        self.assertEqual("raw", normalized["selection_method"])
        self.assertEqual(row["scan_time"], normalized["first_seen_at"])
        self.assertEqual(row["scan_time"], normalized["last_seen_at"])
        self.assertEqual(1, normalized["times_recommended"])
        self.assertEqual(row["scan_time"], normalized["entry_timestamp"])
        self.assertEqual(12.5, normalized["realized_pnl"])
        self.assertEqual(31.0, normalized["maximum_favorable_excursion"])
        self.assertEqual(-14.0, normalized["maximum_adverse_excursion"])

    def test_backfill_groups_setup_history_and_is_idempotent(self):
        base = {
            "ticker": "SPY",
            "strategy": "bull call debit spread",
            "expiration": "2026-07-31",
            "long_strike": 600,
            "short_strike": 605,
            "setup_key": None,
            "scan_run_id": None,
            "scanner_version": None,
            "selection_method": None,
            "first_seen_at": None,
            "last_seen_at": None,
            "times_recommended": None,
            "entry_timestamp": None,
            "maximum_favorable_excursion": None,
            "maximum_adverse_excursion": None,
        }
        rows = [
            {"id": 1, "scan_time": "2026-07-01T12:00:00+00:00", **base},
            {"id": 2, "scan_time": "2026-07-02T12:00:00+00:00", **base},
        ]
        updates = build_history_backfill_updates(rows)
        self.assertEqual(2, len(updates))
        self.assertEqual(2, updates[0]["values"]["times_recommended"])
        self.assertEqual(rows[0]["scan_time"], updates[1]["values"]["first_seen_at"])
        self.assertEqual(rows[1]["scan_time"], updates[0]["values"]["last_seen_at"])

        updated_rows = []
        by_id = {update["id"]: update["values"] for update in updates}
        for row in rows:
            updated_rows.append({**row, **by_id[row["id"]]})
        self.assertEqual([], build_history_backfill_updates(updated_rows))

    def test_backfill_fetches_and_writes_in_batches(self):
        class FakeClient:
            def __init__(self):
                self.rows = [{"id": value} for value in range(1, 6)]
                self.start = 0
                self.end = 0
                self.pending_update = None
                self.pending_id = None
                self.updated_ids = []

            def table(self, _):
                return self

            def select(self, _):
                self.pending_update = None
                return self

            def order(self, _):
                return self

            def range(self, start, end):
                self.start, self.end = start, end
                return self

            def update(self, values):
                self.pending_update = values
                return self

            def eq(self, _, value):
                self.pending_id = value
                return self

            def execute(self):
                if self.pending_update is not None:
                    self.updated_ids.append(self.pending_id)
                    return SimpleNamespace(data=[{"id": self.pending_id}])
                return SimpleNamespace(data=self.rows[self.start:self.end + 1])

        client = FakeClient()
        self.assertEqual(5, len(fetch_scan_history(client, batch_size=2)))
        updates = [
            {"id": value, "values": {"scanner_version": "legacy"}}
            for value in range(1, 6)
        ]
        self.assertEqual(5, apply_backfill(client, updates, batch_size=2))
        self.assertEqual([1, 2, 3, 4, 5], client.updated_ids)


class FillAndValuationTests(unittest.TestCase):
    def test_opening_fill_state_is_separate_from_limit(self):
        values = opening_order_update_values(
            {
                "status": "filled",
                "filled_at": "2026-07-14T14:01:00+00:00",
                "filled_avg_price": "2.00",
            }
        )
        self.assertEqual(2.0, values["opening_filled_avg_price"])
        self.assertEqual(2.0, values["entry_price"])
        self.assertEqual("open", values["position_status"])

    def test_terminal_opening_order_is_not_active(self):
        values = opening_order_update_values(
            {"status": "canceled", "filled_at": None, "filled_avg_price": None}
        )
        order = filled_order(**values)
        self.assertEqual("canceled", order["position_status"])
        self.assertFalse(paper_order_is_active(order))

    def test_take_profit_uses_fill_instead_of_submitted_limit(self):
        order = filled_order(limit_price=1.0, opening_filled_avg_price=2.0)
        decision = paper_exit_decision_for_order(order, current_value_per_share=3.5)
        self.assertEqual(4.0, decision.target_value_per_share)
        self.assertFalse(decision.should_close)

    def test_unfilled_order_cannot_auto_exit(self):
        order = filled_order(
            opening_order_status="accepted",
            opening_filled_avg_price=None,
            position_status="pending",
        )
        self.assertFalse(paper_order_is_active(order))
        self.assertFalse(
            paper_exit_decision_for_order(order, current_value_per_share=9.0).should_close
        )

    def test_closed_historical_order_does_not_make_active_leg_ambiguous(self):
        active = filled_order(id=2)
        closed = filled_order(id=1, position_status="closed", close_order_status="filled")
        counts = active_symbol_order_counts([closed, active])
        self.assertTrue(paper_order_is_active(active))
        self.assertTrue(all(count == 1 for count in counts.values()))


class ClientOrderIdTests(unittest.TestCase):
    def test_different_short_strikes_do_not_collide(self):
        first = ranking_test_scored("SPY", "bull call debit spread", 80, 605, 600)
        second = ranking_test_scored("SPY", "bull call debit spread", 80, 606, 600)
        first_id = scan_client_order_id(first, "run-123")
        second_id = scan_client_order_id(second, "run-123")
        self.assertNotEqual(first_id, second_id)
        self.assertLessEqual(len(first_id), 48)

    def test_iron_condor_leg_changes_do_not_collide(self):
        base = ranking_test_scored(
            "SPY", "iron condor", 80, 610, 590, option_type="mixed"
        )
        first = replace(
            base,
            trade=replace(
                base.trade,
                put_long_strike=580,
                put_short_strike=585,
                call_short_strike=615,
                call_long_strike=620,
            ),
        )
        second = replace(
            first,
            trade=replace(first.trade, call_short_strike=616, call_long_strike=621),
        )
        self.assertNotEqual(
            scan_client_order_id(first, "run-123"),
            scan_client_order_id(second, "run-123"),
        )


class AtomicExitTests(unittest.TestCase):
    def test_successful_claim_submits_and_records(self):
        recorded = []
        result = submit_claimed_paper_exit(
            claim=lambda: {"id": 1},
            submit=lambda _: ({"id": "close-1", "status": "accepted"}, []),
            record_accepted=lambda claimed, close: recorded.append((claimed, close)),
            record_rejected=lambda *_: None,
        )
        self.assertTrue(result.claimed and result.submitted and result.recorded)
        self.assertEqual(1, len(recorded))

    def test_zero_row_claim_never_submits(self):
        submissions = []
        result = submit_claimed_paper_exit(
            claim=lambda: None,
            submit=lambda row: submissions.append(row),
            record_accepted=lambda *_: None,
            record_rejected=lambda *_: None,
        )
        self.assertFalse(result.claimed)
        self.assertEqual([], submissions)

    def test_two_callers_only_submit_once(self):
        state = {"available": True, "submissions": 0}

        def claim():
            if not state["available"]:
                return None
            state["available"] = False
            return {"id": 1, "close_client_order_id": "close-stable"}

        def submit(_):
            state["submissions"] += 1
            return {"id": "close-1", "status": "accepted"}, []

        arguments = {
            "claim": claim,
            "submit": submit,
            "record_accepted": lambda *_: None,
            "record_rejected": lambda *_: None,
        }
        first = submit_claimed_paper_exit(**arguments)
        second = submit_claimed_paper_exit(**arguments)
        self.assertTrue(first.submitted)
        self.assertFalse(second.submitted)
        self.assertEqual(1, state["submissions"])

    def test_accepted_order_survives_database_recording_failure(self):
        claimed = {"id": 1, "close_client_order_id": "close-stable"}

        def fail_record(*_):
            raise RuntimeError("database unavailable")

        result = submit_claimed_paper_exit(
            claim=lambda: claimed,
            submit=lambda _: ({"id": "close-1", "status": "accepted"}, []),
            record_accepted=fail_record,
            record_rejected=lambda *_: None,
        )
        self.assertTrue(result.claimed and result.submitted)
        self.assertFalse(result.recorded)
        self.assertEqual("close-stable", claimed["close_client_order_id"])

    def test_rejected_closing_order_is_recorded_without_success(self):
        rejected = []
        result = submit_claimed_paper_exit(
            claim=lambda: {"id": 1},
            submit=lambda _: (None, ["rejected"]),
            record_accepted=lambda *_: None,
            record_rejected=lambda claimed, message: rejected.append((claimed, message)),
        )
        self.assertTrue(result.claimed)
        self.assertFalse(result.submitted)
        self.assertTrue(result.recorded)
        self.assertEqual("rejected", rejected[0][1])


class CloseRefreshTests(unittest.TestCase):
    def test_newly_filled_close_updates_returned_order(self):
        order = filled_order(close_order_id="close-1", close_order_status="accepted")

        class FakeTable:
            def update(self, values):
                self.values = values
                return self

            def eq(self, *_):
                return self

            def execute(self):
                return SimpleNamespace(data=[self.values])

        class FakeSupabase:
            def table(self, _):
                return FakeTable()

        with (
            patch(
                "history_tracker.get_alpaca_order",
                return_value=(
                    {
                        "status": "filled",
                        "filled_avg_price": "3.00",
                        "filled_at": "2026-07-14T15:00:00+00:00",
                    },
                    [],
                ),
            ),
            patch("history_tracker.supabase", FakeSupabase()),
        ):
            refreshed, errors = _refresh_paper_close_order(order)

        self.assertEqual([], errors)
        self.assertEqual("filled", refreshed["close_order_status"])
        self.assertEqual("closed", refreshed["position_status"])
        self.assertEqual(100.0, refreshed["realized_pnl"])
        self.assertFalse(paper_order_is_active(refreshed))


class ExpirationAndHistorySelectionTests(unittest.TestCase):
    def test_same_day_expiration_is_not_ready_intraday(self):
        today = date(2026, 7, 14)
        self.assertFalse(expiration_is_ready(today, today, include_today=False))
        self.assertTrue(expiration_is_ready(today, today, include_today=True))

    def test_reserved_strategy_candidates_count_toward_ticker_cap(self):
        candidates = [
            ranking_test_scored("SPY", "bull call debit spread", 95, 605, 600),
            ranking_test_scored(
                "SPY", "bear put debit spread", 94, 590, 595, option_type="put"
            ),
            ranking_test_scored(
                "SPY", "put credit spread", 93, 585, 580, option_type="put"
            ),
            ranking_test_scored("NVDA", "bull call debit spread", 92, 205, 200),
            ranking_test_scored("QQQ", "call credit spread", 91, 510, 515),
        ]
        selected = select_history_candidates(
            candidates,
            limit=5,
            per_ticker=2,
            per_strategy=1,
        )
        self.assertLessEqual(
            sum(item.trade.ticker == "SPY" for item in selected),
            2,
        )


if __name__ == "__main__":
    unittest.main()
