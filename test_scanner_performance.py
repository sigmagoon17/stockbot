import inspect
import os
import time
import unittest
from collections import Counter
from datetime import date, timedelta
from unittest.mock import patch

from scheduled_scan import configured_expiration_coverage, main as scheduled_scan_main
from scanner_tracking import setup_key_for_trade
from stock2dupe import (
    EXPIRATION_COVERAGE_EXHAUSTIVE,
    EXPIRATION_COVERAGE_FAST_WEEKLY,
    OptionContract,
    ScanPreferences,
    _build_condor_wings_for_type,
    _combine_condor_wings,
    _condor_diagnostics_from_construction,
    _condor_wing_trade,
    build_iron_condor,
    build_iron_condors_with_diagnostics,
    condor_diagnostics,
    days_to_expiration,
    get_option_chain,
    get_option_chain_result,
    passes_condor_filters,
    passes_condor_wing_filters,
    score_trade,
    select_expirations,
)


def expiration(days: int) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def contract(
    option_type: str,
    strike: float,
    expiration_value: str,
    *,
    ticker: str = "SPY",
    liquid: bool = True,
    valid_delta: bool = True,
    real_quote: bool = True,
) -> OptionContract:
    if option_type == "put":
        bid = round(strike * 0.05, 2)
        delta = -0.20 if valid_delta else -0.35
    else:
        bid = round((150 - strike) * 0.05, 2)
        delta = 0.20 if valid_delta else 0.35
    return OptionContract(
        ticker=ticker,
        expiration=expiration_value,
        option_type=option_type,
        strike=float(strike),
        bid=max(0.10, bid),
        ask=max(0.15, round(bid + 0.05, 2)),
        delta=delta,
        gamma=0.01,
        theta=-0.02,
        vega=0.08,
        implied_volatility=0.20,
        open_interest=500 if liquid else 100,
        volume=100 if liquid else 10,
        quote_source="bid/ask" if real_quote else "last price estimate",
    )


def synthetic_large_chain() -> list[OptionContract]:
    chain = []
    for expiration_value in (expiration(28), expiration(35), expiration(42)):
        for option_type in ("put", "call"):
            for index, strike in enumerate(range(70, 131)):
                chain.append(
                    contract(
                        option_type,
                        strike,
                        expiration_value,
                        liquid=index % 11 != 0,
                        valid_delta=index % 13 != 0,
                        real_quote=index % 17 != 0,
                    )
                )
    return chain


def reference_condor_wings_for_type(
    option_chain,
    option_type,
    underlying_price,
    earnings_date,
    volatility_rank,
):
    contracts = [
        contract_row
        for contract_row in option_chain
        if contract_row.option_type == option_type
    ]
    eligible_wings = []
    stats = Counter()
    operations = 0
    for short_contract in contracts:
        for long_contract in contracts:
            operations += 1
            if short_contract.ticker != long_contract.ticker:
                continue
            if short_contract.expiration != long_contract.expiration:
                continue
            if option_type == "put" and short_contract.strike <= long_contract.strike:
                continue
            if option_type == "call" and short_contract.strike >= long_contract.strike:
                continue
            width = abs(short_contract.strike - long_contract.strike)
            if width <= 0 or width > 5:
                continue
            dte = days_to_expiration(short_contract.expiration)
            if dte <= 0:
                continue
            credit = round(short_contract.bid - long_contract.ask, 2)
            if credit <= 0 or width - credit <= 0:
                continue

            stats["raw_wings_built"] += 1
            wing = _condor_wing_trade(
                short_contract,
                long_contract,
                underlying_price,
                earnings_date,
                volatility_rank,
            )
            passed, reasons = passes_condor_wing_filters(wing)
            if "open interest is below 200" in reasons or "volume is below 25" in reasons:
                stats["rejected_for_liquidity"] += 1
            if "delta is outside the credit range of 0.10 to 0.30" in reasons:
                stats["rejected_for_delta"] += 1
            if "quote is estimated from last price" in reasons:
                stats["rejected_for_estimated_quotes"] += 1
            if passed:
                eligible_wings.append(wing)
    return eligible_wings, stats, operations


def condor_signature(trade, preferences):
    passed, reasons = passes_condor_filters(trade, preferences)
    scored = score_trade(trade, preferences)
    return {
        "setup_key": setup_key_for_trade(trade),
        "strikes": (
            trade.put_long_strike,
            trade.put_short_strike,
            trade.call_short_strike,
            trade.call_long_strike,
        ),
        "credit": trade.credit,
        "max_risk": trade.max_risk,
        "bid": trade.bid,
        "ask": trade.ask,
        "cushions": (
            trade.put_expected_move_cushion,
            trade.call_expected_move_cushion,
            trade.minimum_expected_move_cushion,
        ),
        "passed": passed,
        "reasons": reasons,
        "category_scores": scored.category_scores,
        "quant_score": scored.quant_score,
        "total_score": scored.total_score,
    }


class ExpirationCoverageTests(unittest.TestCase):
    def test_implicit_defaults_preserve_exhaustive_coverage(self):
        preferences = ScanPreferences(500, "neutral", "moderate")
        self.assertEqual(
            EXPIRATION_COVERAGE_EXHAUSTIVE,
            preferences.expiration_coverage,
        )
        for function in (
            select_expirations,
            get_option_chain_result,
            get_option_chain,
        ):
            self.assertEqual(
                EXPIRATION_COVERAGE_EXHAUSTIVE,
                inspect.signature(function)
                .parameters["expiration_coverage"]
                .default,
            )

    def test_select_expirations_implicit_mode_is_exhaustive(self):
        available = [expiration(days) for days in range(1, 71)]
        self.assertEqual(
            [expiration(days) for days in range(21, 61)],
            select_expirations(available),
        )

    def test_fast_weekly_daily_expirations_choose_nearest_in_each_bucket(self):
        available = [expiration(days) for days in range(1, 71)]
        selected = select_expirations(
            available,
            expiration_coverage=EXPIRATION_COVERAGE_FAST_WEEKLY,
        )
        self.assertEqual(
            [expiration(days) for days in (21, 28, 35, 42, 49, 56)],
            selected,
        )
        self.assertLessEqual(len(selected), 7)

    def test_fast_weekly_monthly_only_expirations_preserve_available_dates(self):
        available = [expiration(days) for days in (22, 37, 52)]
        self.assertEqual(
            available,
            select_expirations(
                available,
                expiration_coverage=EXPIRATION_COVERAGE_FAST_WEEKLY,
            ),
        )

    def test_exhaustive_preserves_every_expiration_in_range(self):
        available = [expiration(days) for days in range(1, 71)]
        selected = select_expirations(
            available,
            expiration_coverage=EXPIRATION_COVERAGE_EXHAUSTIVE,
        )
        self.assertEqual([expiration(days) for days in range(21, 61)], selected)

    def test_nearest_five_and_specific_expiration_take_precedence(self):
        available = [expiration(days) for days in range(1, 15)]
        self.assertEqual(
            available[:5],
            select_expirations(
                available,
                nearest_expiration=True,
                expiration_coverage=EXPIRATION_COVERAGE_EXHAUSTIVE,
            ),
        )
        target = date.today() + timedelta(days=9)
        self.assertEqual(
            [target.isoformat()],
            select_expirations(
                available,
                test_expiration=target,
                nearest_expiration=True,
            ),
        )


class ScheduledExpirationCoverageTests(unittest.TestCase):
    def test_missing_environment_value_selects_exhaustive(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                EXPIRATION_COVERAGE_EXHAUSTIVE,
                configured_expiration_coverage(),
            )

    def test_explicit_fast_weekly_is_preserved(self):
        with patch.dict(
            os.environ,
            {"SCAN_EXPIRATION_COVERAGE": EXPIRATION_COVERAGE_FAST_WEEKLY},
            clear=True,
        ):
            self.assertEqual(
                EXPIRATION_COVERAGE_FAST_WEEKLY,
                configured_expiration_coverage(),
            )

    def test_explicit_exhaustive_is_preserved(self):
        with patch.dict(
            os.environ,
            {"SCAN_EXPIRATION_COVERAGE": EXPIRATION_COVERAGE_EXHAUSTIVE},
            clear=True,
        ):
            self.assertEqual(
                EXPIRATION_COVERAGE_EXHAUSTIVE,
                configured_expiration_coverage(),
            )

    def test_invalid_environment_value_fails_clearly(self):
        invalid = "every_other_friday"
        with patch.dict(
            os.environ,
            {"SCAN_EXPIRATION_COVERAGE": invalid},
            clear=True,
        ), patch("scheduled_scan.get_option_chain") as option_chain_loader:
            with self.assertRaisesRegex(ValueError, invalid):
                scheduled_scan_main()
            option_chain_loader.assert_not_called()


class CondorPerformanceEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chain = synthetic_large_chain()
        cls.preferences = ScanPreferences(500, "neutral", "moderate")

    def reference_construction(self):
        put_wings, put_stats, put_operations = reference_condor_wings_for_type(
            self.chain, "put", 100.0, None, 55
        )
        call_wings, call_stats, call_operations = reference_condor_wings_for_type(
            self.chain, "call", 100.0, None, 55
        )
        (
            condors,
            pairing_stats,
            put_expirations,
            call_expirations,
            common_expirations,
        ) = _combine_condor_wings(
            put_wings,
            call_wings,
            100.0,
            55,
            self.preferences,
        )
        diagnostics = _condor_diagnostics_from_construction(
            self.chain,
            self.preferences,
            condors,
            put_wings,
            call_wings,
            put_stats,
            call_stats,
            pairing_stats,
            put_expirations,
            call_expirations,
            common_expirations,
        )
        return condors, diagnostics, put_operations + call_operations

    def test_optimized_construction_matches_reference_exactly(self):
        reference_condors, reference_diagnostics, _ = self.reference_construction()
        optimized = build_iron_condors_with_diagnostics(
            self.chain, 100.0, None, 55, self.preferences
        )
        self.assertEqual(
            [condor_signature(row, self.preferences) for row in reference_condors],
            [condor_signature(row, self.preferences) for row in optimized.condors],
        )
        self.assertEqual(reference_diagnostics, optimized.diagnostics)

    def test_combined_result_matches_compatibility_wrappers(self):
        combined = build_iron_condors_with_diagnostics(
            self.chain, 100.0, None, 55, self.preferences
        )
        self.assertEqual(
            combined.condors,
            build_iron_condor(self.chain, 100.0, None, 55, self.preferences),
        )
        self.assertEqual(
            combined.diagnostics,
            condor_diagnostics(self.chain, 100.0, None, 55, self.preferences),
        )

    def test_bounded_search_considers_dramatically_fewer_pairs(self):
        reference_started = time.perf_counter()
        _, _, reference_operations = self.reference_construction()
        reference_seconds = time.perf_counter() - reference_started

        optimized_started = time.perf_counter()
        optimized = build_iron_condors_with_diagnostics(
            self.chain, 100.0, None, 55, self.preferences
        )
        optimized_seconds = time.perf_counter() - optimized_started
        optimized_operations = optimized.operation_counts[
            "contract_pairs_considered"
        ]
        print(
            "Synthetic condor benchmark: "
            f"reference={reference_operations} pairs/{reference_seconds:.6f}s, "
            f"optimized={optimized_operations} pairs/{optimized_seconds:.6f}s"
        )
        self.assertEqual(
            reference_operations,
            optimized.operation_counts["reference_contract_pairs"],
        )
        self.assertLess(optimized_operations, reference_operations // 5)


if __name__ == "__main__":
    unittest.main()
