import unittest
from dataclasses import replace
from datetime import date, timedelta

from stock2dupe import (
    OptionContract,
    ScanPreferences,
    Trade,
    build_call_credit_spreads,
    build_iron_condor,
    build_put_credit_spreads,
    combine_condor_wings,
    condor_diagnostics,
    diversify_scored_trades,
    passes_filters,
    scan_trades,
    score_trade,
    select_execution_candidates,
    strategy_fit_score,
)


UP_MOVE = {
    "1D Move %": 4.0,
    "5D Move %": 5.0,
    "Move vs 20D Vol": 2.0,
}
DOWN_MOVE = {
    "1D Move %": -4.0,
    "5D Move %": -5.0,
    "Move vs 20D Vol": 2.0,
}


def representative_trade(strategy: str) -> Trade:
    common = {
        "expiration": "2026-08-21",
        "volatility_rank": 55,
        "earnings_before_exp": False,
        "open_interest": 1000,
        "volume": 500,
        "dte": 30,
        "bid": 1.0,
        "ask": 1.1,
        "underlying_price": 100.0,
        "expected_move": 5.0,
        "short_bid": 1.5,
        "short_ask": 1.6,
        "long_bid": 0.4,
        "long_ask": 0.5,
        "quote_source": "bid/ask",
    }
    if strategy == "put credit spread":
        return Trade(
            ticker="SPY",
            strategy=strategy,
            option_type="put",
            delta=-0.20,
            credit=1.0,
            max_risk=4.0,
            short_strike=90.0,
            long_strike=85.0,
            short_delta=-0.20,
            long_delta=-0.10,
            **common,
        )
    if strategy == "call credit spread":
        return Trade(
            ticker="QQQ",
            strategy=strategy,
            option_type="call",
            delta=0.20,
            credit=1.0,
            max_risk=4.0,
            short_strike=110.0,
            long_strike=115.0,
            short_delta=0.20,
            long_delta=0.10,
            **common,
        )
    if strategy == "bull call debit spread":
        return Trade(
            ticker="AAPL",
            strategy=strategy,
            option_type="call",
            delta=0.50,
            credit=0.0,
            max_risk=1.5,
            short_strike=105.0,
            long_strike=100.0,
            short_delta=0.30,
            long_delta=0.50,
            entry_type="debit",
            max_profit=3.5,
            **common,
        )
    if strategy == "bear put debit spread":
        return Trade(
            ticker="MSFT",
            strategy=strategy,
            option_type="put",
            delta=-0.50,
            credit=0.0,
            max_risk=1.5,
            short_strike=95.0,
            long_strike=100.0,
            short_delta=-0.30,
            long_delta=-0.50,
            entry_type="debit",
            max_profit=3.5,
            **common,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def future_expiration(days: int = 30) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def option_contract(
    option_type: str,
    strike: float,
    bid: float,
    ask: float,
    delta: float,
    *,
    ticker: str = "DIA",
    expiration: str | None = None,
    open_interest: int = 500,
    volume: int = 100,
    quote_source: str = "bid/ask",
) -> OptionContract:
    return OptionContract(
        ticker=ticker,
        expiration=expiration or future_expiration(),
        option_type=option_type,
        strike=strike,
        bid=bid,
        ask=ask,
        delta=delta,
        gamma=0,
        theta=0,
        vega=0,
        implied_volatility=0.10,
        open_interest=open_interest,
        volume=volume,
        quote_source=quote_source,
    )


def valid_condor_chain(
    *,
    ticker: str = "DIA",
    expiration: str | None = None,
    call_long_strike: float = 115,
) -> list[OptionContract]:
    expiration = expiration or future_expiration()
    return [
        option_contract(
            "put", 85, 0.45, 0.50, -0.10,
            ticker=ticker, expiration=expiration,
        ),
        option_contract(
            "put", 90, 1.00, 1.05, -0.20,
            ticker=ticker, expiration=expiration,
        ),
        option_contract(
            "call", 110, 1.00, 1.05, 0.20,
            ticker=ticker, expiration=expiration,
        ),
        option_contract(
            "call", call_long_strike, 0.45, 0.50, 0.10,
            ticker=ticker, expiration=expiration,
        ),
    ]


def condor_preferences(**updates) -> ScanPreferences:
    values = {
        "max_risk": 500,
        "outlook": "neutral",
        "risk_tolerance": "moderate",
    }
    values.update(updates)
    return ScanPreferences(**values)


def valid_condor(
    *,
    ticker: str = "DIA",
    volatility_rank: float = 20,
    call_long_strike: float = 115,
) -> Trade:
    condors = build_iron_condor(
        valid_condor_chain(
            ticker=ticker,
            call_long_strike=call_long_strike,
        ),
        100.0,
        None,
        volatility_rank,
        condor_preferences(),
    )
    if len(condors) != 1:
        raise AssertionError(f"Expected one deterministic condor, got {len(condors)}")
    return condors[0]


def wing_trade(
    option_type: str,
    expiration: str,
    *,
    ticker: str = "DIA",
    open_interest: int = 500,
    volume: int = 100,
) -> Trade:
    is_put = option_type == "put"
    short_strike = 90.0 if is_put else 110.0
    long_strike = 85.0 if is_put else 115.0
    return Trade(
        ticker=ticker,
        strategy=("put credit spread" if is_put else "call credit spread"),
        expiration=expiration,
        option_type=option_type,
        delta=(-0.20 if is_put else 0.20),
        volatility_rank=55,
        earnings_before_exp=False,
        open_interest=open_interest,
        volume=volume,
        dte=30,
        bid=0.50,
        ask=0.60,
        credit=0.50,
        max_risk=4.50,
        underlying_price=100.0,
        expected_move=2.0,
        short_strike=short_strike,
        long_strike=long_strike,
        short_bid=1.00,
        short_ask=1.05,
        long_bid=0.45,
        long_ask=0.50,
        short_delta=(-0.20 if is_put else 0.20),
        long_delta=(-0.10 if is_put else 0.10),
        quote_source="bid/ask",
    )


class NonCondorCharacterizationTests(unittest.TestCase):
    CASES = (
        ("put credit spread", "bullish", UP_MOVE),
        ("call credit spread", "bearish", DOWN_MOVE),
        ("bull call debit spread", "bullish", UP_MOVE),
        ("bear put debit spread", "bearish", DOWN_MOVE),
    )

    def test_current_non_condor_filter_and_score_outputs(self):
        expected_scores = {
            "put credit spread": {
                "Expected Move": 25,
                "Realized Volatility Rank": 12,
                "Liquidity": 20,
                "DTE": 15,
                "Delta/Probability": 15,
                "Profit/Risk": 10,
                "Strategy Fit": 10,
            },
            "call credit spread": {
                "Expected Move": 25,
                "Realized Volatility Rank": 12,
                "Liquidity": 20,
                "DTE": 15,
                "Delta/Probability": 15,
                "Profit/Risk": 10,
                "Strategy Fit": 10,
            },
            "bull call debit spread": {
                "Expected Move": 25,
                "Realized Volatility Rank": 20,
                "Liquidity": 20,
                "DTE": 15,
                "Delta/Probability": 15,
                "Profit/Risk": 20,
                "Strategy Fit": 10,
            },
            "bear put debit spread": {
                "Expected Move": 25,
                "Realized Volatility Rank": 20,
                "Liquidity": 20,
                "DTE": 15,
                "Delta/Probability": 15,
                "Profit/Risk": 20,
                "Strategy Fit": 10,
            },
        }
        expected_quant = {
            "put credit spread": 86,
            "call credit spread": 86,
            "bull call debit spread": 100,
            "bear put debit spread": 100,
        }

        for strategy, outlook, price_move in self.CASES:
            with self.subTest(strategy=strategy):
                trade = representative_trade(strategy)
                preferences = ScanPreferences(500, outlook, "moderate")
                passed, reasons = passes_filters(trade, preferences)
                scored = score_trade(
                    trade,
                    preferences,
                    price_move=price_move,
                    event_label="neutral",
                )

                self.assertTrue(passed)
                self.assertEqual([], reasons)
                self.assertEqual(expected_scores[strategy], scored.category_scores)
                self.assertEqual(expected_quant[strategy], scored.quant_score)
                self.assertEqual(10, strategy_fit_score(trade, preferences))
                self.assertEqual(8, scored.raw_price_move_adjustment)
                self.assertEqual(8, scored.effective_price_move_adjustment)
                self.assertEqual(
                    min(100, expected_quant[strategy] + 8),
                    scored.total_score,
                )

    def test_current_non_condor_neutral_scan_order(self):
        strategies = [case[0] for case in self.CASES]
        trades = [representative_trade(strategy) for strategy in strategies]
        passing, rejected = scan_trades(
            trades,
            ScanPreferences(500, "neutral", "moderate"),
        )
        self.assertEqual([], rejected)
        self.assertEqual(
            [
                "bull call debit spread",
                "bear put debit spread",
                "put credit spread",
                "call credit spread",
            ],
            [scored.trade.strategy for scored in passing],
        )
        self.assertEqual([92, 92, 80, 80], [scored.quant_score for scored in passing])


class CondorConstructionTests(unittest.TestCase):
    def test_two_ten_percent_wings_pass_combined_twenty_percent_rule(self):
        chain = valid_condor_chain()
        preferences = condor_preferences()
        put_wing = build_put_credit_spreads(chain, 100, None, 55, preferences)[0]
        call_wing = build_call_credit_spreads(chain, 100, None, 55, preferences)[0]
        self.assertEqual(0.10, put_wing.credit / 5)
        self.assertEqual(0.10, call_wing.credit / 5)
        self.assertFalse(passes_filters(put_wing, preferences)[0])
        self.assertFalse(passes_filters(call_wing, preferences)[0])

        condor = valid_condor(volatility_rank=55)
        self.assertEqual(1.0, condor.credit)
        self.assertEqual(0.20, condor.credit / 5)
        self.assertTrue(passes_filters(condor, preferences)[0])

    def test_call_below_old_global_top_five_pairs_with_matching_expiration(self):
        crowded_expiration = future_expiration(35)
        matching_expiration = future_expiration(42)
        high_ranked_calls = [
            wing_trade(
                "call",
                crowded_expiration,
                open_interest=2000,
                volume=500,
            )
            for _ in range(5)
        ]
        lower_ranked_matching_call = wing_trade(
            "call",
            matching_expiration,
            open_interest=200,
            volume=25,
        )
        matching_put = wing_trade("put", matching_expiration)
        preferences = condor_preferences()
        self.assertTrue(
            all(
                score_trade(call, preferences).total_score
                > score_trade(lower_ranked_matching_call, preferences).total_score
                for call in high_ranked_calls
            )
        )

        condors = combine_condor_wings(
            [matching_put],
            high_ranked_calls + [lower_ranked_matching_call],
            100,
            55,
            preferences,
        )
        self.assertEqual([matching_expiration], [trade.expiration for trade in condors])

    def test_different_expirations_are_never_combined(self):
        condors = combine_condor_wings(
            [wing_trade("put", future_expiration(30))],
            [wing_trade("call", future_expiration(37))],
            100,
            55,
            condor_preferences(),
        )
        self.assertEqual([], condors)

    def test_larger_wing_controls_maximum_risk(self):
        condor = valid_condor(call_long_strike=113)
        self.assertEqual(5.0, condor.put_short_strike - condor.put_long_strike)
        self.assertEqual(3.0, condor.call_long_strike - condor.call_short_strike)
        self.assertEqual(4.0, condor.max_risk)

    def test_diagnostics_show_wing_and_final_filter_flow(self):
        diagnostics = condor_diagnostics(
            valid_condor_chain(),
            100,
            None,
            20,
            condor_preferences(),
        )
        self.assertEqual(1, diagnostics.raw_put_wings_built)
        self.assertEqual(1, diagnostics.raw_call_wings_built)
        self.assertEqual(1, diagnostics.expirations_with_both_sides)
        self.assertEqual(1, diagnostics.pairs_attempted)
        self.assertEqual(1, diagnostics.condors_built)
        self.assertEqual(1, diagnostics.condors_passing_final_filters)
        self.assertIsNotNone(diagnostics.highest_passing_condor_score)


class CondorFilterTests(unittest.TestCase):
    def test_exact_twenty_cent_four_leg_width_passes(self):
        condor = valid_condor()
        self.assertEqual(0.20, round(condor.ask - condor.bid, 2))
        self.assertTrue(passes_filters(condor, condor_preferences())[0])

    def test_width_above_configured_four_leg_threshold_rejects(self):
        condor = replace(valid_condor(), ask=1.41)
        passed, reasons = passes_filters(condor, condor_preferences())
        self.assertFalse(passed)
        self.assertIn(
            "condor four-leg bid/ask spread is wider than configured maximum",
            reasons,
        )

    def test_both_short_strikes_outside_expected_move_pass(self):
        condor = valid_condor()
        self.assertGreaterEqual(condor.put_expected_move_cushion, 0)
        self.assertGreaterEqual(condor.call_expected_move_cushion, 0)
        self.assertTrue(passes_filters(condor, condor_preferences())[0])

    def test_put_short_inside_expected_move_has_put_specific_reason(self):
        condor = replace(
            valid_condor(),
            put_expected_move_cushion=-0.01,
            minimum_expected_move_cushion=-0.01,
        )
        passed, reasons = passes_filters(condor, condor_preferences())
        self.assertFalse(passed)
        self.assertIn(
            "condor put short strike is inside the expected move",
            reasons,
        )
        self.assertNotIn(
            "condor call short strike is inside the expected move",
            reasons,
        )

    def test_call_short_inside_expected_move_has_call_specific_reason(self):
        condor = replace(
            valid_condor(),
            call_expected_move_cushion=-0.01,
            minimum_expected_move_cushion=-0.01,
        )
        passed, reasons = passes_filters(condor, condor_preferences())
        self.assertFalse(passed)
        self.assertIn(
            "condor call short strike is inside the expected move",
            reasons,
        )
        self.assertNotIn(
            "condor put short strike is inside the expected move",
            reasons,
        )

    def test_low_realized_volatility_dia_condor_is_not_hard_rejected(self):
        condor = valid_condor(ticker="DIA", volatility_rank=20)
        passed, reasons = passes_filters(condor, condor_preferences())
        self.assertTrue(passed)
        self.assertNotIn("realized volatility rank is below 35", reasons)

    def test_illiquid_and_estimated_legs_are_rejected(self):
        illiquid_chain = valid_condor_chain()
        illiquid_chain[0] = replace(illiquid_chain[0], open_interest=100)
        self.assertEqual(
            [],
            build_iron_condor(
                illiquid_chain, 100, None, 55, condor_preferences()
            ),
        )
        illiquid_diagnostics = condor_diagnostics(
            illiquid_chain, 100, None, 55, condor_preferences()
        )
        self.assertEqual(1, illiquid_diagnostics.put_wings_rejected_for_liquidity)

        estimated_chain = valid_condor_chain()
        estimated_chain[3] = replace(
            estimated_chain[3],
            quote_source="last price estimate",
        )
        self.assertEqual(
            [],
            build_iron_condor(
                estimated_chain, 100, None, 55, condor_preferences()
            ),
        )
        estimated_diagnostics = condor_diagnostics(
            estimated_chain, 100, None, 55, condor_preferences()
        )
        self.assertEqual(1, estimated_diagnostics.wings_rejected_for_estimated_quotes)


class CondorScoringAndRankingTests(unittest.TestCase):
    def test_best_valid_condor_reaches_existing_scoring(self):
        condor = valid_condor(volatility_rank=55)
        preferences = condor_preferences()
        expected = score_trade(condor, preferences)
        passing, rejected = scan_trades([condor], preferences)
        self.assertEqual([], rejected)
        self.assertEqual(1, len(passing))
        self.assertEqual(expected.category_scores, passing[0].category_scores)
        self.assertEqual(expected.quant_score, passing[0].quant_score)
        self.assertEqual(expected.total_score, passing[0].total_score)

    def test_condors_still_use_diversification_and_execution_selection(self):
        preferences = condor_preferences()
        dia = valid_condor(ticker="DIA", volatility_rank=55)
        similar_dia = replace(
            dia,
            put_long_strike=dia.put_long_strike - 1,
            put_short_strike=dia.put_short_strike - 1,
            call_short_strike=dia.call_short_strike + 1,
            call_long_strike=dia.call_long_strike + 1,
        )
        spy = replace(dia, ticker="SPY")
        diversified = diversify_scored_trades(
            [
                score_trade(dia, preferences),
                score_trade(similar_dia, preferences),
                score_trade(spy, preferences),
            ]
        )
        self.assertEqual(1, sum(row.trade.ticker == "DIA" for row in diversified))
        selected = select_execution_candidates(diversified, limit=3)
        self.assertEqual(
            len(selected),
            len({row.trade.ticker for row in selected}),
        )


if __name__ == "__main__":
    unittest.main()
