import os
import time

try:
    from alpaca_client import (
        leg_key_from_legs,
        submit_scored_multileg_orders,
        trade_multileg_order_details,
    )
except ImportError:
    leg_key_from_legs = None
    submit_scored_multileg_orders = None
    trade_multileg_order_details = None
from event_analysis import get_deep_event_analysis, get_event_analysis
from history_tracker import (
    append_alpaca_paper_orders,
    append_alpaca_paper_snapshots,
    append_scan_history,
    append_trade_snapshots,
    fetch_alpaca_paper_leg_keys,
    update_expired_history,
)
from stock2dupe import (
    EXPIRATION_COVERAGE_FAST_WEEKLY,
    ScanPreferences,
    build_bear_put_debit_spread,
    build_bull_call_debit_spread,
    build_call_credit_spreads,
    build_iron_condors_with_diagnostics,
    build_put_credit_spreads,
    get_option_chain,
    scan_trades,
    select_execution_candidates,
    select_history_candidates,
)
from scanner_tracking import new_scan_run_id
from stock_universe import prefilter_tickers

def selected_deep_analysis_tickers(
    scored_trades,
    price_moves,
    max_tickers: int = 5,
    top_trade_count: int = 10,
):
    selected = []

    for scored in scored_trades[:top_trade_count]:
        ticker = scored.trade.ticker
        if scored.total_score >= 70 and ticker not in selected:
            selected.append(ticker)

    unusual_movers = sorted(
        price_moves.items(),
        key=lambda item: abs(float(item[1].get("Move vs 20D Vol", 0) or 0)),
        reverse=True,
    )
    for ticker, move in unusual_movers:
        move_multiple = abs(float(move.get("Move vs 20D Vol", 0) or 0))
        if move_multiple >= 1.5 and ticker not in selected:
            selected.append(ticker)
        if len(selected) >= max_tickers:
            break

    return selected[:max_tickers]


def env_int(name: str, default: int) -> int:
    try:
        value = os.getenv(name)
        return int(value) if value else default
    except ValueError:
        return default


def top_unplaced_paper_candidates(scored_trades, limit: int = 3):
    if leg_key_from_legs is None or trade_multileg_order_details is None:
        return [], [
            {
                "Candidate": "Duplicate Check",
                "Symbol": "",
                "Status": "Error",
                "Message": "Alpaca multi-leg duplicate helpers are unavailable.",
            }
        ]

    existing_leg_keys, errors = fetch_alpaca_paper_leg_keys()
    selected = []
    skipped = []

    for scored in scored_trades:
        trade = scored.trade
        try:
            legs, _, _ = trade_multileg_order_details(scored)
            leg_key = leg_key_from_legs(legs)
        except ValueError as error:
            skipped.append(
                {
                    "Candidate": f"{trade.ticker} {trade.strategy}",
                    "Symbol": "",
                    "Status": "Skipped",
                    "Message": str(error),
                }
            )
            continue

        if leg_key in existing_leg_keys:
            skipped.append(
                {
                    "Candidate": (
                        f"{trade.ticker} {trade.strategy} {trade.expiration} "
                        f"score {scored.total_score}"
                    ),
                    "Symbol": "Multi-leg order",
                    "Status": "Skipped",
                    "Message": "Same expiration and legs were already paper traded.",
                }
            )
            continue

        selected.append(scored)
        existing_leg_keys.add(leg_key)
        if len(selected) == limit:
            break

    skipped.extend(
        {
            "Candidate": "Duplicate Check",
            "Symbol": "",
            "Status": "Error",
            "Message": error,
        }
        for error in errors
    )
    return selected, skipped


def main() -> int:
    tickers = [
        ticker.strip().upper()
        for ticker in os.getenv(
            "SCAN_TICKERS", "AAPL,SPY,QQQ,NVDA,MSFT,COHR"
        ).split(",")
        if ticker.strip()
    ]
    if os.getenv("SCAN_PREFILTER", "").lower() in {"1", "true", "yes"}:
        selected_tickers, prefilter_results = prefilter_tickers(
            tickers,
            max_selected=env_int("SCAN_PREFILTER_MAX_TICKERS", 35),
            min_price=float(os.getenv("SCAN_PREFILTER_MIN_PRICE", "20")),
            min_average_volume=env_int("SCAN_PREFILTER_MIN_VOLUME", 1000000),
            min_volatility_rank=float(
                os.getenv("SCAN_PREFILTER_MIN_VOL_RANK", "20")
            ),
        )
        print(
            f"Prefilter selected {len(selected_tickers)} of {len(tickers)} tickers."
        )
        for result in sorted(
            prefilter_results,
            key=lambda item: item.score,
            reverse=True,
        )[:10]:
            print(
                f"Prefilter {result.ticker}: "
                f"{'pass' if result.passed else 'skip'} "
                f"score {result.score} - {result.reason}"
            )
        tickers = selected_tickers

    if not tickers:
        print("No tickers available after prefilter.")
        return 1

    preferences = ScanPreferences(
        max_risk=float(os.getenv("SCAN_MAX_RISK", "500")),
        outlook=os.getenv("SCAN_OUTLOOK", "neutral"),
        risk_tolerance=os.getenv("SCAN_RISK_TOLERANCE", "moderate"),
        price_move_mode=os.getenv("PRICE_MOVE_MODE", "Full"),
        expiration_coverage=os.getenv(
            "SCAN_EXPIRATION_COVERAGE",
            EXPIRATION_COVERAGE_FAST_WEEKLY,
        ),
    )
    trades = []
    event_analyses = {}
    event_adjustments = {}
    event_labels = {}
    price_moves = {}
    errors = update_expired_history()
    scan_started = time.perf_counter()

    for ticker in tickers:
        try:
            (
                price,
                option_chain,
                earnings_date,
                volatility_rank,
                price_move,
            ) = get_option_chain(
                ticker,
                expiration_coverage=preferences.expiration_coverage,
            )
            event_analysis = get_event_analysis(ticker, preferences.outlook)
            event_analyses[ticker] = event_analysis
            event_adjustments[ticker] = event_analysis.adjustment
            event_labels[ticker] = event_analysis.label
            price_moves[ticker] = price_move
            condor_result = build_iron_condors_with_diagnostics(
                option_chain, price, earnings_date, volatility_rank, preferences
            )

            trades.extend(condor_result.condors)
            trades.extend(
                build_call_credit_spreads(
                    option_chain, price, earnings_date, volatility_rank, preferences
                )
            )
            trades.extend(
                build_put_credit_spreads(
                    option_chain, price, earnings_date, volatility_rank, preferences
                )
            )
            trades.extend(
                build_bull_call_debit_spread(
                    option_chain, price, earnings_date, volatility_rank, preferences
                )
            )
            trades.extend(
                build_bear_put_debit_spread(
                    option_chain, price, earnings_date, volatility_rank, preferences
                )
            )
            print(
                f"{ticker}: fetched {len(option_chain)} contracts; "
                f"condors built {condor_result.diagnostics.built_condors}; "
                f"blocker: {condor_result.diagnostics.top_reason}; "
                f"coverage: {preferences.expiration_coverage}"
            )
        except Exception as error:
            errors.append(f"{ticker}: {error}")

    scored_trades, _ = scan_trades(
        trades, preferences, event_adjustments, price_moves, event_labels
    )
    deep_tickers = selected_deep_analysis_tickers(scored_trades, price_moves)
    for ticker in deep_tickers:
        deep_analysis = get_deep_event_analysis(ticker, preferences.outlook)
        event_analyses[ticker] = deep_analysis
        event_adjustments[ticker] = deep_analysis.adjustment
        event_labels[ticker] = deep_analysis.label
        print(
            f"{ticker}: deep news analysis {deep_analysis.label} "
            f"({deep_analysis.adjustment:+d})"
        )

    if deep_tickers:
        scored_trades, _ = scan_trades(
            trades, preferences, event_adjustments, price_moves, event_labels
        )

    execution_candidates = select_execution_candidates(scored_trades, limit=3)
    history_candidates = select_history_candidates(scored_trades)
    scan_run_id = new_scan_run_id()
    errors.extend(
        append_scan_history(
            history_candidates,
            event_analyses,
            price_moves,
            execution_candidates=execution_candidates,
            scan_run_id=scan_run_id,
        )
    )
    if os.getenv("ALPACA_AUTO_PAPER_TRADE", "").lower() in {"1", "true", "yes"}:
        if submit_scored_multileg_orders is None:
            print("Warning: Alpaca paper trading helper is unavailable.")
        else:
            paper_candidates, duplicate_results = top_unplaced_paper_candidates(
                execution_candidates, limit=3
            )
            paper_results = submit_scored_multileg_orders(
                paper_candidates,
                quantity=env_int("ALPACA_PAPER_TRADE_QUANTITY", 1),
                limit=3,
                exit_policy=os.getenv("ALPACA_PAPER_EXIT_POLICY", "none"),
                scan_run_id=scan_run_id,
            )
            paper_results = duplicate_results + paper_results
            errors.extend(append_alpaca_paper_orders(paper_results))
            for result in paper_results:
                print(
                    "Alpaca paper trade: "
                    f"{result['Status']} | {result['Symbol']} | {result['Message']}"
                )
    errors.extend(append_trade_snapshots())
    errors.extend(append_alpaca_paper_snapshots())

    print(f"Saved {len(history_candidates)} candidates from {len(scored_trades)} passing trades.")
    print(f"Total scan elapsed: {time.perf_counter() - scan_started:.3f} seconds")
    for error in errors:
        print(f"Warning: {error}")

    return 1 if not trades and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
