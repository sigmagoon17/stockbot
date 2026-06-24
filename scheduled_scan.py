import os
from collections import Counter

from event_analysis import get_event_analysis
from history_tracker import append_scan_history, update_expired_history
from stock2dupe import (
    ScanPreferences,
    build_bear_put_debit_spread,
    build_bull_call_debit_spread,
    build_call_credit_spreads,
    build_iron_condor,
    build_put_credit_spreads,
    get_option_chain,
    scan_trades,
)


def select_history_candidates(scored_trades, limit: int = 25, per_ticker: int = 4):
    selected = []
    selected_ids = set()
    selected_by_strategy = Counter()
    selected_by_ticker = Counter()

    for scored in scored_trades:
        strategy = scored.trade.strategy
        if selected_by_strategy[strategy] >= 1:
            continue
        selected.append(scored)
        selected_ids.add(id(scored))
        selected_by_strategy[strategy] += 1

    for scored in scored_trades:
        ticker = scored.trade.ticker
        if id(scored) in selected_ids or selected_by_ticker[ticker] >= per_ticker:
            continue
        selected.append(scored)
        selected_ids.add(id(scored))
        selected_by_ticker[ticker] += 1
        if len(selected) == limit:
            return selected

    for scored in scored_trades:
        if id(scored) in selected_ids:
            continue
        selected.append(scored)
        selected_ids.add(id(scored))
        if len(selected) == limit:
            break

    return selected


def main() -> int:
    tickers = [
        ticker.strip().upper()
        for ticker in os.getenv(
            "SCAN_TICKERS", "AAPL,SPY,QQQ,NVDA,MSFT,COHR"
        ).split(",")
        if ticker.strip()
    ]
    preferences = ScanPreferences(
        max_risk=float(os.getenv("SCAN_MAX_RISK", "500")),
        outlook=os.getenv("SCAN_OUTLOOK", "neutral"),
        risk_tolerance=os.getenv("SCAN_RISK_TOLERANCE", "moderate"),
    )
    trades = []
    event_analyses = {}
    event_adjustments = {}
    price_moves = {}
    errors = update_expired_history(include_today=True)

    for ticker in tickers:
        try:
            (
                price,
                option_chain,
                earnings_date,
                volatility_rank,
                price_move,
            ) = get_option_chain(ticker)
            event_analysis = get_event_analysis(ticker, preferences.outlook)
            event_analyses[ticker] = event_analysis
            event_adjustments[ticker] = event_analysis.adjustment
            price_moves[ticker] = price_move

            trades.extend(
                build_iron_condor(
                    option_chain, price, earnings_date, volatility_rank, preferences
                )
            )
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
            print(f"{ticker}: fetched {len(option_chain)} contracts")
        except Exception as error:
            errors.append(f"{ticker}: {error}")

    scored_trades, _ = scan_trades(trades, preferences, event_adjustments)
    history_candidates = select_history_candidates(scored_trades)
    errors.extend(
        append_scan_history(history_candidates, event_analyses, price_moves)
    )

    print(f"Saved {len(history_candidates)} candidates from {len(scored_trades)} passing trades.")
    for error in errors:
        print(f"Warning: {error}")

    return 1 if not trades and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
