import os
from collections import Counter

from event_analysis import get_deep_event_analysis, get_event_analysis
from history_tracker import (
    append_scan_history,
    append_trade_snapshots,
    update_expired_history,
)
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
    event_labels = {}
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
            event_labels[ticker] = event_analysis.label
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

    history_candidates = select_history_candidates(scored_trades)
    errors.extend(
        append_scan_history(history_candidates, event_analyses, price_moves)
    )
    errors.extend(append_trade_snapshots())

    print(f"Saved {len(history_candidates)} candidates from {len(scored_trades)} passing trades.")
    for error in errors:
        print(f"Warning: {error}")

    return 1 if not trades and errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
