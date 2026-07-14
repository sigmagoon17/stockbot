from collections import Counter
from datetime import date
import hmac
import os
import pandas as pd
import streamlit as st
import datetime
import supabase
import time
import yfinance as yf
from types import SimpleNamespace

from alpaca_client import (
    get_alpaca_account,
    get_alpaca_positions,
    get_recent_alpaca_orders,
    recent_alpaca_order_results,
)

try:
    from alpaca_client import (
        leg_key_from_legs,
        submit_multileg_order,
        submit_scored_multileg_orders,
        trade_multileg_order_details,
    )
except ImportError:
    leg_key_from_legs = None
    submit_multileg_order = None
    submit_scored_multileg_orders = None
    trade_multileg_order_details = None
from history_tracker import (
    add_manual_position,
    append_scan_history as save_history,
    close_candidate,
    close_manual_position,
    delete_manual_position,
    fetch_completed_history,
    fetch_open_history,
    manual_position_rows_with_marks,
    update_expired_history as update_history,
)

try:
    from history_tracker import (
        append_alpaca_paper_orders,
        append_alpaca_paper_snapshots,
        fetch_alpaca_paper_leg_keys,
        fetch_alpaca_paper_orders,
        fetch_alpaca_paper_snapshots,
    )
except ImportError:
    def append_alpaca_paper_orders(order_results):
        return [
            "Alpaca paper order tracking helpers are unavailable while the app finishes redeploying."
        ]

    def append_alpaca_paper_snapshots():
        return [
            "Alpaca paper snapshot helpers are unavailable while the app finishes redeploying."
        ]

    def fetch_alpaca_paper_leg_keys():
        return set(), [
            "Alpaca duplicate-check helpers are unavailable while the app finishes redeploying."
        ]

    def fetch_alpaca_paper_orders(limit=250):
        return [], [
            "Alpaca paper order chart helpers are unavailable while the app finishes redeploying."
        ]

    def fetch_alpaca_paper_snapshots(limit=500):
        return [], [
            "Alpaca paper snapshot chart helpers are unavailable while the app finishes redeploying."
        ]

from stock2dupe import (
    CONTRACT_MULTIPLIER,
    ScanPreferences,
    build_call_credit_spreads,
    build_iron_condor,
    build_put_credit_spreads,
    build_bull_call_debit_spread,
    build_bear_put_debit_spread,
    condor_diagnostics,
    get_option_chain,
    scan_trades,
)
from stock_universe import prefilter_tickers

try:
    from event_analysis import (
        analyze_candidate_setup,
        get_deep_event_analysis,
        get_event_analysis,
    )
except ImportError:
    from event_analysis import get_event_analysis

    analyze_candidate_setup = None
    get_deep_event_analysis = None

st.set_page_config(page_title="Options Scanner", layout="wide")

EVENT_ANALYSIS_SUCCESS_TTL_SECONDS = 6 * 60 * 60
EVENT_ANALYSIS_FAILURE_TTL_SECONDS = 5 * 60
DEFAULT_TICKERS = "AAPL, SPY, QQQ, NVDA, MSFT, COHR"
LARGE_PRESET_TICKERS = (
    "SPY, QQQ, IWM, SMH, TQQQ, SQQQ, XLF, XLK, XLE, DIA, "
    "NVDA, TSLA, AAPL, MSFT, META, AMZN, GOOGL, AMD, AVGO, NFLX, "
    "PLTR, ORCL, TSM, MU, ARM, SMCI, QCOM, ASML, AMAT, LRCX, "
    "COIN, HOOD, SOFI, JPM, BAC, GS, WFC, SCHW, "
    "CRWD, PANW, SNOW, SHOP, DDOG, NET, ZS, MDB, "
    "LLY, UNH, ABBV, ISRG, "
    "UBER, ABNB, RCL, CAT, GE, XOM, CVX, RIOT, MARA, RKLB"
)


@st.cache_resource
def event_analysis_cache():
    return {}


@st.cache_resource
def candidate_analysis_cache():
    return {}


@st.cache_resource
def deep_event_analysis_cache():
    return {}


def get_cached_event_analysis(ticker: str, outlook: str):
    cache = event_analysis_cache()
    cache_key = (ticker, outlook)
    now = time.monotonic()
    cached = cache.get(cache_key)
    if cached and now - cached["created_at"] < cached["ttl"]:
        return cached["analysis"]

    analysis = get_event_analysis(ticker, outlook)
    cache[cache_key] = {
        "analysis": analysis,
        "created_at": now,
        "ttl": (
            EVENT_ANALYSIS_SUCCESS_TTL_SECONDS
            if analysis.available
            else EVENT_ANALYSIS_FAILURE_TTL_SECONDS
        ),
    }
    return analysis


def get_cached_deep_event_analysis(ticker: str, outlook: str):
    if get_deep_event_analysis is None:
        return get_cached_event_analysis(ticker, outlook)

    cache = deep_event_analysis_cache()
    cache_key = (ticker, outlook)
    now = time.monotonic()
    cached = cache.get(cache_key)
    if cached and now - cached["created_at"] < cached["ttl"]:
        return cached["analysis"]

    analysis = get_deep_event_analysis(ticker, outlook)
    cache[cache_key] = {
        "analysis": analysis,
        "created_at": now,
        "ttl": (
            EVENT_ANALYSIS_SUCCESS_TTL_SECONDS
            if analysis.available
            else EVENT_ANALYSIS_FAILURE_TTL_SECONDS
        ),
    }
    return analysis


def candidate_analysis_key(scored):
    trade = scored.trade
    return (
        trade.ticker,
        trade.strategy,
        trade.expiration,
        trade.long_strike,
        trade.short_strike,
        trade.entry_type,
        scored.total_score,
        scored.quant_score,
        scored.event_adjustment,
        scored.price_move_adjustment,
    )


def get_cached_candidate_analysis(scored, event_analysis, price_move):
    cache = candidate_analysis_cache()
    cache_key = candidate_analysis_key(scored)
    if cache_key not in cache:
        if analyze_candidate_setup is None:
            cache[cache_key] = SimpleNamespace(
                verdict="watch",
                confidence="low",
                summary="AI candidate review is temporarily unavailable while the app finishes redeploying.",
                strengths=[],
                risks=[],
                action="Run the scan again after Streamlit finishes updating the app.",
                available=False,
            )
        else:
            cache[cache_key] = analyze_candidate_setup(
                scored, event_analysis, price_move
            )
    return cache[cache_key]


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


def apply_deep_event_analysis(
    scored_trades,
    trades,
    preferences,
    event_analyses,
    event_adjustments,
    event_labels,
    price_moves,
):
    deep_tickers = selected_deep_analysis_tickers(scored_trades, price_moves)
    if not deep_tickers:
        return scored_trades, event_analyses, event_adjustments, event_labels, []

    messages = []
    for ticker in deep_tickers:
        deep_analysis = get_cached_deep_event_analysis(ticker, preferences.outlook)
        event_analyses[ticker] = deep_analysis
        event_adjustments[ticker] = deep_analysis.adjustment
        event_labels[ticker] = deep_analysis.label
        messages.append(
            f"{ticker}: deep news analysis {deep_analysis.label} "
            f"({deep_analysis.adjustment:+d})"
        )

    rescored_trades, _ = scan_trades(
        trades, preferences, event_adjustments, price_moves, event_labels
    )
    return rescored_trades, event_analyses, event_adjustments, event_labels, messages


st.markdown(
    """
    <style>
        .stApp {
            background: #f6f8f6;
            color: #15221e;
        }
        .block-container {
            max-width: 1440px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }
        [data-testid="stSidebar"] {
            background: #eef2ef;
            border-right: 1px solid #d8e0db;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #d8e0db;
            border-radius: 6px;
            padding: 0.7rem 0.85rem;
        }
        [data-testid="stMetricLabel"] {
            color: #53645c;
        }
        [data-testid="stMetricValue"] {
            color: #0c604e;
        }
        div.stButton > button,
        div.stDownloadButton > button {
            border-radius: 6px;
            font-weight: 600;
        }
        div.stButton > button[kind="primary"] {
            background: #000080;
            border-color: #000080;
        }
        div.stButton > button[kind="primary"]:hover {
            background: #123499;
            border-color: #123499;
        }
        [data-baseweb="tab-list"] {
            gap: 1.25rem;
            border-bottom: 1px solid #d8e0db;
        }
        [data-baseweb="tab"] {
            height: 42px;
            padding: 0 0.2rem;
            font-weight: 600;
        }
        [data-baseweb="tab-highlight"] {
            background-color: #0c604e;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def select_top_candidates(scored_trades, per_ticker: int = 3):
    selected = []
    selected_by_ticker = Counter()

    for scored in scored_trades:
        ticker = scored.trade.ticker
        if selected_by_ticker[ticker] >= per_ticker:
            continue
        selected_by_ticker[ticker] += 1
        selected.append(scored)

    return selected


def select_history_candidates(
    scored_trades, limit: int = 25, per_ticker: int = 4, per_strategy: int = 1
):
    selected = []
    selected_ids = set()
    selected_by_strategy = Counter()

    # Reserve history slots for each strategy before filling by overall rank.
    for scored in scored_trades:
        strategy = scored.trade.strategy
        if selected_by_strategy[strategy] >= per_strategy:
            continue
        selected.append(scored)
        selected_ids.add(id(scored))
        selected_by_strategy[strategy] += 1
        if len(selected) == limit:
            return selected

    for scored in select_top_candidates(scored_trades, per_ticker=per_ticker):
        if id(scored) in selected_ids:
            continue
        selected.append(scored)
        selected_ids.add(id(scored))
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


def scan_watchlist(tickers: list[str], preferences: ScanPreferences):
    trades = []
    ticker_data = []
    condor_diagnostic_rows = []
    errors = []
    event_adjustments = {}
    event_labels = {}
    event_analyses = {}
    price_moves = {}
    progress = st.progress(0, text="Preparing scan")

    for index, ticker in enumerate(tickers, start=1):
        progress.progress(
            index / len(tickers), text=f"Fetching {ticker} option data ({index}/{len(tickers)})"
        )
        try:
            (
                price,
                option_chain,
                earnings_date,
                volatility_rank,
                price_move,
            ) = get_option_chain(
                ticker,
                test_expiration=preferences.test_expiration,
                nearest_expiration=preferences.nearest_expiration,
            )
            price_moves[ticker] = price_move
            ticker_data.append(
                {
                    "Ticker": ticker,
                    "Price": price,
                    "Contracts": len(option_chain),
                    "Volatility Rank": volatility_rank,
                    **price_move,
                    "Earnings Date": earnings_date.isoformat() if earnings_date else "None",
                    "Expiration Used": (
                        "5 nearest"
                        if preferences.nearest_expiration
                        else (
                            preferences.test_expiration.isoformat()
                            if preferences.test_expiration is not None
                            else "All"
                        )
                    ),
                }
            )
            event_analysis = get_cached_event_analysis(ticker, preferences.outlook)
            event_analyses[ticker] = event_analysis
            event_adjustments[ticker] = event_analysis.adjustment
            event_labels[ticker] = event_analysis.label
            ticker_data[-1].update(
                {
                    "Event Label": event_analysis.label.title(),
                    "Event Adjustment": event_analysis.adjustment,
                }
            )
            condor_diagnostic_rows.append(
                condor_diagnostics(
                    option_chain, price, earnings_date, volatility_rank, preferences
                )
            )
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
            
        except Exception as error:
            errors.append(f"{ticker}: {error}")

    progress.empty()
    scored_trades, rejected_trades = scan_trades(
        trades, preferences, event_adjustments, price_moves, event_labels
    )
    deep_messages = []
    (
        scored_trades,
        event_analyses,
        event_adjustments,
        event_labels,
        deep_messages,
    ) = apply_deep_event_analysis(
        scored_trades,
        trades,
        preferences,
        event_analyses,
        event_adjustments,
        event_labels,
        price_moves,
    )
    if deep_messages:
        scored_trades, rejected_trades = scan_trades(
            trades, preferences, event_adjustments, price_moves, event_labels
        )
        for row in ticker_data:
            event_analysis = event_analyses.get(row["Ticker"])
            if event_analysis is not None:
                row["Event Label"] = event_analysis.label.title()
                row["Event Adjustment"] = event_analysis.adjustment
                row["News Depth"] = (
                    "Deep"
                    if row["Ticker"] in {
                        message.split(":", 1)[0] for message in deep_messages
                    }
                    else row.get("News Depth", "Basic")
                )
    for row in ticker_data:
        row.setdefault("News Depth", "Basic")
    return (
        scored_trades,
        rejected_trades,
        trades,
        ticker_data,
        condor_diagnostic_rows,
        errors,
        event_analyses,
        price_moves,
    )


def prefilter_result_rows(results):
    rows = []
    for result in results:
        rows.append(
            {
                "Ticker": result.ticker,
                "Passed": result.passed,
                "Prefilter Score": result.score,
                "Price": result.price,
                "20D Avg Volume": result.average_volume,
                "Volatility Rank": result.volatility_rank,
                "1D Move %": result.one_day_move_percent,
                "5D Move %": result.five_day_move_percent,
                "Reason": result.reason,
            }
        )
    return rows


def expiration_close(ticker: str, expiration: date) -> float | None:
    history = yf.Ticker(ticker).history(
        start=expiration.isoformat(),
        end=(expiration + datetime.timedelta(days=1)).isoformat(),
        auto_adjust=False,
    )
    if history.empty:
        return None
    return round(float(history["Close"].iloc[-1]), 2)


def expiration_pnl(row, closing_price: float) -> float | None:
    strategy = row["strategy"]
    long_strike = float(row["long_strike"])
    short_strike = float(row["short_strike"])
    width = abs(long_strike - short_strike)
    max_risk = float(row["max_risk"])

    if strategy == "bull call debit spread":
        spread_value = max(0, min(closing_price - long_strike, width))
        return round(spread_value * CONTRACT_MULTIPLIER - max_risk, 2)
    if strategy == "bear put debit spread":
        spread_value = max(0, min(long_strike - closing_price, width))
        return round(spread_value * CONTRACT_MULTIPLIER - max_risk, 2)

    credit = float(row["credit"])
    if strategy == "put credit spread":
        spread_loss = max(0, min(short_strike - closing_price, width))
        return round(credit - spread_loss * CONTRACT_MULTIPLIER, 2)
    if strategy == "call credit spread":
        spread_loss = max(0, min(closing_price - short_strike, width))
        return round(credit - spread_loss * CONTRACT_MULTIPLIER, 2)
    if strategy == "iron condor":
        put_short = float(row["put_short_strike"])
        put_long = float(row["put_long_strike"])
        call_short = float(row["call_short_strike"])
        call_long = float(row["call_long_strike"])

        put_width = put_short - put_long
        call_width = call_long - call_short

        put_loss = max(0, min(put_short - closing_price, put_width))
        call_loss = max(0, min(closing_price - call_short, call_width))

        credit = float(row["credit"])
        return round(
            credit - (put_loss + call_loss) * CONTRACT_MULTIPLIER,
            2,
        )
    return None


def update_expired_history() -> list[str]:
    errors = []
    closing_prices = {}
    try:
        response = (
            supabase.table("scan_history")
            .select("*")
            .eq("expiration_status", "open")
            .lt("expiration", date.today().isoformat())
            .execute()
        )
    except Exception as error:
        return [f"Could not read expiration results from Supabase: {error}"]

    for row in response.data:
        expiration = date.fromisoformat(row["expiration"])

        price_key = (row["ticker"], expiration)
        try:
            if price_key not in closing_prices:
                closing_prices[price_key] = expiration_close(*price_key)
            closing_price = closing_prices[price_key]
        except Exception as error:
            errors.append(f"Could not update {row['ticker']} expiration result: {error}")
            continue

        if closing_price is None:
            continue

        pnl = expiration_pnl(row, closing_price)
        if pnl is None:
            update_values = {
                "expiration_close": closing_price,
                "expiration_status": "manual review",
            }
        else:
            update_values = {
                "expiration_close": closing_price,
                "expiration_status": "expired",
                "expiration_pnl": pnl,
            }

        try:
            (
                supabase.table("scan_history")
                .update(update_values)
                .eq("id", row["id"])
                .execute()
            )
        except Exception as error:
            errors.append(f"Could not save {row['ticker']} expiration result: {error}")

    return errors


def candidate_row(scored):
    trade = scored.trade
    return {
        "Ticker": trade.ticker,
        "Strategy": trade.strategy.title(),
        "Expiration": trade.expiration,
        "Short Strike": round(trade.short_strike, 2),
        "Long Strike": round(trade.long_strike, 2),
        "Entry Type": trade.entry_type.title(),
        "Credit": (
            round(trade.credit * CONTRACT_MULTIPLIER, 2)
            if trade.entry_type == "credit"
            else None
        ),
        "Debit": (
            round(trade.max_risk * CONTRACT_MULTIPLIER, 2)
            if trade.entry_type == "debit"
            else None
        ),
        "Max Profit": (
            round(trade.max_profit * CONTRACT_MULTIPLIER, 2)
            if trade.entry_type == "debit"
            else None
        ),
        "Max Risk": round(trade.max_risk * CONTRACT_MULTIPLIER, 2),
        "Setup Score": scored.total_score,
        "Quant Score": scored.quant_score,
        "Event Adjustment": scored.event_adjustment,
        "Price Move Adjustment": scored.price_move_adjustment,
        "Move Setup": scored.price_move_style,
        "Risk Level": scored.risk_level,
        "Volatility Rank": round(trade.volatility_rank, 1),
    }


def candidate_rows(scored_trades):
    return [candidate_row(scored) for scored in select_top_candidates(scored_trades)]


def candidate_column_config():
    return {
        "Credit": st.column_config.NumberColumn(
            "Credit", help="Premium received per spread.", format="$%.2f"
        ),
        "Debit": st.column_config.NumberColumn(
            "Debit", help="Amount paid to open the spread.", format="$%.2f"
        ),
        "Max Profit": st.column_config.NumberColumn(
            "Max Profit", help="Maximum profit per spread at expiration.", format="$%.2f"
        ),
        "Max Risk": st.column_config.NumberColumn(
            "Max Risk", help="Maximum loss per spread at expiration.", format="$%.2f"
        ),
        "Setup Score": st.column_config.NumberColumn(
            "Setup Score",
            help="Final score after quant rules, event analysis, and recent price movement.",
            format="%d / 100",
        ),
        "Quant Score": st.column_config.NumberColumn(
            "Quant Score",
            help="Score from the scanner's mathematical filters before event analysis.",
            format="%d / 100",
        ),
        "Event Adjustment": st.column_config.NumberColumn(
            "Event Adjustment",
            help="Score change from the event-analysis layer. It is zero until event analysis is connected.",
            format="%d",
        ),
        "Price Move Adjustment": st.column_config.NumberColumn(
            "Price Move Adjustment",
            help="Score change from recent stock movement. Directional strategies get credit when the move agrees with them and lose points when it does not.",
            format="%d",
        ),
        "Move Setup": st.column_config.TextColumn(
            "Move Setup",
            help="Whether recent price movement is being treated as trend continuation, mean reversion, or normal movement.",
        ),
        "Volatility Rank": st.column_config.NumberColumn(
            "Volatility Rank",
            help="Current realized volatility compared with the last year of price movement.",
            format="%.1f",
        ),
    }


def paper_trade_key(scored) -> str:
    trade = scored.trade
    long_key = f"{float(trade.long_strike):.3f}".replace(".", "p")
    short_key = f"{float(trade.short_strike):.3f}".replace(".", "p")
    strategy_key = trade.strategy.replace(" ", "-")
    return (
        f"{trade.ticker}-{strategy_key}-{trade.expiration}-"
        f"{long_key}-{short_key}-{date.today().isoformat()}"
    )


def paper_trade_scan_candidates(
    scored_candidates,
    quantity: int,
    limit: int,
) -> list[dict]:
    paper_traded_keys = st.session_state.setdefault("paper_traded_scan_keys", set())
    fresh_candidates = []
    skipped_results = []
    for scored in scored_candidates[:limit]:
        key = paper_trade_key(scored)
        if key in paper_traded_keys:
            trade = scored.trade
            skipped_results.append(
                {
                    "Candidate": (
                        f"{trade.ticker} {trade.strategy} {trade.expiration} "
                        f"score {scored.total_score}"
                    ),
                    "Symbol": "Multi-leg order",
                    "Status": "Skipped",
                    "Message": "Already submitted during this app session.",
                }
            )
            continue
        fresh_candidates.append(scored)
        paper_traded_keys.add(key)

    if submit_scored_multileg_orders is not None:
        return skipped_results + submit_scored_multileg_orders(
            fresh_candidates,
            quantity=quantity,
            limit=limit,
        )

    if not fresh_candidates and not skipped_results:
        return [
            {
                "Candidate": "Latest Scan",
                "Symbol": "",
                "Status": "Skipped",
                "Message": "No candidates were available for paper trading.",
            }
        ]

    return skipped_results + [
        {
            "Candidate": "Latest Scan",
            "Symbol": "",
            "Status": "Error",
            "Message": "Alpaca multi-leg helper is unavailable.",
        }
    ]


def top_unplaced_paper_candidates(scored_trades, limit: int = 3):
    if leg_key_from_legs is None or trade_multileg_order_details is None:
        return [], [
            {
                "Candidate": "Duplicate Check",
                "Symbol": "",
                "Status": "Error",
                "Message": "Alpaca multi-leg duplicate helpers are unavailable while the app finishes redeploying.",
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

    if errors:
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


def symbols_from_leg_key(leg_key: str | None) -> list[str]:
    if not leg_key:
        return []
    return [
        leg_part.split(":", 1)[0]
        for leg_part in leg_key.split("|")
        if leg_part
    ]


def grouped_alpaca_spread_rows(positions: list[dict], paper_history: list[dict]):
    positions_by_symbol = {
        position.get("symbol"): position
        for position in positions
        if position.get("symbol")
    }
    grouped_rows = []

    for order in paper_history:
        symbols = symbols_from_leg_key(order.get("leg_key"))
        if not symbols:
            continue

        matched_positions = [
            positions_by_symbol[symbol]
            for symbol in symbols
            if symbol in positions_by_symbol
        ]
        current_value = sum(
            float(position.get("market_value") or 0)
            for position in matched_positions
        )
        unrealized_pnl = sum(
            float(position.get("unrealized_pl") or 0)
            for position in matched_positions
        )
        grouped_rows.append(
            {
                "Ticker": order.get("ticker"),
                "Strategy": order.get("strategy"),
                "Expiration": order.get("expiration"),
                "Score": order.get("setup_score"),
                "Entry Type": order.get("entry_type"),
                "Limit": order.get("limit_price"),
                "Qty": order.get("quantity"),
                "Current Value": round(current_value, 2),
                "Unrealized P/L": round(unrealized_pnl, 2),
                "Matched Legs": f"{len(matched_positions)}/{len(symbols)}",
                "Status": (
                    "Open"
                    if matched_positions
                    else order.get("status", "No open legs")
                ),
            }
        )

    return grouped_rows


def debit_candidate_rows(scored_trades):
    rows = []
    debit_trades = [
        scored for scored in scored_trades if scored.trade.entry_type == "debit"
    ]

    for scored in debit_trades[:3]:
        trade = scored.trade
        rows.append(
            {
                "Ticker": trade.ticker,
                "Strategy": trade.strategy.title(),
                "Expiration": trade.expiration,
                "Long Strike": trade.long_strike,
                "Short Strike": trade.short_strike,
                "Debit": round(trade.max_risk * CONTRACT_MULTIPLIER, 2),
                "Max Profit": round(trade.max_profit * CONTRACT_MULTIPLIER, 2),
                "Setup Score": scored.total_score,
                "Quant Score": scored.quant_score,
                "Event Adjustment": scored.event_adjustment,
                "Price Move Adjustment": scored.price_move_adjustment,
                "Move Setup": scored.price_move_style,
                "Risk Level": scored.risk_level,
                "Volatility Rank": round(trade.volatility_rank, 1),
            }
        )

    return rows


def credit_candidate_rows(scored_trades):
    rows = []
    credit_trades = [
        scored for scored in scored_trades if scored.trade.entry_type == "credit"
    ]

    for scored in credit_trades[:3]:
        trade = scored.trade
        rows.append(
            {
                "Ticker": trade.ticker,
                "Strategy": trade.strategy.title(),
                "Expiration": trade.expiration,
                "Short Strike": trade.short_strike,
                "Long Strike": trade.long_strike,
                "Credit": round(trade.credit * CONTRACT_MULTIPLIER, 2),
                "Max Risk": round(trade.max_risk * CONTRACT_MULTIPLIER, 2),
                "Setup Score": scored.total_score,
                "Quant Score": scored.quant_score,
                "Event Adjustment": scored.event_adjustment,
                "Price Move Adjustment": scored.price_move_adjustment,
                "Move Setup": scored.price_move_style,
                "Risk Level": scored.risk_level,
                "Volatility Rank": round(trade.volatility_rank, 1),
            }
        )

    return rows


def render_scan_output(scan_output):
    scored_trades = scan_output["scored_trades"]
    rejected_trades = scan_output["rejected_trades"]
    trades = scan_output["trades"]
    ticker_data = scan_output["ticker_data"]
    condor_diagnostic_rows = scan_output.get("condor_diagnostics", [])
    errors = scan_output["errors"]
    event_analyses = scan_output["event_analyses"]
    history_candidates = scan_output["history_candidates"]
    paper_order_results = scan_output.get("paper_order_results", [])
    prefilter_rows = scan_output.get("prefilter_rows", [])
    prefilter_selected_tickers = scan_output.get("prefilter_selected_tickers", [])
    original_ticker_count = scan_output.get("original_ticker_count", len(ticker_data))

    top_score = scored_trades[0].total_score if scored_trades else None
    metric_candidates, metric_score, metric_tracked, metric_tickers = st.columns(4)
    metric_candidates.metric("Passing Candidates", len(scored_trades))
    metric_score.metric("Highest Score", f"{top_score}/100" if top_score else "None")
    metric_tracked.metric("Saved to History", len(history_candidates))
    metric_tickers.metric(
        "Tickers Scanned",
        len(ticker_data),
        delta=(
            f"from {original_ticker_count}"
            if prefilter_rows and original_ticker_count != len(ticker_data)
            else None
        ),
    )

    if errors:
        for error in errors:
            st.warning(error)

    if paper_order_results:
        st.subheader("Alpaca Paper Trade Results")
        st.dataframe(
            pd.DataFrame(paper_order_results),
            width="stretch",
            hide_index=True,
        )

    candidate_analyses = scan_output.get("candidate_analyses", {})
    if candidate_analyses:
        st.subheader("AI Review Of Top Candidates")
        for index, scored in enumerate(scored_trades[:3], start=1):
            trade = scored.trade
            analysis = candidate_analyses.get(candidate_analysis_key(scored))
            if analysis is None:
                continue
            with st.expander(
                f"{index}. {trade.ticker} {trade.strategy.title()} | "
                f"{analysis.verdict.title()} | {analysis.confidence.title()} confidence",
                expanded=index == 1,
            ):
                st.write(analysis.summary)
                strength_column, risk_column = st.columns(2)
                with strength_column:
                    st.markdown("**Strengths**")
                    for strength in analysis.strengths:
                        st.write(f"- {strength}")
                with risk_column:
                    st.markdown("**Risks**")
                    for risk in analysis.risks:
                        st.write(f"- {risk}")
                st.caption(analysis.action)

    candidates_tab, market_tab, diagnostics_tab = st.tabs(
        ["Candidates", "Market Data", "Diagnostics"]
    )

    with candidates_tab:
        st.subheader("Top Candidates")
        candidates = candidate_rows(scored_trades)
        if candidates:
            st.dataframe(
                pd.DataFrame(candidates),
                width="stretch",
                hide_index=True,
                column_config=candidate_column_config(),
            )
            top_25_csv = pd.DataFrame(
                [candidate_row(scored) for scored in history_candidates]
            ).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Tracked Candidates CSV",
                data=top_25_csv,
                file_name="top_25_options_candidates.csv",
                mime="text/csv",
                width="content",
            )

            st.subheader("Candidate Details")
            for scored in select_top_candidates(scored_trades):
                trade = scored.trade
                with st.expander(
                    f"{trade.ticker} | {trade.strategy.title()} | "
                    f"Score {scored.total_score}/100"
                ):
                    st.write(scored.explanation)
                    event_analysis = event_analyses.get(trade.ticker)
                    if event_analysis:
                        st.subheader("AI Event View")
                        event_label, event_adjustment, event_confidence = st.columns(3)
                        event_label.metric("Event Label", event_analysis.label.title())
                        event_adjustment.metric(
                            "Event Adjustment",
                            f"{event_analysis.adjustment:+d}",
                        )
                        event_confidence.metric(
                            "Confidence", event_analysis.confidence.title()
                        )
                        st.write(event_analysis.summary)
                    score_rows = pd.DataFrame(
                        [
                            {"Score Area": area, "Points": points}
                            for area, points in scored.category_scores.items()
                        ]
                    )
                    st.dataframe(score_rows, width="content", hide_index=True)
        else:
            st.info("No candidates passed the current filters.")

        debit_column, credit_column = st.columns(2)
        with debit_column:
            st.subheader("Best Debit Spreads")
            debit_candidates = debit_candidate_rows(scored_trades)
            if debit_candidates:
                st.dataframe(
                    pd.DataFrame(debit_candidates),
                    width="stretch",
                    hide_index=True,
                    column_config=candidate_column_config(),
                )
            else:
                st.info("No debit spreads passed.")

        with credit_column:
            st.subheader("Best Credit Spreads")
            credit_candidates = credit_candidate_rows(scored_trades)
            if credit_candidates:
                st.dataframe(
                    pd.DataFrame(credit_candidates),
                    width="stretch",
                    hide_index=True,
                    column_config=candidate_column_config(),
                )
            else:
                st.info("No credit spreads passed.")

    with market_tab:
        if prefilter_rows:
            st.subheader("Broad Universe Prefilter")
            st.caption(
                f"Selected {len(prefilter_selected_tickers)} of "
                f"{original_ticker_count} input tickers for full option-chain scanning."
            )
            st.dataframe(
                pd.DataFrame(prefilter_rows).sort_values(
                    ["Passed", "Prefilter Score"], ascending=[False, False]
                ),
                width="stretch",
                hide_index=True,
                column_config={
                    "Price": st.column_config.NumberColumn(format="$%.2f"),
                    "20D Avg Volume": st.column_config.NumberColumn(format="%d"),
                    "Volatility Rank": st.column_config.NumberColumn(format="%.1f"),
                    "1D Move %": st.column_config.NumberColumn(format="%.2f%%"),
                    "5D Move %": st.column_config.NumberColumn(format="%.2f%%"),
                },
            )

        st.subheader("Market Data")
        st.dataframe(
            pd.DataFrame(ticker_data),
            width="stretch",
            hide_index=True,
            column_config={
                "1D Move %": st.column_config.NumberColumn(format="%.2f%%"),
                "5D Move %": st.column_config.NumberColumn(format="%.2f%%"),
                "Move vs 20D Vol": st.column_config.NumberColumn(format="%.1fx"),
            },
        )

    with diagnostics_tab:
        st.subheader("Condor Diagnostics")
        if condor_diagnostic_rows:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Ticker": row.ticker,
                            "Put Spreads Built": row.put_spreads_built,
                            "Call Spreads Built": row.call_spreads_built,
                            "Qualified Puts": row.qualified_puts,
                            "Qualified Calls": row.qualified_calls,
                            "Pairs Checked": row.pairs_checked,
                            "Matching Expirations": row.matching_expiration_pairs,
                            "Valid Strike Order": row.valid_order_pairs,
                            "Condors Built": row.built_condors,
                            "Main Blocker": row.top_reason,
                        }
                        for row in condor_diagnostic_rows
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No condor diagnostics were captured for this scan.")

        st.subheader("Strategy Diagnostics")
        strategy_df = pd.DataFrame(
            strategy_rejection_rows(trades, rejected_trades, scored_trades)
        )
        if not strategy_df.empty:
            for ticker in sorted(strategy_df["Ticker"].unique()):
                ticker_strategies = strategy_df[strategy_df["Ticker"] == ticker].drop(
                    columns="Ticker"
                )
                with st.expander(f"{ticker} strategy diagnostics"):
                    st.dataframe(
                        ticker_strategies,
                        width="stretch",
                        hide_index=True,
                        column_config={
                            "Rejected %": st.column_config.NumberColumn(format="%.0f%%")
                        },
                    )

        st.subheader("Ticker Status")
        st.dataframe(
            pd.DataFrame(
                ticker_rejection_rows(trades, rejected_trades, scored_trades)
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Rejected %": st.column_config.NumberColumn(format="%.0f%%")
            },
        )


def strategy_rejection_rows(trades, rejected_trades, scored_trades):
    totals = Counter(
        (trade.ticker, trade.strategy) for trade in trades
    )
    rejected = Counter((trade.ticker, trade.strategy) for trade, _ in rejected_trades)
    passing = Counter((scored.trade.ticker, scored.trade.strategy) for scored in scored_trades)
    reasons_by_strategy = {}
    for trade, reasons in rejected_trades:
        key = (trade.ticker, trade.strategy)

        if key not in reasons_by_strategy:
            reasons_by_strategy[key] = Counter()
        
        for reason in reasons:
            reasons_by_strategy[key][reason] += 1

    rows = []
    for (ticker,strategy), total in totals.items():
        key = (ticker, strategy)
        rejected_count = rejected[key]

        rows.append(
            {
                "Ticker": ticker,
                "Strategy": strategy,
                "Built" : total,
                "Passed": passing[key],
                "Rejected": rejected_count,
                "Rejected %": round(rejected_count / total * 100, 1),
                "Top Rejection Reason": reasons_by_strategy[key].most_common(1)[0][0]
                if key in reasons_by_strategy
                else "None",
            }
        )

    return rows
def ticker_rejection_rows(trades, rejected_trades, scored_trades):
    totals = Counter(trade.ticker for trade in trades)
    rejected = Counter(trade.ticker for trade, _ in rejected_trades)
    passing = Counter(scored.trade.ticker for scored in scored_trades)
    reasons_by_ticker = {}
    

    for trade, reasons in rejected_trades:
        if trade.ticker not in reasons_by_ticker:
            reasons_by_ticker[trade.ticker] = Counter()
        
        for reason in reasons:
            reasons_by_ticker[trade.ticker][reason] +=1
    rows = []
    for ticker, total in totals.items():
        rejected_count = rejected[ticker]
        rows.append(
            {
                "Ticker": ticker,
                "Candidates": total,
                "Passed": passing[ticker],
                "Rejected": rejected_count,
                "Rejected %": round(rejected_count / total * 100, 1),
                "Top Rejection Reason": reasons_by_ticker[ticker].most_common(1)[0][0]
                    if ticker in reasons_by_ticker
                    else "None",
                "Status": "Rejected" if rejected_count / total >= 0.9 and passing[ticker] == 0 else "Eligible",
            }
        )
    return rows


def outcome_summary(results, group_column: str, label: str):
    summary = (
        results.groupby(group_column, observed=True, as_index=False)
        .agg(
            Candidates=("id", "count"),
            Win_Rate=("Outcome P/L", lambda pnl: (pnl > 0).mean() * 100),
            Average_PnL=("Outcome P/L", "mean"),
            Total_PnL=("Outcome P/L", "sum"),
        )
        .rename(
            columns={
                group_column: label,
                "Win_Rate": "Win Rate",
                "Average_PnL": "Average P/L",
                "Total_PnL": "Total P/L",
            }
        )
    )
    summary["Win Rate"] = summary["Win Rate"].round(1)
    summary["Average P/L"] = summary["Average P/L"].round(2)
    summary["Total P/L"] = summary["Total P/L"].round(2)
    return summary


def render_results():
    latest_event_analyses = st.session_state.get("latest_event_analyses", {})
    if latest_event_analyses:
        st.subheader("Latest Event Evidence")
        for ticker, event_analysis in sorted(latest_event_analyses.items()):
            with st.expander(
                f"{ticker} | {event_analysis.label.title()} | "
                f"{event_analysis.adjustment:+d} adjustment"
            ):
                st.write(event_analysis.summary)
                if event_analysis.headlines_used:
                    for headline in event_analysis.headlines_used:
                        st.write(f"- {headline}")
                else:
                    st.caption("No relevant headlines were used for this analysis.")

    st.subheader("Open Candidates")
    open_rows, open_errors = fetch_open_history()
    for error in open_errors:
        st.warning(error)
    if open_rows:
        open_candidates = pd.DataFrame(open_rows)
        open_candidates["Scan Time"] = open_candidates["scan_time_est"].fillna(
            open_candidates["scan_time"]
        )
        if "price_move_adjustment" not in open_candidates:
            open_candidates["price_move_adjustment"] = None
        if "move_setup" not in open_candidates:
            open_candidates["move_setup"] = None
        open_candidates["Entry Cost"] = open_candidates.apply(
            lambda row: row["max_risk"]
            if row["entry_type"] == "debit"
            else row["credit"],
            axis=1,
        )
        open_candidates = open_candidates.reindex(
            columns=[
                "Scan Time",
                "ticker",
                "strategy",
                "expiration",
                "long_strike",
                "short_strike",
                "entry_type",
                "credit",
                "Entry Cost",
                "max_risk",
                "max_profit",
                "setup_score",
                "quant_score",
                "event_adjustment",
                "price_move_adjustment",
                "move_setup",
                "event_label",
                "event_confidence",
            ]
        ).rename(
            columns={
                "ticker": "Ticker",
                "strategy": "Strategy",
                "expiration": "Expiration",
                "long_strike": "Long Strike",
                "short_strike": "Short Strike",
                "entry_type": "Entry Type",
                "credit": "Credit",
                "max_risk": "Max Risk",
                "max_profit": "Max Profit",
                "setup_score": "Setup Score",
                "quant_score": "Quant Score",
                "event_adjustment": "Event Adjustment",
                "price_move_adjustment": "Price Move Adjustment",
                "move_setup": "Move Setup",
                "event_label": "Event Label",
                "event_confidence": "Event Confidence",
            }
        )
        st.dataframe(
            open_candidates,
            width="stretch",
            hide_index=True,
            column_config={
                "Credit": st.column_config.NumberColumn(format="$%.2f"),
                "Entry Cost": st.column_config.NumberColumn(
                    help="Debit paid for debit trades, credit received for credit trades.",
                    format="$%.2f",
                ),
                "Max Risk": st.column_config.NumberColumn(format="$%.2f"),
                "Max Profit": st.column_config.NumberColumn(format="$%.2f"),
            },
        )

        close_options = {
            (
                f"#{row['id']} | {row['ticker']} {row['strategy']} | "
                f"{row['expiration']} | Score {row['setup_score']}"
            ): row
            for row in open_rows
        }
        with st.expander("Close Candidate"):
            with st.form("close_candidate_form"):
                selected_label = st.selectbox("Candidate", list(close_options))
                actual_close_date = st.date_input("Close Date", value=date.today())
                actual_realized_pnl = st.number_input(
                    "Realized P/L Per Contract",
                    value=0.0,
                    step=1.0,
                    format="%.2f",
                )
                close_note = st.text_input("Close Note")
                save_close = st.form_submit_button("Save Close")

            if save_close:
                selected_row = close_options[selected_label]
                close_errors = close_candidate(
                    selected_row["id"],
                    actual_close_date,
                    actual_realized_pnl,
                    close_note,
                )
                if close_errors:
                    for error in close_errors:
                        st.error(error)
                else:
                    st.rerun()
    else:
        st.info("No open candidates are being tracked.")

    st.subheader("Results")
    result_rows, result_errors = fetch_completed_history()
    for error in result_errors:
        st.warning(error)
    if not result_rows:
        metric_columns = st.columns(4)
        metric_columns[0].metric("Completed Candidates", 0)
        metric_columns[1].metric("Win Rate", "N/A")
        metric_columns[2].metric("Average Outcome P/L", "N/A")
        metric_columns[3].metric("Total Outcome P/L", "N/A")

        st.subheader("Performance Scorecard")
        st.caption("Completed candidates will appear here after they expire or are closed.")
        score_tab, strategy_tab, entry_type_tab, event_tab, price_move_tab = st.tabs(
            [
                "Score Bands",
                "Strategies",
                "Debit vs. Credit",
                "AI Event View",
                "Price Move",
            ]
        )
        with score_tab:
            st.info("No completed candidates are available yet.")
        with strategy_tab:
            st.info("No completed candidates are available yet.")
        with entry_type_tab:
            st.info("No completed candidates are available yet.")
        with event_tab:
            st.info("No completed candidates are available yet.")
        with price_move_tab:
            st.info("No completed candidates are available yet.")

        st.subheader("Completed Candidates")
        st.info("No completed candidates are available yet.")
        return

    results = pd.DataFrame(result_rows)
    results["expiration_pnl"] = pd.to_numeric(results["expiration_pnl"])
    results["actual_realized_pnl"] = pd.to_numeric(results["actual_realized_pnl"])
    results["Outcome P/L"] = results["actual_realized_pnl"].fillna(
        results["expiration_pnl"]
    )

    completed_count = len(results)
    win_rate = (results["Outcome P/L"] > 0).mean() * 100
    average_pnl = results["Outcome P/L"].mean()
    total_pnl = results["Outcome P/L"].sum()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Completed Candidates", completed_count)
    metric_columns[1].metric("Win Rate", f"{win_rate:.1f}%")
    metric_columns[2].metric("Average Outcome P/L", f"${average_pnl:.2f}")
    metric_columns[3].metric("Total Outcome P/L", f"${total_pnl:.2f}")

    results["Score Band"] = pd.cut(
        results["setup_score"],
        bins=[-1, 59, 69, 79, 89, 100],
        labels=["Below 60", "60-69", "70-79", "80-89", "90-100"],
    )
    score_results = outcome_summary(results, "Score Band", "Score Band")
    strategy_results = outcome_summary(results, "strategy", "Strategy")
    entry_type_results = outcome_summary(results, "entry_type", "Entry Type")
    if "event_adjustment" not in results:
        results["event_adjustment"] = None
    results["event_adjustment"] = pd.to_numeric(
        results["event_adjustment"], errors="coerce"
    )
    results["Event Adjustment Group"] = "Neutral (0)"
    results.loc[results["event_adjustment"].isna(), "Event Adjustment Group"] = (
        "Not recorded"
    )
    results.loc[
        results["event_adjustment"] < 0, "Event Adjustment Group"
    ] = "Negative"
    results.loc[
        results["event_adjustment"] > 0, "Event Adjustment Group"
    ] = "Positive"
    event_results = outcome_summary(
        results, "Event Adjustment Group", "Event Adjustment"
    )
    if "unusual_move" not in results:
        results["unusual_move"] = None
    results["Price Move Group"] = results["unusual_move"].fillna("Not recorded")
    price_move_results = outcome_summary(
        results, "Price Move Group", "Price Move"
    )
    scorecard_config = {
        "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
        "Average P/L": st.column_config.NumberColumn(format="$%.2f"),
        "Total P/L": st.column_config.NumberColumn(format="$%.2f"),
    }

    st.subheader("Performance Scorecard")
    st.caption(
        f"{completed_count} completed candidates. Treat score patterns as preliminary until each group has a larger sample."
    )
    score_tab, strategy_tab, entry_type_tab, event_tab, price_move_tab = st.tabs(
        [
            "Score Bands",
            "Strategies",
            "Debit vs. Credit",
            "AI Event View",
            "Price Move",
        ]
    )
    with score_tab:
        st.dataframe(
            score_results,
            width="stretch",
            hide_index=True,
            column_config=scorecard_config,
        )
    with strategy_tab:
        st.dataframe(
            strategy_results,
            width="stretch",
            hide_index=True,
            column_config=scorecard_config,
        )
    with entry_type_tab:
        st.dataframe(
            entry_type_results,
            width="stretch",
            hide_index=True,
            column_config=scorecard_config,
        )
    with event_tab:
        st.dataframe(
            event_results,
            width="stretch",
            hide_index=True,
            column_config=scorecard_config,
        )
    with price_move_tab:
        st.dataframe(
            price_move_results,
            width="stretch",
            hide_index=True,
            column_config=scorecard_config,
        )

    st.subheader("Completed Candidates")
    recent_results = results[
        [
            "ticker",
            "strategy",
            "expiration",
            "setup_score",
            "expiration_status",
            "expiration_close",
            "Outcome P/L",
        ]
    ].rename(
        columns={
            "ticker": "Ticker",
            "strategy": "Strategy",
            "expiration": "Expiration",
            "setup_score": "Setup Score",
            "expiration_status": "Outcome Status",
            "expiration_close": "Expiration Close",
            "Outcome P/L": "Outcome P/L",
        }
    )
    st.dataframe(
        recent_results,
        width="stretch",
        hide_index=True,
        column_config={
            "Expiration Close": st.column_config.NumberColumn(format="$%.2f"),
            "Outcome P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_manual_positions():
    title_column, refresh_column = st.columns([4, 1])
    title_column.subheader("My Positions")
    if refresh_column.button("Refresh Quotes", width="stretch"):
        st.rerun()
    st.caption(
        "Private monitor for real positions you enter manually. Values use current Yahoo bid/ask quotes, so treat them as estimates."
    )

    with st.expander("Add Position", expanded=False):
        with st.form("manual_position_form"):
            form_columns = st.columns(3)
            ticker = form_columns[0].text_input("Ticker", value="NVDA").upper()
            strategy = form_columns[1].selectbox(
                "Strategy",
                [
                    "bull call debit spread",
                    "bear put debit spread",
                    "put credit spread",
                    "call credit spread",
                ],
            )
            expiration = form_columns[2].date_input("Expiration", value=date.today())

            strike_columns = st.columns(4)
            long_strike = strike_columns[0].number_input(
                "Long Strike", min_value=0.0, value=200.0, step=1.0
            )
            short_strike = strike_columns[1].number_input(
                "Short Strike", min_value=0.0, value=205.0, step=1.0
            )
            entry_price = strike_columns[2].number_input(
                "Debit/Credit Paid", min_value=0.0, value=1.0, step=0.01
            )
            quantity = strike_columns[3].number_input(
                "Quantity", min_value=1, value=1, step=1
            )
            note = st.text_input("Note")
            submitted = st.form_submit_button("Add Position")

        if submitted:
            add_errors = add_manual_position(
                ticker,
                strategy,
                expiration,
                long_strike,
                short_strike,
                entry_price,
                quantity,
                note,
            )
            if add_errors:
                for error in add_errors:
                    st.error(error)
            else:
                st.success("Position added.")
                st.rerun()

    position_rows, position_errors = manual_position_rows_with_marks()
    for error in position_errors:
        st.warning(error)

    if not position_rows:
        st.info("No open manual positions yet.")
        return

    positions = pd.DataFrame(position_rows)
    display_positions = positions[
        [
            "ticker",
            "strategy",
            "expiration",
            "dte",
            "underlying_price",
            "long_strike",
            "short_strike",
            "entry_price",
            "quantity",
            "current_mark",
            "unrealized_pnl",
            "conservative_value",
            "conservative_pnl",
            "quote_source",
            "recommendation",
            "note",
        ]
    ].rename(
        columns={
            "ticker": "Ticker",
            "strategy": "Strategy",
            "expiration": "Expiration",
            "dte": "DTE",
            "underlying_price": "Stock Price",
            "long_strike": "Long",
            "short_strike": "Short",
            "entry_price": "Entry",
            "quantity": "Qty",
            "current_mark": "Mid Value",
            "unrealized_pnl": "Unrealized P/L",
            "conservative_value": "Close Value",
            "conservative_pnl": "Close P/L",
            "quote_source": "Quote Source",
            "recommendation": "Recommendation",
            "note": "Note",
        }
    )
    st.dataframe(
        display_positions,
        width="stretch",
        hide_index=True,
        column_config={
            "Stock Price": st.column_config.NumberColumn(format="$%.2f"),
            "Long": st.column_config.NumberColumn(format="%.2f"),
            "Short": st.column_config.NumberColumn(format="%.2f"),
            "Entry": st.column_config.NumberColumn(format="$%.2f"),
            "Mid Value": st.column_config.NumberColumn(
                help="Broker-like midpoint value using bid/ask mid prices.",
                format="$%.2f",
            ),
            "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
            "Close Value": st.column_config.NumberColumn(
                help="Conservative estimated value if closing through bid/ask.",
                format="$%.2f",
            ),
            "Close P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    with st.expander("Manage Manual Position"):
        position_options = {
            (
                f"#{row['id']} | {row['ticker']} {row['strategy']} | "
                f"{row['expiration']}"
            ): row
            for row in position_rows
        }
        selected = st.selectbox("Position", list(position_options))
        action_columns = st.columns(2)
        if action_columns[0].button("Mark Closed", width="stretch"):
            selected_id = position_options[selected]["id"]
            errors = close_manual_position(selected_id)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.rerun()
        if action_columns[1].button("Delete Entry", width="stretch"):
            selected_id = position_options[selected]["id"]
            errors = delete_manual_position(selected_id)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.rerun()


def render_alpaca_account_status():
    st.subheader("Alpaca Paper Account")
    st.caption(
        "Read-only connection check. This does not place trades; it only confirms "
        "your paper account keys work."
    )

    account, errors = get_alpaca_account()
    for error in errors:
        st.warning(error)

    if account is None:
        st.info(
            "Use ALPACA_API_KEY and ALPACA_SECRET_KEY in .env. For paper trading, "
            "leave ALPACA_BASE_URL blank or set it to https://paper-api.alpaca.markets."
        )
        return

    if not account.get("_is_paper"):
        st.error(
            "This is not using the paper endpoint. Set ALPACA_BASE_URL to "
            "https://paper-api.alpaca.markets before adding any trading features."
        )
    else:
        st.success("Connected to Alpaca paper trading.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Status", str(account.get("status", "unknown")).title())
    metric_columns[1].metric("Portfolio", f"${float(account.get('portfolio_value', 0)):,.2f}")
    metric_columns[2].metric("Buying Power", f"${float(account.get('buying_power', 0)):,.2f}")
    metric_columns[3].metric("Cash", f"${float(account.get('cash', 0)):,.2f}")

    detail_columns = st.columns(3)
    detail_columns[0].caption(f"Endpoint: {account.get('_base_url')}")
    detail_columns[1].caption(f"Key var: {account.get('_api_key_name')}")
    detail_columns[2].caption(f"Secret var: {account.get('_secret_key_name')}")

    paper_history, paper_history_errors = fetch_alpaca_paper_orders()
    for error in paper_history_errors:
        st.warning(error)

    st.divider()
    st.subheader("Paper Positions")
    st.caption(
        "Alpaca reports multi-leg orders as individual option-leg positions. "
        "Market value and unrealized P/L below are leg-level values from Alpaca."
    )
    positions, position_errors = get_alpaca_positions()
    for error in position_errors:
        st.warning(error)
    if positions:
        positions_df = pd.DataFrame(positions)
        st.dataframe(
            positions_df.reindex(
                columns=[
                    "symbol",
                    "asset_class",
                    "qty",
                    "side",
                    "avg_entry_price",
                    "current_price",
                    "market_value",
                    "unrealized_pl",
                    "unrealized_plpc",
                ]
            ).rename(
                columns={
                    "symbol": "Symbol",
                    "asset_class": "Asset Class",
                    "qty": "Qty",
                    "side": "Side",
                    "avg_entry_price": "Avg Entry",
                    "current_price": "Current",
                    "market_value": "Market Value",
                    "unrealized_pl": "Unrealized P/L",
                    "unrealized_plpc": "Unrealized P/L %",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No Alpaca paper positions are open.")

    st.subheader("Grouped Paper Spreads")
    grouped_rows = grouped_alpaca_spread_rows(positions, paper_history)
    if grouped_rows:
        st.dataframe(
            pd.DataFrame(grouped_rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Limit": st.column_config.NumberColumn(format="$%.2f"),
                "Current Value": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
        grouped_total_value = sum(row["Current Value"] for row in grouped_rows)
        grouped_total_pnl = sum(row["Unrealized P/L"] for row in grouped_rows)
        spread_metric_columns = st.columns(2)
        spread_metric_columns[0].metric(
            "Grouped Spread Value", f"${grouped_total_value:,.2f}"
        )
        spread_metric_columns[1].metric(
            "Grouped Spread P/L", f"${grouped_total_pnl:,.2f}"
        )
        st.caption(
            "Grouped spread totals are built from logged paper-order legs. If the same option leg belongs to multiple spreads, Alpaca's aggregated position can make allocation approximate."
        )
    else:
        st.info("No logged Alpaca paper spreads match current open positions yet.")

    snapshot_columns = st.columns([1, 2])
    if snapshot_columns[0].button(
        "Save Paper P/L Snapshot",
        width="stretch",
        help="Pull current Alpaca paper positions and save spread-level value/P&L to Supabase.",
    ):
        snapshot_errors = append_alpaca_paper_snapshots()
        for error in snapshot_errors:
            st.error(error)
        if not snapshot_errors:
            st.success("Saved the latest Alpaca paper P/L snapshot.")
            st.rerun()

    paper_snapshots, paper_snapshot_errors = fetch_alpaca_paper_snapshots()
    for error in paper_snapshot_errors:
        st.warning(error)

    if paper_snapshots:
        snapshot_df = pd.DataFrame(paper_snapshots)
        snapshot_df["snapshot_time"] = pd.to_datetime(snapshot_df["snapshot_time"])
        snapshot_df["Spread"] = (
            snapshot_df["ticker"].fillna("")
            + " "
            + snapshot_df["strategy"].fillna("")
            + " "
            + snapshot_df["expiration"].astype(str)
        )
        latest_snapshot_df = (
            snapshot_df.sort_values("snapshot_time")
            .groupby("alpaca_paper_order_id", as_index=False)
            .tail(1)
        )
        snapshot_metric_columns = st.columns(3)
        snapshot_metric_columns[0].metric(
            "Tracked Paper Spreads",
            f"{latest_snapshot_df['alpaca_paper_order_id'].nunique()}",
        )
        snapshot_metric_columns[1].metric(
            "Latest Snapshot Value",
            f"${latest_snapshot_df['current_value'].fillna(0).sum():,.2f}",
        )
        snapshot_metric_columns[2].metric(
            "Latest Snapshot P/L",
            f"${latest_snapshot_df['unrealized_pnl'].fillna(0).sum():,.2f}",
        )

        pnl_chart_df = (
            snapshot_df.pivot_table(
                index="snapshot_time",
                columns="Spread",
                values="unrealized_pnl",
                aggfunc="last",
            )
            .sort_index()
        )
        st.line_chart(pnl_chart_df, height=260)
        st.dataframe(
            latest_snapshot_df.reindex(
                columns=[
                    "snapshot_time_est",
                    "ticker",
                    "strategy",
                    "expiration",
                    "current_value",
                    "unrealized_pnl",
                    "unrealized_pnl_percent",
                    "matched_legs",
                    "total_legs",
                ]
            ).rename(
                columns={
                    "snapshot_time_est": "Snapshot Time",
                    "ticker": "Ticker",
                    "strategy": "Strategy",
                    "expiration": "Expiration",
                    "current_value": "Current Value",
                    "unrealized_pnl": "Unrealized P/L",
                    "unrealized_pnl_percent": "Unrealized P/L %",
                    "matched_legs": "Matched Legs",
                    "total_legs": "Total Legs",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Current Value": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P/L": st.column_config.NumberColumn(format="$%.2f"),
                "Unrealized P/L %": st.column_config.NumberColumn(format="%.2f%%"),
            },
        )
    else:
        st.caption(
            "No saved Alpaca paper P/L snapshots yet. Use the snapshot button after the SQL table is created."
        )

    st.subheader("Recent Paper Orders")
    orders, order_errors = get_recent_alpaca_orders()
    for error in order_errors:
        st.warning(error)
    if orders:
        orders_df = pd.DataFrame(orders)
        st.dataframe(
            orders_df.reindex(
                columns=[
                    "submitted_at",
                    "symbol",
                    "asset_class",
                    "side",
                    "qty",
                    "type",
                    "limit_price",
                    "filled_avg_price",
                    "status",
                ]
            ).rename(
                columns={
                    "submitted_at": "Submitted",
                    "symbol": "Symbol",
                    "asset_class": "Asset Class",
                    "side": "Side",
                    "qty": "Qty",
                    "type": "Type",
                    "limit_price": "Limit",
                    "filled_avg_price": "Fill Price",
                    "status": "Status",
                }
            ),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No recent Alpaca paper orders.")

    st.subheader("Paper Order Tracking")
    if paper_history:
        paper_history_df = pd.DataFrame(paper_history)
        paper_history_df["scan_time"] = pd.to_datetime(paper_history_df["scan_time"])
        orders_by_scan = (
            paper_history_df.sort_values("scan_time")
            .groupby("scan_time", as_index=False)
            .agg(
                Orders=("id", "count"),
                Average_Score=("setup_score", "mean"),
            )
        )
        orders_by_scan["Average Score"] = orders_by_scan["Average_Score"].round(1)
        st.line_chart(
            orders_by_scan.set_index("scan_time")[["Orders", "Average Score"]],
            height=260,
        )

        strategy_counts = (
            paper_history_df.groupby("strategy", as_index=False)
            .agg(Orders=("id", "count"))
            .sort_values("Orders", ascending=False)
        )
        st.bar_chart(strategy_counts.set_index("strategy"), height=240)
    else:
        st.info(
            "No Alpaca paper orders have been logged yet. Run the SQL file and then scan with paper auto trading enabled."
        )
        if orders:
            st.warning(
                "Alpaca has recent paper orders, but Supabase has no logged paper-order rows yet."
            )

    if st.button("Backfill Recent Alpaca Orders To Supabase", width="stretch"):
        backfill_results, backfill_errors = recent_alpaca_order_results(limit=50)
        for error in backfill_errors:
            st.error(error)
        save_errors = append_alpaca_paper_orders(backfill_results)
        for error in save_errors:
            st.error(error)
        if not backfill_errors and not save_errors:
            st.success(f"Backfilled {len(backfill_results)} Alpaca paper orders.")
            st.rerun()

    st.divider()
    st.subheader("Paper Trade A Scan Candidate")
    st.caption(
        "Submits the full spread as an Alpaca multi-leg paper order. The order "
        "fills together or not at all."
    )
    scan_output = st.session_state.get("last_scan_output")
    if not scan_output or not scan_output.get("scored_trades"):
        st.info("Run a scan first, then come back here to paper trade a candidate.")
        return
    if submit_multileg_order is None or trade_multileg_order_details is None:
        st.warning(
            "Alpaca multi-leg order helpers are unavailable while the app finishes redeploying."
        )
        return

    paper_candidates = select_history_candidates(scan_output["scored_trades"])[:10]
    if not paper_candidates:
        st.info("The latest scan did not produce paper-tradable candidates.")
        return

    candidate_options = {
        (
            f"{scored.trade.ticker} | {scored.trade.strategy} | "
            f"{scored.trade.expiration} | score {scored.total_score}"
        ): scored
        for scored in paper_candidates
    }
    selected_label = st.selectbox("Candidate", list(candidate_options))
    selected_scored = candidate_options[selected_label]
    selected_trade = selected_scored.trade
    try:
        legs, suggested_limit_price, quantity_type = trade_multileg_order_details(
            selected_scored
        )
    except ValueError as error:
        st.error(str(error))
        return

    preview_columns = st.columns(4)
    preview_columns[0].metric("Order Class", "MLeg")
    preview_columns[1].metric("Legs", len(legs))
    preview_columns[2].metric("Limit Type", quantity_type.title())
    preview_columns[3].metric("Score", selected_scored.total_score)
    st.dataframe(pd.DataFrame(legs), width="stretch", hide_index=True)

    with st.form("paper_option_order_form"):
        order_columns = st.columns(2)
        quantity = order_columns[0].number_input(
            "Contracts", min_value=1, max_value=10, value=1, step=1
        )
        limit_price = order_columns[1].number_input(
            "Limit Price",
            min_value=0.01,
            value=max(round(float(suggested_limit_price), 2), 0.01),
            step=0.01,
        )
        st.caption("Multi-leg option paper orders use limit orders only.")
        confirmation = st.text_input("Type PAPER to submit this paper order")
        submitted = st.form_submit_button("Submit Paper Multi-Leg Order")

    if submitted:
        if confirmation.strip().upper() != "PAPER":
            st.error("Type PAPER before submitting the paper order.")
            return
        order, submit_errors = submit_multileg_order(
            legs,
            quantity=int(quantity),
            limit_price=float(limit_price),
            client_order_id=f"manual-{paper_trade_key(selected_scored)}",
        )
        if submit_errors:
            for error in submit_errors:
                st.error(error)
        else:
            save_errors = append_alpaca_paper_orders(
                [
                    {
                        "Candidate": (
                            f"{selected_trade.ticker} {selected_trade.strategy} "
                            f"{selected_trade.expiration} score {selected_scored.total_score}"
                        ),
                        "Symbol": order.get("symbol")
                        or ("2-leg order" if len(legs) == 2 else "4-leg order"),
                        "Status": order.get("status", "submitted"),
                        "Message": (
                            f"{quantity_type.title()} limit ${float(limit_price):.2f}; "
                            f"order {order.get('id')}"
                        ),
                        "Order ID": order.get("id"),
                        "Client Order ID": order.get("client_order_id"),
                        "Ticker": selected_trade.ticker,
                        "Strategy": selected_trade.strategy,
                        "Expiration": selected_trade.expiration,
                        "Setup Score": selected_scored.total_score,
                        "Entry Type": selected_trade.entry_type,
                        "Limit Price": float(limit_price),
                        "Quantity": int(quantity),
                        "Order Class": order.get("order_class") or "mleg",
                        "Leg Key": leg_key_from_legs(legs)
                        if leg_key_from_legs is not None
                        else None,
                    }
                ]
            )
            st.success(
                f"Paper multi-leg order submitted: "
                f"{order.get('qty')} contract(s), status {order.get('status')}"
            )
            for error in save_errors:
                st.warning(error)
            st.json(
                {
                    "id": order.get("id"),
                    "order_class": order.get("order_class"),
                    "qty": order.get("qty"),
                    "type": order.get("type"),
                    "limit_price": order.get("limit_price"),
                    "status": order.get("status"),
                }
            )


def render_private_results():
    owner_password = os.getenv("OWNER_DASHBOARD_PASSWORD")
    if not owner_password:
        return

    if st.session_state.get("results_unlocked"):
        with st.sidebar:
            if st.button(
                "Refresh AI Event Data",
                help="Clears cached AI event analysis so the next scan fetches fresh headlines.",
                width="stretch",
            ):
                event_analysis_cache().clear()
                deep_event_analysis_cache().clear()
                candidate_analysis_cache().clear()
                st.session_state.pop("latest_event_analyses", None)
                st.session_state.pop("last_scan_output", None)
                st.toast("AI event cache cleared. Run a new scan for fresh analysis.")
            if st.button("Lock Results", width="stretch"):
                st.session_state["results_unlocked"] = False
                st.session_state["admin_clicks"] = 0
                st.session_state["show_admin_prompt"] = False
                st.rerun()
        positions_tab, paper_account_tab, scanner_results_tab = st.tabs(
            ["My Positions", "Paper Account", "Scanner Tracking"]
        )
        with positions_tab:
            render_manual_positions()
        with paper_account_tab:
            render_alpaca_account_status()
        with scanner_results_tab:
            render_results()
        return

    with st.sidebar:
        st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
        if st.button(
            " ",
            icon=":material/lock:",
            help="Owner access",
            key="admin_access_trigger",
            width="content",
        ):
            st.session_state["admin_clicks"] = (
                st.session_state.get("admin_clicks", 0) + 1
            )
            if st.session_state["admin_clicks"] >= 3:
                st.session_state["show_admin_prompt"] = True

        if st.session_state.get("show_admin_prompt"):
            entered_password = st.text_input("Owner Password", type="password")
            if entered_password and hmac.compare_digest(entered_password, owner_password):
                st.session_state["results_unlocked"] = True
                st.session_state["admin_clicks"] = 0
                st.session_state["show_admin_prompt"] = False
                st.rerun()


def render_ready_state():
    st.markdown("### Ready To Scan")
    st.info(
        "Choose your tickers and press Scan Watchlist. The scan may take a minute "
        "because it pulls option chains, checks news, scores setups, and reviews "
        "the top candidates with AI."
    )
    metric_columns = st.columns(3)
    metric_columns[0].metric("Large Preset", "60 tickers")
    metric_columns[1].metric("Strategies", "5 types")
    metric_columns[2].metric("AI Reviews", "Top 3")

    st.markdown("### What You Will See")
    preview_columns = st.columns(3)
    with preview_columns[0]:
        st.markdown("**Candidates**")
        st.caption("Ranked spreads with setup, quant, event, and price-move scores.")
    with preview_columns[1]:
        st.markdown("**Market Data**")
        st.caption("Underlying price, volatility rank, recent move, and event label.")
    with preview_columns[2]:
        st.markdown("**Diagnostics**")
        st.caption("Why tickers or strategies were rejected by the filters.")


st.title("Options Scanner")
st.caption(
    "Yahoo Finance data may be delayed. Volatility Rank uses historical price movement."
)

with st.sidebar:
    st.header("Scanner")
    if "ticker_preset" not in st.session_state:
        st.session_state["ticker_preset"] = "Default"
    if "ticker_text_value" not in st.session_state:
        st.session_state["ticker_text_value"] = DEFAULT_TICKERS

    ticker_preset = st.selectbox(
        "Ticker Preset",
        ["Default", "Large Preset", "Custom"],
        key="ticker_preset",
        help="Use the large preset with the broad universe prefilter turned on.",
    )
    preset_value = (
        LARGE_PRESET_TICKERS if ticker_preset == "Large Preset" else DEFAULT_TICKERS
    )
    if ticker_preset != "Custom":
        st.session_state["ticker_text_value"] = preset_value
    ticker_text = st.text_area(
        "Tickers",
        key="ticker_text_value",
        height=150 if ticker_preset == "Large Preset" else 120,
    )
    outlook = st.selectbox("Outlook", ["neutral", "bullish", "bearish", "income"])
    max_risk = st.number_input("Maximum Risk Per Spread", min_value=50, value=500, step=50)
    risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
    with st.expander("Advanced Expiration Settings"):
        use_nearest_expiration = st.checkbox("Use Nearest Available Expirations")
        use_test_expiration = st.checkbox(
            "Test a Specific Expiration", disabled=use_nearest_expiration
        )
        test_expiration = (
            st.date_input("Test Expiration", min_value=date.today())
            if use_test_expiration
            else None
        )
    with st.expander("Broad Universe Prefilter"):
        use_prefilter = st.checkbox(
            "Prefilter before options scan",
            value=False,
            help=(
                "Use stock-only price, volume, and volatility data to narrow a "
                "large watchlist before pulling option chains."
            ),
        )
        prefilter_max_tickers = st.number_input(
            "Max Tickers For Options Scan",
            min_value=5,
            max_value=100,
            value=35,
            step=5,
            disabled=not use_prefilter,
        )
        prefilter_min_price = st.number_input(
            "Minimum Stock Price",
            min_value=1,
            value=20,
            step=1,
            disabled=not use_prefilter,
        )
        prefilter_min_volume = st.number_input(
            "Minimum 20D Avg Volume",
            min_value=0,
            value=1_000_000,
            step=250_000,
            disabled=not use_prefilter,
        )
        prefilter_min_vol_rank = st.number_input(
            "Minimum Volatility Rank",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            disabled=not use_prefilter,
        )
    if st.session_state.get("results_unlocked"):
        with st.expander("Alpaca Paper Auto Trading"):
            st.checkbox(
                "Paper trade top 3 scan candidates",
                key="auto_paper_trade_scans",
                value=True,
                help=(
                    "After a manual scan, submit Alpaca multi-leg paper orders for "
                    "the three highest-scoring candidates."
                ),
            )
            st.number_input(
                "Contracts Per Paper Order",
                min_value=1,
                max_value=10,
                value=1,
                step=1,
                key="auto_paper_trade_quantity",
            )
    scan_button = st.button("Scan Watchlist", type="primary", width="stretch")

if scan_button:
    tickers = [ticker.strip().upper() for ticker in ticker_text.split(",") if ticker.strip()]
    if not tickers:
        st.error("Enter at least one ticker.")
        st.stop()
    

    history_errors = update_history()
    preferences = ScanPreferences(
        max_risk=float(max_risk),
        outlook=outlook,
        risk_tolerance=risk_tolerance,
        test_expiration=test_expiration,
        nearest_expiration=use_nearest_expiration,
    )
    with st.status("Scanning watchlist...", expanded=True) as scan_status:
        prefilter_rows = []
        original_tickers = tickers
        if use_prefilter:
            st.write(
                f"Prefiltering {len(original_tickers)} tickers before option-chain scan..."
            )
            selected_tickers, prefilter_results = prefilter_tickers(
                original_tickers,
                max_selected=int(prefilter_max_tickers),
                min_price=float(prefilter_min_price),
                min_average_volume=int(prefilter_min_volume),
                min_volatility_rank=float(prefilter_min_vol_rank),
            )
            prefilter_rows = prefilter_result_rows(prefilter_results)
            if not selected_tickers:
                st.error("No tickers passed the broad universe prefilter.")
                st.session_state["last_scan_output"] = {
                    "scored_trades": [],
                    "rejected_trades": [],
                    "trades": [],
                    "ticker_data": [],
                    "condor_diagnostics": [],
                    "errors": ["No tickers passed the broad universe prefilter."],
                    "event_analyses": {},
                    "candidate_analyses": {},
                    "history_candidates": [],
                    "paper_order_results": [],
                    "prefilter_rows": prefilter_rows,
                    "prefilter_selected_tickers": [],
                    "original_ticker_count": len(original_tickers),
                }
                scan_status.update(
                    label="Scan stopped by prefilter", state="error", expanded=False
                )
                st.stop()

            tickers = selected_tickers
            st.write(
                f"Prefilter selected {len(tickers)} of {len(original_tickers)} tickers: "
                + ", ".join(tickers)
            )

        st.write(f"Pulling option chains and market data for {len(tickers)} tickers...")
        (
            scored_trades,
            rejected_trades,
            trades,
            ticker_data,
            condor_diagnostic_rows,
            errors,
            event_analyses,
            price_moves,
        ) = scan_watchlist(tickers, preferences)
        st.session_state["latest_event_analyses"] = event_analyses

        st.write("Saving tracked candidates and updating snapshots...")
        history_candidates = select_history_candidates(scored_trades)
        history_save_errors = save_history(
            history_candidates, event_analyses, price_moves
        )
        errors = history_errors + errors + history_save_errors

        paper_order_results = []
        if (
            st.session_state.get("results_unlocked")
            and st.session_state.get("auto_paper_trade_scans")
        ):
            st.write("Submitting Alpaca paper orders for the top 3 candidates...")
            paper_candidates, duplicate_results = top_unplaced_paper_candidates(
                scored_trades, limit=3
            )
            paper_order_results = paper_trade_scan_candidates(
                paper_candidates,
                quantity=int(st.session_state.get("auto_paper_trade_quantity", 1)),
                limit=3,
            )
            paper_order_results = duplicate_results + paper_order_results
            errors.extend(append_alpaca_paper_orders(paper_order_results))
        elif st.session_state.get("results_unlocked"):
            paper_order_results = [
                {
                    "Candidate": "Latest Scan",
                    "Symbol": "",
                    "Status": "Skipped",
                    "Message": "Alpaca paper auto trading is turned off.",
                }
            ]
        else:
            paper_order_results = [
                {
                    "Candidate": "Latest Scan",
                    "Symbol": "",
                    "Status": "Skipped",
                    "Message": "Unlock the owner panel before scanning to enable Alpaca paper trading.",
                }
            ]

        st.write("Reviewing the top 3 candidates with AI...")
        candidate_analyses = {}
        for scored in scored_trades[:3]:
            trade = scored.trade
            candidate_analyses[candidate_analysis_key(scored)] = (
                get_cached_candidate_analysis(
                    scored,
                    event_analyses.get(trade.ticker),
                    price_moves.get(trade.ticker),
                )
            )
        scan_status.update(label="Scan complete", state="complete", expanded=False)
    st.session_state["last_scan_output"] = {
        "scored_trades": scored_trades,
        "rejected_trades": rejected_trades,
        "trades": trades,
        "ticker_data": ticker_data,
        "condor_diagnostics": condor_diagnostic_rows,
        "errors": errors,
        "event_analyses": event_analyses,
        "candidate_analyses": candidate_analyses,
        "history_candidates": history_candidates,
        "paper_order_results": paper_order_results,
        "prefilter_rows": prefilter_rows,
        "prefilter_selected_tickers": tickers if use_prefilter else [],
        "original_ticker_count": len(original_tickers) if use_prefilter else len(tickers),
    }

if st.session_state.get("last_scan_output"):
    render_scan_output(st.session_state["last_scan_output"])
else:
    render_ready_state()


render_private_results()
