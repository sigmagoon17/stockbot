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

from history_tracker import (
    add_manual_position,
    append_scan_history as save_history,
    append_trade_snapshots as save_trade_snapshots,
    close_candidate,
    close_manual_position,
    delete_manual_position,
    fetch_completed_history,
    fetch_open_history,
    manual_position_rows_with_marks,
    update_expired_history as update_history,
)

from stock2dupe import (
    CONTRACT_MULTIPLIER,
    ScanPreferences,
    build_call_credit_spreads,
    build_iron_condor,
    build_put_credit_spreads,
    build_bull_call_debit_spread,
    build_bear_put_debit_spread,
    get_option_chain,
    scan_trades,
)

try:
    from event_analysis import analyze_candidate_setup, get_event_analysis
except ImportError:
    from event_analysis import get_event_analysis

    analyze_candidate_setup = None

st.set_page_config(page_title="Options Scanner", layout="wide")

EVENT_ANALYSIS_SUCCESS_TTL_SECONDS = 6 * 60 * 60
EVENT_ANALYSIS_FAILURE_TTL_SECONDS = 5 * 60


@st.cache_resource
def event_analysis_cache():
    return {}


@st.cache_resource
def candidate_analysis_cache():
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
    return (
        scored_trades,
        rejected_trades,
        trades,
        ticker_data,
        errors,
        event_analyses,
        price_moves,
    )
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
    errors = scan_output["errors"]
    event_analyses = scan_output["event_analyses"]
    history_candidates = scan_output["history_candidates"]

    top_score = scored_trades[0].total_score if scored_trades else None
    metric_candidates, metric_score, metric_tracked, metric_tickers = st.columns(4)
    metric_candidates.metric("Passing Candidates", len(scored_trades))
    metric_score.metric("Highest Score", f"{top_score}/100" if top_score else "None")
    metric_tracked.metric("Saved to History", len(history_candidates))
    metric_tickers.metric("Tickers Scanned", len(ticker_data))

    if errors:
        for error in errors:
            st.warning(error)

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
                candidate_analysis_cache().clear()
                st.session_state.pop("latest_event_analyses", None)
                st.session_state.pop("last_scan_output", None)
                st.toast("AI event cache cleared. Run a new scan for fresh analysis.")
            if st.button("Lock Results", width="stretch"):
                st.session_state["results_unlocked"] = False
                st.session_state["admin_clicks"] = 0
                st.session_state["show_admin_prompt"] = False
                st.rerun()
        positions_tab, scanner_results_tab = st.tabs(
            ["My Positions", "Scanner Tracking"]
        )
        with positions_tab:
            render_manual_positions()
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


st.title("Options Scanner")
st.caption(
    "Yahoo Finance data may be delayed. Volatility Rank uses historical price movement."
)

with st.sidebar:
    st.header("Scanner")
    ticker_text = st.text_area("Tickers", "AAPL, SPY, QQQ, NVDA, MSFT, COHR")
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
    (
        scored_trades,
        rejected_trades,
        trades,
        ticker_data,
        errors,
        event_analyses,
        price_moves,
    ) = scan_watchlist(tickers, preferences)
    st.session_state["latest_event_analyses"] = event_analyses
    history_candidates = select_history_candidates(scored_trades)
    history_save_errors = save_history(
        history_candidates, event_analyses, price_moves
    )
    snapshot_errors = save_trade_snapshots()
    errors = history_errors + errors + history_save_errors
    errors.extend(snapshot_errors)
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
    st.session_state["last_scan_output"] = {
        "scored_trades": scored_trades,
        "rejected_trades": rejected_trades,
        "trades": trades,
        "ticker_data": ticker_data,
        "errors": errors,
        "event_analyses": event_analyses,
        "candidate_analyses": candidate_analyses,
        "history_candidates": history_candidates,
    }

if st.session_state.get("last_scan_output"):
    render_scan_output(st.session_state["last_scan_output"])


render_private_results()
