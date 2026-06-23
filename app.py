from collections import Counter
from datetime import date
import hmac
import os
import pandas as pd
import streamlit as st
import datetime
import supabase
import yfinance as yf

from history_tracker import (
    append_scan_history as save_history,
    fetch_expired_history,
    fetch_open_history,
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

st.set_page_config(page_title="Options Scanner", layout="wide")


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


def select_history_candidates(scored_trades, limit: int = 25, per_ticker: int = 4):
    selected = select_top_candidates(scored_trades, per_ticker=per_ticker)
    selected = selected[:limit]

    if len(selected) < limit:
        selected_ids = {id(scored) for scored in selected}
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
    progress = st.progress(0, text="Preparing scan")

    for index, ticker in enumerate(tickers, start=1):
        progress.progress(
            index / len(tickers), text=f"Fetching {ticker} option data ({index}/{len(tickers)})"
        )
        try:
            price, option_chain, earnings_date, volatility_rank = get_option_chain(
                ticker,
                test_expiration=preferences.test_expiration,
                nearest_expiration=preferences.nearest_expiration,
            )
            ticker_data.append(
                {
                    "Ticker": ticker,
                    "Price": price,
                    "Contracts": len(option_chain),
                    "Volatility Rank": volatility_rank,
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
    scored_trades, rejected_trades = scan_trades(trades, preferences)
    return scored_trades, rejected_trades, trades, ticker_data, errors
def append_scan_history(scored_trades):
    rows = []
    current_time = datetime.datetime.now().isoformat()
    for scored in scored_trades:
        trade = scored.trade
        scan_info = {
            "scan_time": current_time,
            "ticker": trade.ticker,
            "strategy": trade.strategy,
            "expiration": trade.expiration,
            "long_strike": round(trade.long_strike, 2),
            "short_strike": round(trade.short_strike, 2),
            "put_short_strike": (
                round(trade.put_short_strike, 2)
                if trade.put_short_strike is not None
                else None
            ),
            "put_long_strike":(
                round(trade.put_long_strike, 2)
                if trade.put_long_strike is not None
                else None
            ),
            "call_short_strike": (
                round(trade.call_short_strike, 2)
                if trade.call_short_strike is not None
                else None
            ),
            "call_long_strike": (
                round(trade.call_long_strike, 2)
                if trade.call_long_strike is not None
                else None
            ),
            "underlying_price": round(trade.underlying_price, 2),
            "entry_type": trade.entry_type,
            "credit": round(trade.credit * CONTRACT_MULTIPLIER, 2),
            "max_risk": round(trade.max_risk * CONTRACT_MULTIPLIER, 2),
            "max_profit": round(trade.max_profit * CONTRACT_MULTIPLIER, 2),
            "setup_score": scored.total_score,
            "risk_level": scored.risk_level,
            "dte": trade.dte,
            "volatility_rank": round(trade.volatility_rank, 1),
            "starting_status": "open",
            "expiration_status": "open",
            "expiration_close": None,
            "expiration_pnl": None,

        }
        rows.append(scan_info)
    if not rows:
        return

    def candidate_key(row):
        return (
            row["ticker"],
            row["strategy"],
            row["expiration"],
            row["long_strike"],
            row["short_strike"],
            row["entry_type"],
        )

    existing_rows = (
        supabase.table("scan_history")
        .select("ticker,strategy,expiration,long_strike,short_strike,entry_type")
        .gte("scan_time", date.today().isoformat())
        .execute()
        .data
    )
    existing_keys = {candidate_key(row) for row in existing_rows}
    new_rows = []
    for row in rows:
        key = candidate_key(row)
        if key not in existing_keys:
            new_rows.append(row)
            existing_keys.add(key)

    if new_rows:
        supabase.table("scan_history").insert(new_rows).execute()


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
        "Risk Level": scored.risk_level,
        "Volatility Rank": round(trade.volatility_rank, 1),
    }


def candidate_rows(scored_trades):
    return [candidate_row(scored) for scored in select_top_candidates(scored_trades)]


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
                "Risk Level": scored.risk_level,
                "Volatility Rank": round(trade.volatility_rank, 1),
            }
        )

    return rows


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


def render_results():
    st.subheader("Open Candidates")
    open_rows, open_errors = fetch_open_history()
    for error in open_errors:
        st.warning(error)
    if open_rows:
        open_candidates = pd.DataFrame(open_rows)
        open_candidates["Scan Time"] = open_candidates["scan_time_est"].fillna(
            open_candidates["scan_time"]
        )
        open_candidates = open_candidates[
            [
                "Scan Time",
                "ticker",
                "strategy",
                "expiration",
                "long_strike",
                "short_strike",
                "entry_type",
                "credit",
                "max_risk",
                "max_profit",
                "setup_score",
            ]
        ].rename(
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
            }
        )
        st.dataframe(
            open_candidates,
            width="stretch",
            hide_index=True,
            column_config={
                "Credit": st.column_config.NumberColumn(format="$%.2f"),
                "Max Risk": st.column_config.NumberColumn(format="$%.2f"),
                "Max Profit": st.column_config.NumberColumn(format="$%.2f"),
            },
        )
    else:
        st.info("No open candidates are being tracked.")

    st.subheader("Results")
    result_rows, result_errors = fetch_expired_history()
    for error in result_errors:
        st.warning(error)
    if not result_rows:
        st.info("No expired candidates have results yet.")
        return

    results = pd.DataFrame(result_rows)
    results["expiration_pnl"] = pd.to_numeric(results["expiration_pnl"])

    expired_count = len(results)
    win_rate = (results["expiration_pnl"] > 0).mean() * 100
    average_pnl = results["expiration_pnl"].mean()
    total_pnl = results["expiration_pnl"].sum()
    metric_columns = st.columns(4)
    metric_columns[0].metric("Expired Candidates", expired_count)
    metric_columns[1].metric("Win Rate", f"{win_rate:.1f}%")
    metric_columns[2].metric("Average Expiration P/L", f"${average_pnl:.2f}")
    metric_columns[3].metric("Total Expiration P/L", f"${total_pnl:.2f}")

    strategy_results = (
        results.groupby("strategy", as_index=False)
        .agg(
            Candidates=("id", "count"),
            Win_Rate=("expiration_pnl", lambda pnl: (pnl > 0).mean() * 100),
            Average_PnL=("expiration_pnl", "mean"),
            Total_PnL=("expiration_pnl", "sum"),
        )
        .rename(
            columns={
                "strategy": "Strategy",
                "Win_Rate": "Win Rate",
                "Average_PnL": "Average P/L",
                "Total_PnL": "Total P/L",
            }
        )
    )
    strategy_results["Win Rate"] = strategy_results["Win Rate"].round(1)
    strategy_results["Average P/L"] = strategy_results["Average P/L"].round(2)
    strategy_results["Total P/L"] = strategy_results["Total P/L"].round(2)
    st.dataframe(
        strategy_results,
        width="stretch",
        hide_index=True,
        column_config={
            "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Average P/L": st.column_config.NumberColumn(format="$%.2f"),
            "Total P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    results["Score Band"] = pd.cut(
        results["setup_score"],
        bins=[-1, 59, 69, 79, 89, 100],
        labels=["Below 60", "60-69", "70-79", "80-89", "90-100"],
    )
    score_results = (
        results.groupby("Score Band", observed=True, as_index=False)
        .agg(
            Candidates=("id", "count"),
            Win_Rate=("expiration_pnl", lambda pnl: (pnl > 0).mean() * 100),
            Average_PnL=("expiration_pnl", "mean"),
            Total_PnL=("expiration_pnl", "sum"),
        )
        .rename(
            columns={
                "Win_Rate": "Win Rate",
                "Average_PnL": "Average P/L",
                "Total_PnL": "Total P/L",
            }
        )
    )
    score_results["Win Rate"] = score_results["Win Rate"].round(1)
    score_results["Average P/L"] = score_results["Average P/L"].round(2)
    score_results["Total P/L"] = score_results["Total P/L"].round(2)
    st.dataframe(
        score_results,
        width="stretch",
        hide_index=True,
        column_config={
            "Win Rate": st.column_config.NumberColumn(format="%.1f%%"),
            "Average P/L": st.column_config.NumberColumn(format="$%.2f"),
            "Total P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    recent_results = results[
        [
            "ticker",
            "strategy",
            "expiration",
            "setup_score",
            "expiration_close",
            "expiration_pnl",
        ]
    ].rename(
        columns={
            "ticker": "Ticker",
            "strategy": "Strategy",
            "expiration": "Expiration",
            "setup_score": "Setup Score",
            "expiration_close": "Expiration Close",
            "expiration_pnl": "Expiration P/L",
        }
    )
    st.dataframe(
        recent_results,
        width="stretch",
        hide_index=True,
        column_config={
            "Expiration Close": st.column_config.NumberColumn(format="$%.2f"),
            "Expiration P/L": st.column_config.NumberColumn(format="$%.2f"),
        },
    )


def render_private_results():
    owner_password = os.getenv("OWNER_DASHBOARD_PASSWORD")
    if not owner_password:
        return

    if st.session_state.get("results_unlocked"):
        with st.sidebar:
            if st.button("Lock Results", width="stretch"):
                st.session_state["results_unlocked"] = False
                st.rerun()
        render_results()
        return

    with st.sidebar:
        entered_password = st.text_input("Owner Password", type="password")
        if entered_password and hmac.compare_digest(entered_password, owner_password):
            st.session_state["results_unlocked"] = True
            st.rerun()


st.title("Options Scanner")

with st.sidebar:
    st.header("Scan Settings")
    ticker_text = st.text_area("Tickers", "AAPL, SPY, QQQ, NVDA, MSFT, COHR")
    outlook = st.selectbox("Outlook", ["neutral", "bullish", "bearish", "income"])
    max_risk = st.number_input("Maximum Risk Per Spread", min_value=50, value=500, step=50)
    risk_tolerance = st.selectbox("Risk Tolerance", ["conservative", "moderate", "aggressive"], index=1)
    use_nearest_expiration = st.checkbox("Use Nearest Available Expiration")
    use_test_expiration = st.checkbox(
        "Test a Specific Expiration", disabled=use_nearest_expiration
    )
    test_expiration = (
        st.date_input("Test Expiration", min_value=date.today())
        if use_test_expiration
        else None
    )
    scan_button = st.button("Scan Watchlist", type="primary", width='stretch')

st.caption("Yahoo Finance data can be delayed. Realized volatility rank is a historical-price proxy, not implied-volatility rank.")

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
    scored_trades, rejected_trades, trades, ticker_data, errors = scan_watchlist(tickers, preferences)
    errors = history_errors + errors
    history_candidates = select_history_candidates(scored_trades)
    save_history(history_candidates)
    st.subheader("Strategy Diagnostics")
    strategy_df = pd.DataFrame(
        strategy_rejection_rows(trades, rejected_trades, scored_trades)
    )
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
    if errors:
        for error in errors:
            st.warning(error)

    st.subheader("Data Snapshot")
    st.dataframe(pd.DataFrame(ticker_data), width='stretch', hide_index=True)

    st.subheader("Top Candidates")
    candidates = candidate_rows(scored_trades)
    if candidates:
        st.dataframe(pd.DataFrame(candidates), width='stretch', hide_index=True)
        top_25_csv = pd.DataFrame(
            [candidate_row(scored) for scored in history_candidates]
        ).to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Top 25 CSV",
            data=top_25_csv,
            file_name="top_25_options_candidates.csv",
            mime="text/csv",
            width="content",
        )
        for scored in select_top_candidates(scored_trades):
            trade = scored.trade
            with st.expander(
                f"{trade.ticker} {trade.strategy.title()} | "
                f"Setup Score {scored.total_score}/100"
            ):
                st.write(scored.explanation)
                st.json(scored.category_scores)
    else:
        st.info("No candidates passed the current filters.")

    st.subheader("Best Debit Spread Candidates")
    debit_candidates = debit_candidate_rows(scored_trades)
    scored_debit_candidates = [
        scored for scored in scored_trades if scored.trade.entry_type == "debit"
    ]
    if debit_candidates:
        st.dataframe(
            pd.DataFrame(debit_candidates),
            width="stretch",
            hide_index=True,
        )
        for scored in scored_debit_candidates[:3]:
            trade = scored.trade
            with st.expander(
                f"{trade.ticker} {trade.strategy.title()} | "
                f"Setup Score {scored.total_score}/100"
            ):
                st.write(scored.explanation)
                st.json(scored.category_scores)
    else:
        st.info("No debit spreads passed the current filters.")

    st.subheader("Best Credit Spread Candidates")
    credit_candidates = credit_candidate_rows(scored_trades)
    scored_credit_candidates = [
        scored for scored in scored_trades if scored.trade.entry_type == "credit"
    ]
    if credit_candidates:
        st.dataframe(
            pd.DataFrame(credit_candidates),
            width="stretch",
            hide_index=True,
        )
        for scored in scored_credit_candidates[:3]:
            trade = scored.trade
            with st.expander(
                f"{trade.ticker} {trade.strategy.title()} | "
                f"Setup Score {scored.total_score}/100"
            ):
                st.write(scored.explanation)
                st.json(scored.category_scores)
    else:
        st.info("No credit spreads passed the current filters.")

    st.subheader("Ticker Status")
    st.dataframe(
        pd.DataFrame(ticker_rejection_rows(trades, rejected_trades, scored_trades)),
        width='stretch',
        hide_index=True,
        column_config={"Rejected %": st.column_config.NumberColumn(format="%.0f%%")},
    )


render_private_results()
