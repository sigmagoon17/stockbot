import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

from stock2dupe import CONTRACT_MULTIPLIER


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")
if not SUPABASE_URL or not SUPABASE_SECRET_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SECRET_KEY must be configured.")

supabase = create_client(SUPABASE_URL, SUPABASE_SECRET_KEY)


def numeric_value(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def append_scan_history(scored_trades, event_analyses=None, price_moves=None):
    event_analyses = event_analyses or {}
    price_moves = price_moves or {}
    rows = []
    scan_timestamp = datetime.now(timezone.utc)
    current_time = scan_timestamp.isoformat()
    current_time_est = scan_timestamp.astimezone(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    for scored in scored_trades:
        trade = scored.trade
        event_analysis = event_analyses.get(trade.ticker)
        price_move = price_moves.get(trade.ticker, {})
        rows.append(
            {
                "scan_time": current_time,
                "scan_time_est": current_time_est,
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
                "put_long_strike": (
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
                "quant_score": scored.quant_score,
                "event_adjustment": scored.event_adjustment,
                "price_move_adjustment": scored.price_move_adjustment,
                "move_setup": scored.price_move_style,
                "event_label": (
                    event_analysis.label if event_analysis is not None else None
                ),
                "event_confidence": (
                    event_analysis.confidence if event_analysis is not None else None
                ),
                "event_summary": (
                    event_analysis.summary if event_analysis is not None else None
                ),
                "daily_move_pct": price_move.get("1D Move %"),
                "five_day_move_pct": price_move.get("5D Move %"),
                "move_vs_20d_vol": price_move.get("Move vs 20D Vol"),
                "unusual_move": price_move.get("Unusual Move"),
                "setup_score": scored.total_score,
                "risk_level": scored.risk_level,
                "dte": trade.dte,
                "volatility_rank": round(trade.volatility_rank, 1),
                "starting_status": "open",
                "expiration_status": "open",
                "expiration_close": None,
                "expiration_pnl": None,
            }
        )

    if not rows:
        return []

    def candidate_key(row):
        return (
            row["ticker"],
            row["strategy"],
            row["expiration"],
            row["long_strike"],
            row["short_strike"],
            row["entry_type"],
        )

    try:
        existing_rows = (
            supabase.table("scan_history")
            .select("ticker,strategy,expiration,long_strike,short_strike,entry_type")
            .gte("scan_time", date.today().isoformat())
            .execute()
            .data
        )
    except Exception as error:
        return [f"Could not check existing scan history: {error}"]
    existing_keys = {candidate_key(row) for row in existing_rows}
    new_rows = []
    for row in rows:
        key = candidate_key(row)
        if key not in existing_keys:
            new_rows.append(row)
            existing_keys.add(key)

    if new_rows:
        try:
            supabase.table("scan_history").insert(new_rows).execute()
        except Exception as error:
            return [f"Could not save scan history: {error}"]

    return []


def option_quote(option_table, strike: float) -> tuple[float, float] | None:
    matches = option_table[option_table["strike"].astype(float) == float(strike)]
    if matches.empty:
        return None

    row = matches.iloc[0]
    bid = numeric_value(row.get("bid"))
    ask = numeric_value(row.get("ask"))
    if bid is None or ask is None or bid <= 0 or ask <= 0:
        return None
    return bid, ask


def manual_option_quote(option_table, strike: float) -> dict[str, float] | None:
    matches = option_table[option_table["strike"].astype(float) == float(strike)]
    if matches.empty:
        return None

    row = matches.iloc[0]
    bid = numeric_value(row.get("bid"))
    ask = numeric_value(row.get("ask"))
    last_price = numeric_value(row.get("lastPrice"))

    has_bid_ask = (
        bid is not None
        and ask is not None
        and bid > 0
        and ask > 0
    )
    if has_bid_ask:
        return {
            "bid": bid,
            "ask": ask,
            "mid": (bid + ask) / 2,
            "quote_source": "bid/ask",
        }

    if last_price is not None and last_price > 0:
        return {
            "bid": last_price,
            "ask": last_price,
            "mid": last_price,
            "quote_source": "last price",
        }

    return None


def current_underlying_price(stock: yf.Ticker) -> float | None:
    try:
        fast_info = stock.fast_info
        price = fast_info.get("lastPrice") or fast_info.get("last_price")
        if price is not None:
            return round(float(price), 2)
    except Exception:
        pass

    history = stock.history(period="5d")
    if history.empty:
        return None
    return round(float(history["Close"].iloc[-1]), 2)


def spread_mark_and_pnl(row, chain) -> tuple[float, float] | None:
    strategy = row["strategy"]
    long_strike = float(row["long_strike"])
    short_strike = float(row["short_strike"])
    credit = float(row.get("credit") or 0)
    max_risk = float(row.get("max_risk") or 0)

    if strategy == "bull call debit spread":
        long_quote = option_quote(chain.calls, long_strike)
        short_quote = option_quote(chain.calls, short_strike)
        if long_quote is None or short_quote is None:
            return None
        long_bid, _ = long_quote
        _, short_ask = short_quote
        current_mark = max(0, round(long_bid - short_ask, 2))
        pnl = round(current_mark * CONTRACT_MULTIPLIER - max_risk, 2)
        return current_mark, pnl

    if strategy == "bear put debit spread":
        long_quote = option_quote(chain.puts, long_strike)
        short_quote = option_quote(chain.puts, short_strike)
        if long_quote is None or short_quote is None:
            return None
        long_bid, _ = long_quote
        _, short_ask = short_quote
        current_mark = max(0, round(long_bid - short_ask, 2))
        pnl = round(current_mark * CONTRACT_MULTIPLIER - max_risk, 2)
        return current_mark, pnl

    if strategy == "put credit spread":
        short_quote = option_quote(chain.puts, short_strike)
        long_quote = option_quote(chain.puts, long_strike)
        if short_quote is None or long_quote is None:
            return None
        _, short_ask = short_quote
        long_bid, _ = long_quote
        current_mark = max(0, round(short_ask - long_bid, 2))
        pnl = round(credit - current_mark * CONTRACT_MULTIPLIER, 2)
        return current_mark, pnl

    if strategy == "call credit spread":
        short_quote = option_quote(chain.calls, short_strike)
        long_quote = option_quote(chain.calls, long_strike)
        if short_quote is None or long_quote is None:
            return None
        _, short_ask = short_quote
        long_bid, _ = long_quote
        current_mark = max(0, round(short_ask - long_bid, 2))
        pnl = round(credit - current_mark * CONTRACT_MULTIPLIER, 2)
        return current_mark, pnl

    if strategy == "iron condor":
        put_short = row.get("put_short_strike")
        put_long = row.get("put_long_strike")
        call_short = row.get("call_short_strike")
        call_long = row.get("call_long_strike")
        if any(strike is None for strike in (put_short, put_long, call_short, call_long)):
            return None

        put_short_quote = option_quote(chain.puts, float(put_short))
        put_long_quote = option_quote(chain.puts, float(put_long))
        call_short_quote = option_quote(chain.calls, float(call_short))
        call_long_quote = option_quote(chain.calls, float(call_long))
        if any(
            quote is None
            for quote in (
                put_short_quote,
                put_long_quote,
                call_short_quote,
                call_long_quote,
            )
        ):
            return None

        _, put_short_ask = put_short_quote
        put_long_bid, _ = put_long_quote
        _, call_short_ask = call_short_quote
        call_long_bid, _ = call_long_quote
        put_mark = max(0, put_short_ask - put_long_bid)
        call_mark = max(0, call_short_ask - call_long_bid)
        current_mark = round(put_mark + call_mark, 2)
        pnl = round(credit - current_mark * CONTRACT_MULTIPLIER, 2)
        return current_mark, pnl

    return None


def snapshot_exit_signal(row, unrealized_pnl: float) -> str:
    max_profit = numeric_value(row.get("max_profit")) or numeric_value(row.get("credit")) or 0
    max_risk = numeric_value(row.get("max_risk")) or 0

    if max_profit > 0 and unrealized_pnl >= max_profit * 0.75:
        return "take_profit_75"
    if max_profit > 0 and unrealized_pnl >= max_profit * 0.50:
        return "take_profit_50"
    if max_risk > 0 and unrealized_pnl <= -max_risk * 0.50:
        return "stop_loss_50"
    return "hold"


def updated_pnl_extremes(row, unrealized_pnl: float) -> tuple[float, float]:
    previous_high = numeric_value(row.get("highest_unrealized_pnl"))
    previous_low = numeric_value(row.get("lowest_unrealized_pnl"))

    highest = (
        unrealized_pnl
        if previous_high is None
        else max(previous_high, unrealized_pnl)
    )
    lowest = (
        unrealized_pnl
        if previous_low is None
        else min(previous_low, unrealized_pnl)
    )
    return round(highest, 2), round(lowest, 2)


def append_trade_snapshots() -> list[str]:
    errors = []
    open_rows, open_errors = fetch_open_history(limit=1000)
    if open_errors:
        return open_errors
    if not open_rows:
        return []

    snapshot_timestamp = datetime.now(timezone.utc)
    snapshot_time = snapshot_timestamp.isoformat()
    snapshot_time_est = snapshot_timestamp.astimezone(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %I:%M:%S %p %Z")

    stocks = {}
    chains = {}
    prices = {}
    snapshot_rows = []
    pnl_extreme_updates = []

    for row in open_rows:
        ticker = row["ticker"]
        expiration = row["expiration"]
        key = (ticker, expiration)

        try:
            if ticker not in stocks:
                stocks[ticker] = yf.Ticker(ticker)
            stock = stocks[ticker]

            if ticker not in prices:
                prices[ticker] = current_underlying_price(stock)
            underlying_price = prices[ticker]

            if key not in chains:
                chains[key] = stock.option_chain(expiration)
            mark_and_pnl = spread_mark_and_pnl(row, chains[key])
        except Exception as error:
            errors.append(f"Could not snapshot {ticker} {expiration}: {error}")
            continue

        if underlying_price is None or mark_and_pnl is None:
            continue

        current_mark, unrealized_pnl = mark_and_pnl
        max_profit = numeric_value(row.get("max_profit")) or numeric_value(row.get("credit")) or 0
        max_risk = numeric_value(row.get("max_risk")) or 0
        expiration_date = date.fromisoformat(expiration)
        highest_unrealized_pnl, lowest_unrealized_pnl = updated_pnl_extremes(
            row, unrealized_pnl
        )

        snapshot_rows.append(
            {
                "scan_history_id": row["id"],
                "snapshot_time": snapshot_time,
                "snapshot_time_est": snapshot_time_est,
                "ticker": ticker,
                "strategy": row["strategy"],
                "expiration": expiration,
                "dte": (expiration_date - date.today()).days,
                "underlying_price": underlying_price,
                "current_spread_mark": round(current_mark * CONTRACT_MULTIPLIER, 2),
                "unrealized_pnl": unrealized_pnl,
                "pnl_percent_of_max_profit": (
                    round(unrealized_pnl / max_profit * 100, 2)
                    if max_profit > 0
                    else None
                ),
                "pnl_percent_of_max_risk": (
                    round(unrealized_pnl / max_risk * 100, 2)
                    if max_risk > 0
                    else None
                ),
                "exit_signal": snapshot_exit_signal(row, unrealized_pnl),
            }
        )
        pnl_extreme_updates.append(
            {
                "id": row["id"],
                "highest_unrealized_pnl": highest_unrealized_pnl,
                "lowest_unrealized_pnl": lowest_unrealized_pnl,
            }
        )

    if not snapshot_rows:
        return errors

    try:
        supabase.table("trade_snapshots").insert(snapshot_rows).execute()
    except Exception as error:
        errors.append(f"Could not save trade snapshots: {error}")

    for update in pnl_extreme_updates:
        try:
            (
                supabase.table("scan_history")
                .update(
                    {
                        "highest_unrealized_pnl": update["highest_unrealized_pnl"],
                        "lowest_unrealized_pnl": update["lowest_unrealized_pnl"],
                    }
                )
                .eq("id", update["id"])
                .execute()
            )
        except Exception as error:
            errors.append(
                f"Could not update P/L extremes for scan_history "
                f"{update['id']}: {error}"
            )

    return errors


def add_manual_position(
    ticker: str,
    strategy: str,
    expiration: date,
    long_strike: float,
    short_strike: float,
    entry_price: float,
    quantity: int,
    note: str,
) -> list[str]:
    try:
        supabase.table("manual_positions").insert(
            {
                "ticker": ticker.upper().strip(),
                "strategy": strategy,
                "expiration": expiration.isoformat(),
                "long_strike": round(long_strike, 2),
                "short_strike": round(short_strike, 2),
                "entry_price": round(entry_price, 2),
                "quantity": int(quantity),
                "note": note.strip() or None,
            }
        ).execute()
        return []
    except Exception as error:
        return [f"Could not add manual position: {error}"]


def fetch_manual_positions(status: str = "open") -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("manual_positions")
            .select("*")
            .eq("status", status)
            .order("created_at", desc=True)
            .execute()
        )
        return response.data, []
    except Exception as error:
        return [], [f"Could not load manual positions: {error}"]


def close_manual_position(record_id: int) -> list[str]:
    try:
        (
            supabase.table("manual_positions")
            .update({"status": "closed"})
            .eq("id", record_id)
            .execute()
        )
        return []
    except Exception as error:
        return [f"Could not close manual position: {error}"]


def delete_manual_position(record_id: int) -> list[str]:
    try:
        (
            supabase.table("manual_positions")
            .delete()
            .eq("id", record_id)
            .execute()
        )
        return []
    except Exception as error:
        return [f"Could not delete manual position: {error}"]


def quote_midpoint(quote: tuple[float, float]) -> float:
    bid, ask = quote
    return (bid + ask) / 2


def manual_position_mark_and_pnl(row, chain) -> dict[str, float] | None:
    strategy = row["strategy"]
    long_strike = float(row["long_strike"])
    short_strike = float(row["short_strike"])
    entry_price = float(row["entry_price"])
    quantity = int(row.get("quantity") or 1)

    if strategy == "bull call debit spread":
        long_quote = manual_option_quote(chain.calls, long_strike)
        short_quote = manual_option_quote(chain.calls, short_strike)
        if long_quote is None or short_quote is None:
            return None
        conservative_mark = max(0, round(long_quote["bid"] - short_quote["ask"], 2))
        midpoint_mark = max(
            0,
            round(long_quote["mid"] - short_quote["mid"], 2),
        )
        return {
            "conservative_mark": conservative_mark,
            "conservative_pnl": round(
                (conservative_mark - entry_price) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "midpoint_mark": midpoint_mark,
            "midpoint_pnl": round(
                (midpoint_mark - entry_price) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "quote_source": (
                "last price"
                if "last price" in {long_quote["quote_source"], short_quote["quote_source"]}
                else "bid/ask"
            ),
        }

    if strategy == "bear put debit spread":
        long_quote = manual_option_quote(chain.puts, long_strike)
        short_quote = manual_option_quote(chain.puts, short_strike)
        if long_quote is None or short_quote is None:
            return None
        conservative_mark = max(0, round(long_quote["bid"] - short_quote["ask"], 2))
        midpoint_mark = max(
            0,
            round(long_quote["mid"] - short_quote["mid"], 2),
        )
        return {
            "conservative_mark": conservative_mark,
            "conservative_pnl": round(
                (conservative_mark - entry_price) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "midpoint_mark": midpoint_mark,
            "midpoint_pnl": round(
                (midpoint_mark - entry_price) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "quote_source": (
                "last price"
                if "last price" in {long_quote["quote_source"], short_quote["quote_source"]}
                else "bid/ask"
            ),
        }

    if strategy == "put credit spread":
        short_quote = manual_option_quote(chain.puts, short_strike)
        long_quote = manual_option_quote(chain.puts, long_strike)
        if short_quote is None or long_quote is None:
            return None
        conservative_mark = max(0, round(short_quote["ask"] - long_quote["bid"], 2))
        midpoint_mark = max(
            0,
            round(short_quote["mid"] - long_quote["mid"], 2),
        )
        return {
            "conservative_mark": conservative_mark,
            "conservative_pnl": round(
                (entry_price - conservative_mark) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "midpoint_mark": midpoint_mark,
            "midpoint_pnl": round(
                (entry_price - midpoint_mark) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "quote_source": (
                "last price"
                if "last price" in {long_quote["quote_source"], short_quote["quote_source"]}
                else "bid/ask"
            ),
        }

    if strategy == "call credit spread":
        short_quote = manual_option_quote(chain.calls, short_strike)
        long_quote = manual_option_quote(chain.calls, long_strike)
        if short_quote is None or long_quote is None:
            return None
        conservative_mark = max(0, round(short_quote["ask"] - long_quote["bid"], 2))
        midpoint_mark = max(
            0,
            round(short_quote["mid"] - long_quote["mid"], 2),
        )
        return {
            "conservative_mark": conservative_mark,
            "conservative_pnl": round(
                (entry_price - conservative_mark) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "midpoint_mark": midpoint_mark,
            "midpoint_pnl": round(
                (entry_price - midpoint_mark) * CONTRACT_MULTIPLIER * quantity,
                2,
            ),
            "quote_source": (
                "last price"
                if "last price" in {long_quote["quote_source"], short_quote["quote_source"]}
                else "bid/ask"
            ),
        }

    return None


def manual_position_recommendation(row, pnl: float | None, dte: int) -> str:
    if pnl is None:
        return "Needs quote"

    entry_price = float(row["entry_price"])
    quantity = int(row.get("quantity") or 1)
    entry_value = entry_price * CONTRACT_MULTIPLIER * quantity

    if entry_value > 0 and pnl >= entry_value * 0.5:
        return "Consider taking profit"
    if entry_value > 0 and pnl <= -entry_value * 0.5:
        return "Review risk"
    if dte <= 7:
        return "Expiration close"
    return "Hold"


def manual_position_rows_with_marks() -> tuple[list[dict], list[str]]:
    positions, errors = fetch_manual_positions()
    if errors or not positions:
        return [], errors

    rows = []
    stocks = {}
    chains = {}
    prices = {}

    for position in positions:
        ticker = position["ticker"]
        expiration = position["expiration"]
        key = (ticker, expiration)
        try:
            if ticker not in stocks:
                stocks[ticker] = yf.Ticker(ticker)
            stock = stocks[ticker]
            if ticker not in prices:
                prices[ticker] = current_underlying_price(stock)
            if key not in chains:
                chains[key] = stock.option_chain(expiration)
            mark_and_pnl = manual_position_mark_and_pnl(position, chains[key])
        except Exception as error:
            errors.append(f"Could not price {ticker} manual position: {error}")
            mark_and_pnl = None

        expiration_date = date.fromisoformat(expiration)
        dte = (expiration_date - date.today()).days
        midpoint_mark = None
        midpoint_pnl = None
        conservative_mark = None
        conservative_pnl = None
        quote_source = None
        if mark_and_pnl is not None:
            midpoint_mark = mark_and_pnl["midpoint_mark"]
            midpoint_pnl = mark_and_pnl["midpoint_pnl"]
            conservative_mark = mark_and_pnl["conservative_mark"]
            conservative_pnl = mark_and_pnl["conservative_pnl"]
            quote_source = mark_and_pnl["quote_source"]

        rows.append(
            {
                "id": position["id"],
                "ticker": ticker,
                "strategy": position["strategy"],
                "expiration": expiration,
                "dte": dte,
                "long_strike": position["long_strike"],
                "short_strike": position["short_strike"],
                "entry_price": position["entry_price"],
                "quantity": position["quantity"],
                "underlying_price": prices.get(ticker),
                "current_mark": (
                    round(midpoint_mark * CONTRACT_MULTIPLIER, 2)
                    if midpoint_mark is not None
                    else None
                ),
                "unrealized_pnl": midpoint_pnl,
                "conservative_value": (
                    round(conservative_mark * CONTRACT_MULTIPLIER, 2)
                    if conservative_mark is not None
                    else None
                ),
                "conservative_pnl": conservative_pnl,
                "quote_source": quote_source,
                "recommendation": manual_position_recommendation(
                    position, midpoint_pnl, dte
                ),
                "note": position.get("note"),
            }
        )

    return rows, errors


def expiration_close(ticker: str, expiration: date) -> float | None:
    history = yf.Ticker(ticker).history(
        start=expiration.isoformat(),
        end=(expiration + timedelta(days=1)).isoformat(),
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
        condor_strikes = (
            row.get("put_short_strike"),
            row.get("put_long_strike"),
            row.get("call_short_strike"),
            row.get("call_long_strike"),
        )
        if any(strike is None for strike in condor_strikes):
            return None

        put_short, put_long, call_short, call_long = map(float, condor_strikes)
        put_loss = max(0, min(put_short - closing_price, put_short - put_long))
        call_loss = max(0, min(closing_price - call_short, call_long - call_short))
        return round(credit - (put_loss + call_loss) * CONTRACT_MULTIPLIER, 2)

    return None


def update_expired_history(include_today: bool = False) -> list[str]:
    errors = []
    closing_prices = {}
    try:
        query = (
            supabase.table("scan_history")
            .select("*")
            .eq("expiration_status", "open")
        )
        if include_today:
            query = query.lte("expiration", date.today().isoformat())
        else:
            query = query.lt("expiration", date.today().isoformat())
        response = query.execute()
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

        try:
            pnl = expiration_pnl(row, closing_price)
        except (TypeError, ValueError) as error:
            errors.append(f"Could not calculate {row['ticker']} expiration P/L: {error}")
            continue

        update_values = {
            "expiration_close": closing_price,
            "expiration_status": "manual review" if pnl is None else "expired",
        }
        if pnl is not None:
            update_values["expiration_pnl"] = pnl

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


def fetch_completed_history() -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("scan_history")
            .select("*")
            .in_("expiration_status", ["expired", "closed early"])
            .order("expiration", desc=True)
            .execute()
        )
        return response.data, []
    except Exception as error:
        return [], [f"Could not load results from Supabase: {error}"]


def fetch_open_history(limit: int = 100) -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("scan_history")
            .select("*")
            .eq("expiration_status", "open")
            .order("scan_time", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data, []
    except Exception as error:
        return [], [f"Could not load open candidates from Supabase: {error}"]


def close_candidate(
    record_id: int,
    close_date: date,
    realized_pnl: float,
    note: str,
) -> list[str]:
    try:
        (
            supabase.table("scan_history")
            .update(
                {
                    "expiration_status": "closed early",
                    "actual_close_date": close_date.isoformat(),
                    "actual_realized_pnl": round(realized_pnl, 2),
                    "close_note": note.strip() or None,
                }
            )
            .eq("id", record_id)
            .execute()
        )
        return []
    except Exception as error:
        return [f"Could not close candidate: {error}"]
