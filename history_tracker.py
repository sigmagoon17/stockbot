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


def append_scan_history(scored_trades, event_analyses=None):
    event_analyses = event_analyses or {}
    rows = []
    scan_timestamp = datetime.now(timezone.utc)
    current_time = scan_timestamp.isoformat()
    current_time_est = scan_timestamp.astimezone(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    for scored in scored_trades:
        trade = scored.trade
        event_analysis = event_analyses.get(trade.ticker)
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
                "event_label": (
                    event_analysis.label if event_analysis is not None else None
                ),
                "event_confidence": (
                    event_analysis.confidence if event_analysis is not None else None
                ),
                "event_summary": (
                    event_analysis.summary if event_analysis is not None else None
                ),
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


def fetch_open_history() -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("scan_history")
            .select("*")
            .eq("expiration_status", "open")
            .order("scan_time", desc=True)
            .limit(100)
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
