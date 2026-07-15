import os
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import yfinance as yf
from dotenv import load_dotenv
from supabase import create_client

from alpaca_client import (
    get_alpaca_order,
    get_alpaca_order_by_client_id,
    get_alpaca_positions,
    spread_width_from_order_legs,
    submit_multileg_order,
)
from paper_exit import (
    calculate_filled_trade_economics,
    closing_legs_from_leg_key,
    evaluate_paper_exit,
    submit_claimed_paper_exit,
)
from scanner_tracking import (
    SCANNER_VERSION,
    SELECTION_DIVERSIFIED,
    SELECTION_EXECUTION,
    git_commit_sha,
    new_scan_run_id,
    normalize_history_row,
    setup_key_for_trade,
    utc_now_iso,
)
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


def filled_economics_values(
    entry_type: str,
    spread_width_per_share,
    opening_filled_avg_price,
) -> dict:
    economics = calculate_filled_trade_economics(
        entry_type,
        numeric_value(spread_width_per_share),
        numeric_value(opening_filled_avg_price),
    )
    return {
        "spread_width_per_share": economics.spread_width_per_share,
        "filled_max_profit_per_share": economics.filled_max_profit_per_share,
        "filled_max_risk_per_share": economics.filled_max_risk_per_share,
        "fill_validation_error": economics.validation_error,
    }


def append_alpaca_paper_orders(order_results: list[dict]) -> list[str]:
    rows = []
    scan_timestamp = datetime.now(timezone.utc)
    current_time = scan_timestamp.isoformat()
    current_time_est = scan_timestamp.astimezone(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %I:%M:%S %p %Z")

    for result in order_results:
        if result.get("Status") in {"Skipped", "Error"}:
            continue

        opening_status = str(
            result.get("Opening Order Status") or result.get("Status") or "unknown"
        ).lower()
        opening_fill = numeric_value(result.get("Opening Filled Avg Price"))
        if opening_fill is not None:
            opening_fill = abs(opening_fill)
        opening_filled_at = result.get("Opening Filled At")
        spread_width = numeric_value(result.get("Spread Width Per Share"))
        fill_economics = filled_economics_values(
            result.get("Entry Type") or "",
            spread_width,
            opening_fill if opening_status == "filled" else None,
        )
        if (
            opening_status == "filled"
            and opening_fill is not None
            and opening_fill > 0
        ):
            position_status = "open"
        elif opening_status in {"canceled", "rejected", "expired"}:
            position_status = opening_status
        else:
            position_status = "pending"

        rows.append(
            {
                "scan_time": current_time,
                "scan_time_est": current_time_est,
                "entry_timestamp": opening_filled_at,
                "order_id": result.get("Order ID"),
                "client_order_id": result.get("Client Order ID"),
                "ticker": result.get("Ticker"),
                "strategy": result.get("Strategy"),
                "expiration": result.get("Expiration"),
                "setup_score": result.get("Setup Score"),
                "ticker_score": result.get("Ticker Score"),
                "quant_score": result.get("Quant Score"),
                "setup_key": result.get("Setup Key"),
                "scan_run_id": result.get("Scan Run ID"),
                "execution_rank": result.get("Execution Rank"),
                "selection_method": result.get("Selection Method"),
                "entry_type": result.get("Entry Type"),
                "limit_price": result.get("Limit Price"),
                "entry_price": opening_fill,
                "opening_order_status": opening_status,
                "opening_filled_at": opening_filled_at,
                "opening_filled_avg_price": opening_fill,
                **fill_economics,
                "max_profit": result.get("Max Profit"),
                "max_risk": result.get("Max Risk"),
                "quantity": result.get("Quantity"),
                "order_class": result.get("Order Class"),
                "symbol": result.get("Symbol"),
                "status": result.get("Status"),
                "message": result.get("Message"),
                "leg_key": result.get("Leg Key"),
                "exit_policy": result.get("Exit Policy") or "none",
                "position_status": position_status,
            }
        )

    if not rows:
        return []

    try:
        existing_response = (
            supabase.table("alpaca_paper_orders")
            .select("order_id,client_order_id,leg_key,position_status")
            .execute()
        )
        existing_order_keys = {
            value
            for row in existing_response.data
            for value in (
                row.get("order_id"),
                row.get("client_order_id"),
            )
            if value
        }
        existing_leg_keys = active_leg_keys(existing_response.data)

        new_rows = []
        queued_keys = set()
        for row in rows:
            order_keys = {row.get("order_id"), row.get("client_order_id")}
            order_keys.discard(None)
            leg_key = row.get("leg_key")
            if (
                order_keys & existing_order_keys
                or order_keys & queued_keys
                or leg_key in existing_leg_keys
                or leg_key in queued_keys
            ):
                continue

            queued_keys.update(order_keys)
            if leg_key:
                queued_keys.add(leg_key)
            new_rows.append(row)

        if new_rows:
            supabase.table("alpaca_paper_orders").insert(new_rows).execute()
        return []
    except Exception as error:
        return [f"Could not save Alpaca paper orders: {error}"]


def fetch_alpaca_paper_orders(limit: int = 250) -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("alpaca_paper_orders")
            .select("*")
            .order("scan_time", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data, []
    except Exception as error:
        return [], [f"Could not load Alpaca paper order history: {error}"]


def fetch_alpaca_paper_leg_keys() -> tuple[set[str], list[str]]:
    try:
        response = (
            supabase.table("alpaca_paper_orders")
            .select("leg_key")
            .in_("position_status", ["pending", "open"])
            .not_.is_("leg_key", "null")
            .execute()
        )
        return {
            row["leg_key"]
            for row in response.data
            if row.get("leg_key")
        }, []
    except Exception as error:
        return set(), [f"Could not load existing Alpaca paper leg keys: {error}"]


def active_leg_keys(order_rows: list[dict]) -> set[str]:
    return {
        row["leg_key"]
        for row in order_rows
        if row.get("leg_key")
        and str(row.get("position_status") or "").lower() in {"pending", "open"}
    }


def symbols_from_leg_key(leg_key: str | None) -> list[str]:
    if not leg_key:
        return []
    return [
        leg_part.split(":", 1)[0]
        for leg_part in leg_key.split("|")
        if leg_part
    ]


def spread_width_for_tracked_order(order: dict) -> float | None:
    stored_width = numeric_value(order.get("spread_width_per_share"))
    if stored_width is not None:
        return stored_width
    symbols = symbols_from_leg_key(order.get("leg_key"))
    if not symbols:
        return None
    return spread_width_from_order_legs(
        [{"symbol": symbol} for symbol in symbols],
        str(order.get("strategy") or "").lower(),
    )


OPENING_TERMINAL_STATUSES = {"canceled", "rejected", "expired"}
CLOSE_RETRYABLE_STATUSES = {"rejected", "canceled", "expired"}
CLOSE_BLOCKING_STATUSES = {
    "accepted",
    "new",
    "pending_new",
    "partially_filled",
    "filled",
    "submitting",
}


def opening_order_update_values(
    alpaca_order: dict,
    tracked_order: dict | None = None,
) -> dict:
    status = str(alpaca_order.get("status") or "unknown").lower()
    filled_price = numeric_value(alpaca_order.get("filled_avg_price"))
    if filled_price is not None:
        filled_price = abs(filled_price)
    filled_at = alpaca_order.get("filled_at")
    values = {
        "opening_order_status": status,
        "opening_filled_at": filled_at,
        "opening_filled_avg_price": filled_price,
    }
    if status == "filled" and filled_price is not None and filled_price > 0:
        tracked_order = tracked_order or {}
        values.update(
            {
                "position_status": "open",
                "entry_timestamp": filled_at,
                "entry_price": filled_price,
                **filled_economics_values(
                    tracked_order.get("entry_type") or "",
                    spread_width_for_tracked_order(tracked_order),
                    filled_price,
                ),
            }
        )
    elif status in OPENING_TERMINAL_STATUSES:
        values["position_status"] = status
    else:
        values["position_status"] = "pending"
    return values


def paper_order_is_active(order: dict) -> bool:
    opening_status = str(
        order.get("opening_order_status") or order.get("status") or ""
    ).lower()
    position_status = str(order.get("position_status") or "").lower()
    close_status = str(order.get("close_order_status") or "").lower()
    return (
        opening_status == "filled"
        and (numeric_value(order.get("opening_filled_avg_price")) or 0) > 0
        and position_status == "open"
        and close_status != "filled"
    )


def close_attempt_is_allowed(order: dict, attempt_run_id: str | None = None) -> bool:
    close_status = str(order.get("close_order_status") or "").lower()
    if close_status in CLOSE_BLOCKING_STATUSES:
        return False
    if close_status and close_status not in CLOSE_RETRYABLE_STATUSES:
        return False
    if close_status in CLOSE_RETRYABLE_STATUSES and not bool(
        order.get("exit_retryable")
    ):
        return False
    if attempt_run_id and order.get("close_attempt_run_id") == attempt_run_id:
        return False
    return paper_order_is_active(order)


def close_client_order_id_for_attempt(
    order: dict,
    attempt_number: int,
    exit_reason: str,
) -> str:
    identity = str(order.get("setup_key") or "unknown")[:12]
    return (
        f"close-{identity}-{order.get('id')}-a{attempt_number}-{exit_reason}"
    )[:48]


def active_symbol_order_counts(paper_orders: list[dict]) -> dict[str, int]:
    counts = {}
    for order in paper_orders:
        if not paper_order_is_active(order):
            continue
        for symbol in set(symbols_from_leg_key(order.get("leg_key"))):
            counts[symbol] = counts.get(symbol, 0) + 1
    return counts


def paper_exit_decision_for_order(
    order: dict,
    current_value_per_share: float | None,
):
    if not paper_order_is_active(order):
        return evaluate_paper_exit(
            entry_type=order.get("entry_type") or "",
            entry_price_per_share=0,
            max_profit_per_share=abs(numeric_value(order.get("max_profit")) or 0),
            current_value_per_share=None,
            policy="none",
        )
    filled_max_profit = numeric_value(order.get("filled_max_profit_per_share"))
    if filled_max_profit is None or filled_max_profit <= 0:
        return evaluate_paper_exit(
            entry_type=order.get("entry_type") or "",
            entry_price_per_share=0,
            max_profit_per_share=0,
            current_value_per_share=None,
            policy="none",
        )
    close_status = order.get("close_order_status")
    effective_close_status = (
        None if close_attempt_is_allowed(order) else close_status
    )
    return evaluate_paper_exit(
        entry_type=order.get("entry_type") or "",
        entry_price_per_share=abs(
            numeric_value(order.get("opening_filled_avg_price")) or 0
        ),
        max_profit_per_share=abs(filled_max_profit),
        current_value_per_share=current_value_per_share,
        policy=order.get("exit_policy"),
        close_order_status=effective_close_status,
    )


def append_alpaca_paper_snapshots() -> list[str]:
    errors = []
    paper_orders, order_errors = fetch_alpaca_paper_orders(limit=1000)
    errors.extend(order_errors)
    if order_errors:
        return errors
    if not paper_orders:
        return []

    refreshed_orders = []
    for original_order in paper_orders:
        order, opening_errors = _refresh_paper_opening_order(original_order)
        errors.extend(opening_errors)
        order, close_errors = _refresh_paper_close_order(order)
        errors.extend(close_errors)
        if str(order.get("close_order_status") or "").lower() == "filled":
            continue
        refreshed_orders.append(order)

    positions, position_errors = get_alpaca_positions()
    errors.extend(position_errors)
    if position_errors:
        return errors

    positions_by_symbol = {
        position.get("symbol"): position
        for position in positions
        if position.get("symbol")
    }
    snapshot_timestamp = datetime.now(timezone.utc)
    snapshot_time = snapshot_timestamp.isoformat()
    snapshot_time_est = snapshot_timestamp.astimezone(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    rows = []
    pnl_extreme_updates = []
    symbol_order_counts = active_symbol_order_counts(refreshed_orders)

    for order in refreshed_orders:
        if not paper_order_is_active(order):
            continue

        symbols = symbols_from_leg_key(order.get("leg_key"))
        if not symbols:
            continue

        matched_positions = [
            positions_by_symbol[symbol]
            for symbol in symbols
            if symbol in positions_by_symbol
        ]
        current_value = sum(
            numeric_value(position.get("market_value")) or 0
            for position in matched_positions
        )
        unrealized_pnl = sum(
            numeric_value(position.get("unrealized_pl")) or 0
            for position in matched_positions
        )
        risk_basis = (
            abs(numeric_value(order.get("filled_max_risk_per_share")) or 0)
            * int(order.get("quantity") or 1)
            * CONTRACT_MULTIPLIER
        )
        quantity = int(order.get("quantity") or 1)
        all_legs_matched = len(matched_positions) == len(symbols)
        ambiguous_allocation = any(
            symbol_order_counts.get(symbol, 0) > 1 for symbol in symbols
        )
        current_value_per_share = (
            round(abs(current_value) / (quantity * CONTRACT_MULTIPLIER), 4)
            if all_legs_matched and not ambiguous_allocation and quantity > 0
            else None
        )
        exit_decision = paper_exit_decision_for_order(
            order,
            current_value_per_share,
        )
        previous_mfe = numeric_value(order.get("maximum_favorable_excursion"))
        previous_mae = numeric_value(order.get("maximum_adverse_excursion"))
        mfe = unrealized_pnl if previous_mfe is None else max(previous_mfe, unrealized_pnl)
        mae = unrealized_pnl if previous_mae is None else min(previous_mae, unrealized_pnl)
        pnl_extreme_updates.append(
            {
                "id": order.get("id"),
                "maximum_favorable_excursion": round(mfe, 2),
                "maximum_adverse_excursion": round(mae, 2),
            }
        )

        rows.append(
            {
                "alpaca_paper_order_id": order.get("id"),
                "snapshot_time": snapshot_time,
                "snapshot_time_est": snapshot_time_est,
                "order_id": order.get("order_id"),
                "client_order_id": order.get("client_order_id"),
                "ticker": order.get("ticker"),
                "strategy": order.get("strategy"),
                "expiration": order.get("expiration"),
                "entry_type": order.get("entry_type"),
                "limit_price": order.get("limit_price"),
                "quantity": order.get("quantity"),
                "current_value": round(current_value, 2),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_percent": (
                    round(unrealized_pnl / risk_basis * 100, 2)
                    if risk_basis > 0
                    else None
                ),
                "matched_legs": len(matched_positions),
                "total_legs": len(symbols),
                "leg_key": order.get("leg_key"),
                "exit_policy": order.get("exit_policy") or "none",
                "target_value_per_share": exit_decision.target_value_per_share,
                "current_value_per_share": current_value_per_share,
                "exit_signal": exit_decision.exit_reason or "hold",
            }
        )

        if exit_decision.should_close:
            errors.extend(
                _submit_paper_exit(
                    order,
                    exit_decision.exit_reason,
                    current_value_per_share,
                    snapshot_time,
                )
            )

    if not rows:
        return []

    try:
        supabase.table("alpaca_paper_position_snapshots").insert(rows).execute()
    except Exception as error:
        return [f"Could not save Alpaca paper position snapshots: {error}"]

    for update in pnl_extreme_updates:
        try:
            (
                supabase.table("alpaca_paper_orders")
                .update(
                    {
                        "maximum_favorable_excursion": update["maximum_favorable_excursion"],
                        "maximum_adverse_excursion": update["maximum_adverse_excursion"],
                    }
                )
                .eq("id", update["id"])
                .execute()
            )
        except Exception as error:
            errors.append(f"Could not update paper P/L extremes: {error}")
    return errors


def _submit_paper_exit(
    order: dict,
    exit_reason: str,
    current_value_per_share: float,
    signal_time: str,
) -> list[str]:
    attempt_run_id = signal_time
    if not close_attempt_is_allowed(order, attempt_run_id):
        return []
    closing_legs = closing_legs_from_leg_key(order.get("leg_key") or "")
    if not closing_legs:
        return [f"Could not close Alpaca paper order {order.get('order_id')}: missing legs."]

    attempt_number = int(order.get("close_attempt_count") or 0) + 1
    client_order_id = close_client_order_id_for_attempt(
        order,
        attempt_number,
        exit_reason,
    )
    claim_errors = []

    def claim() -> dict | None:
        try:
            response = supabase.rpc(
                "claim_alpaca_paper_exit",
                {
                    "p_order_id": order["id"],
                    "p_exit_reason": exit_reason,
                    "p_signal_time": signal_time,
                    "p_close_client_order_id": client_order_id,
                    "p_attempt_number": attempt_number,
                    "p_attempt_run_id": attempt_run_id,
                },
            ).execute()
        except Exception as error:
            claim_errors.append(
                f"Could not reserve paper exit for order {order.get('order_id')}: {error}"
            )
            return None
        claimed_rows = response.data or []
        if len(claimed_rows) == 0:
            return None
        if len(claimed_rows) != 1:
            claim_errors.append(
                f"Exit claim for order {order.get('order_id')} returned "
                f"{len(claimed_rows)} rows; no order was submitted."
            )
            return None
        return claimed_rows[0]

    def submit(claimed_order: dict) -> tuple[dict | None, list[str]]:
        return submit_multileg_order(
            closing_legs,
            quantity=int(claimed_order.get("quantity") or 1),
            limit_price=max(0.01, round(current_value_per_share, 2)),
            client_order_id=client_order_id,
        )

    def record_accepted(claimed_order: dict, close_order: dict) -> None:
        response = (
            supabase.table("alpaca_paper_orders")
            .update(
                {
                    "close_order_id": close_order.get("id"),
                    "close_order_status": close_order.get("status") or "accepted",
                    "close_order_submitted_at": signal_time,
                    "exit_retryable": False,
                }
            )
            .eq("id", claimed_order["id"])
            .eq("close_client_order_id", client_order_id)
            .execute()
        )
        if len(response.data or []) != 1:
            raise RuntimeError(
                "accepted closing order update did not match exactly one tracked row"
            )

    def record_rejected(claimed_order: dict, message: str) -> None:
        (
            supabase.table("alpaca_paper_orders")
            .update(
                {
                    "close_order_status": "rejected",
                    "exit_reason": "order_rejected",
                    "last_exit_error": message,
                    "exit_retryable": True,
                }
            )
            .eq("id", claimed_order["id"])
            .eq("close_client_order_id", client_order_id)
            .execute()
        )

    result = submit_claimed_paper_exit(
        claim=claim,
        submit=submit,
        record_accepted=record_accepted,
        record_rejected=record_rejected,
    )
    errors = claim_errors + list(result.errors)
    return [
        f"Paper exit for order {order.get('order_id')}: {message}"
        for message in errors
    ]


def _refresh_paper_opening_order(order: dict) -> tuple[dict, list[str]]:
    status = str(order.get("opening_order_status") or "").lower()
    has_fill = (numeric_value(order.get("opening_filled_avg_price")) or 0) > 0
    if status in OPENING_TERMINAL_STATUSES:
        return order, []
    if status == "filled" and has_fill:
        if (
            numeric_value(order.get("filled_max_profit_per_share")) is not None
            and numeric_value(order.get("filled_max_risk_per_share")) is not None
        ):
            return order, []
        fill_values = filled_economics_values(
            order.get("entry_type") or "",
            spread_width_for_tracked_order(order),
            order.get("opening_filled_avg_price"),
        )
        if fill_values["filled_max_profit_per_share"] is None:
            return order, []
        try:
            (
                supabase.table("alpaca_paper_orders")
                .update(fill_values)
                .eq("id", order["id"])
                .execute()
            )
        except Exception as error:
            return order, [
                f"Could not save fill economics for opening order "
                f"{order.get('order_id')}: {error}"
            ]
        return {**order, **fill_values}, []
    if not order.get("order_id"):
        return order, ["Could not refresh an Alpaca opening order without an order ID."]

    alpaca_order, errors = get_alpaca_order(order["order_id"])
    if errors:
        return order, errors
    update_values = opening_order_update_values(alpaca_order, order)
    try:
        (
            supabase.table("alpaca_paper_orders")
            .update(update_values)
            .eq("id", order["id"])
            .execute()
        )
    except Exception as error:
        return order, [
            f"Could not refresh Alpaca opening order {order['order_id']}: {error}"
        ]
    return {**order, **update_values}, []


def _refresh_paper_close_order(order: dict) -> tuple[dict, list[str]]:
    tracked_close_status = str(order.get("close_order_status") or "").lower()
    if (
        tracked_close_status == "filled"
        and str(order.get("position_status") or "").lower() == "closed"
    ):
        return order, []
    if (
        tracked_close_status == "submitting"
        and not order.get("close_order_id")
        and order.get("close_client_order_id")
    ):
        recovered_order, recovery_errors = get_alpaca_order_by_client_id(
            order["close_client_order_id"]
        )
        if recovery_errors:
            return order, recovery_errors
        recovery_values = {
            "close_order_id": recovered_order.get("id"),
            "close_order_status": recovered_order.get("status") or "accepted",
            "close_order_submitted_at": (
                recovered_order.get("submitted_at") or order.get("exit_signal_time")
            ),
            "last_exit_error": None,
            "exit_retryable": False,
        }
        try:
            response = (
                supabase.table("alpaca_paper_orders")
                .update(recovery_values)
                .eq("id", order["id"])
                .eq("close_client_order_id", order["close_client_order_id"])
                .execute()
            )
            if len(response.data or []) != 1:
                raise RuntimeError(
                    "recovered closing order update did not match exactly one tracked row"
                )
        except Exception as error:
            return order, [
                "Recovered Alpaca closing order but could not save it: " + str(error)
            ]
        order = {**order, **recovery_values}

    if not order.get("close_order_id"):
        return order, []

    close_order, errors = get_alpaca_order(order["close_order_id"])
    if errors:
        return order, errors
    status = str(
        close_order.get("status") or order.get("close_order_status") or "unknown"
    ).lower()
    close_fill_price = numeric_value(close_order.get("filled_avg_price"))
    if close_fill_price is not None:
        close_fill_price = abs(close_fill_price)
    update_values = {
        "close_order_status": status,
        "exit_fill_price": close_fill_price,
        "last_exit_error": None,
        "exit_retryable": status in CLOSE_RETRYABLE_STATUSES,
    }
    if status == "filled":
        fill_price = close_fill_price or 0
        entry_price = abs(
            numeric_value(order.get("opening_filled_avg_price")) or 0
        )
        quantity = int(order.get("quantity") or 1)
        if order.get("entry_type") == "credit":
            realized_pnl = (entry_price - fill_price) * quantity * CONTRACT_MULTIPLIER
        else:
            realized_pnl = (fill_price - entry_price) * quantity * CONTRACT_MULTIPLIER
        max_risk = abs(
            numeric_value(order.get("filled_max_risk_per_share")) or 0
        )
        update_values.update(
            {
                "position_status": "closed",
                "exit_retryable": False,
                "exit_fill_time": close_order.get("filled_at") or utc_now_iso(),
                "realized_pnl": round(realized_pnl, 2),
                "realized_return_on_risk": (
                    round(realized_pnl / (max_risk * quantity * CONTRACT_MULTIPLIER) * 100, 2)
                    if max_risk > 0
                    else None
                ),
            }
        )
    elif status in {"rejected", "canceled", "expired"}:
        update_values["last_exit_error"] = (
            close_order.get("reject_reason") or f"Closing order ended with status {status}."
        )

    try:
        (
            supabase.table("alpaca_paper_orders")
            .update(update_values)
            .eq("id", order["id"])
            .execute()
        )
        return {**order, **update_values}, []
    except Exception as error:
        return order, [
            f"Could not refresh Alpaca close order {order['close_order_id']}: {error}"
        ]


def fetch_alpaca_paper_snapshots(limit: int = 500) -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("alpaca_paper_position_snapshots")
            .select("*")
            .order("snapshot_time", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data, []
    except Exception as error:
        return [], [f"Could not load Alpaca paper position snapshots: {error}"]


def append_scan_history(
    scored_trades,
    event_analyses=None,
    price_moves=None,
    execution_candidates=None,
    scan_run_id=None,
):
    event_analyses = event_analyses or {}
    price_moves = price_moves or {}
    rows = []
    execution_candidates = execution_candidates or []
    scan_run_id = scan_run_id or new_scan_run_id()
    commit_sha = git_commit_sha()
    scan_timestamp = datetime.now(timezone.utc)
    current_time = scan_timestamp.isoformat()
    current_time_est = scan_timestamp.astimezone(
        ZoneInfo("America/New_York")
    ).strftime("%Y-%m-%d %I:%M:%S %p %Z")
    execution_ranks = {
        setup_key_for_trade(scored.trade): scored.execution_rank or rank
        for rank, scored in enumerate(execution_candidates, start=1)
    }
    setup_keys = [setup_key_for_trade(scored.trade) for scored in scored_trades]
    existing_by_setup = {}
    if setup_keys:
        try:
            existing_rows = (
                supabase.table("scan_history")
                .select("setup_key,scan_time,first_seen_at,last_seen_at,times_recommended")
                .in_("setup_key", list(set(setup_keys)))
                .execute()
                .data
            )
            for existing in existing_rows:
                key = existing.get("setup_key")
                if not key:
                    continue
                state = existing_by_setup.setdefault(
                    key,
                    {"count": 0, "first_seen_at": existing.get("scan_time")},
                )
                state["count"] += 1
                first_seen = existing.get("first_seen_at") or existing.get("scan_time")
                if first_seen and (
                    not state.get("first_seen_at") or first_seen < state["first_seen_at"]
                ):
                    state["first_seen_at"] = first_seen
        except Exception as error:
            return [
                "Could not read setup history metadata. Run "
                f"supabase_scanner_tracking_upgrade.sql first: {error}"
            ]

    for scored in scored_trades:
        trade = scored.trade
        setup_key = setup_key_for_trade(trade)
        prior = existing_by_setup.get(setup_key, {})
        times_recommended = int(prior.get("count", 0)) + 1
        first_seen_at = prior.get("first_seen_at") or current_time
        execution_rank = execution_ranks.get(setup_key)
        event_analysis = event_analyses.get(trade.ticker)
        price_move = price_moves.get(trade.ticker, {})
        rows.append(
            {
                "scan_time": current_time,
                "scan_time_est": current_time_est,
                "scan_run_id": scan_run_id,
                "setup_key": setup_key,
                "scanner_version": SCANNER_VERSION,
                "git_commit_sha": commit_sha,
                "raw_rank": scored.raw_rank,
                "diversified_rank": scored.diversified_rank,
                "execution_rank": execution_rank,
                "execution_selected": execution_rank is not None,
                "selection_method": (
                    SELECTION_EXECUTION if execution_rank is not None else SELECTION_DIVERSIFIED
                ),
                "first_seen_at": first_seen_at,
                "last_seen_at": current_time,
                "times_recommended": times_recommended,
                "ticker": trade.ticker,
                "strategy": trade.strategy,
                "expiration": trade.expiration,
                "option_type": trade.option_type,
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
                "event_adjustment": (
                    getattr(event_analysis, "adjustment", scored.event_adjustment)
                    if event_analysis is not None
                    else scored.event_adjustment
                ),
                "raw_price_move_adjustment": scored.raw_price_move_adjustment,
                "effective_price_move_adjustment": scored.effective_price_move_adjustment,
                "price_move_adjustment": scored.effective_price_move_adjustment,
                "base_score_without_price_move": scored.base_score_without_price_move,
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
                "entry_timestamp": current_time,
                "entry_price": round(
                    (
                        trade.max_risk
                        if trade.entry_type == "debit"
                        else trade.credit
                    )
                    * CONTRACT_MULTIPLIER,
                    2,
                ),
                "exit_timestamp": None,
                "exit_price": None,
                "exit_reason": None,
                "realized_pnl": None,
                "realized_return_on_risk": None,
                "closing_underlying_price": None,
                "days_held": None,
                "maximum_favorable_excursion": None,
                "maximum_adverse_excursion": None,
                "last_update_error": None,
                "update_retryable": False,
            }
        )

    if not rows:
        return []

    try:
        supabase.table("scan_history").insert(rows).execute()
    except Exception as error:
        return [f"Could not save scan history: {error}"]

    for row in rows:
        try:
            (
                supabase.table("scan_history")
                .update(
                    {
                        "last_seen_at": current_time,
                        "times_recommended": row["times_recommended"],
                    }
                )
                .eq("setup_key", row["setup_key"])
                .execute()
            )
        except Exception as error:
            return [f"Saved scan occurrence but could not update setup frequency: {error}"]

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
                        "maximum_favorable_excursion": update["highest_unrealized_pnl"],
                        "maximum_adverse_excursion": update["lowest_unrealized_pnl"],
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


def expiration_is_ready(
    expiration: date,
    today: date | None = None,
    include_today: bool = False,
) -> bool:
    today = today or date.today()
    return expiration <= today if include_today else expiration < today


def update_expired_history(include_today: bool = False) -> list[str]:
    errors = []
    closing_prices = {}
    try:
        query = (
            supabase.table("scan_history")
            .select("*")
            .in_("expiration_status", ["open", "expiration_update_failed"])
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
            message = f"Could not update {row['ticker']} expiration result: {error}"
            errors.append(message)
            _mark_expiration_update_failed(row, message)
            continue

        if closing_price is None:
            message = (
                f"Could not update {row['ticker']} expiration result: "
                "closing market data is unavailable."
            )
            errors.append(message)
            _mark_expiration_update_failed(row, message)
            continue

        try:
            pnl = expiration_pnl(row, closing_price)
        except (TypeError, ValueError) as error:
            message = f"Could not calculate {row['ticker']} expiration P/L: {error}"
            errors.append(message)
            _mark_expiration_update_failed(row, message)
            continue

        update_values = expiration_result_values(
            row, expiration, closing_price, pnl
        )
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


def _mark_expiration_update_failed(row: dict, message: str) -> None:
    try:
        (
            supabase.table("scan_history")
            .update(expiration_failure_values(message))
            .eq("id", row["id"])
            .execute()
        )
    except Exception:
        pass


def expiration_failure_values(message: str) -> dict:
    return {
        "expiration_status": "expiration_update_failed",
        "exit_reason": "data_unavailable",
        "last_update_error": message,
        "update_retryable": True,
    }


def expiration_result_values(
    row: dict,
    expiration: date,
    closing_price: float,
    pnl: float | None,
) -> dict:
    entry_price = numeric_value(row.get("entry_price"))
    if entry_price is None:
        entry_price = (
            numeric_value(row.get("max_risk"))
            if row.get("entry_type") == "debit"
            else numeric_value(row.get("credit"))
        )
    max_risk = numeric_value(row.get("max_risk")) or 0
    entry_timestamp = row.get("entry_timestamp") or row.get("scan_time")
    try:
        entry_date = datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00")).date()
        days_held = max(0, (expiration - entry_date).days)
    except (AttributeError, TypeError, ValueError):
        days_held = None
    exit_price = None
    if pnl is not None and entry_price is not None:
        if row.get("entry_type") == "debit":
            exit_price = round(entry_price + pnl, 2)
        else:
            exit_price = round(entry_price - pnl, 2)
    return {
        "expiration_close": closing_price,
        "closing_underlying_price": closing_price,
        "expiration_status": "manual review" if pnl is None else "expired",
        "starting_status": "closed" if pnl is not None else "manual review",
        "exit_timestamp": datetime.combine(
            expiration, datetime.min.time(), tzinfo=timezone.utc
        ).isoformat(),
        "exit_price": exit_price,
        "exit_reason": "expiration" if pnl is not None else "data_unavailable",
        "realized_pnl": pnl,
        "realized_return_on_risk": (
            round(pnl / max_risk * 100, 2)
            if pnl is not None and max_risk > 0
            else None
        ),
        "days_held": days_held,
        "maximum_favorable_excursion": row.get("highest_unrealized_pnl"),
        "maximum_adverse_excursion": row.get("lowest_unrealized_pnl"),
        "last_update_error": None,
        "update_retryable": False,
    }


def test_expiration_tracking_values() -> None:
    row = {
        "entry_type": "debit",
        "entry_price": 200,
        "max_risk": 200,
        "scan_time": "2026-07-01T12:00:00+00:00",
        "highest_unrealized_pnl": 80,
        "lowest_unrealized_pnl": -25,
    }
    completed = expiration_result_values(row, date(2026, 7, 31), 150, 100)
    assert completed["expiration_status"] == "expired"
    assert completed["starting_status"] == "closed"
    assert completed["exit_reason"] == "expiration"
    assert completed["realized_return_on_risk"] == 50
    assert completed["maximum_favorable_excursion"] == 80
    assert completed["maximum_adverse_excursion"] == -25

    failed = expiration_failure_values("quote unavailable")
    assert failed["expiration_status"] == "expiration_update_failed"
    assert failed["update_retryable"] is True
    assert failed["last_update_error"] == "quote unavailable"
    print("Expiration tracking tests passed.")


def fetch_completed_history() -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("scan_history")
            .select("*")
            .in_("expiration_status", ["expired", "closed early"])
            .order("expiration", desc=True)
            .execute()
        )
        return [normalize_history_row(row) for row in response.data], []
    except Exception as error:
        return [], [f"Could not load results from Supabase: {error}"]


def fetch_open_history(limit: int = 100) -> tuple[list[dict], list[str]]:
    try:
        response = (
            supabase.table("scan_history")
            .select("*")
            .in_("expiration_status", ["open", "expiration_update_failed"])
            .order("scan_time", desc=True)
            .limit(limit)
            .execute()
        )
        return [normalize_history_row(row) for row in response.data], []
    except Exception as error:
        return [], [f"Could not load open candidates from Supabase: {error}"]


def close_candidate(
    record_id: int,
    close_date: date,
    realized_pnl: float,
    note: str,
) -> list[str]:
    try:
        response = (
            supabase.table("scan_history")
            .select("max_risk,entry_timestamp,scan_time")
            .eq("id", record_id)
            .limit(1)
            .execute()
        )
        row = response.data[0] if response.data else {}
        max_risk = numeric_value(row.get("max_risk")) or 0
        entry_timestamp = row.get("entry_timestamp") or row.get("scan_time")
        try:
            entry_date = datetime.fromisoformat(entry_timestamp.replace("Z", "+00:00")).date()
            days_held = max(0, (close_date - entry_date).days)
        except (AttributeError, TypeError, ValueError):
            days_held = None
        (
            supabase.table("scan_history")
            .update(
                {
                    "expiration_status": "closed early",
                    "starting_status": "closed",
                    "actual_close_date": close_date.isoformat(),
                    "actual_realized_pnl": round(realized_pnl, 2),
                    "exit_timestamp": datetime.combine(
                        close_date, datetime.min.time(), tzinfo=timezone.utc
                    ).isoformat(),
                    "exit_reason": "manual",
                    "realized_pnl": round(realized_pnl, 2),
                    "realized_return_on_risk": (
                        round(realized_pnl / max_risk * 100, 2)
                        if max_risk > 0
                        else None
                    ),
                    "days_held": days_held,
                    "close_note": note.strip() or None,
                }
            )
            .eq("id", record_id)
            .execute()
        )
        return []
    except Exception as error:
        return [f"Could not close candidate: {error}"]
