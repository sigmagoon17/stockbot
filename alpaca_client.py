import math
import os
import re
from datetime import date, timedelta

import requests
from dotenv import load_dotenv
from scanner_tracking import SELECTION_EXECUTION, setup_key_for_trade


load_dotenv()

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
DATA_BASE_URL = "https://data.alpaca.markets"


def _first_env_value(*names: str) -> tuple[str | None, str | None]:
    for name in names:
        value = os.getenv(name)
        if value:
            return value, name
    return None, None


def alpaca_config_status() -> dict:
    api_key, api_key_name = _first_env_value("ALPACA_API_KEY", "APCA_API_KEY_ID")
    secret_key, secret_key_name = _first_env_value(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET",
        "APCA_API_SECRET_KEY",
    )
    base_url = (
        os.getenv("ALPACA_BASE_URL")
        or os.getenv("APCA_API_BASE_URL")
        or PAPER_BASE_URL
    ).rstrip("/")

    missing = []
    if not api_key:
        missing.append("ALPACA_API_KEY")
    if not secret_key:
        missing.append("ALPACA_SECRET_KEY")

    return {
        "api_key": api_key,
        "api_key_name": api_key_name,
        "secret_key": secret_key,
        "secret_key_name": secret_key_name,
        "base_url": base_url,
        "missing": missing,
        "is_paper": "paper-api.alpaca.markets" in base_url,
    }


def alpaca_headers() -> tuple[dict | None, list[str]]:
    config = alpaca_config_status()
    if config["missing"]:
        return None, [
            "Missing Alpaca setting(s): "
            + ", ".join(config["missing"])
            + ". Add them to .env locally and Streamlit/GitHub secrets online."
        ]

    return {
        "APCA-API-KEY-ID": config["api_key"],
        "APCA-API-SECRET-KEY": config["secret_key"],
    }, []


def alpaca_request(method: str, path: str, **kwargs) -> tuple[dict | list | None, list[str]]:
    config = alpaca_config_status()
    headers, errors = alpaca_headers()
    if errors:
        return None, errors

    try:
        response = requests.request(
            method,
            f"{config['base_url']}{path}",
            headers=headers,
            timeout=15,
            **kwargs,
        )
        response.raise_for_status()
        if response.text:
            return response.json(), []
        return {}, []
    except requests.HTTPError as error:
        detail = ""
        try:
            detail = f" {response.json()}"
        except Exception:
            detail = f" {response.text[:200]}"
        return None, [f"Alpaca rejected the request: {error}.{detail}"]
    except requests.RequestException as error:
        return None, [f"Could not reach Alpaca: {error}"]


def alpaca_data_request(
    method: str,
    path: str,
    **kwargs,
) -> tuple[dict | list | None, list[str]]:
    headers, errors = alpaca_headers()
    if errors:
        return None, errors

    try:
        response = requests.request(
            method,
            f"{DATA_BASE_URL}{path}",
            headers=headers,
            timeout=20,
            **kwargs,
        )
        response.raise_for_status()
        if response.text:
            return response.json(), []
        return {}, []
    except requests.HTTPError as error:
        detail = ""
        try:
            detail = f" {response.json()}"
        except Exception:
            detail = f" {response.text[:200]}"
        return None, [f"Alpaca market data rejected the request: {error}.{detail}"]
    except requests.RequestException as error:
        return None, [f"Could not reach Alpaca market data: {error}"]


def get_stock_daily_bars(
    ticker: str,
    lookback_days: int = 430,
) -> tuple[list[dict], list[str]]:
    start = date.today() - timedelta(days=lookback_days)
    payload, errors = alpaca_data_request(
        "GET",
        "/v2/stocks/bars",
        params={
            "symbols": ticker.upper(),
            "timeframe": "1Day",
            "start": start.isoformat(),
            "adjustment": "all",
            "feed": os.getenv("ALPACA_DATA_FEED", "iex"),
            "limit": 10000,
            "sort": "asc",
        },
    )
    if errors:
        return [], errors
    return (payload or {}).get("bars", {}).get(ticker.upper(), []), []


def get_alpaca_account() -> tuple[dict | None, list[str]]:
    config = alpaca_config_status()
    account, errors = alpaca_request("GET", "/v2/account")
    if errors:
        return None, errors

    account["_base_url"] = config["base_url"]
    account["_is_paper"] = config["is_paper"]
    account["_api_key_name"] = config["api_key_name"]
    account["_secret_key_name"] = config["secret_key_name"]
    return account, []


def option_symbol(
    ticker: str,
    expiration: str,
    option_type: str,
    strike: float,
) -> str:
    expiration_code = expiration.replace("-", "")[2:]
    type_code = "C" if option_type.lower() == "call" else "P"
    strike_code = f"{int(round(float(strike) * 1000)):08d}"
    return f"{ticker.upper()}{expiration_code}{type_code}{strike_code}"


def scored_trade_paper_key(scored, trading_day: date | None = None) -> str:
    trading_day = trading_day or date.today()
    return f"{setup_key_for_trade(scored.trade)}-{trading_day:%Y%m%d}"


def scan_client_order_id(scored, scan_run_id: str | None = None) -> str:
    setup_hash = setup_key_for_trade(scored.trade)[:24]
    run_suffix = (
        scan_run_id.replace("-", "")[:12]
        if scan_run_id
        else date.today().strftime("%Y%m%d")
    )
    return f"scan-{setup_hash}-{run_suffix}"[:48]


def leg_key_from_legs(legs: list[dict]) -> str:
    return "|".join(
        (
            f"{leg.get('symbol')}:{leg.get('side')}:"
            f"{leg.get('position_intent')}:{leg.get('ratio_qty')}"
        )
        for leg in legs
    )


def ticker_from_option_symbol(symbol: str) -> str:
    for index, character in enumerate(symbol):
        if character.isdigit():
            return symbol[:index]
    return symbol


def expiration_from_option_symbol(symbol: str) -> str | None:
    for index, character in enumerate(symbol):
        if character.isdigit():
            expiration_code = symbol[index:index + 6]
            if len(expiration_code) == 6:
                return f"20{expiration_code[:2]}-{expiration_code[2:4]}-{expiration_code[4:6]}"
            return None
    return None


def strategy_from_order_legs(legs: list[dict]) -> str:
    symbols = [leg.get("symbol", "") for leg in legs]
    sides = [leg.get("side") for leg in legs]
    if len(legs) == 4:
        return "iron condor"
    if len(legs) != 2:
        return "multi-leg option order"

    option_type = "call" if "C" in symbols[0][-9:-8] else "put"
    strikes = [int(symbol[-8:]) / 1000 for symbol in symbols]

    if option_type == "call":
        if sides == ["buy", "sell"] and strikes[0] < strikes[1]:
            return "bull call debit spread"
        if sides == ["buy", "sell"] and strikes[0] > strikes[1]:
            return "call credit spread"
    if option_type == "put":
        if sides == ["buy", "sell"] and strikes[0] > strikes[1]:
            return "bear put debit spread"
        if sides == ["buy", "sell"] and strikes[0] < strikes[1]:
            return "put credit spread"
    return "multi-leg option order"


ENTRY_TYPE_BY_STRATEGY = {
    "bull call debit spread": "debit",
    "bear put debit spread": "debit",
    "put credit spread": "credit",
    "call credit spread": "credit",
    "iron condor": "credit",
}


def entry_type_for_strategy(strategy: str) -> str | None:
    return ENTRY_TYPE_BY_STRATEGY.get(str(strategy or "").lower())


def spread_width_from_order_legs(legs: list[dict], strategy: str) -> float | None:
    symbols = [leg.get("symbol", "") for leg in legs if leg.get("symbol")]
    if len(symbols) == 2:
        return round(abs(int(symbols[0][-8:]) - int(symbols[1][-8:])) / 1000, 4)
    if len(symbols) == 4 and strategy == "iron condor":
        put_strikes = [int(symbol[-8:]) / 1000 for symbol in symbols if symbol[-9] == "P"]
        call_strikes = [int(symbol[-8:]) / 1000 for symbol in symbols if symbol[-9] == "C"]
        if len(put_strikes) != 2 or len(call_strikes) != 2:
            return None
        return round(
            max(
                abs(put_strikes[0] - put_strikes[1]),
                abs(call_strikes[0] - call_strikes[1]),
            ),
            4,
        )
    return None


def trade_spread_width_per_share(trade) -> float:
    if trade.strategy == "iron condor":
        put_width = abs(float(trade.put_short_strike) - float(trade.put_long_strike))
        call_width = abs(float(trade.call_long_strike) - float(trade.call_short_strike))
        return round(max(put_width, call_width), 4)
    return round(abs(float(trade.short_strike) - float(trade.long_strike)), 4)


def paper_order_result_from_alpaca_order(order: dict) -> dict | None:
    legs = order.get("legs") or []
    if not legs:
        return None

    normalized_legs = [
        {
            "symbol": leg.get("symbol"),
            "side": leg.get("side"),
            "position_intent": leg.get("position_intent"),
            "ratio_qty": leg.get("ratio_qty") or "1",
        }
        for leg in legs
        if leg.get("symbol")
    ]
    if not normalized_legs:
        return None

    first_symbol = normalized_legs[0]["symbol"]
    strategy = strategy_from_order_legs(normalized_legs)
    return {
        "Candidate": "Alpaca backfill",
        "Symbol": order.get("symbol") or (
            "2-leg order" if len(normalized_legs) == 2 else "4-leg order"
        ),
        "Status": order.get("status", "accepted"),
        "Message": f"Backfilled Alpaca order {order.get('id')}",
        "Order ID": order.get("id"),
        "Client Order ID": order.get("client_order_id"),
        "Ticker": ticker_from_option_symbol(first_symbol),
        "Strategy": strategy,
        "Expiration": expiration_from_option_symbol(first_symbol),
        "Setup Score": None,
        "Entry Type": entry_type_for_strategy(strategy),
        "Limit Price": float(order.get("limit_price") or 0),
        "Spread Width Per Share": spread_width_from_order_legs(
            normalized_legs, strategy
        ),
        "Max Profit": None,
        "Max Risk": None,
        "Quantity": int(float(order.get("qty") or 1)),
        "Order Class": order.get("order_class") or "mleg",
        "Leg Key": leg_key_from_legs(normalized_legs),
        "Exit Policy": "none",
        "Opening Order Status": order.get("status"),
        "Opening Filled At": order.get("filled_at"),
        "Opening Filled Avg Price": order.get("filled_avg_price"),
    }


def recent_alpaca_order_results(limit: int = 50) -> tuple[list[dict], list[str]]:
    orders, errors = get_recent_alpaca_orders(limit)
    if errors:
        return [], errors
    return [
        result
        for result in (
            paper_order_result_from_alpaca_order(order)
            for order in orders
        )
        if result is not None
    ], []


def get_alpaca_positions() -> tuple[list[dict], list[str]]:
    positions, errors = alpaca_request("GET", "/v2/positions")
    if errors:
        return [], errors
    return positions or [], []


def nonzero_alpaca_position_quantities(
    positions: list[dict],
) -> tuple[dict[str, float], list[str]]:
    quantities = {}
    errors = []
    for position in positions:
        if not isinstance(position, dict):
            errors.append("Alpaca returned malformed position data.")
            continue
        symbol = str(position.get("symbol") or "").strip().upper()
        raw_quantity = position.get("qty")
        if not symbol:
            errors.append("Alpaca returned a position without an option symbol.")
            continue
        try:
            quantity = float(raw_quantity)
        except (TypeError, ValueError):
            errors.append(
                f"Alpaca returned an invalid position quantity for {symbol}: "
                f"{raw_quantity!r}."
            )
            continue
        if not math.isfinite(quantity):
            errors.append(
                f"Alpaca returned an invalid position quantity for {symbol}: "
                f"{raw_quantity!r}."
            )
            continue
        if quantity != 0:
            quantities[symbol] = quantity
    return quantities, errors


def _display_position_quantity(quantity: float) -> str:
    return f"{quantity:g}"


def _position_overlap_message(leg: dict, quantity: float) -> str:
    symbol = leg["symbol"]
    required_intent = leg["position_intent"]
    inferred_intent = None
    if quantity > 0 and leg.get("side") == "sell":
        inferred_intent = "sell_to_close"
    elif quantity < 0 and leg.get("side") == "buy":
        inferred_intent = "buy_to_close"

    if inferred_intent:
        explanation = (
            f"the candidate requires {required_intent} but Alpaca would infer "
            f"{inferred_intent}"
        )
    else:
        explanation = (
            f"the candidate requires {required_intent}, and opening another setup "
            "against an existing broker leg could cause Alpaca to infer a closing intent"
        )
    return (
        f"Skipped because {symbol} already has broker position qty "
        f"{_display_position_quantity(quantity)}; {explanation}, causing a "
        "position_intent mismatch."
    )


def get_open_alpaca_orders() -> tuple[list[dict], list[str]]:
    orders, errors = alpaca_request(
        "GET",
        "/v2/orders",
        params={
            "status": "open",
            "limit": 500,
            "direction": "desc",
            "nested": "true",
        },
    )
    if errors:
        return [], errors
    return orders or [], []


def alpaca_open_order_symbols(
    orders: list[dict],
) -> tuple[set[str], list[str]]:
    symbols = set()
    errors = []
    for order in orders:
        if not isinstance(order, dict):
            errors.append("Alpaca returned malformed open-order data.")
            continue
        order_legs = order.get("legs")
        if order_legs is None:
            order_legs = [order]
        if not isinstance(order_legs, list):
            errors.append("Alpaca returned malformed open-order leg data.")
            continue
        for leg in order_legs:
            if not isinstance(leg, dict):
                errors.append("Alpaca returned malformed open-order leg data.")
                continue
            symbol = str(leg.get("symbol") or "").strip().upper()
            if not symbol:
                errors.append("Alpaca returned an open order without an option symbol.")
                continue
            symbols.add(symbol)
    return symbols, errors


def get_alpaca_opening_preflight_state() -> tuple[dict | None, list[str]]:
    config = alpaca_config_status()
    if not config["is_paper"]:
        return None, [
            "Refusing broker preflight because Alpaca is not using the paper endpoint; "
            "no broker requests or paper orders were submitted."
        ]
    positions, position_errors = get_alpaca_positions()
    if position_errors:
        return None, [
            "Alpaca position lookup failed; no paper orders were submitted. "
            + "; ".join(position_errors)
        ]
    position_quantities, quantity_errors = nonzero_alpaca_position_quantities(
        positions
    )
    if quantity_errors:
        return None, [
            "Alpaca position data could not be validated; no paper orders were "
            "submitted. " + "; ".join(quantity_errors)
        ]

    open_orders, order_errors = get_open_alpaca_orders()
    if order_errors:
        return None, [
            "Alpaca open-order lookup failed; no paper orders were submitted. "
            + "; ".join(order_errors)
        ]
    open_order_symbols, symbol_errors = alpaca_open_order_symbols(open_orders)
    if symbol_errors:
        return None, [
            "Alpaca open-order data could not be validated; no paper orders were "
            "submitted. " + "; ".join(symbol_errors)
        ]
    return {
        "position_quantities": position_quantities,
        "open_order_symbols": open_order_symbols,
    }, []


def opening_legs_conflict_message(
    legs: list[dict],
    preflight_state: dict,
    reserved_symbols: set[str] | None = None,
) -> tuple[str | None, str | None]:
    position_quantities = preflight_state["position_quantities"]
    open_order_symbols = preflight_state["open_order_symbols"]
    reserved_symbols = reserved_symbols or set()

    for leg in legs:
        symbol = str(leg.get("symbol") or "").strip().upper()
        if symbol in position_quantities:
            normalized_leg = {**leg, "symbol": symbol}
            return symbol, _position_overlap_message(
                normalized_leg, position_quantities[symbol]
            )
        if symbol in open_order_symbols:
            return symbol, (
                f"Skipped because {symbol} is already reserved by an open Alpaca "
                "order; no legs were submitted, preventing a position_intent conflict."
            )
        if symbol in reserved_symbols:
            return symbol, (
                f"Skipped because {symbol} is already reserved by a higher-ranked "
                "candidate in this scan; no legs were submitted, preventing a "
                "position_intent conflict."
            )
    return None, None


def validate_opening_option_legs(legs: list[dict]) -> list[str]:
    errors = []
    if not isinstance(legs, list) or len(legs) not in {2, 4}:
        return ["The strategy must contain exactly two or four option legs."]

    symbols = []
    for leg in legs:
        if not isinstance(leg, dict):
            errors.append("A strategy leg is malformed.")
            continue
        symbol = str(leg.get("symbol") or "").strip().upper()
        side = leg.get("side")
        intent = leg.get("position_intent")
        try:
            ratio_quantity = int(leg.get("ratio_qty"))
        except (TypeError, ValueError):
            ratio_quantity = 0
        if not re.fullmatch(r"[A-Z0-9]{1,6}\d{6}[CP]\d{8}", symbol):
            errors.append(f"Invalid option symbol: {symbol or '[missing]' }.")
        if (side, intent) not in {
            ("buy", "buy_to_open"),
            ("sell", "sell_to_open"),
        }:
            errors.append(
                f"Invalid opening side or position intent for {symbol or '[missing]'}."
            )
        if ratio_quantity <= 0:
            errors.append(f"Invalid leg quantity for {symbol or '[missing]' }.")
        symbols.append(symbol)
    if len(set(symbols)) != len(symbols):
        errors.append("The strategy contains duplicate option symbols.")
    return errors


def get_recent_alpaca_orders(limit: int = 10) -> tuple[list[dict], list[str]]:
    orders, errors = alpaca_request(
        "GET",
        "/v2/orders",
        params={
            "status": "all",
            "limit": limit,
            "direction": "desc",
            "nested": "false",
        },
    )
    if errors:
        return [], errors
    return orders or [], []


def get_alpaca_order(order_id: str) -> tuple[dict | None, list[str]]:
    order, errors = alpaca_request("GET", f"/v2/orders/{order_id}")
    if errors:
        return None, errors
    return order or {}, []


def get_alpaca_order_by_client_id(
    client_order_id: str,
) -> tuple[dict | None, list[str]]:
    order, errors = alpaca_request(
        "GET",
        "/v2/orders:by_client_order_id",
        params={"client_order_id": client_order_id},
    )
    if errors:
        return None, errors
    return order or {}, []


def submit_option_order(
    symbol: str,
    side: str,
    quantity: int,
    order_type: str = "limit",
    limit_price: float | None = None,
    client_order_id: str | None = None,
) -> tuple[dict | None, list[str]]:
    config = alpaca_config_status()
    if not config["is_paper"]:
        return None, [
            "Refusing to submit an order because Alpaca is not using the paper endpoint."
        ]

    payload = {
        "symbol": symbol,
        "qty": str(int(quantity)),
        "side": side,
        "type": order_type,
        "time_in_force": "day",
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id[:48]
    if order_type == "limit":
        if limit_price is None or limit_price <= 0:
            return None, ["Limit orders require a limit price greater than 0."]
        payload["limit_price"] = f"{limit_price:.2f}"

    order, errors = alpaca_request("POST", "/v2/orders", json=payload)
    if errors:
        return None, errors
    return order, []


def trade_multileg_order_details(scored) -> tuple[list[dict], float, str]:
    trade = scored.trade
    quantity_type = "credit" if trade.entry_type == "credit" else "debit"

    if trade.strategy in {"bull call debit spread", "bear put debit spread"}:
        option_type = "call" if trade.strategy == "bull call debit spread" else "put"
        legs = [
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, option_type, trade.long_strike
                ),
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, option_type, trade.short_strike
                ),
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
        ]
        return legs, round(float(trade.max_risk), 2), quantity_type

    if trade.strategy in {"put credit spread", "call credit spread"}:
        option_type = "put" if trade.strategy == "put credit spread" else "call"
        legs = [
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, option_type, trade.long_strike
                ),
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, option_type, trade.short_strike
                ),
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
        ]
        return legs, round(float(trade.credit), 2), quantity_type

    if trade.strategy == "iron condor":
        if (
            trade.put_long_strike is None
            or trade.put_short_strike is None
            or trade.call_short_strike is None
            or trade.call_long_strike is None
        ):
            raise ValueError("Iron condor is missing one or more leg strikes.")

        legs = [
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, "put", trade.put_long_strike
                ),
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, "put", trade.put_short_strike
                ),
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, "call", trade.call_short_strike
                ),
                "ratio_qty": "1",
                "side": "sell",
                "position_intent": "sell_to_open",
            },
            {
                "symbol": option_symbol(
                    trade.ticker, trade.expiration, "call", trade.call_long_strike
                ),
                "ratio_qty": "1",
                "side": "buy",
                "position_intent": "buy_to_open",
            },
        ]
        return legs, round(float(trade.credit), 2), quantity_type

    raise ValueError(f"Unsupported multi-leg strategy: {trade.strategy}")


def submit_multileg_order(
    legs: list[dict],
    quantity: int,
    limit_price: float,
    client_order_id: str | None = None,
) -> tuple[dict | None, list[str]]:
    config = alpaca_config_status()
    if not config["is_paper"]:
        return None, [
            "Refusing to submit an order because Alpaca is not using the paper endpoint."
        ]
    if limit_price <= 0:
        return None, ["Multi-leg orders require a limit price greater than 0."]

    payload = {
        "order_class": "mleg",
        "qty": str(int(quantity)),
        "type": "limit",
        "limit_price": f"{limit_price:.2f}",
        "time_in_force": "day",
        "legs": legs,
    }
    if client_order_id:
        payload["client_order_id"] = client_order_id[:48]

    order, errors = alpaca_request("POST", "/v2/orders", json=payload)
    if errors:
        return None, errors
    return order, []


def validate_manual_limit_price(
    entry_type: str,
    limit_price,
    spread_width_per_share,
) -> list[str]:
    normalized_entry_type = str(entry_type or "").strip().lower()
    if normalized_entry_type not in {"debit", "credit"}:
        return [f"Unsupported manual entry type: {entry_type!r}."]

    try:
        width = float(spread_width_per_share)
    except (TypeError, ValueError):
        return ["Manual spread width must be a finite number greater than $0.00."]
    if not math.isfinite(width) or width <= 0:
        return ["Manual spread width must be a finite number greater than $0.00."]

    entry_label = normalized_entry_type.title()
    range_message = (
        f"{entry_label} limit must be greater than $0.00 and less than the "
        f"${width:.2f} spread width."
    )
    try:
        price = float(limit_price)
    except (TypeError, ValueError):
        return [range_message]
    if not math.isfinite(price) or price <= 0 or price >= width:
        return [range_message]
    return []


def submit_manual_multileg_order(
    legs: list[dict],
    quantity: int,
    limit_price: float,
    client_order_id: str | None = None,
    expected_entry_type: str | None = None,
    expected_spread_width: float | None = None,
) -> tuple[dict | None, list[str], str | None]:
    if expected_entry_type is not None or expected_spread_width is not None:
        limit_errors = validate_manual_limit_price(
            expected_entry_type,
            limit_price,
            expected_spread_width,
        )
        if limit_errors:
            return None, limit_errors, None
    leg_errors = validate_opening_option_legs(legs)
    if leg_errors:
        return None, leg_errors, None
    preflight_state, preflight_errors = get_alpaca_opening_preflight_state()
    if preflight_errors:
        return None, preflight_errors, None

    _, conflict_message = opening_legs_conflict_message(legs, preflight_state)
    if conflict_message:
        return None, [], conflict_message

    order, errors = submit_multileg_order(
        legs,
        quantity=quantity,
        limit_price=limit_price,
        client_order_id=client_order_id,
    )
    return order, errors, None


def submit_scored_multileg_orders(
    scored_candidates,
    quantity: int = 1,
    limit: int = 3,
    exit_policy: str = "none",
    scan_run_id: str | None = None,
) -> list[dict]:
    selected_candidates = list(scored_candidates)[:limit]
    if not selected_candidates:
        return [
            {
                "Candidate": "Latest Scan",
                "Symbol": "",
                "Status": "Skipped",
                "Message": "No candidates were available for paper trading.",
            }
        ]

    preflight_state, preflight_errors = get_alpaca_opening_preflight_state()
    if preflight_errors:
        return [
            {
                "Candidate": "Broker Position Safety Check",
                "Symbol": "",
                "Status": "Error",
                "Message": "; ".join(preflight_errors),
            }
        ]

    recent_orders, recent_errors = get_recent_alpaca_orders(limit=100)
    recent_client_order_ids = {
        order.get("client_order_id")
        for order in recent_orders
        if order.get("client_order_id")
    }

    results = [
        {
            "Candidate": "Recent Orders",
            "Symbol": "",
            "Status": "Error",
            "Message": error,
        }
        for error in recent_errors
    ]

    reserved_symbols = set()

    for scored in selected_candidates:
        trade = scored.trade
        candidate_label = (
            f"{trade.ticker} {trade.strategy} {trade.expiration} "
            f"score {scored.total_score}"
        )
        try:
            legs, limit_price, quantity_type = trade_multileg_order_details(scored)
            leg_key = leg_key_from_legs(legs)
        except ValueError as error:
            results.append(
                {
                    "Candidate": candidate_label,
                    "Symbol": "",
                    "Status": "Error",
                    "Message": str(error),
                }
            )
            continue

        symbol, conflict_message = opening_legs_conflict_message(
            legs, preflight_state, reserved_symbols
        )
        if conflict_message:
            results.append(
                {
                    "Candidate": candidate_label,
                    "Symbol": symbol,
                    "Status": "Skipped",
                    "Message": conflict_message,
                }
            )
            continue

        candidate_symbols = {leg["symbol"].upper() for leg in legs}
        reserved_symbols.update(candidate_symbols)

        client_order_id = scan_client_order_id(scored, scan_run_id)
        if client_order_id in recent_client_order_ids:
            results.append(
                {
                    "Candidate": candidate_label,
                    "Symbol": "2-leg order" if len(legs) == 2 else "4-leg order",
                    "Status": "Skipped",
                    "Message": "Already submitted recently.",
                }
            )
            continue

        order, errors = submit_multileg_order(
            legs,
            quantity=quantity,
            limit_price=limit_price,
            client_order_id=client_order_id,
        )
        if errors:
            results.append(
                {
                    "Candidate": candidate_label,
                    "Symbol": "2-leg order" if len(legs) == 2 else "4-leg order",
                    "Status": "Error",
                    "Message": "; ".join(errors),
                }
            )
            continue

        recent_client_order_ids.add(client_order_id)
        results.append(
            {
                "Candidate": candidate_label,
                "Symbol": order.get("symbol") or (
                    "2-leg order" if len(legs) == 2 else "4-leg order"
                ),
                "Status": order.get("status", "submitted"),
                "Message": (
                    f"{quantity_type.title()} limit ${limit_price:.2f}; "
                    f"order {order.get('id')}"
                ),
                "Order ID": order.get("id"),
                "Client Order ID": order.get("client_order_id"),
                "Ticker": trade.ticker,
                "Strategy": trade.strategy,
                "Expiration": trade.expiration,
                "Setup Score": scored.total_score,
                "Ticker Score": scored.normalized_ticker_score,
                "Quant Score": scored.quant_score,
                "Setup Key": setup_key_for_trade(trade),
                "Scan Run ID": scan_run_id,
                "Execution Rank": scored.execution_rank,
                "Selection Method": SELECTION_EXECUTION,
                "Entry Type": trade.entry_type,
                "Limit Price": limit_price,
                "Spread Width Per Share": trade_spread_width_per_share(trade),
                "Max Profit": round(
                    float(
                        trade.max_profit
                        if trade.entry_type == "debit"
                        else trade.credit
                    ),
                    2,
                ),
                "Max Risk": round(float(trade.max_risk), 2),
                "Quantity": quantity,
                "Order Class": order.get("order_class") or "mleg",
                "Leg Key": leg_key,
                "Exit Policy": exit_policy,
                "Opening Order Status": order.get("status"),
                "Opening Filled At": order.get("filled_at"),
                "Opening Filled Avg Price": order.get("filled_avg_price"),
            }
        )

    return results


def submit_scored_debit_long_leg_orders(
    scored_candidates,
    quantity: int = 1,
    limit: int = 3,
) -> list[dict]:
    return submit_scored_multileg_orders(scored_candidates, quantity, limit)
