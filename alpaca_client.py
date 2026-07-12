import os
from datetime import date, timedelta

import requests
from dotenv import load_dotenv


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
    trade = scored.trade
    trading_day = trading_day or date.today()
    strike_key = f"{float(trade.long_strike):.3f}".replace(".", "p")
    return (
        f"{trade.ticker}-{trade.expiration}-{trade.option_type}-"
        f"{strike_key}-{trading_day.isoformat()}"
    )


def scan_client_order_id(scored) -> str:
    return f"scan-{scored_trade_paper_key(scored)}"[:48]


def leg_key_from_legs(legs: list[dict]) -> str:
    return "|".join(
        (
            f"{leg.get('symbol')}:{leg.get('side')}:"
            f"{leg.get('position_intent')}:{leg.get('ratio_qty')}"
        )
        for leg in legs
    )


def get_alpaca_positions() -> tuple[list[dict], list[str]]:
    positions, errors = alpaca_request("GET", "/v2/positions")
    if errors:
        return [], errors
    return positions or [], []


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


def submit_scored_multileg_orders(
    scored_candidates,
    quantity: int = 1,
    limit: int = 3,
) -> list[dict]:
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

    selected_candidates = list(scored_candidates)[:limit]

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

        client_order_id = scan_client_order_id(scored)
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
                "Entry Type": trade.entry_type,
                "Limit Price": limit_price,
                "Quantity": quantity,
                "Order Class": order.get("order_class") or "mleg",
                "Leg Key": leg_key,
            }
        )

    if not selected_candidates:
        results.append(
            {
                "Candidate": "Latest Scan",
                "Symbol": "",
                "Status": "Skipped",
                "Message": "No candidates were available for paper trading.",
            }
        )

    return results


def submit_scored_debit_long_leg_orders(
    scored_candidates,
    quantity: int = 1,
    limit: int = 3,
) -> list[dict]:
    return submit_scored_multileg_orders(scored_candidates, quantity, limit)
