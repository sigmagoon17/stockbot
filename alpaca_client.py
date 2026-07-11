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


def submit_scored_debit_long_leg_orders(
    scored_candidates,
    quantity: int = 1,
    limit: int = 3,
) -> list[dict]:
    recent_orders, recent_errors = get_recent_alpaca_orders(limit=100)
    recent_symbols = {
        order.get("symbol")
        for order in recent_orders
        if order.get("side") == "buy"
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

    debit_candidates = [
        scored
        for scored in scored_candidates
        if scored.trade.entry_type == "debit"
    ][:limit]

    for scored in debit_candidates:
        trade = scored.trade
        symbol = option_symbol(
            trade.ticker,
            trade.expiration,
            trade.option_type,
            trade.long_strike,
        )
        candidate_label = (
            f"{trade.ticker} {trade.strategy} {trade.expiration} "
            f"{trade.long_strike:g} {trade.option_type}"
        )
        if symbol in recent_symbols:
            results.append(
                {
                    "Candidate": candidate_label,
                    "Symbol": symbol,
                    "Status": "Skipped",
                    "Message": "Already submitted recently.",
                }
            )
            continue

        order, errors = submit_option_order(
            symbol,
            side="buy",
            quantity=quantity,
            order_type="limit",
            limit_price=float(trade.long_ask),
            client_order_id=f"scan-{scored_trade_paper_key(scored)}",
        )
        if errors:
            results.append(
                {
                    "Candidate": candidate_label,
                    "Symbol": symbol,
                    "Status": "Error",
                    "Message": "; ".join(errors),
                }
            )
            continue

        recent_symbols.add(symbol)
        results.append(
            {
                "Candidate": candidate_label,
                "Symbol": symbol,
                "Status": order.get("status", "submitted"),
                "Message": f"Order {order.get('id')}",
            }
        )

    if not debit_candidates:
        results.append(
            {
                "Candidate": "Latest Scan",
                "Symbol": "",
                "Status": "Skipped",
                "Message": "No debit candidates were available for paper trading.",
            }
        )

    return results
