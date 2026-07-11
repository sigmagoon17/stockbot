import os

import requests
from dotenv import load_dotenv


load_dotenv()

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


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
    order_type: str = "market",
    limit_price: float | None = None,
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
    if order_type == "limit":
        if limit_price is None or limit_price <= 0:
            return None, ["Limit orders require a limit price greater than 0."]
        payload["limit_price"] = f"{limit_price:.2f}"

    order, errors = alpaca_request("POST", "/v2/orders", json=payload)
    if errors:
        return None, errors
    return order, []
