from dataclasses import dataclass
from datetime import date
import math
from pathlib import Path
import yfinance as yf
from collections import Counter

try:
    from alpaca_client import get_stock_daily_bars
except ImportError:
    def get_stock_daily_bars(ticker: str, lookback_days: int = 430):
        return [], ["Alpaca stock bars helper is unavailable."]


CONTRACT_MULTIPLIER = 100
MAX_SETUP_SCORE = 125
YFINANCE_CACHE_DIR = Path(__file__).with_name(".yfinance-cache")
YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)


@dataclass(frozen=True)
class ScanPreferences:
    max_risk: float
    outlook: str #bullish, bearish, neutral, income
    risk_tolerance: str #conservative, moderate, aggressive
    test_expiration: date | None = None
    nearest_expiration: bool = False



@dataclass(frozen=True)
class OptionContract:
    ticker: str
    expiration: str
    option_type: str
    strike: float
    bid: float
    ask: float
    delta: float
    gamma: float
    theta: float
    vega: float
    implied_volatility: float
    open_interest: int
    volume: int
    quote_source: str = "bid/ask"


def normal_cdf(value: float) -> float:
    return (1 + math.erf(value / math.sqrt(2))) / 2


def estimated_delta(
    underlying_price: float,
    strike: float,
    implied_volatility: float,
    dte: int,
    option_type: str,
) -> float:
    time_to_expiration = dte / 365
    volatility_time = implied_volatility * math.sqrt(time_to_expiration)
    d1 = (
        math.log(underlying_price / strike)
        + (0.04 + implied_volatility ** 2 / 2) * time_to_expiration
    ) / volatility_time
    call_delta = normal_cdf(d1)
    return call_delta if option_type == "call" else call_delta - 1


def integer_or_zero(value) -> int:
    return 0 if value is None or math.isnan(float(value)) else int(value)


def open_interest_or_volume(row) -> int:
    open_interest = integer_or_zero(row["openInterest"])
    volume = integer_or_zero(row["volume"])
    return open_interest if open_interest > 0 else volume


def option_bid_ask_from_row(row) -> tuple[float, float, str] | None:
    bid = float(row["bid"])
    ask = float(row["ask"])
    if bid > 0 and ask > 0:
        return bid, ask, "bid/ask"

    last_price = float(row["lastPrice"])
    if last_price <= 0 or math.isnan(last_price):
        return None

    synthetic_spread = max(0.02, min(0.04, last_price * 0.01))
    synthetic_bid = max(0.01, last_price - synthetic_spread / 2)
    synthetic_ask = last_price + synthetic_spread / 2
    return round(synthetic_bid, 2), round(synthetic_ask, 2), "last price estimate"


def get_underlying_price(stock: yf.Ticker) -> float:
    try:
        fast_info = stock.fast_info
        price = fast_info.get("lastPrice") or fast_info.get("last_price")
        if price is not None and not math.isnan(float(price)):
            return float(price)
    except Exception:
        pass

    history = stock.history(period="5d")
    if history.empty:
        raise ValueError("Yahoo Finance did not return an underlying stock price.")
    return float(history["Close"].iloc[-1])


def alpaca_price_history_metrics(ticker: str) -> tuple[float, float, dict[str, float | str]]:
    bars, errors = get_stock_daily_bars(ticker)
    if errors or len(bars) < 80:
        raise ValueError(
            "Alpaca did not return enough daily stock bars for price metrics."
        )

    closing_prices = [float(bar["c"]) for bar in bars if bar.get("c") is not None]
    if len(closing_prices) < 80:
        raise ValueError(
            "Alpaca did not return enough closing prices for price metrics."
        )

    daily_returns = [
        closing_prices[index] / closing_prices[index - 1] - 1
        for index in range(1, len(closing_prices))
        if closing_prices[index - 1] > 0
    ]
    if len(daily_returns) < 60:
        raise ValueError(
            "Alpaca did not return enough daily returns for volatility ranking."
        )

    rolling_volatility = []
    for index in range(20, len(daily_returns) + 1):
        window = daily_returns[index - 20:index]
        average_return = sum(window) / len(window)
        variance = sum((value - average_return) ** 2 for value in window) / (
            len(window) - 1
        )
        rolling_volatility.append(math.sqrt(variance) * math.sqrt(252))

    if len(rolling_volatility) < 60:
        raise ValueError(
            "Alpaca did not return enough volatility history for ranking."
        )

    current_volatility = rolling_volatility[-1]
    lowest_volatility = min(rolling_volatility)
    highest_volatility = max(rolling_volatility)

    if highest_volatility == lowest_volatility:
        volatility_rank = 50.0
    else:
        volatility_rank = round(
            (current_volatility - lowest_volatility)
            / (highest_volatility - lowest_volatility)
            * 100,
            1,
        )

    latest_daily_return = daily_returns[-1]
    baseline_returns = daily_returns[-21:-1]
    average_baseline_return = sum(baseline_returns) / len(baseline_returns)
    baseline_variance = sum(
        (value - average_baseline_return) ** 2 for value in baseline_returns
    ) / (len(baseline_returns) - 1)
    baseline_daily_volatility = math.sqrt(baseline_variance)
    move_multiple = (
        abs(latest_daily_return) / baseline_daily_volatility
        if baseline_daily_volatility > 0
        else 0.0
    )
    five_day_return = closing_prices[-1] / closing_prices[-6] - 1
    price_move = {
        "1D Move %": round(latest_daily_return * 100, 2),
        "5D Move %": round(five_day_return * 100, 2),
        "Move vs 20D Vol": round(move_multiple, 1),
        "Unusual Move": "Yes" if move_multiple >= 2 else "No",
        "Price Source": "Alpaca",
    }
    return closing_prices[-1], volatility_rank, price_move


def price_history_metrics(stock: yf.Ticker) -> tuple[float, dict[str, float | str]]:
    history = stock.history(period="1y", auto_adjust=True)
    if history.empty:
        raise ValueError("Yahoo Finance did not return price history.")

    closing_prices = history["Close"]
    daily_returns = history["Close"].pct_change().dropna()
    rolling_volatility = (daily_returns.rolling(20).std() * math.sqrt(252)).dropna()

    if len(rolling_volatility) < 60:
        raise ValueError("Yahoo Finance did not return enough price history for volatility ranking.")

    current_volatility = rolling_volatility.iloc[-1]
    lowest_volatility = rolling_volatility.min()
    highest_volatility = rolling_volatility.max()

    if highest_volatility == lowest_volatility:
        volatility_rank = 50.0
    else:
        volatility_rank = round(
            (current_volatility - lowest_volatility)
            / (highest_volatility - lowest_volatility)
            * 100,
            1,
        )

    latest_daily_return = float(daily_returns.iloc[-1])
    baseline_daily_volatility = float(daily_returns.iloc[-21:-1].std())
    move_multiple = (
        abs(latest_daily_return) / baseline_daily_volatility
        if baseline_daily_volatility > 0
        else 0.0
    )
    five_day_return = float(closing_prices.iloc[-1] / closing_prices.iloc[-6] - 1)
    price_move = {
        "1D Move %": round(latest_daily_return * 100, 2),
        "5D Move %": round(five_day_return * 100, 2),
        "Move vs 20D Vol": round(move_multiple, 1),
        "Unusual Move": "Yes" if move_multiple >= 2 else "No",
        "Price Source": "Yahoo Finance",
    }
    return volatility_rank, price_move


def realized_volatility_rank(stock: yf.Ticker) -> float:
    volatility_rank, _ = price_history_metrics(stock)
    return volatility_rank


def get_option_chain(
    ticker: str,
    test_expiration: date | None = None,
    nearest_expiration: bool = False,
) -> tuple[float, list[OptionContract], date | None, float, dict[str, float | str]]:
    stock = yf.Ticker(ticker)
    earnings_date = None
    etf_tickers = ["SPY", "QQQ"]
    if ticker not in etf_tickers:
        try:
            earnings_table = stock.get_earnings_dates(limit= 12)

            future_dates = [
                earnings.date()
                for earnings in earnings_table.index
                if earnings.date() >= date.today()
            ]
            earnings_date = min(future_dates) if future_dates else None
        except Exception:
            earnings_date = None

    try:
        underlying_price, volatility_rank, price_move = alpaca_price_history_metrics(
            ticker
        )
    except Exception:
        underlying_price = get_underlying_price(stock)
        volatility_rank, price_move = price_history_metrics(stock)
    contracts = []

    available_expirations = [
        expiration
        for expiration in stock.options
        if days_to_expiration(expiration) >= 0
    ]
    if test_expiration is not None:
        selected_expiration = test_expiration.isoformat()
        if selected_expiration not in available_expirations:
            raise ValueError(
                f"Yahoo Finance does not offer {selected_expiration} for {ticker}."
            )
        expirations_to_fetch = [selected_expiration]
    elif nearest_expiration:
        expirations_to_fetch = sorted(
            available_expirations, key=days_to_expiration
        )[:5]
    else:
        expirations_to_fetch = [
            expiration
            for expiration in available_expirations
            if 21 <= days_to_expiration(expiration) <= 60
        ]

    for expiration in expirations_to_fetch:
        dte = days_to_expiration(expiration)
        chain = stock.option_chain(expiration)
        for option_type, table in (("call", chain.calls), ("put", chain.puts)):
            for _, row in table.iterrows():
                strike = float(row["strike"])
                bid_ask = option_bid_ask_from_row(row)
                if bid_ask is None:
                    continue
                bid, ask, quote_source = bid_ask
                implied_volatility = float(row["impliedVolatility"])

                if implied_volatility <= 0:
                    continue

                contracts.append(
                    OptionContract(
                        ticker=ticker,
                        expiration=expiration,
                        option_type=option_type,
                        strike=strike,
                        bid=bid,
                        ask=ask,
                        delta=estimated_delta(
                            underlying_price,
                            strike,
                            implied_volatility,
                            dte,
                            option_type,
                        ),
                        gamma=0,
                        theta=0,
                        vega=0,
                        implied_volatility=implied_volatility,
                        open_interest=open_interest_or_volume(row),
                        volume=integer_or_zero(row["volume"]),
                        quote_source=quote_source,
                    )
                )

    if not contracts:
        expiration_range = (
            "the selected test expiration"
            if test_expiration is not None or nearest_expiration
            else "the 21 to 60 DTE range"
        )
        raise ValueError(
            f"Yahoo Finance returned no usable option contracts for {expiration_range}."
        )

    return underlying_price, contracts, earnings_date, volatility_rank, price_move


def days_to_expiration(expiration: str) -> int:
    return (date.fromisoformat(expiration) - date.today()).days


def expected_move(underlying_price: float, implied_volatility: float, dte: int) -> float:
    return round(underlying_price * implied_volatility * math.sqrt(dte / 365), 2)


@dataclass(frozen=True)
class Trade:
    ticker: str
    strategy: str
    expiration: str
    option_type: str
    delta: float
    volatility_rank: float
    earnings_before_exp: bool
    open_interest: int
    volume: int
    dte: int
    bid: float
    ask: float
    credit: float
    max_risk: float
    underlying_price: float
    expected_move: float
    short_strike: float
    long_strike: float
    short_bid: float
    short_ask: float
    long_bid: float
    long_ask: float
    short_delta: float
    long_delta: float
    entry_type: str = "credit"
    max_profit: float = 0.0
    put_short_strike: float | None = None
    put_long_strike: float | None = None
    call_short_strike: float | None = None
    call_long_strike: float | None = None
    quote_source: str = "bid/ask"


@dataclass(frozen=True)
class ScoredTrade:
    trade: Trade
    risk_level: str
    category_scores: dict[str, int]
    quant_score: int
    event_adjustment: int
    price_move_adjustment: int
    price_move_style: str
    total_score: int
    reasons: list[str]
    explanation: str


@dataclass(frozen=True)
class CondorDiagnostics:
    ticker: str
    put_spreads_built: int
    call_spreads_built: int
    qualified_puts: int
    qualified_calls: int
    pairs_checked: int
    matching_expiration_pairs: int
    valid_order_pairs: int
    built_condors: int
    top_reason: str

def strategy_fit_score(trade: Trade, preferences: ScanPreferences) -> int:
    if preferences.outlook == "bullish":
        if trade.strategy == "put credit spread":
            return 10
        if trade.strategy == "iron condor":
            return 4
        if trade.strategy == "call credit spread":
            return -10
        if trade.strategy == "bull call debit spread":
            return 10
    if preferences.outlook == "bearish":
        if trade.strategy == "call credit spread":
            return 10
        if trade.strategy == "iron condor":
            return 4
        if trade.strategy == "put credit spread":
            return -10
        if trade.strategy == "bull call debit spread":
            return -5
        if trade.strategy == "bear put debit spread":
            return 10
        
    if preferences.outlook in ["neutral", "income"]:
        if trade.strategy == "iron condor":
            return 10
        if trade.strategy in ["put credit spread", "call credit spread"]:
            return 3
    return 0
    


def bid_ask_spread(trade: Trade) -> float:
    return round(trade.ask - trade.bid, 2)


def expected_move_cushion(trade: Trade) -> float:
    if trade.option_type == "mixed":
        put_side_cushion = trade.underlying_price - trade.short_strike - trade.expected_move
        call_side_cushion = trade.long_strike - trade.underlying_price - trade.expected_move
        return round(min(put_side_cushion, call_side_cushion), 2)

    distance_from_price = abs(trade.short_strike - trade.underlying_price)
    return round(distance_from_price - trade.expected_move, 2)


def spread_width(trade: Trade) -> float:
    if trade.option_type == "mixed":
        return round(trade.credit + trade.max_risk, 2)
    return round(abs(trade.short_strike - trade.long_strike), 2)

def credit_to_width_ratio(trade: Trade) -> float:
    if trade.entry_type != "credit":
        return 0
    
    width = spread_width(trade)
    return trade.credit / width if width else 0
        
def reward_to_risk_ratio(trade: Trade) -> float:
    if trade.entry_type != "debit":
        return 0
    return trade.max_profit / trade.max_risk if trade.max_risk else 0
def cushion_percent_of_expected_move(trade: Trade) -> float:
    if trade.expected_move <= 0:
        return 0

    return expected_move_cushion(trade) / trade.expected_move


def risk_level(trade: Trade) -> str:
    delta = abs(trade.delta)
    credit_ratio = credit_to_width_ratio(trade)
    debit_ratio = reward_to_risk_ratio(trade)
    cushion_ratio = cushion_percent_of_expected_move(trade)

    extreme_signals = 0
    moderate_signals = 0
    if trade.entry_type == "credit":
        if delta > 0.25:
            extreme_signals += 1
        elif delta > 0.20:
            moderate_signals += 1
        if trade.volatility_rank >= 80:
            extreme_signals += 1
        elif trade.volatility_rank >= 60:
            moderate_signals += 1
        if credit_ratio > 0.40:
            extreme_signals += 1
        elif credit_ratio > 0.30:
            moderate_signals += 1
        if cushion_ratio < 0:
            extreme_signals += 1
        elif cushion_ratio < 0.30:
            moderate_signals += 1
    elif trade.entry_type == "debit":
        if debit_ratio < 0.5:
            extreme_signals += 1
        elif debit_ratio < 0.7:
            moderate_signals += 1

    if trade.dte < 30 or trade.dte > 45:
        moderate_signals += 1

    if bid_ask_spread(trade) > 0.12:
        moderate_signals += 1

    if extreme_signals >= 2 or (extreme_signals >= 1 and moderate_signals >= 2):
        return "Extreme"
    if moderate_signals >= 2 or extreme_signals == 1:
        return "Moderate"
    return "Conservative"


def passes_filters(trade: Trade, preferences: ScanPreferences) -> tuple[bool, list[str]]:
    rejection_reasons = []

    if trade.quote_source != "bid/ask":
        rejection_reasons.append("quote is estimated from last price")

    if trade.earnings_before_exp:
        rejection_reasons.append("earnings occur before expiration")

    if trade.entry_type == "credit":
        if not 0.10 <= abs(trade.delta) <= 0.30:
            rejection_reasons.append("delta is outside the credit range of 0.10 to 0.30")
    elif trade.entry_type == "debit":
        if not 0.40 <= abs(trade.delta) <= 0.60:
            rejection_reasons.append("delta is outside the debit range of 0.40 to 0.60")
    if trade.open_interest < 200:
        rejection_reasons.append("open interest is below 200")

    if trade.volume < 25:
        rejection_reasons.append("volume is below 25")

    if trade.volatility_rank < 35:
        rejection_reasons.append("realized volatility rank is below 35")

    if preferences.test_expiration is not None:
        if trade.expiration != preferences.test_expiration.isoformat():
            rejection_reasons.append("expiration does not match the test expiration")
    elif preferences.nearest_expiration:
        pass
    elif not 21 <= trade.dte <= 60:
        rejection_reasons.append("DTE is outside the 21 to 60 day range")

    if bid_ask_spread(trade) >= 0.20:
        rejection_reasons.append("bid/ask spread is wider than 0.20")

    if trade.max_risk <= 0:
        rejection_reasons.append("max risk must be greater than 0")

    if trade.entry_type == "credit":
        if credit_to_width_ratio(trade) < 0.15:
            rejection_reasons.append("credit is below 15% of spread width")
    
    if trade.max_risk * CONTRACT_MULTIPLIER > preferences.max_risk:
        rejection_reasons.append("max risk too high")
    
    if trade.entry_type == "credit":
        if expected_move_cushion(trade) < 0:
            rejection_reasons.append("short strike is inside the expected move")
    
    if trade.entry_type == "debit" and trade.max_risk > 0:
        if trade.max_profit / trade.max_risk <= 0.3:
            rejection_reasons.append("minimum reward to risk ratio is below 0.3")
    
    return len(rejection_reasons) == 0, rejection_reasons


def score_expected_move(trade: Trade) -> int:
    cushion_ratio = cushion_percent_of_expected_move(trade)
    if trade.entry_type == "debit":
        if trade.expected_move <= 0:
            return 0

        if trade.strategy == "bear put debit spread":
            target_ratio = (
                trade.underlying_price - trade.short_strike
            ) / trade.expected_move
        else:
            target_ratio = (
                trade.short_strike - trade.underlying_price
            ) / trade.expected_move

        if 0.50 <= target_ratio <= 1.00:
            return 25
        if 0.25 <= target_ratio < 0.50 or 1.00 < target_ratio <= 1.25:
            return 16
        if 0 < target_ratio < 0.25 or 1.25 < target_ratio <= 1.50:
            return 8
        return 0

    if cushion_ratio >= 0.50:
        return 25
    if cushion_ratio >= 0.3:
        return 20
    if cushion_ratio >= 0.15:
        return 14
    if cushion_ratio >= 0:
        return 8
    return 0


def score_volatility_rank(trade: Trade) -> int:
    if trade.entry_type == "credit":
        if trade.volatility_rank >= 80:
            return 20
        if trade.volatility_rank >= 60:
            return 16
        if trade.volatility_rank >= 45:
            return 12
        if trade.volatility_rank >= 35:
            return 10
    elif trade.entry_type == "debit":
        if trade.volatility_rank >= 60:
            return 12
        if 50 <= trade.volatility_rank < 60:
            return 20
        if trade.volatility_rank >= 40:
            return 16
        if trade.volatility_rank >= 35:
            return 10
    return 0


def score_liquidity(trade: Trade) -> int:
    open_interest_score = min(trade.open_interest / 1000, 1) * 8
    volume_score = min(trade.volume / 250, 1) * 7
    spread_score = 5 if bid_ask_spread(trade) <= 0.10 else 2
    return round(open_interest_score + volume_score + spread_score)


def score_dte(trade: Trade) -> int:
    if 30 <= trade.dte <= 45:
        return 15
    if 21 <= trade.dte < 30:
        return 12
    if 46 <= trade.dte <= 60:
        return 8
    return 0


def score_delta_probability(trade: Trade) -> int:
    delta = abs(trade.delta)
    if trade.entry_type == "credit":
        if 0.16 <= delta <= 0.22:
            return 15
        if 0.10 <= delta < 0.16:
            return 12
        if 0.22 < delta <= 0.30:
            return 8
    elif trade.entry_type == "debit":
        if 0.45 <= delta <= 0.55:
            return 15
        if 0.55 < delta <= 0.60:
            return 12
        if 0.40 <= delta < 0.45:
            return 8
    return 0


def score_profit_risk(trade: Trade) -> int:
    
    ratio = credit_to_width_ratio(trade)
    if trade.entry_type == "credit":     
        if 0.30 <= ratio <= 0.40:
            return 20
        if 0.25 <= ratio < 0.30:
            return 16
        if 0.20 <= ratio < 0.25:
            return 10
        if ratio > 0.40:
            return 8
    elif trade.entry_type == "debit":
        ratio = trade.max_profit / trade.max_risk
        if 0.30 <= ratio <= 0.40:
            return 16
        if 0.25 <= ratio < 0.30:
            return 10
        if 0.20 <= ratio < 0.25:
            return 8
        if ratio > 0.40:
            return 20
    return 4


def price_move_signal(
    trade: Trade,
    price_move: dict[str, float | str] | None,
    event_label: str = "neutral",
) -> tuple[int, str]:
    if not price_move:
        return 0, "None"

    try:
        daily_move = float(price_move.get("1D Move %", 0))
        five_day_move = float(price_move.get("5D Move %", 0))
        move_multiple = float(price_move.get("Move vs 20D Vol", 0))
    except (TypeError, ValueError):
        return 0, "None"

    if move_multiple < 1.5 and abs(five_day_move) < 3:
        return 0, "Normal Move"

    magnitude = 3
    if move_multiple >= 2:
        magnitude = 6
    if move_multiple >= 3:
        magnitude = 8
    if abs(five_day_move) >= 5:
        magnitude = min(10, magnitude + 2)

    if daily_move > 0:
        move_direction = "up"
    elif daily_move < 0:
        move_direction = "down"
    elif five_day_move > 0:
        move_direction = "up"
    elif five_day_move < 0:
        move_direction = "down"
    else:
        return 0, "Normal Move"

    bullish_strategies = {"put credit spread", "bull call debit spread"}
    bearish_strategies = {"call credit spread", "bear put debit spread"}
    clean_reversion_event = event_label.lower() in {"neutral", "supportive"}

    if trade.strategy in bullish_strategies:
        if move_direction == "up":
            return magnitude, "Trend Continuation"
        if clean_reversion_event:
            return 5, "Mean Reversion"
        return -magnitude, "Against Setup"

    if trade.strategy in bearish_strategies:
        if move_direction == "down":
            return magnitude, "Trend Continuation"
        if clean_reversion_event:
            return 5, "Mean Reversion"
        return -magnitude, "Against Setup"

    if trade.strategy == "iron condor":
        return -magnitude, "Unusual Move"
    return 0, "None"


def price_move_adjustment(
    trade: Trade,
    price_move: dict[str, float | str] | None,
    event_label: str = "neutral",
) -> int:
    adjustment, _ = price_move_signal(trade, price_move, event_label)
    return adjustment


def score_trade(
    trade: Trade,
    preferences: ScanPreferences,
    event_adjustment: int = 0,
    price_move: dict[str, float | str] | None = None,
    event_label: str = "neutral",
) -> ScoredTrade:
    category_scores = {
        "Expected Move": score_expected_move(trade),
        "Volatility Rank": score_volatility_rank(trade),
        "Liquidity": score_liquidity(trade),
        "DTE": score_dte(trade),
        "Delta/Probability": score_delta_probability(trade),
        "Profit/Risk": score_profit_risk(trade),
        "Strategy Fit": strategy_fit_score(trade, preferences),
    }
    raw_total_score = sum(category_scores.values())
    quant_score = max(0, min(100, round(raw_total_score / MAX_SETUP_SCORE * 100)))
    price_adjustment, price_style = price_move_signal(
        trade, price_move, event_label
    )
    total_score = max(0, min(100, quant_score + event_adjustment + price_adjustment))
    reasons = passing_reasons(trade)

    return ScoredTrade(
        trade=trade,
        risk_level=risk_level(trade),
        category_scores=category_scores,
        quant_score=quant_score,
        event_adjustment=event_adjustment,
        price_move_adjustment=price_adjustment,
        price_move_style=price_style,
        total_score=total_score,
        reasons=reasons,
        explanation=beginner_explanation(trade, category_scores),
    )


def passing_reasons(trade: Trade) -> list[str]:
    reasons = [
        "No earnings risk before expiration",
        "Liquidity passes minimum open interest and volume checks",
        "Bid/ask spread is tight enough for Phase 1",
    ]

    if trade.volatility_rank >= 60:
        reasons.append("Recent realized volatility is elevated")
    else:
        reasons.append("Recent realized volatility is elevated enough to pass the scanner")

    if trade.entry_type == "credit":
        reasons.append("Delta is in a realistic premium-selling range")
        if expected_move_cushion(trade) > 0:
            reasons.append("Short strike is outside the expected move")
    elif trade.entry_type == "debit":
        reasons.append("Delta is in a realistic bullish range")

    if trade.entry_type == "credit":
        reasons.append("Credit is at least 15% of the spread width")
    elif trade.entry_type == "debit":
        reasons.append("Profit/risk ratio is at least 0.3")
    return reasons


def beginner_explanation(trade: Trade, category_scores: dict[str, int]) -> str:
    cushion = expected_move_cushion(trade)
    credit_width_ratio = credit_to_width_ratio(trade)
    article = "an" if trade.strategy[0].lower() in "aeiou" else "a"
    risk = risk_level(trade)
    max_profit = trade.max_profit
    if trade.entry_type == "debit":
        reward_to_risk = max_profit / trade.max_risk
        if trade.strategy == "bear put debit spread":
            downside_target = trade.underlying_price - trade.short_strike
            return (
                f"{trade.ticker} is a bear put debit spread candidate with "
                f"{risk.lower()} risk. It pays ${trade.max_risk:.2f} to target up to "
                f"${trade.max_profit:.2f} in profit, a {reward_to_risk:.1f}:1 "
                f"reward-to-risk ratio. The short put target is about "
                f"${downside_target:.2f} below the current stock price."
            )

        upside_target = trade.short_strike - trade.underlying_price
        return (
            f"{trade.ticker} is a bull call debit spread candidate with "
            f"{risk.lower()} risk. It pays ${trade.max_risk:.2f} to target up to "
            f"${trade.max_profit:.2f} in profit, a {reward_to_risk:.1f}:1 "
            f"reward-to-risk ratio. The short call target is about "
            f"${upside_target:.2f} above the current stock price."
        )





    return (
        f"{trade.ticker} is {article} {trade.strategy} candidate with {risk.lower()} "
        f"risk because it has enough "
        f"liquidity, no earnings before expiration, and options premiums are "
        f"elevated. The short strike has about ${cushion:.2f} of cushion beyond "
        f"the expected move. The credit is {credit_width_ratio:.0%} of the spread "
        f"width, and the strongest score area is {best_category(category_scores)}."
    )


def best_category(category_scores: dict[str, int]) -> str:
    return max(category_scores, key=category_scores.get)


def spread_summary(trade: Trade) -> str:
    if trade.entry_type == "debit":
        return (
            f"Expiration: {trade.expiration} | "
            f"Long: {trade.long_strike} {trade.option_type} | "
            f"Short: {trade.short_strike} {trade.option_type} | "
            f"Width: ${spread_width(trade):.2f} | "
            f"Debit: ${trade.max_risk:.2f} | "
            f"Max profit: ${trade.max_profit:.2f}"
        )

    if trade.option_type == "mixed":
        return (
            f"Expiration: {trade.expiration} | "
            f"Reference short strike: {trade.short_strike} | "
            f"Reference long strike: {trade.long_strike} | "
            f"Width: ${spread_width(trade):.2f} | Credit: ${trade.credit:.2f}"
        )

    return (
        f"Expiration: {trade.expiration} | "
        f"Short: {trade.short_strike} {trade.option_type} | "
        f"Long: {trade.long_strike} {trade.option_type} | "
        f"Width: ${spread_width(trade):.2f} | Credit: ${trade.credit:.2f}"
    )


def scan_trades(
    trades: list[Trade],
    preferences: ScanPreferences,
    event_adjustments: dict[str, int] | None = None,
    price_moves: dict[str, dict[str, float | str]] | None = None,
    event_labels: dict[str, str] | None = None,
) -> tuple[list[ScoredTrade], list[tuple[Trade, list[str]]]]:
    passing = []
    rejected = []
    if event_adjustments is None:
        event_adjustments = {}
    if price_moves is None:
        price_moves = {}
    if event_labels is None:
        event_labels = {}
    for trade in trades:
        passed, reasons = passes_filters(trade, preferences)

        if passed:
            event_adjustment = event_adjustments.get(trade.ticker, 0)
            price_move = price_moves.get(trade.ticker)
            event_label = event_labels.get(trade.ticker, "neutral")
            passing.append(
                score_trade(
                    trade,
                    preferences,
                    event_adjustment,
                    price_move,
                    event_label,
                )
            )
        else:
            rejected.append((trade, reasons))

    passing.sort(key=lambda scored_trade: scored_trade.total_score, reverse=True)
    return passing, rejected

def build_put_credit_spreads(
    option_chain, underlying_price: float, earnings_date, volatility_rank: float, preferences
):
    puts = []
    trades = []
    for contract in option_chain:
        if contract.option_type == "put":
            puts.append(contract)
    for short_put in puts:
        if not 0.10 <= abs(short_put.delta) <= 0.30:
            continue

        if short_put.open_interest < 200 or short_put.volume < 25:
            continue

        for long_put in puts:  #the 2 for statements test every "pair"
            if short_put.ticker != long_put.ticker:
                continue
            if short_put.expiration != long_put.expiration:
                continue
            #short strike must be higher than long for credit spread, short bid long ask
            if short_put.strike <= long_put.strike:
                continue
            
            credit = round(short_put.bid - long_put.ask, 2)
            width = short_put.strike - long_put.strike
            if width > 5:
                continue
            max_risk = round(width - credit, 2)

            if credit <= 0:
                continue
            if max_risk <= 0:
                continue
            expiration_date = date.fromisoformat(short_put.expiration)
            dte = days_to_expiration(short_put.expiration)
            if dte <= 0:
                continue
            if earnings_date is not None:
                earnings_before_exp = date.today() <= earnings_date <= expiration_date
            else:
                earnings_before_exp = False
            trade = Trade(
            short_put.ticker,
            "put credit spread",
            short_put.expiration,
            "put",
            short_put.delta,
            volatility_rank,
            earnings_before_exp,
            min(short_put.open_interest, long_put.open_interest),
            min(short_put.volume, long_put.volume),
            dte,
            credit,
            round(short_put.ask - long_put.bid, 2),
            credit,
            max_risk,
            underlying_price,
            expected_move(underlying_price, short_put.implied_volatility, dte),
            short_put.strike,
            long_put.strike,
            short_put.bid,
            short_put.ask,
            long_put.bid,
            long_put.ask,
            short_put.delta,
            long_put.delta,
            quote_source=(
                "bid/ask"
                if short_put.quote_source == "bid/ask"
                and long_put.quote_source == "bid/ask"
                else "last price estimate"
            ),
            )
            trades.append(trade)
    return trades

def build_call_credit_spreads(
    option_chain, underlying_price: float, earnings_date, volatility_rank: float, preferences
):
    calls = []
    trades = []
    for contract in option_chain:
        if contract.option_type == "call":
            calls.append(contract)
    for short_call in calls:
        if not 0.10 <= abs(short_call.delta) <= 0.30:
            continue

        if short_call.open_interest < 200 or short_call.volume < 25:
            continue
        
        for long_call in calls:  #the 2 for statements test every "pair"
            if short_call.ticker != long_call.ticker:
                continue
            if short_call.expiration != long_call.expiration:
                continue
            #short strike must be higher than long for credit spread, short bid long ask
            if short_call.strike >= long_call.strike:
                continue
            
            credit = round(short_call.bid - long_call.ask, 2)
            width = long_call.strike - short_call.strike
            if width > 5:
                continue
            max_risk = round(width - credit, 2)

            if credit <= 0:
                continue
            if max_risk <= 0:
                continue

            expiration_date = date.fromisoformat(short_call.expiration)
            dte = days_to_expiration(short_call.expiration)
            if dte <= 0:
                continue
            if earnings_date is not None:
                earnings_before_exp = date.today() <= earnings_date <= expiration_date
            else:
                earnings_before_exp = False

            trade = Trade(
            short_call.ticker,
            "call credit spread",
            short_call.expiration,
            "call",
            short_call.delta,
            volatility_rank,
            earnings_before_exp,
            min(short_call.open_interest, long_call.open_interest),
            min(short_call.volume, long_call.volume),
            dte,
            credit,
            round(short_call.ask - long_call.bid, 2),
            credit,
            max_risk,
            underlying_price,
            expected_move(underlying_price, short_call.implied_volatility, dte),
            short_call.strike,
            long_call.strike,
            short_call.bid,
            short_call.ask,
            long_call.bid,
            long_call.ask,
            short_call.delta,
            long_call.delta,
            quote_source=(
                "bid/ask"
                if short_call.quote_source == "bid/ask"
                and long_call.quote_source == "bid/ask"
                else "last price estimate"
            ),
            )
            trades.append(trade)
    return trades
def build_iron_condor(
    option_chain, underlying_price: float, earnings_date, volatility_rank: float, preferences
):
    trades = []
    iron_condors = []
    credit_put = build_put_credit_spreads(
        option_chain, underlying_price, earnings_date, volatility_rank, preferences
    )
    credit_call = build_call_credit_spreads(
        option_chain, underlying_price, earnings_date, volatility_rank, preferences
    )
    qualified_puts = [
        trade for trade in credit_put
        if passes_filters(trade, preferences)[0]
    ]
    qualified_calls = [
        trade for trade in credit_call
        if passes_filters(trade, preferences)[0]
    ]
    top_puts = sorted(
        qualified_puts,
        key=lambda trade: score_trade(trade, preferences).total_score,
        reverse=True
    )[:5]

    top_calls = sorted(
        qualified_calls,
        key=lambda trade: score_trade(trade, preferences).total_score,
        reverse=True
    )[:5]
    for call_spread in top_calls:
        for put_spread in top_puts:
            if call_spread.expiration !=  put_spread.expiration:
                continue
            if call_spread.ticker != put_spread.ticker:
                continue
            if call_spread.short_strike <= put_spread.short_strike:
                continue
            total_credit = round(call_spread.credit + put_spread.credit, 2)
            put_width = put_spread.short_strike - put_spread.long_strike
            call_width = call_spread.long_strike - call_spread.short_strike
            max_width = max(put_width, call_width)
            max_risk = round(max_width - total_credit, 2)
            combined_bid = round(put_spread.bid + call_spread.bid, 2)
            combined_ask = round(put_spread.ask + call_spread.ask, 2)

            if total_credit <= 0:
                continue
            if max_risk <=0:
                continue

            trade = Trade(
                put_spread.ticker,
                "iron condor",
                put_spread.expiration,
                "mixed",
                max(abs(put_spread.delta), abs(call_spread.delta)),
                volatility_rank,
                put_spread.earnings_before_exp or call_spread.earnings_before_exp,
                min(put_spread.open_interest, call_spread.open_interest),
                min(put_spread.volume, call_spread.volume),
                put_spread.dte,
                combined_bid,
                combined_ask,
                combined_bid,
                max_risk,
                underlying_price,
                max(put_spread.expected_move, call_spread.expected_move),
                put_spread.short_strike,
                call_spread.short_strike,
                put_spread.short_bid,
                put_spread.short_ask,
                call_spread.short_bid,
                call_spread.short_ask,
                put_spread.short_delta,
                call_spread.short_delta,
                put_short_strike=put_spread.short_strike,
                put_long_strike=put_spread.long_strike,
                call_short_strike=call_spread.short_strike,
                call_long_strike=call_spread.long_strike,
                quote_source=(
                    "bid/ask"
                    if put_spread.quote_source == "bid/ask"
                    and call_spread.quote_source == "bid/ask"
                    else "last price estimate"
                ),
            )      
            trades.append(trade)  
    return trades


def condor_diagnostics(
    option_chain,
    underlying_price: float,
    earnings_date,
    volatility_rank: float,
    preferences,
) -> CondorDiagnostics:
    put_spreads = build_put_credit_spreads(
        option_chain, underlying_price, earnings_date, volatility_rank, preferences
    )
    call_spreads = build_call_credit_spreads(
        option_chain, underlying_price, earnings_date, volatility_rank, preferences
    )
    qualified_puts = [
        trade for trade in put_spreads
        if passes_filters(trade, preferences)[0]
    ]
    qualified_calls = [
        trade for trade in call_spreads
        if passes_filters(trade, preferences)[0]
    ]

    top_puts = sorted(
        qualified_puts,
        key=lambda trade: score_trade(trade, preferences).total_score,
        reverse=True,
    )[:5]
    top_calls = sorted(
        qualified_calls,
        key=lambda trade: score_trade(trade, preferences).total_score,
        reverse=True,
    )[:5]

    pairs_checked = 0
    matching_expiration_pairs = 0
    valid_order_pairs = 0
    nonpositive_credit_pairs = 0
    nonpositive_risk_pairs = 0
    built_condors = 0

    for call_spread in top_calls:
        for put_spread in top_puts:
            pairs_checked += 1
            if call_spread.expiration != put_spread.expiration:
                continue
            if call_spread.ticker != put_spread.ticker:
                continue
            matching_expiration_pairs += 1
            if call_spread.short_strike <= put_spread.short_strike:
                continue
            valid_order_pairs += 1

            total_credit = round(call_spread.credit + put_spread.credit, 2)
            put_width = put_spread.short_strike - put_spread.long_strike
            call_width = call_spread.long_strike - call_spread.short_strike
            max_width = max(put_width, call_width)
            max_risk = round(max_width - total_credit, 2)

            if total_credit <= 0:
                nonpositive_credit_pairs += 1
                continue
            if max_risk <= 0:
                nonpositive_risk_pairs += 1
                continue
            built_condors += 1

    if not put_spreads:
        top_reason = "no put credit spreads built"
    elif not call_spreads:
        top_reason = "no call credit spreads built"
    elif not qualified_puts and not qualified_calls:
        top_reason = "no put or call side passed filters"
    elif not qualified_puts:
        top_reason = "no put side passed filters"
    elif not qualified_calls:
        top_reason = "no call side passed filters"
    elif pairs_checked == 0:
        top_reason = "no qualified put/call pairs to combine"
    elif matching_expiration_pairs == 0:
        top_reason = "no matching expiration between qualified sides"
    elif valid_order_pairs == 0:
        top_reason = "put/call short strikes overlap or are reversed"
    elif nonpositive_credit_pairs:
        top_reason = "combined credit was not positive"
    elif nonpositive_risk_pairs:
        top_reason = "combined max risk was not positive"
    elif built_condors == 0:
        top_reason = "no condors survived builder checks"
    else:
        top_reason = "condors built"

    ticker = option_chain[0].ticker if option_chain else "Unknown"
    return CondorDiagnostics(
        ticker=ticker,
        put_spreads_built=len(put_spreads),
        call_spreads_built=len(call_spreads),
        qualified_puts=len(qualified_puts),
        qualified_calls=len(qualified_calls),
        pairs_checked=pairs_checked,
        matching_expiration_pairs=matching_expiration_pairs,
        valid_order_pairs=valid_order_pairs,
        built_condors=built_condors,
        top_reason=top_reason,
    )


def build_bull_call_debit_spread(
        option_chain, underlying_price: float, earnings_date, volatility_rank: float, preferences
):
    calls = [contract for contract in option_chain if contract.option_type == "call"]
    trades = []
    for long_call in calls:
        if not 0.40 <= abs(long_call.delta) <= 0.60:
            continue
        for short_call in calls:
            if short_call.delta >= long_call.delta:
                continue
            if short_call.ticker != long_call.ticker:
                continue
            if short_call.expiration != long_call.expiration:
                continue
            if long_call.strike >= short_call.strike:
                continue
            debit = round(long_call.ask - short_call.bid, 2)
            width = short_call.strike - long_call.strike
            if width > 5:
                continue
            max_profit = width - debit
            if debit <= 0 or max_profit <= 0:
                continue
            expiration_date = date.fromisoformat(short_call.expiration)
            dte = days_to_expiration(short_call.expiration)
            if dte <= 0:
                continue
            if earnings_date is not None:
                earnings_before_exp = date.today() <= earnings_date <= expiration_date
            else:
                earnings_before_exp = False

            trade = Trade(
                ticker=long_call.ticker,
                strategy="bull call debit spread",
                expiration=long_call.expiration,
                option_type="call",
                delta=long_call.delta,
                volatility_rank=volatility_rank,
                earnings_before_exp=earnings_before_exp,
                open_interest=min(long_call.open_interest, short_call.open_interest),
                volume=min(long_call.volume, short_call.volume),
                dte=dte,
                bid=round(long_call.bid - short_call.ask, 2),
                ask=debit,
                credit=0.0,
                max_risk=debit,
                underlying_price=underlying_price,
                expected_move=expected_move(
                    underlying_price, long_call.implied_volatility, dte
                ),
                short_strike=short_call.strike,
                long_strike=long_call.strike,
                short_bid=short_call.bid,
                short_ask=short_call.ask,
                long_bid=long_call.bid,
                long_ask=long_call.ask,
                short_delta=short_call.delta,
                long_delta=long_call.delta,
                entry_type="debit",
                max_profit=max_profit,
                quote_source=(
                    "bid/ask"
                    if long_call.quote_source == "bid/ask"
                    and short_call.quote_source == "bid/ask"
                    else "last price estimate"
                ),
            )
            trades.append(trade)

    return trades

def build_bear_put_debit_spread(
        option_chain, underlying_price: float, earnings_date, volatility_rank: float, preferences
):
    puts = [contract for contract in option_chain if contract.option_type == "put"]
    trades = []
    for long_put in puts:
        if not 0.40 <= abs(long_put.delta) <= 0.60:
            continue
        for short_put in puts:
            if short_put.delta <= long_put.delta:
                continue
            if short_put.ticker != long_put.ticker:
                continue
            if short_put.strike >= long_put.strike:
                continue
            if short_put.expiration != long_put.expiration:
                continue
            debit = round(long_put.ask - short_put.bid, 2)
            width = long_put.strike - short_put.strike
            max_profit = width-debit
            if width > 5:
                continue
            if debit <= 0 or max_profit <= 0:
                continue
            expiration_date = date.fromisoformat(short_put.expiration)
            dte = days_to_expiration(short_put.expiration)
            if dte <= 0:
                continue
            if earnings_date is not None:
                earnings_before_exp = date.today() <= earnings_date <= expiration_date
            else:
                earnings_before_exp = False

            trade = Trade(
                ticker=long_put.ticker,
                strategy="bear put debit spread",
                expiration=long_put.expiration,
                option_type="put",
                delta=long_put.delta,
                volatility_rank=volatility_rank,
                earnings_before_exp=earnings_before_exp,
                open_interest=min(long_put.open_interest, short_put.open_interest),
                volume=min(long_put.volume, short_put.volume),
                dte=dte,
                bid=round(long_put.bid - short_put.ask, 2),
                ask=debit,
                credit=0.0,
                max_risk=debit,
                underlying_price=underlying_price,
                expected_move=expected_move(
                    underlying_price, long_put.implied_volatility, dte
                ),
                short_strike=short_put.strike,
                long_strike=long_put.strike,
                short_bid=short_put.bid,
                short_ask=short_put.ask,
                long_bid=long_put.bid,
                long_ask=long_put.ask,
                short_delta=short_put.delta,
                long_delta=long_put.delta,
                entry_type="debit",
                max_profit=max_profit,
                quote_source=(
                    "bid/ask"
                    if long_put.quote_source == "bid/ask"
                    and short_put.quote_source == "bid/ask"
                    else "last price estimate"
                ),
            )
            trades.append(trade)

    return trades
            




def fake_option_chain() -> list[OptionContract]:
    return [
        #              ticker  expiration    type    strike  bid   ask   delta   gamma  theta  vega  IV    OI    vol
        OptionContract("TEST", "2026-07-31", "put",  175,    0.10, 0.15, -0.08,  0.014, -0.02, 0.08, 0.34, 900,  180),
        OptionContract("TEST", "2026-07-31", "put",  180,    1.20, 1.25, -0.18,  0.023, -0.04, 0.14, 0.38, 1400, 320),
        OptionContract("TEST", "2026-07-31", "put",  195,    1.50, 1.55, -0.30,  0.030, -0.06, 0.20, 0.32, 1200, 250),
        OptionContract("TEST", "2026-07-31", "put",  200,    3.95, 4.00, -0.50,  0.040, -0.08, 0.28, 0.30, 1500, 300),
        OptionContract("TEST", "2026-07-31", "call", 210,    1.20, 1.25,  0.18,  0.023, -0.04, 0.14, 0.38, 1400, 320),
        OptionContract("TEST", "2026-07-31", "call", 215,    0.10, 0.15,  0.08,  0.014, -0.02, 0.08, 0.34, 900,  180),
        OptionContract("CONDOR", "2026-07-31", "put",  170,   0.12, 0.15, -0.08,  0.014, -0.02, 0.08, 0.34, 900,  180),
        OptionContract("CONDOR", "2026-07-31", "put",  175,   1.20, 1.23, -0.18,  0.023, -0.04, 0.14, 0.38, 1400, 320),
        OptionContract("CONDOR", "2026-07-31", "call", 225,   1.20, 1.23,  0.18,  0.023, -0.04, 0.14, 0.38, 1400, 320),
        OptionContract("CONDOR", "2026-07-31", "call", 230,   0.12, 0.15,  0.08,  0.014, -0.02, 0.08, 0.34, 900,  180),
        OptionContract("AAPL", "2026-07-31", "put",  175,    0.70, 0.76, -0.09,  0.018, -0.03, 0.11, 0.31, 950,  180),
        OptionContract("AAPL", "2026-07-31", "put",  180,    1.08, 1.15, -0.15,  0.023, -0.04, 0.14, 0.33, 1500, 310),
        OptionContract("AAPL", "2026-07-31", "put",  185,    2.20, 2.30, -0.24,  0.029, -0.06, 0.18, 0.35, 2100, 500),
        OptionContract("AAPL", "2026-07-31", "put",  190,    3.10, 3.25, -0.38,  0.034, -0.08, 0.22, 0.36, 2500, 760),
        OptionContract("AAPL", "2026-07-31", "call", 200,    3.25, 3.40,  0.42,  0.035, -0.08, 0.23, 0.34, 2800, 820),
        OptionContract("AAPL", "2026-07-31", "call", 205,    1.95, 2.04,  0.27,  0.030, -0.06, 0.19, 0.33, 1900, 430),
        OptionContract("AAPL", "2026-07-31", "call", 210,    1.80, 1.90,  0.18,  0.024, -0.04, 0.15, 0.32, 1300, 260),
        OptionContract("AAPL", "2026-07-31", "call", 215,    0.70, 0.78,  0.11,  0.018, -0.03, 0.12, 0.31, 840,  150),
        OptionContract("NVDA", "2026-07-24", "put",  125,    0.95, 1.05, -0.10,  0.020, -0.05, 0.18, 0.51, 900,  210),
        OptionContract("NVDA", "2026-07-24", "put",  130,    1.55, 1.62, -0.16,  0.026, -0.07, 0.23, 0.53, 1600, 390),
        OptionContract("NVDA", "2026-07-24", "put",  135,    2.65, 2.74, -0.27,  0.034, -0.10, 0.30, 0.55, 2500, 720),
        OptionContract("NVDA", "2026-07-24", "call", 155,    3.85, 4.05,  0.31,  0.036, -0.11, 0.31, 0.54, 2700, 800),
        OptionContract("NVDA", "2026-07-24", "call", 160,    2.65, 2.78,  0.22,  0.030, -0.09, 0.26, 0.53, 2100, 610),
        OptionContract("NVDA", "2026-07-24", "call", 165,    1.85, 1.96,  0.15,  0.024, -0.07, 0.21, 0.51, 1500, 350),
        OptionContract("NVDA", "2026-07-24", "call", 170,    1.20, 1.30,  0.10,  0.019, -0.05, 0.17, 0.50, 950,  220),
    ]


def sample_trades() -> list[Trade]:
    return [
        #     ticker  strategy              expiration    type     delta  IVR  earn?  OI    vol  DTE  bid   ask   credit  risk  price  exp_move  short  long  short_bid  short_ask  long_bid  long_ask  short_delta  long_delta
        Trade("SNDK", "iron condor",        "2026-07-24", "mixed",  0.18, 62,  False, 620,  140, 35, 1.79, 1.87, 1.80,   3.20, 70,   8,       83,    88,   2.65,       2.74,       0.85,      0.87,      0.18,        0.08),
        Trade("AAPL", "put credit spread",  "2026-07-31", "put",   -0.20, 54,  False, 1800, 420, 38, 1.12, 1.19, 1.15,   3.85, 195,  11,      181,   176,  2.10,       2.18,       0.95,      0.99,     -0.20,       -0.12),
        Trade("NVDA", "call credit spread", "2026-07-24", "call",   0.16, 76,  False, 2200, 650, 31, 2.20, 2.30, 2.25,   7.75, 142,  14,      164,   174,  3.80,       3.95,       1.55,      1.65,      0.16,        0.09),
        Trade("TSLA", "iron condor",        "2026-07-31", "mixed",  0.19, 81,  True,  1200, 300, 42, 2.60, 2.72, 2.66,   7.34, 180,  22,      215,   225,  4.90,       5.05,       2.24,      2.33,      0.19,        0.10),
        Trade("AMD",  "put credit spread",  "2026-07-24", "put",   -0.21, 45,  False, 155,  80,  32, 1.05, 1.12, 1.08,   3.92, 118,  9,       105,   100,  2.00,       2.08,       0.92,      0.96,     -0.21,       -0.13),
        Trade("META", "call credit spread", "2026-07-31", "call",   0.34, 67,  False, 900,  210, 36, 1.70, 1.78, 1.74,   3.26, 510,  28,      548,   553,  3.10,       3.22,       1.36,      1.44,      0.34,        0.25),
        Trade("MSFT", "put credit spread",  "2026-07-24", "put",   -0.17, 38,  False, 500,  12,  28, 0.90, 0.98, 0.92,   4.08, 470,  16,      450,   445,  1.80,       1.90,       0.88,      0.92,     -0.17,       -0.10),
        Trade("COIN", "call credit spread", "2026-08-07", "call",   0.23, 88,  False, 700,  180, 49, 2.10, 2.34, 2.20,   7.80, 245,  30,      286,   296,  4.70,       4.94,       2.50,      2.60,      0.23,        0.14),
        Trade("AMZN", "put credit spread",  "2026-07-17", "put",   -0.14, 41,  False, 350,  70,  24, 0.75, 0.86, 0.80,   4.20, 185,  10,      176,   171,  1.40,       1.49,       0.60,      0.63,     -0.14,       -0.08),
    ]


def test_event_adjustments() -> None:
    preferences = ScanPreferences(
        max_risk=500,
        outlook="neutral",
        risk_tolerance="moderate",
    )
    scored_trades, _ = scan_trades(
        sample_trades(), preferences, event_adjustments={"AAPL": -5}
    )

    aapl_trade = next(
        scored for scored in scored_trades if scored.trade.ticker == "AAPL"
    )
    expected_aapl_score = max(0, min(100, aapl_trade.quant_score - 5))
    assert aapl_trade.event_adjustment == -5
    assert aapl_trade.total_score == expected_aapl_score

    for scored in scored_trades:
        if scored.trade.ticker != "AAPL":
            assert scored.event_adjustment == 0

    print(
        "Event adjustment test passed: "
        f"AAPL {aapl_trade.quant_score} + ({aapl_trade.event_adjustment}) "
        f"= {aapl_trade.total_score}"
    )

def print_rejections(trades, rejected_trades, scored_trades):
    passing_by_ticker = Counter(
        scored.trade.ticker for scored in scored_trades
    )
    total_by_ticker = Counter(trade.ticker for trade in trades)
    rejected_by_ticker = Counter(
        trade.ticker for trade, reasons in rejected_trades
    )
    for ticker, total_count in total_by_ticker.items():
        rejected_count = rejected_by_ticker[ticker]
        rejection_rate = rejected_count / total_count
        
        if rejection_rate >= 0.9 and passing_by_ticker[ticker] == 0:
            print(f"{ticker}: {rejection_rate:.0%} of candidates were rejected")



def print_report(scored_trades: list[ScoredTrade], rejected_trades: list[tuple[Trade, list[str]]]) -> None:
    print("AI Options Scanner - Phase 1 CLI Prototype")
    print("=" * 48)
    printed_by_ticker = Counter()
    display_index = 0
    
    print("\nPassing trades ranked by quant score:")
    if not scored_trades:
        print("No trades passed the Phase 1 filters.")
    else:
        for scored in scored_trades:
            trade = scored.trade

            if printed_by_ticker[trade.ticker] >= 4:
                continue
            printed_by_ticker[trade.ticker] +=1
            display_index += 1
            print(
                f"\n{display_index}. {trade.ticker} {trade.strategy} - "
                f"Setup Score: {scored.total_score}/100 - Risk: {scored.risk_level}"
            )
            print(spread_summary(trade))
            print("Score breakdown:")
            for category, score in scored.category_scores.items():
                print(f"  {category}: +{score}")

            print("Why it passed:")
            for reason in scored.reasons:
                print(f"  - {reason}")

            print(f"Beginner explanation: {scored.explanation}")

    print("\nRejected trades:")
    if not rejected_trades:
        print("No trades were rejected.")
    else:
        rejection_counts = {}

        for trade, reasons in rejected_trades:
            for reason in reasons:
                if reason not in rejection_counts:
                    rejection_counts[reason] = 0
                rejection_counts[reason] += 1
        
        for reason, count in rejection_counts.items():
            print(f"{count} rejected because {reason}")
                

# print why whole stock was rejected 
def main() -> None:
    
    ticker_list = ["AAPL", "SPY", "QQQ", "NVDA", "MSFT", "COHR"]
    
    trades = []
    preferences = ScanPreferences(
        max_risk=500,
        outlook="neutral",
        risk_tolerance="moderate"
    )
    for ticker in ticker_list:
        
        (
            underlying_price,
            option_chain,
            earnings_date,
            volatility_rank,
            price_move,
        ) = get_option_chain(ticker)
        print(f"{ticker} reference price: ${underlying_price:.2f}")
        print(f"{ticker} usable option contracts fetched: {len(option_chain)}")
        print(f"{ticker} realized volatility rank: {volatility_rank:.1f}")
        print(f"{ticker} latest daily move: {price_move['1D Move %']:+.2f}%")

        
        trades.extend(build_iron_condor(option_chain, underlying_price, earnings_date, volatility_rank, preferences))
        trades.extend(build_call_credit_spreads(option_chain, underlying_price, earnings_date, volatility_rank, preferences))
        trades.extend(build_put_credit_spreads(option_chain, underlying_price, earnings_date, volatility_rank, preferences))
        trades.extend(build_bear_put_debit_spread(option_chain, underlying_price, earnings_date, volatility_rank, preferences))
        trades.extend(build_bull_call_debit_spread(option_chain, underlying_price, earnings_date, volatility_rank, preferences))
    scored_trades, rejected_trades = scan_trades(trades, preferences)
    print_rejections(trades, rejected_trades, scored_trades)
    print_report(scored_trades, rejected_trades,)

if __name__ == "__main__":
    main()
