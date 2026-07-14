from bisect import bisect_left, bisect_right
from dataclasses import dataclass, replace
from datetime import date
import math
from pathlib import Path
import time
import yfinance as yf
from collections import Counter

try:
    from alpaca_client import get_stock_daily_bars
except ImportError:
    def get_stock_daily_bars(ticker: str, lookback_days: int = 430):
        return [], ["Alpaca stock bars helper is unavailable."]


CONTRACT_MULTIPLIER = 100
MAX_SETUP_SCORE = 125
PRICE_MOVE_MODES = {"Full", "Conservative", "Shadow", "Off"}
EXPIRATION_COVERAGE_FAST_WEEKLY = "fast_weekly"
EXPIRATION_COVERAGE_EXHAUSTIVE = "exhaustive"
EXPIRATION_COVERAGE_MODES = {
    EXPIRATION_COVERAGE_FAST_WEEKLY,
    EXPIRATION_COVERAGE_EXHAUSTIVE,
}
BULLISH_STRATEGIES = {
    "put credit spread",
    "bull call debit spread",
}
BEARISH_STRATEGIES = {
    "call credit spread",
    "bear put debit spread",
}
NEUTRAL_STRATEGIES = {"iron condor"}
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
    price_move_mode: str = "Full"
    condor_max_wings_per_side: int = 15
    condor_max_bid_ask_width: float = 0.40
    condor_max_bid_ask_to_credit_ratio: float | None = None
    expiration_coverage: str = EXPIRATION_COVERAGE_EXHAUSTIVE



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


@dataclass(frozen=True)
class OptionChainResult:
    underlying_price: float
    contracts: list[OptionContract]
    earnings_date: date | None
    volatility_rank: float
    price_move: dict[str, float | str]
    expirations_fetched: tuple[str, ...]
    expiration_coverage: str
    timings: dict[str, float]

    def legacy_tuple(self):
        return (
            self.underlying_price,
            self.contracts,
            self.earnings_date,
            self.volatility_rank,
            self.price_move,
        )


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


def select_expirations(
    available_expirations: list[str] | tuple[str, ...],
    test_expiration: date | None = None,
    nearest_expiration: bool = False,
    expiration_coverage: str = EXPIRATION_COVERAGE_EXHAUSTIVE,
) -> list[str]:
    dte_by_expiration = {
        expiration: days_to_expiration(expiration)
        for expiration in available_expirations
    }
    available = sorted(
        (
            expiration
            for expiration in available_expirations
            if dte_by_expiration[expiration] >= 0
        ),
        key=lambda expiration: (dte_by_expiration[expiration], expiration),
    )

    if test_expiration is not None:
        selected_expiration = test_expiration.isoformat()
        if selected_expiration not in available:
            raise ValueError(
                f"Yahoo Finance does not offer {selected_expiration} for this ticker."
            )
        return [selected_expiration]

    if nearest_expiration:
        return available[:5]

    if expiration_coverage not in EXPIRATION_COVERAGE_MODES:
        raise ValueError(
            f"Unsupported expiration coverage mode: {expiration_coverage}"
        )

    in_range = [
        expiration
        for expiration in available
        if 21 <= dte_by_expiration[expiration] <= 60
    ]
    if expiration_coverage == EXPIRATION_COVERAGE_EXHAUSTIVE:
        return in_range

    nearest_by_bucket = {}
    for expiration in in_range:
        bucket = (dte_by_expiration[expiration] - 21) // 7
        nearest_by_bucket.setdefault(bucket, expiration)
    return list(nearest_by_bucket.values())[:7]


def get_option_chain_result(
    ticker: str,
    test_expiration: date | None = None,
    nearest_expiration: bool = False,
    expiration_coverage: str = EXPIRATION_COVERAGE_EXHAUSTIVE,
) -> OptionChainResult:
    timings = {}
    stock = yf.Ticker(ticker)
    underlying_started = time.perf_counter()
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
    timings["underlying_history_data"] = time.perf_counter() - underlying_started
    contracts = []

    expiration_list_started = time.perf_counter()
    available_expirations = list(stock.options)
    if (
        test_expiration is not None
        and test_expiration.isoformat() not in available_expirations
    ):
        raise ValueError(
            f"Yahoo Finance does not offer {test_expiration.isoformat()} for {ticker}."
        )
    expirations_to_fetch = select_expirations(
        available_expirations,
        test_expiration=test_expiration,
        nearest_expiration=nearest_expiration,
        expiration_coverage=expiration_coverage,
    )
    timings["expiration_list_retrieval"] = (
        time.perf_counter() - expiration_list_started
    )

    download_started = time.perf_counter()
    dte_by_expiration = {
        expiration: days_to_expiration(expiration)
        for expiration in expirations_to_fetch
    }
    for expiration in expirations_to_fetch:
        dte = dte_by_expiration[expiration]
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
    timings["option_chain_downloads"] = time.perf_counter() - download_started

    if not contracts:
        expiration_range = (
            "the selected test expiration"
            if test_expiration is not None or nearest_expiration
            else "the 21 to 60 DTE range"
        )
        raise ValueError(
            f"Yahoo Finance returned no usable option contracts for {expiration_range}."
        )

    return OptionChainResult(
        underlying_price=underlying_price,
        contracts=contracts,
        earnings_date=earnings_date,
        volatility_rank=volatility_rank,
        price_move=price_move,
        expirations_fetched=tuple(expirations_to_fetch),
        expiration_coverage=expiration_coverage,
        timings=timings,
    )


def get_option_chain(
    ticker: str,
    test_expiration: date | None = None,
    nearest_expiration: bool = False,
    expiration_coverage: str = EXPIRATION_COVERAGE_EXHAUSTIVE,
) -> tuple[float, list[OptionContract], date | None, float, dict[str, float | str]]:
    return get_option_chain_result(
        ticker,
        test_expiration=test_expiration,
        nearest_expiration=nearest_expiration,
        expiration_coverage=expiration_coverage,
    ).legacy_tuple()


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
    put_expected_move_cushion: float | None = None
    call_expected_move_cushion: float | None = None
    minimum_expected_move_cushion: float | None = None
    condor_bid_ask_to_credit_ratio: float | None = None


@dataclass(frozen=True)
class ScoredTrade:
    trade: Trade
    risk_level: str
    category_scores: dict[str, int]
    quant_score: int
    event_adjustment: int
    raw_price_move_adjustment: int
    effective_price_move_adjustment: int
    price_move_adjustment: int
    price_move_style: str
    base_score_without_price_move: int
    total_score: int
    reasons: list[str]
    explanation: str
    normalized_ticker_score: int = 0
    raw_rank: int | None = None
    diversified_rank: int | None = None
    execution_rank: int | None = None


@dataclass(frozen=True)
class CondorDiagnostics:
    ticker: str
    raw_put_wings_built: int
    raw_call_wings_built: int
    put_wings_rejected_for_liquidity: int
    call_wings_rejected_for_liquidity: int
    wings_rejected_for_delta: int
    wings_rejected_for_estimated_quotes: int
    eligible_put_expirations: int
    eligible_call_expirations: int
    expirations_with_both_sides: int
    pairs_attempted: int
    pairs_rejected_for_mismatched_expiration: int
    pairs_rejected_for_strike_overlap: int
    condors_built: int
    condors_rejected_for_nonpositive_combined_credit: int
    condors_rejected_for_max_risk: int
    condors_rejected_for_combined_credit_to_width: int
    condors_rejected_for_put_expected_move: int
    condors_rejected_for_call_expected_move: int
    condors_rejected_for_bid_ask_width: int
    condors_passing_final_filters: int
    highest_passing_condor_score: int | None
    highest_built_condor_score: int | None
    primary_blocker: str

    @property
    def built_condors(self) -> int:
        return self.condors_built

    @property
    def top_reason(self) -> str:
        return self.primary_blocker


@dataclass(frozen=True)
class CondorConstructionResult:
    condors: list[Trade]
    diagnostics: CondorDiagnostics
    timings: dict[str, float]
    operation_counts: dict[str, int]


def strategy_direction(strategy: str) -> str:
    normalized = strategy.lower().strip()
    if normalized in BULLISH_STRATEGIES:
        return "bullish"
    if normalized in BEARISH_STRATEGIES:
        return "bearish"
    if normalized in NEUTRAL_STRATEGIES:
        return "neutral"
    return "unknown"


def scored_trade_sort_key(scored: ScoredTrade) -> tuple[int, int, int]:
    return (
        scored.total_score,
        scored.normalized_ticker_score,
        scored.quant_score,
    )

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
        if trade.minimum_expected_move_cushion is not None:
            return round(trade.minimum_expected_move_cushion, 2)
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


def condor_put_cushion(trade: Trade) -> float:
    if trade.put_expected_move_cushion is not None:
        return round(trade.put_expected_move_cushion, 2)
    put_short_strike = (
        trade.put_short_strike
        if trade.put_short_strike is not None
        else trade.short_strike
    )
    return round(
        trade.underlying_price - put_short_strike - trade.expected_move,
        2,
    )


def condor_call_cushion(trade: Trade) -> float:
    if trade.call_expected_move_cushion is not None:
        return round(trade.call_expected_move_cushion, 2)
    call_short_strike = (
        trade.call_short_strike
        if trade.call_short_strike is not None
        else trade.long_strike
    )
    return round(
        call_short_strike - trade.underlying_price - trade.expected_move,
        2,
    )


def condor_bid_ask_to_credit_ratio(trade: Trade) -> float | None:
    if trade.credit <= 0:
        return None
    return round(bid_ask_spread(trade) / trade.credit, 4)


def condor_wing_rejection_reasons(trade: Trade) -> list[str]:
    reasons = []
    width = spread_width(trade)
    if trade.quote_source != "bid/ask":
        reasons.append("quote is estimated from last price")
    if not 0.10 <= abs(trade.delta) <= 0.30:
        reasons.append("delta is outside the credit range of 0.10 to 0.30")
    if trade.open_interest < 200:
        reasons.append("open interest is below 200")
    if trade.volume < 25:
        reasons.append("volume is below 25")
    if trade.dte <= 0:
        reasons.append("DTE must be greater than zero")
    if width <= 0 or width > 5:
        reasons.append("wing width is outside the supported range")
    if trade.credit <= 0:
        reasons.append("wing credit must be greater than zero")
    return reasons


def passes_condor_wing_filters(trade: Trade) -> tuple[bool, list[str]]:
    reasons = condor_wing_rejection_reasons(trade)
    return len(reasons) == 0, reasons


def condor_rejection_reasons(
    trade: Trade,
    preferences: ScanPreferences,
) -> list[str]:
    reasons = []
    max_width = spread_width(trade)
    put_cushion = condor_put_cushion(trade)
    call_cushion = condor_call_cushion(trade)
    four_leg_bid_ask_width = bid_ask_spread(trade)
    bid_ask_credit_ratio = condor_bid_ask_to_credit_ratio(trade)

    if trade.quote_source != "bid/ask":
        reasons.append("quote is estimated from last price")
    if trade.credit <= 0:
        reasons.append("condor combined credit must be greater than zero")
    if trade.max_risk <= 0:
        reasons.append("max risk must be greater than 0")
    if trade.max_risk * CONTRACT_MULTIPLIER > preferences.max_risk:
        reasons.append("condor max risk too high")
    if max_width <= 0 or trade.credit / max_width < 0.15:
        reasons.append("condor credit is below 15% of maximum wing width")
    if put_cushion < 0:
        reasons.append("condor put short strike is inside the expected move")
    if call_cushion < 0:
        reasons.append("condor call short strike is inside the expected move")
    if trade.earnings_before_exp:
        reasons.append("earnings occur before expiration")
    if trade.open_interest < 200:
        reasons.append("open interest is below 200")
    if trade.volume < 25:
        reasons.append("volume is below 25")

    if preferences.test_expiration is not None:
        if trade.expiration != preferences.test_expiration.isoformat():
            reasons.append("expiration does not match the test expiration")
    elif preferences.nearest_expiration:
        pass
    elif not 21 <= trade.dte <= 60:
        reasons.append("DTE is outside the 21 to 60 day range")

    if four_leg_bid_ask_width > preferences.condor_max_bid_ask_width:
        reasons.append("condor four-leg bid/ask spread is wider than configured maximum")
    if (
        preferences.condor_max_bid_ask_to_credit_ratio is not None
        and bid_ask_credit_ratio is not None
        and bid_ask_credit_ratio
        > preferences.condor_max_bid_ask_to_credit_ratio
    ):
        reasons.append("condor bid/ask-to-credit ratio is too high")
    return reasons


def passes_condor_filters(
    trade: Trade,
    preferences: ScanPreferences,
) -> tuple[bool, list[str]]:
    reasons = condor_rejection_reasons(trade, preferences)
    return len(reasons) == 0, reasons


def passes_filters(trade: Trade, preferences: ScanPreferences) -> tuple[bool, list[str]]:
    if trade.strategy == "iron condor":
        return passes_condor_filters(trade, preferences)

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


def apply_price_move_mode(raw_adjustment: int, mode: str) -> tuple[int, int]:
    normalized_mode = mode.title()
    if normalized_mode not in PRICE_MOVE_MODES:
        normalized_mode = "Full"
    if normalized_mode == "Off":
        return raw_adjustment, 0
    if normalized_mode == "Shadow":
        return raw_adjustment, 0
    if normalized_mode == "Conservative":
        return raw_adjustment, min(raw_adjustment, 3)
    return raw_adjustment, raw_adjustment


def score_trade(
    trade: Trade,
    preferences: ScanPreferences,
    event_adjustment: int = 0,
    price_move: dict[str, float | str] | None = None,
    event_label: str = "neutral",
) -> ScoredTrade:
    category_scores = {
        "Expected Move": score_expected_move(trade),
        "Realized Volatility Rank": score_volatility_rank(trade),
        "Liquidity": score_liquidity(trade),
        "DTE": score_dte(trade),
        "Delta/Probability": score_delta_probability(trade),
        "Profit/Risk": score_profit_risk(trade),
        "Strategy Fit": strategy_fit_score(trade, preferences),
    }
    raw_total_score = sum(category_scores.values())
    quant_score = max(0, min(100, round(raw_total_score / MAX_SETUP_SCORE * 100)))
    calculated_price_adjustment, price_style = price_move_signal(
        trade, price_move, event_label
    )
    raw_price_adjustment, effective_price_adjustment = apply_price_move_mode(
        calculated_price_adjustment,
        preferences.price_move_mode,
    )
    base_score_without_price_move = max(
        0, min(100, quant_score + event_adjustment)
    )
    total_score = max(
        0, min(100, base_score_without_price_move + effective_price_adjustment)
    )
    reasons = passing_reasons(trade)

    return ScoredTrade(
        trade=trade,
        risk_level=risk_level(trade),
        category_scores=category_scores,
        quant_score=quant_score,
        event_adjustment=event_adjustment,
        raw_price_move_adjustment=raw_price_adjustment,
        effective_price_move_adjustment=effective_price_adjustment,
        price_move_adjustment=effective_price_adjustment,
        price_move_style=price_style,
        base_score_without_price_move=base_score_without_price_move,
        total_score=total_score,
        reasons=reasons,
        explanation=beginner_explanation(trade, category_scores),
    )


def apply_normalized_ticker_scores(scored_trades: list[ScoredTrade]) -> list[ScoredTrade]:
    scores_by_ticker = {}
    for scored in scored_trades:
        scores_by_ticker.setdefault(scored.trade.ticker, []).append(scored.total_score)

    normalized = []
    for scored in scored_trades:
        ticker_scores = scores_by_ticker[scored.trade.ticker]
        if len(ticker_scores) == 1:
            ticker_score = 50
        else:
            lower_or_equal_count = sum(
                score <= scored.total_score for score in ticker_scores
            )
            ticker_score = round(lower_or_equal_count / len(ticker_scores) * 100)
        normalized.append(
            replace(scored, normalized_ticker_score=max(0, min(100, ticker_score)))
        )
    return normalized


def similar_spread(left: ScoredTrade, right: ScoredTrade) -> bool:
    left_trade = left.trade
    right_trade = right.trade
    if (
        left_trade.ticker != right_trade.ticker
        or left_trade.strategy != right_trade.strategy
        or left_trade.expiration != right_trade.expiration
    ):
        return False

    if left_trade.strategy == "iron condor":
        left_strikes = (
            left_trade.put_long_strike,
            left_trade.put_short_strike,
            left_trade.call_short_strike,
            left_trade.call_long_strike,
        )
        right_strikes = (
            right_trade.put_long_strike,
            right_trade.put_short_strike,
            right_trade.call_short_strike,
            right_trade.call_long_strike,
        )
        if any(value is None for value in left_strikes + right_strikes):
            return False
        return all(
            abs(float(left_value) - float(right_value)) <= 2
            for left_value, right_value in zip(left_strikes, right_strikes)
        )

    left_width = spread_width(left_trade)
    right_width = spread_width(right_trade)
    left_center = (left_trade.short_strike + left_trade.long_strike) / 2
    right_center = (right_trade.short_strike + right_trade.long_strike) / 2
    return (
        abs(left_width - right_width) <= 1
        and abs(left_center - right_center) <= 2
    )


def remove_similar_spreads(scored_trades: list[ScoredTrade]) -> list[ScoredTrade]:
    selected = []
    for scored in sorted(
        scored_trades,
        key=scored_trade_sort_key,
        reverse=True,
    ):
        if any(similar_spread(scored, existing) for existing in selected):
            continue
        selected.append(scored)
    return selected


def diversify_scored_trades(scored_trades: list[ScoredTrade]) -> list[ScoredTrade]:
    deduped = remove_similar_spreads(scored_trades)
    normalized = apply_normalized_ticker_scores(deduped)
    best_by_ticker_strategy = {}

    for scored in normalized:
        key = (scored.trade.ticker, scored.trade.strategy)
        current_best = best_by_ticker_strategy.get(key)
        if current_best is None or scored_trade_sort_key(scored) > scored_trade_sort_key(
            current_best
        ):
            best_by_ticker_strategy[key] = scored

    ranked = sorted(
        best_by_ticker_strategy.values(),
        key=scored_trade_sort_key,
        reverse=True,
    )
    return [
        replace(scored, diversified_rank=rank)
        for rank, scored in enumerate(ranked, start=1)
    ]


def select_execution_candidates(
    scored_trades: list[ScoredTrade],
    limit: int = 3,
    max_per_ticker: int = 1,
) -> list[ScoredTrade]:
    if limit <= 0 or max_per_ticker <= 0:
        return []

    selected = []
    selected_by_ticker = Counter()
    for scored in sorted(scored_trades, key=scored_trade_sort_key, reverse=True):
        ticker = scored.trade.ticker
        if selected_by_ticker[ticker] >= max_per_ticker:
            continue
        selected.append(replace(scored, execution_rank=len(selected) + 1))
        selected_by_ticker[ticker] += 1
        if len(selected) >= limit:
            break
    return selected


def select_history_candidates(
    scored_trades: list[ScoredTrade],
    limit: int = 25,
    per_ticker: int = 4,
    per_strategy: int = 1,
) -> list[ScoredTrade]:
    if limit <= 0 or per_ticker <= 0 or per_strategy < 0:
        return []

    ranked = sorted(scored_trades, key=scored_trade_sort_key, reverse=True)
    selected = []
    selected_ids = set()
    selected_by_strategy = Counter()
    selected_by_ticker = Counter()

    # Reserve a small number of slots for each strategy, while still applying
    # the same ticker cap used by the overall fill pass.
    for scored in ranked:
        strategy = scored.trade.strategy
        ticker = scored.trade.ticker
        scored_id = id(scored)
        if (
            selected_by_strategy[strategy] >= per_strategy
            or selected_by_ticker[ticker] >= per_ticker
            or scored_id in selected_ids
        ):
            continue
        selected.append(scored)
        selected_ids.add(scored_id)
        selected_by_strategy[strategy] += 1
        selected_by_ticker[ticker] += 1
        if len(selected) >= limit:
            return selected

    for scored in ranked:
        ticker = scored.trade.ticker
        scored_id = id(scored)
        if scored_id in selected_ids or selected_by_ticker[ticker] >= per_ticker:
            continue
        selected.append(scored)
        selected_ids.add(scored_id)
        selected_by_ticker[ticker] += 1
        if len(selected) >= limit:
            break

    return selected


def execution_selection_diagnostics(
    scored_trades: list[ScoredTrade],
    selected_trades: list[ScoredTrade],
    requested_limit: int = 3,
) -> list[dict[str, str | int]]:
    selected_tickers = {scored.trade.ticker for scored in selected_trades}
    selected_keys = {
        (scored.trade.ticker, scored.trade.strategy, scored.trade.expiration)
        for scored in selected_trades
    }
    diagnostics = []
    for scored in scored_trades:
        key = (
            scored.trade.ticker,
            scored.trade.strategy,
            scored.trade.expiration,
        )
        if key in selected_keys:
            continue
        if scored.trade.ticker in selected_tickers:
            diagnostics.append(
                {
                    "Ticker": scored.trade.ticker,
                    "Strategy": scored.trade.strategy.title(),
                    "Reason": (
                        "Ticker already selected; this was a lower-ranked strategy "
                        "for the same ticker."
                    ),
                }
            )

    diagnostics.append(
        {
            "Ticker": "All",
            "Strategy": "Near duplicates",
            "Reason": "Duplicate or similar spreads were removed before execution selection.",
        }
    )
    if len(selected_trades) < requested_limit:
        diagnostics.append(
            {
                "Ticker": "All",
                "Strategy": "Execution selection",
                "Reason": (
                    f"Only {len(selected_trades)} distinct ticker(s) were available "
                    f"for {requested_limit} requested slots."
                ),
            }
        )
    return diagnostics


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
    timing_by_ticker: dict[str, float] | None = None,
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
        trade_started = time.perf_counter() if timing_by_ticker is not None else None
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
        if timing_by_ticker is not None:
            timing_by_ticker[trade.ticker] = (
                timing_by_ticker.get(trade.ticker, 0.0)
                + time.perf_counter()
                - trade_started
            )

    passing.sort(key=scored_trade_sort_key, reverse=True)
    passing = [
        replace(scored, raw_rank=rank)
        for rank, scored in enumerate(passing, start=1)
    ]
    return diversify_scored_trades(passing), rejected

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
def _condor_wing_trade(
    short_contract: OptionContract,
    long_contract: OptionContract,
    underlying_price: float,
    earnings_date,
    volatility_rank: float,
    *,
    dte: int | None = None,
    expiration_date: date | None = None,
    earnings_before_exp: bool | None = None,
    expected_move_value: float | None = None,
) -> Trade:
    is_put = short_contract.option_type == "put"
    width = (
        short_contract.strike - long_contract.strike
        if is_put
        else long_contract.strike - short_contract.strike
    )
    credit = round(short_contract.bid - long_contract.ask, 2)
    expiration_date = expiration_date or date.fromisoformat(short_contract.expiration)
    dte = dte if dte is not None else days_to_expiration(short_contract.expiration)
    if earnings_before_exp is None:
        earnings_before_exp = (
            date.today() <= earnings_date <= expiration_date
            if earnings_date is not None
            else False
        )
    return Trade(
        ticker=short_contract.ticker,
        strategy=("put credit spread" if is_put else "call credit spread"),
        expiration=short_contract.expiration,
        option_type=short_contract.option_type,
        delta=short_contract.delta,
        volatility_rank=volatility_rank,
        earnings_before_exp=earnings_before_exp,
        open_interest=min(
            short_contract.open_interest,
            long_contract.open_interest,
        ),
        volume=min(short_contract.volume, long_contract.volume),
        dte=dte,
        bid=credit,
        ask=round(short_contract.ask - long_contract.bid, 2),
        credit=credit,
        max_risk=round(width - credit, 2),
        underlying_price=underlying_price,
        expected_move=(
            expected_move_value
            if expected_move_value is not None
            else expected_move(
                underlying_price,
                short_contract.implied_volatility,
                dte,
            )
        ),
        short_strike=short_contract.strike,
        long_strike=long_contract.strike,
        short_bid=short_contract.bid,
        short_ask=short_contract.ask,
        long_bid=long_contract.bid,
        long_ask=long_contract.ask,
        short_delta=short_contract.delta,
        long_delta=long_contract.delta,
        quote_source=(
            "bid/ask"
            if short_contract.quote_source == "bid/ask"
            and long_contract.quote_source == "bid/ask"
            else "last price estimate"
        ),
    )


def _build_condor_wings_for_type(
    option_chain,
    option_type: str,
    underlying_price: float,
    earnings_date,
    volatility_rank: float,
    operation_stats: Counter | None = None,
) -> tuple[list[Trade], Counter]:
    contracts = [
        contract
        for contract in option_chain
        if contract.option_type == option_type
    ]
    eligible_wings = []
    stats = Counter()
    grouped_contracts = {}
    for contract in contracts:
        grouped_contracts.setdefault(
            (contract.ticker, contract.expiration, contract.option_type), []
        ).append(contract)
    strike_indexes = {}
    for key, group in grouped_contracts.items():
        by_strike = sorted(
            enumerate(group),
            key=lambda item: (item[1].strike, item[0]),
        )
        strike_indexes[key] = (
            [contract.strike for _, contract in by_strike],
            by_strike,
        )

    dte_by_expiration = {}
    date_by_expiration = {}
    earnings_by_expiration = {}
    expected_move_by_contract = {}

    for short_contract in contracts:
        key = (
            short_contract.ticker,
            short_contract.expiration,
            short_contract.option_type,
        )
        strikes, by_strike = strike_indexes[key]
        expiration = short_contract.expiration
        if expiration not in dte_by_expiration:
            dte_by_expiration[expiration] = days_to_expiration(expiration)
            date_by_expiration[expiration] = date.fromisoformat(expiration)
            earnings_by_expiration[expiration] = (
                date.today() <= earnings_date <= date_by_expiration[expiration]
                if earnings_date is not None
                else False
            )
        dte = dte_by_expiration[expiration]
        if dte <= 0:
            continue
        expected_move_key = (
            expiration,
            short_contract.strike,
            short_contract.implied_volatility,
        )
        if expected_move_key not in expected_move_by_contract:
            expected_move_by_contract[expected_move_key] = expected_move(
                underlying_price,
                short_contract.implied_volatility,
                dte,
            )
        if option_type == "put":
            first = bisect_left(strikes, short_contract.strike - 5)
            last = bisect_left(strikes, short_contract.strike)
        else:
            first = bisect_right(strikes, short_contract.strike)
            last = bisect_right(strikes, short_contract.strike + 5)
        protective_longs = sorted(
            by_strike[first:last],
            key=lambda item: item[0],
        )
        if operation_stats is not None:
            operation_stats["reference_contract_pairs"] += len(contracts)
            operation_stats["contract_pairs_considered"] += len(protective_longs)

        for _, long_contract in protective_longs:
            width = abs(short_contract.strike - long_contract.strike)
            credit = round(short_contract.bid - long_contract.ask, 2)
            if credit <= 0 or width - credit <= 0:
                continue

            stats["raw_wings_built"] += 1
            wing = _condor_wing_trade(
                short_contract,
                long_contract,
                underlying_price,
                earnings_date,
                volatility_rank,
                dte=dte_by_expiration[expiration],
                expiration_date=date_by_expiration[expiration],
                earnings_before_exp=earnings_by_expiration[expiration],
                expected_move_value=expected_move_by_contract[expected_move_key],
            )
            passed, reasons = passes_condor_wing_filters(wing)
            if "open interest is below 200" in reasons or "volume is below 25" in reasons:
                stats["rejected_for_liquidity"] += 1
            if "delta is outside the credit range of 0.10 to 0.30" in reasons:
                stats["rejected_for_delta"] += 1
            if "quote is estimated from last price" in reasons:
                stats["rejected_for_estimated_quotes"] += 1
            if passed:
                eligible_wings.append(wing)

    return eligible_wings, stats


def _group_ranked_condor_wings(
    wings: list[Trade],
    preferences: ScanPreferences,
) -> dict[tuple[str, str], list[Trade]]:
    grouped = {}
    for wing in wings:
        grouped.setdefault((wing.ticker, wing.expiration), []).append(wing)
    limit = max(0, int(preferences.condor_max_wings_per_side))
    rank_cache = {}

    def rank_key(trade: Trade) -> tuple:
        if trade not in rank_cache:
            rank_cache[trade] = (
                score_trade(trade, preferences).total_score,
                trade.credit,
                -bid_ask_spread(trade),
                -abs(trade.delta),
            )
        return rank_cache[trade]

    return {
        key: sorted(
            group,
            key=rank_key,
            reverse=True,
        )[:limit]
        for key, group in grouped.items()
    }


def _condor_from_wings(
    put_spread: Trade,
    call_spread: Trade,
    underlying_price: float,
    volatility_rank: float,
) -> Trade:
    total_credit = round(call_spread.credit + put_spread.credit, 2)
    put_width = put_spread.short_strike - put_spread.long_strike
    call_width = call_spread.long_strike - call_spread.short_strike
    max_width = max(put_width, call_width)
    max_risk = round(max_width - total_credit, 2)
    combined_bid = round(put_spread.bid + call_spread.bid, 2)
    combined_ask = round(put_spread.ask + call_spread.ask, 2)
    combined_expected_move = max(
        put_spread.expected_move,
        call_spread.expected_move,
    )
    put_cushion = round(
        underlying_price - put_spread.short_strike - combined_expected_move,
        2,
    )
    call_cushion = round(
        call_spread.short_strike - underlying_price - combined_expected_move,
        2,
    )
    minimum_cushion = min(put_cushion, call_cushion)
    bid_ask_credit_ratio = (
        round((combined_ask - combined_bid) / total_credit, 4)
        if total_credit > 0
        else None
    )
    return Trade(
        ticker=put_spread.ticker,
        strategy="iron condor",
        expiration=put_spread.expiration,
        option_type="mixed",
        delta=max(abs(put_spread.delta), abs(call_spread.delta)),
        volatility_rank=volatility_rank,
        earnings_before_exp=(
            put_spread.earnings_before_exp
            or call_spread.earnings_before_exp
        ),
        open_interest=min(put_spread.open_interest, call_spread.open_interest),
        volume=min(put_spread.volume, call_spread.volume),
        dte=put_spread.dte,
        bid=combined_bid,
        ask=combined_ask,
        credit=total_credit,
        max_risk=max_risk,
        underlying_price=underlying_price,
        expected_move=combined_expected_move,
        short_strike=put_spread.short_strike,
        long_strike=call_spread.short_strike,
        short_bid=put_spread.short_bid,
        short_ask=put_spread.short_ask,
        long_bid=call_spread.short_bid,
        long_ask=call_spread.short_ask,
        short_delta=put_spread.short_delta,
        long_delta=call_spread.short_delta,
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
        put_expected_move_cushion=put_cushion,
        call_expected_move_cushion=call_cushion,
        minimum_expected_move_cushion=minimum_cushion,
        condor_bid_ask_to_credit_ratio=bid_ask_credit_ratio,
    )


def _combine_condor_wings(
    put_wings: list[Trade],
    call_wings: list[Trade],
    underlying_price: float,
    volatility_rank: float,
    preferences: ScanPreferences,
) -> tuple[list[Trade], Counter, int, int, int]:
    put_groups = _group_ranked_condor_wings(put_wings, preferences)
    call_groups = _group_ranked_condor_wings(call_wings, preferences)
    common_keys = set(put_groups) & set(call_groups)
    stats = Counter()
    condors = []

    for put_key, ranked_puts in put_groups.items():
        for call_key, ranked_calls in call_groups.items():
            if put_key != call_key:
                stats["mismatched_expiration"] += len(ranked_puts) * len(ranked_calls)

    for key in sorted(common_keys):
        for call_spread in call_groups[key]:
            for put_spread in put_groups[key]:
                stats["pairs_attempted"] += 1
                if call_spread.short_strike <= put_spread.short_strike:
                    stats["strike_overlap"] += 1
                    continue
                total_credit = round(call_spread.credit + put_spread.credit, 2)
                if total_credit <= 0:
                    stats["nonpositive_credit"] += 1
                    continue
                condor = _condor_from_wings(
                    put_spread,
                    call_spread,
                    underlying_price,
                    volatility_rank,
                )
                if condor.max_risk <= 0:
                    stats["nonpositive_risk"] += 1
                    continue
                condors.append(condor)

    return (
        condors,
        stats,
        len(put_groups),
        len(call_groups),
        len(common_keys),
    )


def combine_condor_wings(
    put_wings: list[Trade],
    call_wings: list[Trade],
    underlying_price: float,
    volatility_rank: float,
    preferences: ScanPreferences,
) -> list[Trade]:
    condors, _, _, _, _ = _combine_condor_wings(
        put_wings,
        call_wings,
        underlying_price,
        volatility_rank,
        preferences,
    )
    return condors


def _condor_diagnostics_from_construction(
    option_chain,
    preferences: ScanPreferences,
    condors: list[Trade],
    put_wings: list[Trade],
    call_wings: list[Trade],
    put_stats: Counter,
    call_stats: Counter,
    pairing_stats: Counter,
    put_expirations: int,
    call_expirations: int,
    common_expirations: int,
) -> CondorDiagnostics:
    rejection_counts = Counter()
    passing_scores = []
    built_scores = []
    for condor in condors:
        scored = score_trade(condor, preferences)
        built_scores.append(scored.total_score)
        passed, reasons = passes_condor_filters(condor, preferences)
        if passed:
            passing_scores.append(scored.total_score)
        else:
            rejection_counts.update(reasons)

    if put_stats["raw_wings_built"] == 0:
        primary_blocker = "no structurally valid put wings existed"
    elif call_stats["raw_wings_built"] == 0:
        primary_blocker = "no structurally valid call wings existed"
    elif not put_wings:
        primary_blocker = "all put wings failed construction-safety filters"
    elif not call_wings:
        primary_blocker = "all call wings failed construction-safety filters"
    elif common_expirations == 0:
        primary_blocker = "eligible put and call wings had no matching expiration"
    elif pairing_stats["pairs_attempted"] == pairing_stats["strike_overlap"]:
        primary_blocker = "put and call short strikes overlapped"
    elif not condors:
        primary_blocker = "combined credit or maximum risk was nonpositive"
    elif not passing_scores:
        primary_blocker = (
            rejection_counts.most_common(1)[0][0]
            if rejection_counts
            else "all built condors failed final filtering"
        )
    else:
        primary_blocker = (
            "condors passed final filters; final display depends on ranking"
        )

    ticker = option_chain[0].ticker if option_chain else "Unknown"
    return CondorDiagnostics(
        ticker=ticker,
        raw_put_wings_built=put_stats["raw_wings_built"],
        raw_call_wings_built=call_stats["raw_wings_built"],
        put_wings_rejected_for_liquidity=put_stats["rejected_for_liquidity"],
        call_wings_rejected_for_liquidity=call_stats["rejected_for_liquidity"],
        wings_rejected_for_delta=(
            put_stats["rejected_for_delta"]
            + call_stats["rejected_for_delta"]
        ),
        wings_rejected_for_estimated_quotes=(
            put_stats["rejected_for_estimated_quotes"]
            + call_stats["rejected_for_estimated_quotes"]
        ),
        eligible_put_expirations=put_expirations,
        eligible_call_expirations=call_expirations,
        expirations_with_both_sides=common_expirations,
        pairs_attempted=pairing_stats["pairs_attempted"],
        pairs_rejected_for_mismatched_expiration=pairing_stats[
            "mismatched_expiration"
        ],
        pairs_rejected_for_strike_overlap=pairing_stats["strike_overlap"],
        condors_built=len(condors),
        condors_rejected_for_nonpositive_combined_credit=pairing_stats[
            "nonpositive_credit"
        ],
        condors_rejected_for_max_risk=rejection_counts["condor max risk too high"],
        condors_rejected_for_combined_credit_to_width=rejection_counts[
            "condor credit is below 15% of maximum wing width"
        ],
        condors_rejected_for_put_expected_move=rejection_counts[
            "condor put short strike is inside the expected move"
        ],
        condors_rejected_for_call_expected_move=rejection_counts[
            "condor call short strike is inside the expected move"
        ],
        condors_rejected_for_bid_ask_width=rejection_counts[
            "condor four-leg bid/ask spread is wider than configured maximum"
        ],
        condors_passing_final_filters=len(passing_scores),
        highest_passing_condor_score=(
            max(passing_scores) if passing_scores else None
        ),
        highest_built_condor_score=(max(built_scores) if built_scores else None),
        primary_blocker=primary_blocker,
    )


def build_iron_condors_with_diagnostics(
    option_chain,
    underlying_price: float,
    earnings_date,
    volatility_rank: float,
    preferences: ScanPreferences,
) -> CondorConstructionResult:
    operation_stats = Counter()
    wing_started = time.perf_counter()
    put_wings, put_stats = _build_condor_wings_for_type(
        option_chain,
        "put",
        underlying_price,
        earnings_date,
        volatility_rank,
        operation_stats,
    )
    call_wings, call_stats = _build_condor_wings_for_type(
        option_chain,
        "call",
        underlying_price,
        earnings_date,
        volatility_rank,
        operation_stats,
    )
    wing_seconds = time.perf_counter() - wing_started

    pairing_started = time.perf_counter()
    (
        condors,
        pairing_stats,
        put_expirations,
        call_expirations,
        common_expirations,
    ) = _combine_condor_wings(
        put_wings,
        call_wings,
        underlying_price,
        volatility_rank,
        preferences,
    )
    pairing_seconds = time.perf_counter() - pairing_started

    diagnostics_started = time.perf_counter()
    diagnostics = _condor_diagnostics_from_construction(
        option_chain,
        preferences,
        condors,
        put_wings,
        call_wings,
        put_stats,
        call_stats,
        pairing_stats,
        put_expirations,
        call_expirations,
        common_expirations,
    )
    diagnostics_seconds = time.perf_counter() - diagnostics_started
    return CondorConstructionResult(
        condors=condors,
        diagnostics=diagnostics,
        timings={
            "wing_construction": wing_seconds,
            "pairing": pairing_seconds,
            "diagnostics": diagnostics_seconds,
        },
        operation_counts=dict(operation_stats),
    )


def build_iron_condor(
    option_chain,
    underlying_price: float,
    earnings_date,
    volatility_rank: float,
    preferences,
):
    return build_iron_condors_with_diagnostics(
        option_chain,
        underlying_price,
        earnings_date,
        volatility_rank,
        preferences,
    ).condors


def condor_diagnostics(
    option_chain,
    underlying_price: float,
    earnings_date,
    volatility_rank: float,
    preferences,
) -> CondorDiagnostics:
    return build_iron_condors_with_diagnostics(
        option_chain,
        underlying_price,
        earnings_date,
        volatility_rank,
        preferences,
    ).diagnostics


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


def ranking_test_trade(
    ticker: str,
    strategy: str,
    short_strike: float,
    long_strike: float,
    expiration: str = "2026-07-31",
    option_type: str = "call",
) -> Trade:
    return Trade(
        ticker=ticker,
        strategy=strategy,
        expiration=expiration,
        option_type=option_type,
        delta=0.2,
        volatility_rank=55,
        earnings_before_exp=False,
        open_interest=1000,
        volume=500,
        dte=30,
        bid=1.0,
        ask=1.1,
        credit=1.0,
        max_risk=4.0,
        underlying_price=100.0,
        expected_move=5.0,
        short_strike=short_strike,
        long_strike=long_strike,
        short_bid=1.0,
        short_ask=1.1,
        long_bid=0.4,
        long_ask=0.5,
        short_delta=0.2,
        long_delta=0.1,
    )


def ranking_test_scored(
    ticker: str,
    strategy: str,
    score: int,
    short_strike: float,
    long_strike: float,
    expiration: str = "2026-07-31",
    option_type: str = "call",
    **trade_updates,
) -> ScoredTrade:
    trade = ranking_test_trade(
        ticker,
        strategy,
        short_strike,
        long_strike,
        expiration,
        option_type,
    )
    if trade_updates:
        trade = replace(trade, **trade_updates)
    return ScoredTrade(
        trade=trade,
        risk_level="Moderate",
        category_scores={},
        quant_score=score,
        event_adjustment=0,
        raw_price_move_adjustment=0,
        effective_price_move_adjustment=0,
        price_move_adjustment=0,
        price_move_style="normal",
        base_score_without_price_move=score,
        total_score=score,
        reasons=[],
        explanation="ranking test",
    )


def test_diversified_ranking_single_candidate_neutral_score() -> None:
    ranked = diversify_scored_trades(
        [ranking_test_scored("AAPL", "bull call debit spread", 82, 105, 100)]
    )
    assert len(ranked) == 1
    assert ranked[0].normalized_ticker_score == 50


def test_diversified_ranking_tied_scores_share_top_percentile() -> None:
    ranked = diversify_scored_trades(
        [
            ranking_test_scored("AAPL", "bull call debit spread", 80, 105, 100),
            ranking_test_scored("AAPL", "bear put debit spread", 80, 95, 100, option_type="put"),
        ]
    )
    assert {scored.normalized_ticker_score for scored in ranked} == {100}


def test_diversified_ranking_removes_duplicate_spreads() -> None:
    ranked = diversify_scored_trades(
        [
            ranking_test_scored("SPY", "bull call debit spread", 70, 605, 600),
            ranking_test_scored("SPY", "bull call debit spread", 90, 606, 601),
            ranking_test_scored("SPY", "bull call debit spread", 75, 607, 602),
        ]
    )
    assert len(ranked) == 1
    assert ranked[0].total_score == 90
    assert ranked[0].trade.short_strike == 606


def test_diversified_ranking_keeps_several_strategies_for_one_ticker() -> None:
    ranked = diversify_scored_trades(
        [
            ranking_test_scored("NVDA", "bull call debit spread", 88, 210, 205),
            ranking_test_scored("NVDA", "bear put debit spread", 81, 190, 195, option_type="put"),
            ranking_test_scored("NVDA", "put credit spread", 76, 180, 175, option_type="put"),
        ]
    )
    assert [scored.trade.strategy for scored in ranked] == [
        "bull call debit spread",
        "bear put debit spread",
        "put credit spread",
    ]


def test_diversified_ranking_keeps_multiple_tickers() -> None:
    ranked = diversify_scored_trades(
        [
            ranking_test_scored("SPY", "bull call debit spread", 86, 605, 600),
            ranking_test_scored("NVDA", "bull call debit spread", 84, 210, 205),
            ranking_test_scored("TSLA", "bear put debit spread", 82, 190, 195, option_type="put"),
        ]
    )
    assert [scored.trade.ticker for scored in ranked] == ["SPY", "NVDA", "TSLA"]


def test_diversified_ranking_iron_condor_similarity() -> None:
    first = ranking_test_scored(
        "SPY",
        "iron condor",
        72,
        600,
        610,
        option_type="mixed",
        put_long_strike=580,
        put_short_strike=585,
        call_short_strike=615,
        call_long_strike=620,
    )
    better_duplicate = ranking_test_scored(
        "SPY",
        "iron condor",
        88,
        601,
        611,
        option_type="mixed",
        put_long_strike=581,
        put_short_strike=586,
        call_short_strike=616,
        call_long_strike=621,
    )
    ranked = diversify_scored_trades([first, better_duplicate])
    assert len(ranked) == 1
    assert ranked[0].total_score == 88


def test_diversified_ranking_highest_score_survives_deduplication() -> None:
    ranked = diversify_scored_trades(
        [
            ranking_test_scored("QQQ", "call credit spread", 78, 505, 510),
            ranking_test_scored("QQQ", "call credit spread", 91, 506, 511),
            ranking_test_scored("QQQ", "call credit spread", 83, 507, 512),
        ]
    )
    assert len(ranked) == 1
    assert ranked[0].total_score == 91


def test_diversified_ranking() -> None:
    test_diversified_ranking_single_candidate_neutral_score()
    test_diversified_ranking_tied_scores_share_top_percentile()
    test_diversified_ranking_removes_duplicate_spreads()
    test_diversified_ranking_keeps_several_strategies_for_one_ticker()
    test_diversified_ranking_keeps_multiple_tickers()
    test_diversified_ranking_iron_condor_similarity()
    test_diversified_ranking_highest_score_survives_deduplication()
    print("Diversified ranking tests passed.")


def test_execution_selection() -> None:
    displayed = diversify_scored_trades(
        [
            ranking_test_scored("SPY", "bull call debit spread", 91, 605, 600),
            ranking_test_scored("SPY", "bear put debit spread", 89, 590, 595, option_type="put"),
            ranking_test_scored("NVDA", "bull call debit spread", 88, 210, 205),
            ranking_test_scored("QQQ", "put credit spread", 84, 480, 475, option_type="put"),
        ]
    )
    assert len([item for item in displayed if item.trade.ticker == "SPY"]) == 2

    selected = select_execution_candidates(displayed, limit=3)
    assert [item.trade.ticker for item in selected] == ["SPY", "NVDA", "QQQ"]
    assert len({item.trade.ticker for item in selected}) == len(selected)
    assert [item.execution_rank for item in selected] == [1, 2, 3]

    two_tickers = select_execution_candidates(displayed[:3], limit=3)
    assert len(two_tickers) == 2
    assert two_tickers[0].total_score == 91

    tied = [
        replace(
            ranking_test_scored("AAPL", "bull call debit spread", 80, 105, 100),
            normalized_ticker_score=80,
            quant_score=75,
        ),
        replace(
            ranking_test_scored("MSFT", "bull call debit spread", 80, 505, 500),
            normalized_ticker_score=90,
            quant_score=70,
        ),
        replace(
            ranking_test_scored("NVDA", "bull call debit spread", 80, 205, 200),
            normalized_ticker_score=90,
            quant_score=78,
        ),
    ]
    tied_selected = select_execution_candidates(tied, limit=3)
    assert [item.trade.ticker for item in tied_selected] == ["NVDA", "MSFT", "AAPL"]

    assert strategy_direction("put credit spread") == "bullish"
    assert strategy_direction("bull call debit spread") == "bullish"
    assert strategy_direction("call credit spread") == "bearish"
    assert strategy_direction("bear put debit spread") == "bearish"
    assert strategy_direction("iron condor") == "neutral"
    assert strategy_direction("calendar spread") == "unknown"
    print("Execution selection tests passed.")


def test_price_move_modes() -> None:
    assert apply_price_move_mode(8, "Full") == (8, 8)
    assert apply_price_move_mode(-6, "Full") == (-6, -6)
    assert apply_price_move_mode(8, "Conservative") == (8, 3)
    assert apply_price_move_mode(-6, "Conservative") == (-6, -6)
    assert apply_price_move_mode(8, "Shadow") == (8, 0)
    assert apply_price_move_mode(8, "Off") == (8, 0)

    trade = ranking_test_trade("AAPL", "bull call debit spread", 105, 100)
    price_move = {"1D Move %": 4, "5D Move %": 5, "Move vs 20D Vol": 2}
    preferences = ScanPreferences(500, "bullish", "moderate", price_move_mode="Shadow")
    scored = score_trade(trade, preferences, price_move=price_move)
    assert scored.raw_price_move_adjustment > 0
    assert scored.effective_price_move_adjustment == 0
    assert scored.price_move_adjustment == 0
    assert scored.total_score == scored.base_score_without_price_move
    print("Price move mode tests passed.")


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
