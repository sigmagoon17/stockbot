from dataclasses import dataclass
import math

import yfinance as yf


@dataclass(frozen=True)
class PrefilterResult:
    ticker: str
    passed: bool
    score: int
    price: float | None
    average_volume: int | None
    volatility_rank: float | None
    one_day_move_percent: float | None
    five_day_move_percent: float | None
    reason: str


def dedupe_tickers(tickers: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for ticker in tickers:
        normalized = ticker.strip().upper()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(normalized)
    return cleaned


def stock_prefilter_score(
    price: float,
    average_volume: int,
    volatility_rank: float,
    one_day_move_percent: float,
    five_day_move_percent: float,
) -> int:
    score = 0

    if price >= 20:
        score += 20
    elif price >= 10:
        score += 10

    if average_volume >= 5_000_000:
        score += 25
    elif average_volume >= 2_000_000:
        score += 20
    elif average_volume >= 1_000_000:
        score += 12

    if 45 <= volatility_rank <= 80:
        score += 30
    elif 30 <= volatility_rank < 45 or 80 < volatility_rank <= 90:
        score += 20
    elif volatility_rank >= 20:
        score += 10

    abs_one_day_move = abs(one_day_move_percent)
    if 0.5 <= abs_one_day_move <= 3:
        score += 15
    elif abs_one_day_move < 0.5:
        score += 8
    elif abs_one_day_move <= 5:
        score += 6

    if abs(five_day_move_percent) <= 8:
        score += 10
    elif abs(five_day_move_percent) <= 12:
        score += 5

    return min(score, 100)


def stock_prefilter_result(
    ticker: str,
    min_price: float,
    min_average_volume: int,
    min_volatility_rank: float,
) -> PrefilterResult:
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="6mo", auto_adjust=True)
        if history.empty or len(history) < 70:
            return PrefilterResult(
                ticker,
                False,
                0,
                None,
                None,
                None,
                None,
                None,
                "not enough price history",
            )

        closes = history["Close"].dropna()
        volumes = history["Volume"].dropna()
        daily_returns = closes.pct_change().dropna()
        rolling_volatility = (
            daily_returns.rolling(20).std() * math.sqrt(252)
        ).dropna()

        if len(closes) < 6 or len(volumes) < 20 or len(rolling_volatility) < 40:
            return PrefilterResult(
                ticker,
                False,
                0,
                None,
                None,
                None,
                None,
                None,
                "not enough usable price data",
            )

        price = float(closes.iloc[-1])
        average_volume = int(volumes.tail(20).mean())
        current_volatility = float(rolling_volatility.iloc[-1])
        lowest_volatility = float(rolling_volatility.min())
        highest_volatility = float(rolling_volatility.max())
        if highest_volatility == lowest_volatility:
            volatility_rank = 50.0
        else:
            volatility_rank = round(
                (current_volatility - lowest_volatility)
                / (highest_volatility - lowest_volatility)
                * 100,
                1,
            )
        one_day_move_percent = round(float(daily_returns.iloc[-1]) * 100, 2)
        five_day_move_percent = round(float(closes.iloc[-1] / closes.iloc[-6] - 1) * 100, 2)

        rejection_reasons = []
        if price < min_price:
            rejection_reasons.append(f"price below ${min_price:.0f}")
        if average_volume < min_average_volume:
            rejection_reasons.append(
                f"20D average volume below {min_average_volume:,}"
            )
        if volatility_rank < min_volatility_rank:
            rejection_reasons.append(
                f"volatility rank below {min_volatility_rank:.0f}"
            )

        score = stock_prefilter_score(
            price,
            average_volume,
            volatility_rank,
            one_day_move_percent,
            five_day_move_percent,
        )

        return PrefilterResult(
            ticker=ticker,
            passed=not rejection_reasons,
            score=score,
            price=round(price, 2),
            average_volume=average_volume,
            volatility_rank=volatility_rank,
            one_day_move_percent=one_day_move_percent,
            five_day_move_percent=five_day_move_percent,
            reason=", ".join(rejection_reasons) if rejection_reasons else "passed",
        )
    except Exception as error:
        return PrefilterResult(
            ticker,
            False,
            0,
            None,
            None,
            None,
            None,
            None,
            str(error),
        )


def prefilter_tickers(
    tickers: list[str],
    max_selected: int = 35,
    min_price: float = 20,
    min_average_volume: int = 1_000_000,
    min_volatility_rank: float = 20,
) -> tuple[list[str], list[PrefilterResult]]:
    results = [
        stock_prefilter_result(
            ticker,
            min_price=min_price,
            min_average_volume=min_average_volume,
            min_volatility_rank=min_volatility_rank,
        )
        for ticker in dedupe_tickers(tickers)
    ]
    selected = [
        result.ticker
        for result in sorted(
            (result for result in results if result.passed),
            key=lambda result: result.score,
            reverse=True,
        )[:max_selected]
    ]
    return selected, results
