from collections import Counter
from dataclasses import dataclass
from time import perf_counter
from typing import Callable


PUT_CREDIT_SPREADS = "Put Credit Spreads"
CALL_CREDIT_SPREADS = "Call Credit Spreads"
BULL_CALL_DEBIT_SPREADS = "Bull Call Debit Spreads"
BEAR_PUT_DEBIT_SPREADS = "Bear Put Debit Spreads"
IRON_CONDORS = "Iron Condors"

STRATEGY_OPTIONS = (
    PUT_CREDIT_SPREADS,
    CALL_CREDIT_SPREADS,
    BULL_CALL_DEBIT_SPREADS,
    BEAR_PUT_DEBIT_SPREADS,
    IRON_CONDORS,
)

# Preserve the builder order used before strategies became selectable.
STRATEGY_BUILD_ORDER = (
    IRON_CONDORS,
    CALL_CREDIT_SPREADS,
    PUT_CREDIT_SPREADS,
    BULL_CALL_DEBIT_SPREADS,
    BEAR_PUT_DEBIT_SPREADS,
)


def run_selected_strategy_builders(selected_strategies, builders):
    selected = set(STRATEGY_OPTIONS if selected_strategies is None else selected_strategies)
    unsupported = selected.difference(STRATEGY_OPTIONS)
    if unsupported:
        raise ValueError(f"Unsupported strategies: {sorted(unsupported)}")
    return {
        strategy: builders[strategy]()
        for strategy in STRATEGY_BUILD_ORDER
        if strategy in selected
    }


def _analysis_and_cache_status(result):
    if isinstance(result, tuple) and len(result) == 2:
        return result
    return result, "unknown"


@dataclass(frozen=True)
class PostSelectionAnalysisResult:
    execution_candidates: list
    event_analyses: dict
    candidate_analyses: dict
    event_seconds_by_ticker: dict[str, float]
    review_seconds_by_ticker: dict[str, float]
    diagnostics: dict[str, int]


def analyze_top_candidates(
    execution_candidates,
    price_moves,
    load_ticker_event: Callable,
    review_candidate: Callable,
    candidate_key: Callable,
    unavailable_event: Callable,
    unavailable_review: Callable,
) -> PostSelectionAnalysisResult:
    stable_candidates = list(execution_candidates)
    unique_tickers = list(
        dict.fromkeys(scored.trade.ticker for scored in stable_candidates)
    )
    event_analyses = {}
    candidate_analyses = {}
    event_seconds_by_ticker = Counter()
    review_seconds_by_ticker = Counter()
    event_cache = Counter()
    review_cache = Counter()

    for ticker in unique_tickers:
        started = perf_counter()
        try:
            analysis, cache_status = _analysis_and_cache_status(
                load_ticker_event(ticker)
            )
        except Exception as error:
            analysis = unavailable_event(ticker, error)
            cache_status = "miss"
        event_seconds_by_ticker[ticker] += perf_counter() - started
        event_analyses[ticker] = analysis
        event_cache[cache_status] += 1

    for scored in stable_candidates:
        ticker = scored.trade.ticker
        started = perf_counter()
        try:
            analysis, cache_status = _analysis_and_cache_status(
                review_candidate(
                    scored,
                    event_analyses.get(ticker),
                    price_moves.get(ticker),
                )
            )
        except Exception as error:
            analysis = unavailable_review(scored, error)
            cache_status = "miss"
        review_seconds_by_ticker[ticker] += perf_counter() - started
        candidate_analyses[candidate_key(scored)] = analysis
        review_cache[cache_status] += 1

    return PostSelectionAnalysisResult(
        execution_candidates=stable_candidates,
        event_analyses=event_analyses,
        candidate_analyses=candidate_analyses,
        event_seconds_by_ticker=dict(event_seconds_by_ticker),
        review_seconds_by_ticker=dict(review_seconds_by_ticker),
        diagnostics={
            "execution_candidates": len(stable_candidates),
            "unique_candidate_tickers": len(unique_tickers),
            "ticker_event_calls": len(unique_tickers),
            "candidate_review_calls": len(stable_candidates),
            "ticker_event_cache_hits": event_cache["hit"],
            "ticker_event_cache_misses": event_cache["miss"],
            "candidate_review_cache_hits": review_cache["hit"],
            "candidate_review_cache_misses": review_cache["miss"],
        },
    )
