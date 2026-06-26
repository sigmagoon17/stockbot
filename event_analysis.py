from dataclasses import dataclass
from dotenv import load_dotenv
from openai import APIError, OpenAI
import json
import os
import requests


load_dotenv()
client = OpenAI()

@dataclass(frozen=True)
class EventAnalysis:
    adjustment: int
    confidence: str
    label: str
    summary: str
    headlines_used: list[str]
    available: bool = True


@dataclass(frozen=True)
class CandidateAnalysis:
    verdict: str
    confidence: str
    summary: str
    strengths: list[str]
    risks: list[str]
    action: str
    available: bool = True


def neutral_event_analysis(ticker, headlines, summary, available=True):
    return EventAnalysis(
        adjustment=0,
        confidence="low",
        label="neutral",
        summary=summary.format(ticker=ticker),
        headlines_used=headlines,
        available=available,
    )


def unavailable_candidate_analysis(summary):
    return CandidateAnalysis(
        verdict="watch",
        confidence="low",
        summary=summary,
        strengths=[],
        risks=[],
        action="Review the scanner numbers manually before acting.",
        available=False,
    )


def test_openai_connection():
    
    response = client.responses.create(
        model = "gpt-4.1-mini",
        input = "reply with exactly: connection works"
    )
    print(response.output_text)

def get_recent_headlines(ticker):
    headlines = []
    api_key = os.getenv("MARKETAUX_API_KEY")
    if not api_key:
        raise ValueError("MARKETAUX_API_KEY is missing.")

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "api_token": api_key,
        "symbols": ticker,
        "limit": 3,
        "filter_entities": "true",
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])

    for story in data.get("data", []):
        matching_entities = [
            entity
            for entity in story.get("entities", [])
            if entity.get("symbol") == ticker
        ]
        if not matching_entities:
            continue

        time_published = story.get("published_at", "unknown time")
        source = story.get("source", "unknown source")
        title = story.get("title", "untitled")
        description = story.get("description") or story.get("snippet") or ""
        if description:
            description = f" | {description[:350]}"

        headlines.append(f"{time_published} | {source}: {title}{description}")
    return headlines
    

def analyze_events(ticker, scanner_outlook, headlines):
    if not headlines:
        return neutral_event_analysis(
            ticker,
            [],
            "No relevant recent headlines were available for {ticker}.",
        )

    headline_text = "\n".join(
        f"{number}, {headline}"
        for number, headline in enumerate(headlines, start=1)
    )
    try:

        prompt = f"""
        you are an event-risk analyst for an options scanner.

        ticker: {ticker}
        scanner outlook: {scanner_outlook}

        recent headlines:
        {headline_text}

        use only the headlines provided.

        Do not treat the absence of negative news as supportive evidence.

        If headlines are mixed, weakly related, duplicated, or do not clearly support or conflict with the scanner outlook, use:
        - adjustment: 0
        - label: neutral
        - confidence: low

        Only use a non-zero adjustment when a supplied headline gives concrete, ticker-relevant evidence.

        headline_numbers must only contain numbers from the supplied headline list.

        If the adjustment is 0 because no headline is material evidence, headline_numbers must be an empty list.
        return only valid json with these fields, do not wrap the json in markdown code fences:
        -adjustment: an integer from -10 to 10
        -confidence: low, medium, or high
        -label: supportive, neutral, or caution
        -summary: one short explanation
        -headline_numbers: a list of the 1-based headline numbers you relied on.
        """

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        print(response.output_text)
        raw_output = response.output_text.strip()
        raw_output = (
            raw_output
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(raw_output)

        adjustment = int(data["adjustment"])
        confidence = data["confidence"].lower()
        label = data["label"].lower()
        summary = data["summary"]
        headline_numbers = data["headline_numbers"]

        if not -10 <= adjustment <= 10:
            raise ValueError("AI adjustment must be between -10 and 10.")

        if confidence not in ["low", "medium", "high"]:
            raise ValueError("AI confidence is invalid.")

        if label not in ["supportive", "neutral", "caution"]:
            raise ValueError("AI label is invalid.")

        if not isinstance(headline_numbers, list):
            raise ValueError("AI headlines_used must be a list.")
        if not all(
            isinstance(number, int) and 1 <= number <= len(headlines)
            for number in headline_numbers
        ):
            raise ValueError("AI used an invalid headline number.")
        if adjustment != 0 and not headline_numbers:
            raise ValueError("A non-zero AI adjustment requires headlines")
        
        headlines_used = [headlines[number - 1] for number in headline_numbers]
        analysis = EventAnalysis(
            adjustment,
            confidence,
            label,
            summary,
            headlines_used,
        )
    
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        analysis = neutral_event_analysis(
            ticker,
            headlines,
            "AI could not analyze the available headlines for {ticker}.",
            available=False,
        )
    return analysis


def get_event_analysis(ticker, scanner_outlook):
    try:
        headlines = get_recent_headlines(ticker)
        return analyze_events(ticker, scanner_outlook, headlines)
    except (APIError, requests.RequestException, RuntimeError, ValueError):
        return neutral_event_analysis(
            ticker,
            [],
            "Event analysis was unavailable for {ticker}.",
            available=False,
        )


def analyze_candidate_setup(scored_trade, event_analysis=None, price_move=None):
    trade = scored_trade.trade
    price_move = price_move or {}
    event_summary = (
        event_analysis.summary
        if event_analysis is not None
        else "No event analysis was available."
    )
    prompt = f"""
    You are reviewing an options scanner candidate for a beginner-friendly trading dashboard.

    Use only these scanner facts. Do not invent live prices, news, or probabilities.

    Ticker: {trade.ticker}
    Strategy: {trade.strategy}
    Entry type: {trade.entry_type}
    Expiration: {trade.expiration}
    DTE: {trade.dte}
    Underlying price: {trade.underlying_price:.2f}
    Short strike: {trade.short_strike}
    Long strike: {trade.long_strike}
    Credit: {trade.credit * 100:.2f}
    Max risk: {trade.max_risk * 100:.2f}
    Max profit: {trade.max_profit * 100:.2f}
    Delta: {trade.delta:.2f}
    Volatility rank: {trade.volatility_rank:.1f}
    Quant score: {scored_trade.quant_score}
    Event adjustment: {scored_trade.event_adjustment}
    Price move adjustment: {scored_trade.price_move_adjustment}
    Final setup score: {scored_trade.total_score}
    Risk level: {scored_trade.risk_level}
    Recent 1D move percent: {price_move.get("1D Move %", "unknown")}
    Recent 5D move percent: {price_move.get("5D Move %", "unknown")}
    Move setup: {scored_trade.price_move_style}
    Event summary: {event_summary}

    Return only valid JSON with these fields:
    - verdict: one of strong, good, watch, avoid
    - confidence: low, medium, or high
    - summary: one beginner-friendly sentence
    - strengths: list of 2 short strings
    - risks: list of 2 short strings
    - action: one short sentence explaining what a user should double-check next
    """
    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )
        raw_output = response.output_text.strip()
        raw_output = (
            raw_output
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        data = json.loads(raw_output)

        verdict = data["verdict"].lower()
        confidence = data["confidence"].lower()
        strengths = data["strengths"]
        risks = data["risks"]

        if verdict not in ["strong", "good", "watch", "avoid"]:
            raise ValueError("Candidate verdict is invalid.")
        if confidence not in ["low", "medium", "high"]:
            raise ValueError("Candidate confidence is invalid.")
        if not isinstance(strengths, list) or not isinstance(risks, list):
            raise ValueError("Candidate strengths and risks must be lists.")

        return CandidateAnalysis(
            verdict=verdict,
            confidence=confidence,
            summary=data["summary"],
            strengths=[str(item) for item in strengths[:2]],
            risks=[str(item) for item in risks[:2]],
            action=data["action"],
        )
    except (APIError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return unavailable_candidate_analysis(
            f"AI candidate review was unavailable for {trade.ticker}."
        )


if __name__ == "__main__":
    analysis = get_event_analysis("AAPL", "bullish")
    print(analysis)
