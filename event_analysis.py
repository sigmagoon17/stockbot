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


def neutral_event_analysis(ticker, headlines, summary, available=True):
    return EventAnalysis(
        adjustment=0,
        confidence="low",
        label="neutral",
        summary=summary.format(ticker=ticker),
        headlines_used=headlines,
        available=available,
    )


def test_openai_connection():
    
    response = client.responses.create(
        model = "gpt-4.1-mini",
        input = "reply with exactly: connection works"
    )
    print(response.output_text)

def get_recent_headlines(ticker):
    headlines = []
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("ALPHA_VANTAGE_API_KEY is missing.")
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": 5,
        "apikey": api_key,
    }
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data= response.json()
    if "Note" in data or "Information" in data:
        raise RuntimeError(data.get("Note") or data.get("Information"))
    for story in data.get("feed", []):

        matching_tickers = [
            sentiment
            for sentiment in story.get("ticker_sentiment", [])
            if sentiment.get("ticker") == ticker
        ]      
        if not matching_tickers:
            continue
        relevance = max(
            float(sentiment.get("relevance_score", 0))
            for sentiment in matching_tickers
        )
        if relevance < 0.25:
            continue
        time_published = story.get("time_published", "unknown time")
        source = story.get("source", "unknown source")
        title = story.get("title", "untitled")

        headlines.append(f"{time_published} | {source}: {title}")
    return headlines[:5]
    

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

if __name__ == "__main__":
    analysis = get_event_analysis("AAPL", "bullish")
    print(analysis)
