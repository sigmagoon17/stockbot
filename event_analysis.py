from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from openai import APIError, OpenAI
import json
import os
import re
import requests


load_dotenv()
client = OpenAI()

COMPANY_NAMES = {
    "AAPL": ["Apple"],
    "MSFT": ["Microsoft"],
    "NVDA": ["Nvidia", "NVIDIA"],
    "COHR": ["Coherent"],
    "SNDK": ["SanDisk"],
    "SPY": ["S&P 500", "stock market", "Fed", "Treasury yields"],
    "QQQ": ["Nasdaq", "mega-cap tech", "technology stocks", "Fed"],
}

SECTOR_TERMS = {
    "AAPL": ["iPhone", "consumer tech", "App Store"],
    "MSFT": ["cloud", "Azure", "AI software"],
    "NVDA": ["semiconductors", "chips", "AI chips", "SMH"],
    "COHR": ["optical networking", "semiconductors", "AI data centers"],
    "SNDK": ["memory chips", "NAND", "semiconductors"],
    "SPY": ["S&P 500", "Federal Reserve", "inflation", "Treasury yields"],
    "QQQ": ["Nasdaq", "mega-cap tech", "AI stocks", "Treasury yields"],
}

COMPETITOR_TERMS = {
    "AAPL": ["Samsung smartphone", "Google Pixel", "Huawei phone", "Apple competition"],
    "MSFT": ["Amazon AWS", "Google Cloud", "Microsoft cloud competition"],
    "NVDA": [
        "AMD AI chips",
        "Broadcom AI chips",
        "Qualcomm AI chips",
        "Google TPU",
        "Amazon AI chips",
        "Nvidia competition",
        "AI chip competition",
    ],
    "COHR": ["Lumentum", "Applied Optoelectronics", "optical components competition"],
    "SNDK": ["Micron NAND", "Samsung memory chips", "memory chip competition"],
    "SPY": ["S&P 500 selloff", "Federal Reserve rates", "inflation stocks"],
    "QQQ": ["Nasdaq selloff", "mega-cap tech selloff", "AI stock competition"],
}

REPUTABLE_SOURCE_HINTS = [
    "reuters",
    "bloomberg",
    "associated press",
    "ap news",
    "cnbc",
    "wall street journal",
    "wsj",
    "marketwatch",
    "investing.com",
    "yahoo finance",
    "barron's",
    "the fly",
    "seeking alpha",
]

MATERIAL_EVENT_WORDS = [
    "earnings",
    "guidance",
    "outlook",
    "lawsuit",
    "investigation",
    "antitrust",
    "upgrade",
    "downgrade",
    "target",
    "tariff",
    "export",
    "ban",
    "regulator",
    "recall",
    "layoffs",
    "acquisition",
    "merger",
    "selloff",
    "rally",
    "surge",
    "plunge",
    "fed",
    "inflation",
    "yields",
    "rates",
]

RECENT_NEWS_DAYS = 7

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

def normalized_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def story_source_name(story):
    source = story.get("source", "unknown source")
    if isinstance(source, dict):
        return source.get("name") or source.get("domain") or "unknown source"
    return str(source)


def source_quality_score(source):
    source_text = normalized_text(source)
    if any(source_hint in source_text for source_hint in REPUTABLE_SOURCE_HINTS):
        return 3
    return 1


def important_word_score(text):
    text = normalized_text(text)
    return min(
        sum(1 for word in MATERIAL_EVENT_WORDS if word in text),
        4,
    )


def relevance_score(ticker, story, bucket, repeated_theme_count):
    title = story.get("title", "")
    description = story.get("description") or story.get("snippet") or ""
    source = story_source_name(story)
    text = normalized_text(f"{title} {description}")
    ticker_text = ticker.lower()
    company_terms = [term.lower() for term in COMPANY_NAMES.get(ticker, [])]
    sector_terms = [term.lower() for term in SECTOR_TERMS.get(ticker, [])]

    score = 0
    score += source_quality_score(source)
    score += important_word_score(text)
    score += min(repeated_theme_count, 3)

    if ticker_text in text:
        score += 5
    if any(term and term.lower() in text for term in company_terms):
        score += 4
    if any(term and term.lower() in text for term in sector_terms):
        score += 2
    if bucket == "ticker":
        score += 2
    if bucket in ["sector", "market", "competitor"]:
        score += 1
    if bucket == "competitor" and any(
        term.lower() in text
        for term in COMPETITOR_TERMS.get(ticker, [])
    ):
        score += 3

    return score


def story_signature(story):
    title = normalized_text(story.get("title", ""))
    return re.sub(r"[^a-z0-9 ]", "", title)[:100]


def parsed_published_at(story):
    published_at = story.get("published_at")
    if not published_at:
        return None
    try:
        return datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_recent_story(story):
    published_at = parsed_published_at(story)
    if published_at is None:
        return False
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_NEWS_DAYS)
    return published_at >= cutoff


def theme_key(story):
    title = normalized_text(story.get("title", ""))
    words = [
        word
        for word in re.findall(r"[a-z0-9]+", title)
        if len(word) > 4
    ]
    return " ".join(words[:4])


def format_story(story, bucket, score):
    time_published = story.get("published_at", "unknown time")
    source = story_source_name(story)
    title = story.get("title", "untitled")
    description = story.get("description") or story.get("snippet") or ""
    if description:
        description = f" | {description[:350]}"

    return f"[{bucket}; relevance {score}] {time_published} | {source}: {title}{description}"


def marketaux_request(params):
    api_key = os.getenv("MARKETAUX_API_KEY")
    if not api_key:
        raise ValueError("MARKETAUX_API_KEY is missing.")

    url = "https://api.marketaux.com/v1/news/all"
    request_params = {
        "api_token": api_key,
        "language": "en",
        "published_after": (
            datetime.now(timezone.utc) - timedelta(days=RECENT_NEWS_DAYS)
        ).date().isoformat(),
        **params,
    }
    response = requests.get(url, params=request_params, timeout=10)
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(data["error"])
    return data.get("data", [])


def news_buckets_for_ticker(ticker, deep=False):
    buckets = [
        (
            "ticker",
            {
                "symbols": ticker,
                "limit": 4 if not deep else 5,
                "filter_entities": "true",
            },
        )
    ]

    if not deep:
        return buckets

    company_terms = COMPANY_NAMES.get(ticker, [])
    if company_terms:
        buckets.append(
            (
                "company",
                {
                    "search": " OR ".join(company_terms[:2]),
                    "limit": 3 if not deep else 5,
                },
            )
        )

    sector_terms = SECTOR_TERMS.get(ticker, [])
    if sector_terms:
        bucket_name = "market" if ticker in ["SPY", "QQQ"] else "sector"
        buckets.append(
            (
                bucket_name,
                {
                    "search": " OR ".join(sector_terms[:3]),
                    "limit": 5,
                },
            )
        )

    competitor_terms = COMPETITOR_TERMS.get(ticker, [])
    if competitor_terms:
        buckets.append(
            (
                "competitor",
                {
                    "search": " OR ".join(competitor_terms[:4]),
                    "limit": 5,
                },
            )
        )

    return buckets


def get_recent_headlines(ticker, deep=False):
    stories = []
    seen_story_keys = set()

    for bucket, params in news_buckets_for_ticker(ticker, deep=deep):
        try:
            bucket_stories = marketaux_request(params)
        except (requests.RequestException, RuntimeError, ValueError):
            continue

        for story in bucket_stories:
            if not is_recent_story(story):
                continue
            signature = story_signature(story)
            if not signature or signature in seen_story_keys:
                continue
            seen_story_keys.add(signature)
            stories.append((bucket, story))

    if not stories:
        return []

    theme_counts = {}
    for _, story in stories:
        key = theme_key(story)
        if key:
            theme_counts[key] = theme_counts.get(key, 0) + 1

    ranked_stories = []
    for bucket, story in stories:
        score = relevance_score(
            ticker,
            story,
            bucket,
            theme_counts.get(theme_key(story), 1),
        )
        ranked_stories.append((score, bucket, story))

    ranked_stories.sort(key=lambda item: item[0], reverse=True)
    limit = 10 if deep else 8
    return [
        format_story(story, bucket, score)
        for score, bucket, story in ranked_stories[:limit]
        if score >= 6
    ]
    

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
        The headlines are pre-ranked by source quality, ticker relevance, repeated themes, and material event keywords.

        Do not treat the absence of negative news as supportive evidence.
        Do not use broad sector or market headlines unless they plausibly affect this ticker or ETF within the option trade timeframe.
        Prefer direct ticker/company headlines over sector headlines.
        Competitor headlines can be cautionary if they show pressure on the ticker's pricing power, growth story, or market share.
        Treat repeated themes from reputable sources as stronger evidence than one isolated weak headline.

        If headlines are mixed, weakly related, duplicated, or do not clearly support or conflict with the scanner outlook, use:
        - adjustment: 0
        - label: neutral
        - confidence: low

        Only use a non-zero adjustment when a supplied headline gives concrete, ticker-relevant, trade-timeframe evidence.

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


def get_deep_event_analysis(ticker, scanner_outlook):
    try:
        headlines = get_recent_headlines(ticker, deep=True)
        if not headlines:
            return neutral_event_analysis(
                ticker,
                [],
                "Deep event analysis found no recent material headlines for {ticker}.",
            )
        return analyze_events(ticker, scanner_outlook, headlines)
    except (APIError, requests.RequestException, RuntimeError, ValueError):
        return neutral_event_analysis(
            ticker,
            [],
            "Deep event analysis was unavailable for {ticker}.",
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
