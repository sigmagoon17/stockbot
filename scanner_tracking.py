import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from uuid import uuid4


SCANNER_VERSION = "2.1-selection-tracking"
SELECTION_RAW = "raw"
SELECTION_DIVERSIFIED = "ticker_strategy_diversified"
SELECTION_EXECUTION = "ticker_capped_execution"


def _normalized_number(value) -> str | None:
    if value is None:
        return None
    return f"{float(value):.4f}"


def setup_key_from_values(
    *,
    ticker,
    strategy,
    expiration,
    option_type,
    long_strike,
    short_strike,
    put_long_strike=None,
    put_short_strike=None,
    call_short_strike=None,
    call_long_strike=None,
) -> str:
    payload = {
        "ticker": str(ticker or "").upper(),
        "strategy": str(strategy or "").lower(),
        "expiration": str(expiration or ""),
        "option_type": str(option_type or "").lower(),
        "long_strike": _normalized_number(long_strike),
        "short_strike": _normalized_number(short_strike),
        "put_long_strike": _normalized_number(put_long_strike),
        "put_short_strike": _normalized_number(put_short_strike),
        "call_short_strike": _normalized_number(call_short_strike),
        "call_long_strike": _normalized_number(call_long_strike),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:32]


def setup_key_for_trade(trade) -> str:
    return setup_key_from_values(
        ticker=trade.ticker,
        strategy=trade.strategy,
        expiration=trade.expiration,
        option_type=trade.option_type,
        long_strike=trade.long_strike,
        short_strike=trade.short_strike,
        put_long_strike=trade.put_long_strike,
        put_short_strike=trade.put_short_strike,
        call_short_strike=trade.call_short_strike,
        call_long_strike=trade.call_long_strike,
    )


def setup_key_for_history_row(row: dict) -> str:
    strategy = str(row.get("strategy") or "").lower()
    inferred_option_type = {
        "put credit spread": "put",
        "bear put debit spread": "put",
        "call credit spread": "call",
        "bull call debit spread": "call",
        "iron condor": "mixed",
    }.get(strategy)
    return row.get("setup_key") or setup_key_from_values(
        ticker=row.get("ticker"),
        strategy=row.get("strategy"),
        expiration=row.get("expiration"),
        option_type=row.get("option_type") or inferred_option_type or row.get("entry_type"),
        long_strike=row.get("long_strike"),
        short_strike=row.get("short_strike"),
        put_long_strike=row.get("put_long_strike"),
        put_short_strike=row.get("put_short_strike"),
        call_short_strike=row.get("call_short_strike"),
        call_long_strike=row.get("call_long_strike"),
    )


def new_scan_run_id() -> str:
    return str(uuid4())


def git_commit_sha() -> str | None:
    for name in ("RAILWAY_GIT_COMMIT_SHA", "GITHUB_SHA", "COMMIT_SHA"):
        value = os.getenv(name)
        if value:
            return value[:40]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        return result.stdout.strip()[:40] or None
    except (OSError, subprocess.SubprocessError):
        return None


def normalize_history_row(row: dict) -> dict:
    normalized = dict(row)
    normalized["setup_key"] = setup_key_for_history_row(normalized)
    if not normalized.get("scan_run_id"):
        normalized["scan_run_id"] = f"legacy-{normalized.get('id', 'unknown')}"
    if not normalized.get("scanner_version"):
        normalized["scanner_version"] = "legacy"
    normalized.setdefault("git_commit_sha", None)
    normalized.setdefault("raw_rank", None)
    normalized.setdefault("diversified_rank", None)
    normalized.setdefault("execution_rank", None)
    normalized.setdefault("execution_selected", False)
    if not normalized.get("selection_method"):
        normalized["selection_method"] = SELECTION_RAW
    if normalized.get("first_seen_at") is None:
        normalized["first_seen_at"] = normalized.get("scan_time")
    if normalized.get("last_seen_at") is None:
        normalized["last_seen_at"] = normalized.get("scan_time")
    if normalized.get("times_recommended") is None:
        normalized["times_recommended"] = 1
    if normalized.get("entry_timestamp") is None:
        normalized["entry_timestamp"] = normalized.get("scan_time")
    normalized.setdefault("entry_price", None)
    normalized.setdefault("exit_timestamp", None)
    normalized.setdefault("exit_price", None)
    normalized.setdefault("exit_reason", None)
    if normalized.get("realized_pnl") is None:
        normalized["realized_pnl"] = normalized.get("actual_realized_pnl")
    normalized.setdefault("realized_return_on_risk", None)
    normalized.setdefault("closing_underlying_price", normalized.get("expiration_close"))
    normalized.setdefault("days_held", None)
    if normalized.get("maximum_favorable_excursion") is None:
        normalized["maximum_favorable_excursion"] = normalized.get(
            "highest_unrealized_pnl"
        )
    if normalized.get("maximum_adverse_excursion") is None:
        normalized["maximum_adverse_excursion"] = normalized.get(
            "lowest_unrealized_pnl"
        )
    normalized.setdefault("last_update_error", None)
    normalized.setdefault("update_retryable", False)
    return normalized


def build_history_backfill_updates(rows: list[dict]) -> list[dict]:
    normalized_rows = [normalize_history_row(row) for row in rows]
    setup_stats = {}

    for row in normalized_rows:
        setup_key = row["setup_key"]
        scan_time = row.get("scan_time")
        state = setup_stats.setdefault(
            setup_key,
            {"first_seen_at": scan_time, "last_seen_at": scan_time, "count": 0},
        )
        state["count"] += 1
        if scan_time and (
            not state["first_seen_at"] or scan_time < state["first_seen_at"]
        ):
            state["first_seen_at"] = scan_time
        if scan_time and (
            not state["last_seen_at"] or scan_time > state["last_seen_at"]
        ):
            state["last_seen_at"] = scan_time

    updates = []
    for original, normalized in zip(rows, normalized_rows):
        stats = setup_stats[normalized["setup_key"]]
        desired = {
            "setup_key": normalized["setup_key"],
            "scan_run_id": normalized["scan_run_id"],
            "scanner_version": normalized["scanner_version"],
            "selection_method": normalized["selection_method"],
            "first_seen_at": stats["first_seen_at"],
            "last_seen_at": stats["last_seen_at"],
            "times_recommended": stats["count"],
            "entry_timestamp": normalized["entry_timestamp"],
            "maximum_favorable_excursion": normalized.get(
                "maximum_favorable_excursion"
            ),
            "maximum_adverse_excursion": normalized.get(
                "maximum_adverse_excursion"
            ),
        }
        changed = {
            field: value
            for field, value in desired.items()
            if original.get(field) != value
        }
        if changed:
            updates.append({"id": original.get("id"), "values": changed})

    return updates


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def test_setup_tracking() -> None:
    values = {
        "ticker": "SPY",
        "strategy": "bull call debit spread",
        "expiration": "2026-08-21",
        "option_type": "call",
        "long_strike": 600,
        "short_strike": 605,
    }
    first = setup_key_from_values(**values)
    second = setup_key_from_values(**values)
    different = setup_key_from_values(**{**values, "short_strike": 606})
    assert first == second
    assert first != different

    first_run = new_scan_run_id()
    second_run = new_scan_run_id()
    assert first_run != second_run
    assert first == setup_key_from_values(**values)

    legacy = normalize_history_row(
        {
            "id": 42,
            "scan_time": "2026-07-01T12:00:00+00:00",
            **values,
        }
    )
    assert legacy["scan_run_id"] == "legacy-42"
    assert legacy["setup_key"] == first
    assert legacy["times_recommended"] == 1
    print("Setup/history compatibility tests passed.")
