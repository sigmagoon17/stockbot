from dataclasses import dataclass


EXIT_NONE = "none"
EXIT_TP50 = "tp50"
EXIT_TP75 = "tp75"
EXIT_POLICIES = {EXIT_NONE, EXIT_TP50, EXIT_TP75}


@dataclass(frozen=True)
class PaperExitDecision:
    should_close: bool
    exit_reason: str | None
    target_value_per_share: float | None
    current_value_per_share: float | None
    message: str


def normalized_exit_policy(policy: str | None) -> str:
    normalized = str(policy or EXIT_NONE).lower().strip()
    return normalized if normalized in EXIT_POLICIES else EXIT_NONE


def take_profit_fraction(policy: str | None) -> float | None:
    normalized = normalized_exit_policy(policy)
    if normalized == EXIT_TP50:
        return 0.50
    if normalized == EXIT_TP75:
        return 0.75
    return None


def take_profit_target_per_share(
    entry_type: str,
    entry_price_per_share: float,
    max_profit_per_share: float,
    policy: str,
) -> float | None:
    fraction = take_profit_fraction(policy)
    if fraction is None:
        return None
    if entry_type.lower() == "debit":
        return round(entry_price_per_share + max_profit_per_share * fraction, 4)
    if entry_type.lower() == "credit":
        return round(entry_price_per_share * (1 - fraction), 4)
    return None


def evaluate_paper_exit(
    *,
    entry_type: str,
    entry_price_per_share: float,
    max_profit_per_share: float,
    current_value_per_share: float | None,
    policy: str | None,
    close_order_status: str | None = None,
) -> PaperExitDecision:
    normalized_policy = normalized_exit_policy(policy)
    target = take_profit_target_per_share(
        entry_type,
        entry_price_per_share,
        max_profit_per_share,
        normalized_policy,
    )
    if target is None:
        return PaperExitDecision(False, None, None, current_value_per_share, "Automatic exit disabled.")
    if close_order_status:
        return PaperExitDecision(
            False,
            None,
            target,
            current_value_per_share,
            f"Closing order already recorded with status {close_order_status}.",
        )
    if current_value_per_share is None:
        return PaperExitDecision(False, None, target, None, "Current spread value unavailable.")

    if entry_type.lower() == "debit":
        triggered = current_value_per_share >= target
    elif entry_type.lower() == "credit":
        triggered = current_value_per_share <= target
    else:
        triggered = False

    return PaperExitDecision(
        triggered,
        normalized_policy if triggered else None,
        target,
        current_value_per_share,
        "Take-profit threshold reached." if triggered else "Take-profit threshold not reached.",
    )


def closing_legs_from_leg_key(leg_key: str) -> list[dict]:
    closing_legs = []
    for leg_part in (leg_key or "").split("|"):
        if not leg_part:
            continue
        symbol, side, _, ratio_qty = leg_part.split(":", 3)
        closing_legs.append(
            {
                "symbol": symbol,
                "ratio_qty": ratio_qty or "1",
                "side": "sell" if side == "buy" else "buy",
                "position_intent": "sell_to_close" if side == "buy" else "buy_to_close",
            }
        )
    return closing_legs


def test_paper_exit_logic() -> None:
    assert take_profit_target_per_share("debit", 1.66, 3.34, EXIT_TP50) == 3.33
    assert take_profit_target_per_share("credit", 1.20, 1.20, EXIT_TP50) == 0.60
    assert take_profit_target_per_share("credit", 1.20, 1.20, EXIT_TP75) == 0.30

    debit = evaluate_paper_exit(
        entry_type="debit",
        entry_price_per_share=1.66,
        max_profit_per_share=3.34,
        current_value_per_share=3.40,
        policy=EXIT_TP50,
    )
    assert debit.should_close and debit.exit_reason == EXIT_TP50

    credit = evaluate_paper_exit(
        entry_type="credit",
        entry_price_per_share=1.20,
        max_profit_per_share=1.20,
        current_value_per_share=0.55,
        policy=EXIT_TP50,
    )
    assert credit.should_close and credit.exit_reason == EXIT_TP50

    repeated = evaluate_paper_exit(
        entry_type="debit",
        entry_price_per_share=1.66,
        max_profit_per_share=3.34,
        current_value_per_share=3.40,
        policy=EXIT_TP50,
        close_order_status="accepted",
    )
    assert not repeated.should_close

    closing_legs = closing_legs_from_leg_key(
        "AAPL260717C00100000:buy:buy_to_open:1|"
        "AAPL260717C00105000:sell:sell_to_open:1"
    )
    assert [leg["side"] for leg in closing_legs] == ["sell", "buy"]
    assert [leg["position_intent"] for leg in closing_legs] == [
        "sell_to_close",
        "buy_to_close",
    ]
    print("Paper exit tests passed.")
