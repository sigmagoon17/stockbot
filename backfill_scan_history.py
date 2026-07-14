import argparse
import os

from dotenv import load_dotenv
from supabase import create_client

from scanner_tracking import build_history_backfill_updates


def fetch_scan_history(client, batch_size: int) -> list[dict]:
    rows = []
    offset = 0
    while True:
        response = (
            client.table("scan_history")
            .select("*")
            .order("id")
            .range(offset, offset + batch_size - 1)
            .execute()
        )
        batch = response.data or []
        rows.extend(batch)
        if len(batch) < batch_size:
            break
        offset += batch_size
    return rows


def apply_backfill(client, updates: list[dict], batch_size: int) -> int:
    updated = 0
    for start in range(0, len(updates), batch_size):
        batch = updates[start:start + batch_size]
        for update in batch:
            if update["id"] is None:
                continue
            (
                client.table("scan_history")
                .update(update["values"])
                .eq("id", update["id"])
                .execute()
            )
            updated += 1
        print(f"Applied batch {start // batch_size + 1}: {len(batch)} rows requested")
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely backfill scanner tracking metadata in scan_history."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write updates. Without this flag the script is a dry run.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect and report changes without writing them (the default).",
    )
    parser.add_argument("--batch-size", type=int, default=200)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")

    load_dotenv()
    url = os.getenv("SUPABASE_URL")
    secret = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not secret:
        raise SystemExit("SUPABASE_URL and SUPABASE_SECRET_KEY must be configured")

    client = create_client(url, secret)
    rows = fetch_scan_history(client, args.batch_size)
    updates = build_history_backfill_updates(rows)
    print(f"Inspected {len(rows)} rows; {len(updates)} rows need updates.")

    if not args.apply:
        print("Dry run only. Re-run with --apply to write these updates.")
        return 0

    updated = apply_backfill(client, updates, args.batch_size)
    print(f"Backfill complete: inspected {len(rows)} rows; updated {updated} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
