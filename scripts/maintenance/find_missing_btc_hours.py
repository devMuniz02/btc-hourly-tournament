#!/usr/bin/env python3
"""
Print missing BTC Polymarket timestamps grouped by minute from market start.

The script reads `market_end_iso` and `minutes_from_market_start`, finds the
earliest and latest market-end hours present, then walks one hour at a time for
each expected minute bucket and prints missing timestamps grouped by
`0, 10, 20, 30, 40, 50`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = str(PROJECT_ROOT / "data" / "btc" / "btc_history.csv")
EXPECTED_MINUTES = (0, 10, 20, 30, 40, 50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print missing hourly `market_end_iso` values from a BTC history CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=DEFAULT_CSV,
        help=f"CSV file to inspect. Default: {DEFAULT_CSV}",
    )
    parser.add_argument(
        "--column",
        default="market_end_iso",
        help="CSV column containing the hourly timestamp. Default: market_end_iso",
    )
    parser.add_argument(
        "--minutes-column",
        default="minutes_from_market_start",
        help="CSV column containing minute from market start. Default: minutes_from_market_start",
    )
    parser.add_argument(
        "--end-now",
        action="store_true",
        help="Check through the current UTC hour instead of stopping at the latest CSV hour.",
    )
    return parser.parse_args()


def parse_iso_hour(value: str) -> datetime | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    else:
        dt = dt.astimezone(UTC)
    return dt.replace(minute=0, second=0, microsecond=0)


def parse_expected_minute(value: str) -> int | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    try:
        minute = int(candidate)
    except ValueError:
        return None
    return minute if minute in EXPECTED_MINUTES else None


def load_present_hours(
    csv_path: Path,
    hour_column: str,
    minutes_column: str,
) -> dict[int, set[datetime]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or hour_column not in reader.fieldnames:
            raise ValueError(
                f"Column {hour_column!r} was not found in {csv_path}. "
                f"Available columns: {', '.join(reader.fieldnames or [])}"
            )
        if minutes_column not in reader.fieldnames:
            raise ValueError(
                f"Column {minutes_column!r} was not found in {csv_path}. "
                f"Available columns: {', '.join(reader.fieldnames or [])}"
            )

        grouped_hours: dict[int, set[datetime]] = {
            minute: set() for minute in EXPECTED_MINUTES
        }
        for row in reader:
            dt = parse_iso_hour(row.get(hour_column, ""))
            minute = parse_expected_minute(row.get(minutes_column, ""))
            if dt is not None and minute is not None:
                grouped_hours[minute].add(dt)
        return grouped_hours


def iter_missing_hours(
    start_hour: datetime,
    end_hour: datetime,
    present_hours: set[datetime],
) -> list[datetime]:
    missing: list[datetime] = []
    current = start_hour
    while current <= end_hour:
        if current not in present_hours:
            missing.append(current)
        current += timedelta(hours=1)
    return missing


def main() -> int:
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        return 1

    try:
        present_hours_by_minute = load_present_hours(
            csv_path,
            args.column,
            args.minutes_column,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    all_present_hours = set().union(*present_hours_by_minute.values())
    if not all_present_hours:
        print(
            (
                "No valid grouped hourly timestamps were found in columns "
                f"{args.column!r} and {args.minutes_column!r}."
            ),
            file=sys.stderr,
        )
        return 1

    start_hour = min(all_present_hours)
    latest_csv_hour = max(all_present_hours)
    if args.end_now:
        now_utc = datetime.now(tz=UTC).replace(minute=0, second=0, microsecond=0)
        end_hour = max(latest_csv_hour, now_utc)
    else:
        end_hour = latest_csv_hour

    print(f"CSV: {csv_path}")
    print(f"Column: {args.column}")
    print(f"Minutes column: {args.minutes_column}")
    print(f"Start hour: {start_hour.isoformat()}")
    print(f"End hour: {end_hour.isoformat()}")
    print(f"Present market hours: {len(all_present_hours)}")

    total_missing = 0
    for minute in EXPECTED_MINUTES:
        missing_hours = iter_missing_hours(
            start_hour,
            end_hour,
            present_hours_by_minute[minute],
        )
        total_missing += len(missing_hours)

        print("")
        print(f"Minute {minute}: missing {len(missing_hours)}")
        for hour in missing_hours:
            market_start = hour - timedelta(hours=1)
            sample_time = market_start + timedelta(minutes=minute)
            print(
                f"{hour.isoformat()} | market_start={market_start.isoformat()} | sample_time={sample_time.isoformat()} | minute={minute}"
            )

    print("")
    print(f"Total missing entries across minute groups: {total_missing}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
