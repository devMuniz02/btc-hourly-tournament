#!/usr/bin/env python3
"""
Force a rerun of the current-hour BTC tournament by clearing the hourly success gate
before executing the normal local pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.local import run_local_pipeline
from src.btc_pipeline.path_config import HOURLY_LAST_PREDICTION_PATH


LAST_PREDICTION_PATH = HOURLY_LAST_PREDICTION_PATH
BACKUP_PATH = ROOT / "last_prediction.before_rerun.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Redo the current-hour tournament and regenerate last_prediction.json."
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Run the rerun locally without pushing artifacts to origin/main.",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Run the rerun without committing or pushing generated artifacts.",
    )
    parser.add_argument(
        "--reset-champion-from-challenger",
        action="store_true",
        help="Ignore the stored champion and choose from the current top challenger only.",
    )
    return parser.parse_args()


def load_prediction_record() -> dict[str, Any]:
    if not LAST_PREDICTION_PATH.exists():
        return {}
    return json.loads(LAST_PREDICTION_PATH.read_text(encoding="utf-8"))


def save_prediction_record(record: dict[str, Any]) -> None:
    LAST_PREDICTION_PATH.write_text(
        json.dumps(record, indent=2),
        encoding="utf-8",
    )


def invalidate_current_hour_success() -> None:
    expected_target = run_local_pipeline.expected_target_timestamp()
    record = load_prediction_record()

    if record:
        BACKUP_PATH.write_text(
            json.dumps(record, indent=2),
            encoding="utf-8",
        )

    print(f"expected_target={expected_target}")

    if not record:
        print("No existing last_prediction.json found. Proceeding with a clean rerun.")
        return

    saved_target = str(record.get("target_candle_timestamp", ""))
    saved_status = str(record.get("status", ""))
    print(f"saved_target={saved_target}")
    print(f"saved_status={saved_status}")

    if saved_target == expected_target and saved_status == "success":
        record["status"] = "rerun_requested"
        record["rerun_requested_at"] = datetime.now(timezone.utc).isoformat()
        save_prediction_record(record)
        print("Cleared the current-hour success marker so the tournament will run again.")
        return

    print("Current record does not block the hourly gate. Proceeding with a forced rerun.")


def main() -> int:
    args = parse_args()
    print("Preparing a current-hour rerun.")
    run_local_pipeline.load_dotenv(ROOT / ".env")
    invalidate_current_hour_success()

    command = [sys.executable, "scripts/local/run_local_pipeline.py", "--event-name", "schedule"]
    if args.skip_push:
        command.append("--skip-push")
    if args.skip_git:
        command.append("--skip-git")
    if args.reset_champion_from_challenger:
        command.append("--reset-champion-from-challenger")

    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
