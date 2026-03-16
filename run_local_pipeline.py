#!/usr/bin/env python3
"""
Run the BTC directional bot locally with the same control flow used in GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
LAST_PREDICTION_PATH = ROOT / "last_prediction.json"
ARTIFACT_FILES = [
    ROOT / "history.csv",
    ROOT / "assets" / "dashboard.png",
    ROOT / "last_prediction.json",
]
REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
]


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def validate_required_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def run_python_script(script_name: str) -> int:
    command = [sys.executable, script_name]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return completed.returncode


def load_prediction_record() -> dict[str, str]:
    if not LAST_PREDICTION_PATH.exists():
        return {}
    return json.loads(LAST_PREDICTION_PATH.read_text(encoding="utf-8"))


def expected_target_timestamp() -> str:
    now = datetime.now(timezone.utc)
    target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return target.isoformat()


def should_run_tournament(event_name: str) -> bool:
    if event_name != "schedule":
        return True
    record = load_prediction_record()
    saved_target = record.get("target_candle_timestamp", "")
    saved_status = record.get("status", "")
    expected_target = expected_target_timestamp()

    print(f"expected_target={expected_target}")
    print(f"saved_target={saved_target}")
    print(f"saved_status={saved_status}")

    if saved_target and saved_target == expected_target and saved_status == "success":
        print("Scheduled run already exists for this target candle. Skipping tournament.")
        return False
    return True


def run_git_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def commit_artifacts(commit_message: str) -> None:
    existing_files = [path for path in ARTIFACT_FILES if path.exists()]
    if not existing_files:
        return

    run_git_command("config", "user.name", "local-btc-bot")
    run_git_command("config", "user.email", "local-btc-bot@users.noreply.github.com")

    for path in existing_files:
        run_git_command("add", str(path.relative_to(ROOT)))

    staged_status = run_git_command("diff", "--cached", "--name-only")
    if not staged_status.stdout.strip():
        return

    commit_result = run_git_command("commit", "-m", commit_message)
    if commit_result.returncode != 0:
        print(commit_result.stdout, end="")
        print(commit_result.stderr, end="")
        raise RuntimeError("Failed to commit local artifacts.")


def push_current_head() -> None:
    push_result = run_git_command("push", "origin", "HEAD:main")
    if push_result.returncode != 0:
        print(push_result.stdout, end="")
        print(push_result.stderr, end="")
        raise RuntimeError("Failed to push local artifacts to origin/main.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local BTC directional tournament pipeline."
    )
    parser.add_argument(
        "--event-name",
        default="schedule",
        choices=["schedule", "workflow_dispatch"],
        help="Use schedule to preserve the hourly dedup gate. Use workflow_dispatch to force a full run.",
    )
    parser.add_argument(
        "--skip-push",
        action="store_true",
        help="Commit local artifact changes without pushing them to origin/main.",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Do not commit or push generated artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    validate_required_env()

    validate_exit_code = run_python_script("validate_dashboard.py")

    if not args.skip_git:
        commit_artifacts("Update BTC validation dashboard [skip ci]")
        if not args.skip_push:
            push_current_head()

    if validate_exit_code != 0:
        return validate_exit_code

    tournament_exit_code = 0
    if should_run_tournament(args.event_name):
        tournament_exit_code = run_python_script("main.py")

    if not args.skip_git:
        commit_artifacts("Update BTC bot artifacts [skip ci]")
        if not args.skip_push:
            push_current_head()

    return tournament_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
