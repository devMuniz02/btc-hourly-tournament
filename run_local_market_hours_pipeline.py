#!/usr/bin/env python3
"""
Run the ET market-hours BTC tournament locally with isolated artifacts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
import subprocess

import artifact_sync


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "local_market_hours_pipeline_run.txt"
LAST_PREDICTION_PATH = ROOT / "last_prediction_market_hours.json"
ARTIFACT_FILES = [
    ROOT / "history_market_hours.csv",
    ROOT / "assets" / "dashboard_market_hours.png",
    ROOT / "assets" / "dashboard_market_hours_reverse.png",
    ROOT / "last_prediction_market_hours.json",
]
NON_BLOCKING_LOCAL_FILES = [
    LOG_PATH,
]
REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
]


class Tee(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for stream in self.streams:
            try:
                stream.write(s)
                stream.flush()
            except ValueError:
                continue
        return len(s)

    def flush(self) -> None:
        for stream in self.streams:
            try:
                stream.flush()
            except ValueError:
                continue


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


def log_step(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def validate_required_env() -> None:
    log_step("Validate required environment")
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name, "").strip()]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def run_python_script(script_name: str, script_args: list[str] | None = None) -> int:
    log_step(f"Run {script_name}")
    command = [sys.executable, script_name, *(script_args or [])]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    return process.wait()


def load_prediction_record() -> dict[str, str]:
    if not LAST_PREDICTION_PATH.exists():
        return {}
    return json.loads(LAST_PREDICTION_PATH.read_text(encoding="utf-8"))


def expected_target_timestamp() -> str:
    now = datetime.now(timezone.utc)
    target = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    return target.isoformat()


def should_run_tournament(event_name: str) -> bool:
    log_step("Decide whether to run this hour")
    if event_name != "schedule":
        print("Non-scheduled run requested. Forcing tournament execution.")
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
    print("No successful run exists yet for this target candle. Running tournament.")
    return True


def sync_with_origin_main() -> None:
    log_step("Sync local branch with origin/main")
    artifact_sync.sync_artifacts_from_remote(
        repo_root=ROOT,
        artifact_files=ARTIFACT_FILES,
    )


def commit_artifacts(commit_message: str) -> bool:
    log_step(f"Commit artifacts: {commit_message}")
    return artifact_sync.stage_and_commit_artifacts(
        repo_root=ROOT,
        artifact_files=ARTIFACT_FILES,
        commit_message=commit_message,
    )


def push_current_head(commit_message: str) -> bool:
    log_step("Push artifacts to origin/main")
    return artifact_sync.publish_artifacts_to_origin(
        repo_root=ROOT,
        artifact_files=ARTIFACT_FILES,
        commit_message=commit_message,
    )


def run_pipeline_once(args: argparse.Namespace) -> tuple[int, bool]:
    if not args.skip_git:
        sync_with_origin_main()

    validate_exit_code = run_python_script("validate_dashboard_market_hours.py")

    run_tournament = should_run_tournament(args.event_name)

    if not args.skip_git:
        committed = commit_artifacts("Local run: update BTC market-hours validation dashboard [skip ci]")
        if committed and not args.skip_push and not run_tournament:
            if not push_current_head("Local run: update BTC market-hours validation dashboard [skip ci]"):
                return 0, True

    if validate_exit_code != 0:
        return validate_exit_code, False

    tournament_exit_code = 0
    if run_tournament:
        tournament_args: list[str] = []
        if args.reset_champion_from_challenger:
            tournament_args.append("--reset-champion-from-challenger")
        tournament_exit_code = run_python_script("market_hours_main.py", tournament_args)
        refresh_exit_code = run_python_script("validate_dashboard_market_hours.py")
        if refresh_exit_code != 0 and tournament_exit_code == 0:
            tournament_exit_code = refresh_exit_code

    if not args.skip_git:
        commit_message = (
            "Local run: update BTC market-hours artifacts [skip ci]"
            if run_tournament
            else "Local run: update BTC market-hours validation dashboard [skip ci]"
        )
        committed = commit_artifacts(commit_message)
        if committed and not args.skip_push:
            if not push_current_head(commit_message):
                return 0, True

    return tournament_exit_code, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local BTC market-hours tournament pipeline."
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
    parser.add_argument(
        "--reset-champion-from-challenger",
        action="store_true",
        help="Ignore the current champion comparison and choose from the top challenger only.",
    )
    return parser.parse_args()


def main() -> int:
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            log_step("Load local environment")
            print(f"Writing run log to {LOG_PATH}")
            args = parse_args()
            load_dotenv(ROOT / ".env")
            os.environ["BTC_EXCHANGE_MODE"] = "binance"
            validate_required_env()
            max_attempts = 2 if not args.skip_git and not args.skip_push else 1
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    log_step(f"Retry pipeline on latest origin/main (attempt {attempt}/{max_attempts})")
                exit_code, should_retry = run_pipeline_once(args)
                if not should_retry:
                    return exit_code
            raise RuntimeError("Failed to push artifacts because origin/main changed repeatedly during the local run.")


if __name__ == "__main__":
    raise SystemExit(main())
