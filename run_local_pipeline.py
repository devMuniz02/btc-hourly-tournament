#!/usr/bin/env python3
"""
Run the BTC directional bot locally with the same control flow used in GitHub Actions.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GIT_DIR = ROOT / ".git"
LOG_PATH = ROOT / "local_pipeline_run.txt"
LAST_PREDICTION_PATH = ROOT / "last_prediction.json"
ARTIFACT_FILES = [
    ROOT / "history.csv",
    ROOT / "assets" / "dashboard.png",
    ROOT / "last_prediction.json",
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


def run_python_script(script_name: str) -> int:
    log_step(f"Run {script_name}")
    command = [sys.executable, script_name]
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


def run_git_command(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def rebase_in_progress() -> bool:
    return (GIT_DIR / "rebase-merge").exists() or (GIT_DIR / "rebase-apply").exists()


def worktree_clean() -> bool:
    status = run_git_command("status", "--porcelain")
    return not status.stdout.strip()


def has_non_artifact_worktree_changes() -> bool:
    status = run_git_command("status", "--porcelain")
    artifact_paths = {str(path.relative_to(ROOT)).replace("\\", "/") for path in ARTIFACT_FILES}
    non_blocking_paths = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in NON_BLOCKING_LOCAL_FILES
    }
    for line in status.stdout.splitlines():
        path = line[3:].strip().replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip()
        if path and path not in artifact_paths:
            if path in non_blocking_paths:
                continue
            return True
    return False


def commit_artifacts(commit_message: str) -> bool:
    log_step(f"Commit artifacts: {commit_message}")
    existing_files = [path for path in ARTIFACT_FILES if path.exists()]
    if not existing_files:
        print("No artifact files exist yet. Nothing to commit.")
        return False

    run_git_command("config", "user.name", "local-btc-bot")
    run_git_command("config", "user.email", "local-btc-bot@users.noreply.github.com")

    for path in existing_files:
        run_git_command("add", str(path.relative_to(ROOT)))

    staged_status = run_git_command("diff", "--cached", "--name-only")
    if not staged_status.stdout.strip():
        print("No staged artifact changes detected.")
        return False

    commit_result = run_git_command("commit", "-m", commit_message)
    if commit_result.returncode != 0:
        print(commit_result.stdout, end="")
        print(commit_result.stderr, end="")
        raise RuntimeError("Failed to commit local artifacts.")
    print(commit_result.stdout, end="")
    return True


def push_current_head() -> None:
    log_step("Push artifacts to origin/main")
    if has_non_artifact_worktree_changes():
        print("Skipping push because the working tree has non-artifact local changes.")
        return
    if rebase_in_progress():
        print("Detected an in-progress git rebase.")
        if worktree_clean():
            continue_result = run_git_command("rebase", "--continue")
            if continue_result.returncode == 0:
                if continue_result.stdout.strip():
                    print(continue_result.stdout, end="")
                if continue_result.stderr.strip():
                    print(continue_result.stderr, end="")
            else:
                abort_result = run_git_command("rebase", "--abort")
                if abort_result.returncode != 0:
                    print(continue_result.stdout, end="")
                    print(continue_result.stderr, end="")
                    print(abort_result.stdout, end="")
                    print(abort_result.stderr, end="")
                    raise RuntimeError("Failed to recover from an existing git rebase.")
                print("Aborted stale git rebase before pushing artifacts.")
        else:
            print("Skipping push because a git rebase is already in progress with local changes.")
            return
    pull_result = run_git_command("pull", "--rebase", "origin", "main")
    if pull_result.returncode != 0:
        print(pull_result.stdout, end="")
        print(pull_result.stderr, end="")
        raise RuntimeError("Failed to rebase local artifacts onto origin/main.")
    if pull_result.stdout.strip():
        print(pull_result.stdout, end="")
    if pull_result.stderr.strip():
        print(pull_result.stderr, end="")

    push_result = run_git_command("push", "origin", "HEAD:main")
    if push_result.returncode != 0:
        print(push_result.stdout, end="")
        print(push_result.stderr, end="")
        raise RuntimeError("Failed to push local artifacts to origin/main.")
    print(push_result.stdout, end="")
    if push_result.stderr.strip():
        print(push_result.stderr, end="")


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
    with LOG_PATH.open("w", encoding="utf-8") as log_file:
        tee = Tee(sys.stdout, log_file)
        with redirect_stdout(tee), redirect_stderr(tee):
            log_step("Load local environment")
            print(f"Writing run log to {LOG_PATH}")
            args = parse_args()
            load_dotenv(ROOT / ".env")
            os.environ["BTC_EXCHANGE_MODE"] = "binance"
            validate_required_env()

            validate_exit_code = run_python_script("validate_dashboard.py")

            run_tournament = should_run_tournament(args.event_name)

            if not args.skip_git:
                committed = commit_artifacts("Local run: update BTC validation dashboard [skip ci]")
                if committed and not args.skip_push and not run_tournament:
                    push_current_head()

            if validate_exit_code != 0:
                return validate_exit_code

            tournament_exit_code = 0
            if run_tournament:
                tournament_exit_code = run_python_script("main.py")
                refresh_exit_code = run_python_script("validate_dashboard.py")
                if refresh_exit_code != 0 and tournament_exit_code == 0:
                    tournament_exit_code = refresh_exit_code

            if not args.skip_git:
                commit_message = (
                    "Local run: update BTC bot artifacts [skip ci]"
                    if run_tournament
                    else "Local run: update BTC validation dashboard [skip ci]"
                )
                committed = commit_artifacts(commit_message)
                if committed and not args.skip_push:
                    push_current_head()

            return tournament_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
