#!/usr/bin/env python3
"""
Finalize consolidated promotion and artifact publication after the trader has already placed orders.
"""

from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

import artifact_sync
import pandas as pd
from pipelines.consolidated import config as consolidated_config
from pipelines.consolidated import main as consolidated_main


ROOT = Path(__file__).resolve().parent
LOG_DIR = consolidated_config.ARTIFACT_DIR / "background_logs"
LOG_PREFIX = "background-publish"
ARTIFACT_FILES = tuple(consolidated_config.tracked_output_files())
REQUIRED_ENV_VARS = [
    "MLFLOW_TRACKING_URI",
    "MLFLOW_TRACKING_USERNAME",
    "MLFLOW_TRACKING_PASSWORD",
]
MAX_RETAINED_LOGS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize deferred consolidated promotion and artifact publication."
    )
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-git", action="store_true")
    return parser.parse_args()


def log_step(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


def resolve_hourly_log_path(
    log_dir: Path,
    prefix: str,
) -> Path:
    timestamp = pd.Timestamp.utcnow()
    timestamp = timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    timestamp = timestamp.floor("h")
    return log_dir / f"{prefix}-{timestamp.strftime('%Y%m%d-%H00Z')}.log"


def prune_hourly_logs(
    log_dir: Path,
    prefix: str,
    *,
    keep_last: int = MAX_RETAINED_LOGS,
) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        log_dir.glob(f"{prefix}-*.log"),
        key=lambda path: path.name,
        reverse=True,
    )
    for stale_path in candidates[keep_last:]:
        try:
            stale_path.unlink()
        except FileNotFoundError:
            continue


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def validate_required_env() -> None:
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


def stage_and_commit_artifacts(commit_message: str) -> bool:
    log_step(f"Commit artifacts: {commit_message}")
    existing_files = [path for path in ARTIFACT_FILES if path.exists()]
    if not existing_files:
        print("No consolidated artifact files exist yet. Nothing to commit.")
        return False

    artifact_sync.run_git_command_in_dir(ROOT, "config", "user.name", "local-btc-bot")
    artifact_sync.run_git_command_in_dir(
        ROOT, "config", "user.email", "local-btc-bot@users.noreply.github.com"
    )

    for path in existing_files:
        relative_path = artifact_sync.normalize_git_path(str(path.relative_to(ROOT)))
        add_args = ["add", relative_path]
        if path.suffix.lower() == ".log":
            add_args = ["add", "-f", relative_path]
        add_result = artifact_sync.run_git_command_in_dir(ROOT, *add_args)
        if add_result.returncode != 0:
            artifact_sync.print_git_result(add_result)
            raise RuntimeError(f"Failed to stage artifact '{relative_path}'.")

    staged_status = artifact_sync.run_git_command_in_dir(ROOT, "diff", "--cached", "--name-only")
    if staged_status.returncode != 0:
        artifact_sync.print_git_result(staged_status)
        raise RuntimeError("Failed to inspect staged consolidated artifacts.")
    if not staged_status.stdout.strip():
        print("No staged consolidated artifact changes detected.")
        return False

    commit_result = artifact_sync.run_git_command_in_dir(ROOT, "commit", "-m", commit_message)
    if commit_result.returncode != 0:
        artifact_sync.print_git_result(commit_result)
        raise RuntimeError("Failed to commit local consolidated artifacts.")
    artifact_sync.print_git_result(commit_result)
    return True


def push_current_head(commit_message: str) -> bool:
    log_step("Push artifacts to origin/main")
    return artifact_sync.publish_artifacts_to_origin(
        repo_root=ROOT,
        artifact_files=ARTIFACT_FILES,
        commit_message=commit_message,
    )


def main() -> int:
    args = parse_args()
    consolidated_config.ensure_output_dirs()
    log_path = resolve_hourly_log_path(LOG_DIR, LOG_PREFIX)
    prune_hourly_logs(LOG_DIR, LOG_PREFIX)
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        raise RuntimeError(f"Deferred consolidated publish manifest not found: {manifest_path}")

    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = log_file
        sys.stderr = log_file
        try:
            log_step("Load local environment")
            load_dotenv(ROOT / ".env")
            os.environ["BTC_EXCHANGE_MODE"] = "binance"
            validate_required_env()

            log_step("Load deferred consolidated publish payload")
            with manifest_path.open("rb") as handle:
                pending_publish = pickle.load(handle)

            log_step("Promote deferred consolidated champions")
            execution = consolidated_main.finalize_pending_publish(pending_publish)
            consolidated_main.persist_execution_outputs(execution)

            validate_exit_code = run_python_script("pipelines/consolidated/validate_dashboard.py")
            if validate_exit_code != 0:
                raise RuntimeError(
                    f"Consolidated dashboard refresh failed with exit code {validate_exit_code}."
                )

            if not args.skip_git:
                committed = stage_and_commit_artifacts("BTC consolidated artifacts [skip ci]")
                if committed and not args.skip_push:
                    if not push_current_head("BTC consolidated artifacts [skip ci]"):
                        raise RuntimeError(
                            "Failed to push consolidated artifacts after deferred publish."
                        )
            return 0
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            try:
                manifest_path.unlink(missing_ok=True)
            except Exception:
                pass


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal error: {exc}")
        traceback.print_exc()
        raise
