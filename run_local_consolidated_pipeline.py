#!/usr/bin/env python3
"""
Run the consolidated BTC tournament locally with isolated consolidated artifacts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path

import artifact_sync
from pipelines.consolidated import config as consolidated_config


ROOT = Path(__file__).resolve().parent
LOG_PATH = ROOT / "local_consolidated_pipeline_run.txt"
LAST_PREDICTION_PATH = consolidated_config.LAST_PREDICTION_PATH
ARTIFACT_FILES = tuple(consolidated_config.tracked_output_files())
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
        print("Non-scheduled run requested. Forcing consolidated execution.")
        return True
    record = load_prediction_record()
    saved_target = str(record.get("target_candle_timestamp", "")).strip()
    saved_status = str(record.get("status", "")).strip()
    expected_target = expected_target_timestamp()

    print(f"expected_target={expected_target}")
    print(f"saved_target={saved_target}")
    print(f"saved_status={saved_status}")

    if saved_target and saved_target == expected_target and saved_status == "success":
        print("Scheduled consolidated run already exists for this target candle. Skipping run.")
        return False
    print("No successful consolidated run exists yet for this target candle. Running pipeline.")
    return True


def sync_with_origin_main() -> None:
    log_step("Sync local branch with origin/main")
    artifact_sync.sync_artifacts_from_remote(
        repo_root=ROOT,
        artifact_files=ARTIFACT_FILES,
    )


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


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def push_current_head(commit_message: str) -> bool:
    log_step("Push artifacts to origin/main")
    with tempfile.TemporaryDirectory() as snapshot_root_raw:
        snapshot_root = Path(snapshot_root_raw)
        for path in ARTIFACT_FILES:
            copy_if_exists(path, snapshot_root / path.relative_to(ROOT))

        for attempt in range(1, 4):
            fetch_result = artifact_sync.run_git_command_in_dir(ROOT, "fetch", "origin", "main")
            if fetch_result.returncode != 0:
                artifact_sync.print_git_result(fetch_result)
                raise RuntimeError("Failed to fetch origin/main before publishing consolidated artifacts.")
            artifact_sync.print_git_result(fetch_result)

            with tempfile.TemporaryDirectory() as worktree_root_raw:
                worktree_root = Path(worktree_root_raw)
                publish_worktree = worktree_root / "publish"
                add_result = artifact_sync.run_git_command_in_dir(
                    ROOT,
                    "worktree",
                    "add",
                    "--detach",
                    str(publish_worktree),
                    "origin/main",
                )
                if add_result.returncode != 0:
                    artifact_sync.print_git_result(add_result)
                    raise RuntimeError("Failed to create temporary publish worktree from origin/main.")
                try:
                    for path in ARTIFACT_FILES:
                        relative_path = path.relative_to(ROOT)
                        artifact_sync.apply_local_artifact_to_repo(
                            publish_worktree / relative_path,
                            snapshot_root / relative_path,
                        )

                    changed = stage_and_commit_artifacts_in_repo(
                        repo_root=publish_worktree,
                        artifact_files=[publish_worktree / path.relative_to(ROOT) for path in ARTIFACT_FILES],
                        commit_message=commit_message,
                    )
                    if not changed:
                        for path in ARTIFACT_FILES:
                            copy_if_exists(publish_worktree / path.relative_to(ROOT), path)
                        return True

                    push_result = artifact_sync.run_git_command_in_dir(
                        publish_worktree,
                        "push",
                        "origin",
                        "HEAD:main",
                    )
                    if push_result.returncode == 0:
                        artifact_sync.print_git_result(push_result)
                        for path in ARTIFACT_FILES:
                            copy_if_exists(publish_worktree / path.relative_to(ROOT), path)
                        return True
                    artifact_sync.print_git_result(push_result)
                finally:
                    remove_result = artifact_sync.run_git_command_in_dir(
                        ROOT,
                        "worktree",
                        "remove",
                        "--force",
                        str(publish_worktree),
                    )
                    if remove_result.returncode != 0:
                        artifact_sync.print_git_result(remove_result)
                        raise RuntimeError("Failed to remove the temporary publish worktree.")
            if attempt < 3:
                print("Retrying consolidated artifact publish on the latest origin/main.")
        raise RuntimeError("Failed to push local consolidated artifacts to origin/main.")


def stage_and_commit_artifacts_in_repo(
    *,
    repo_root: Path,
    artifact_files: list[Path],
    commit_message: str,
) -> bool:
    existing_files = [path for path in artifact_files if path.exists()]
    if not existing_files:
        print("No consolidated artifact files exist yet in publish worktree. Nothing to commit.")
        return False

    artifact_sync.run_git_command_in_dir(repo_root, "config", "user.name", "local-btc-bot")
    artifact_sync.run_git_command_in_dir(
        repo_root, "config", "user.email", "local-btc-bot@users.noreply.github.com"
    )

    for path in existing_files:
        relative_path = artifact_sync.normalize_git_path(str(path.relative_to(repo_root)))
        add_args = ["add", relative_path]
        if path.suffix.lower() == ".log":
            add_args = ["add", "-f", relative_path]
        add_result = artifact_sync.run_git_command_in_dir(repo_root, *add_args)
        if add_result.returncode != 0:
            artifact_sync.print_git_result(add_result)
            raise RuntimeError(f"Failed to stage artifact '{relative_path}' in publish worktree.")

    staged_status = artifact_sync.run_git_command_in_dir(repo_root, "diff", "--cached", "--name-only")
    if staged_status.returncode != 0:
        artifact_sync.print_git_result(staged_status)
        raise RuntimeError("Failed to inspect staged consolidated artifacts in publish worktree.")
    if not staged_status.stdout.strip():
        print("No staged consolidated artifact changes detected in publish worktree.")
        return False

    commit_result = artifact_sync.run_git_command_in_dir(repo_root, "commit", "-m", commit_message)
    if commit_result.returncode != 0:
        artifact_sync.print_git_result(commit_result)
        raise RuntimeError("Failed to commit consolidated artifacts in publish worktree.")
    artifact_sync.print_git_result(commit_result)
    return True


def run_pipeline_once(args: argparse.Namespace) -> tuple[int, bool]:
    if not args.skip_git:
        sync_with_origin_main()

    validate_exit_code = run_python_script("pipelines/consolidated/validate_dashboard.py")
    run_tournament = should_run_tournament(args.event_name)

    if not args.skip_git:
        committed = stage_and_commit_artifacts(
            "Local run: update BTC consolidated validation dashboard [skip ci]"
        )
        if committed and not args.skip_push and not run_tournament:
            if not push_current_head("Local run: update BTC consolidated validation dashboard [skip ci]"):
                return 0, True

    if validate_exit_code != 0:
        return validate_exit_code, False

    tournament_exit_code = 0
    if run_tournament:
        tournament_args: list[str] = []
        if args.reset_champion_from_challenger:
            tournament_args.append("--reset-champion-from-challenger")
        tournament_exit_code = run_python_script("pipelines/consolidated/main.py", tournament_args)
        refresh_exit_code = run_python_script("pipelines/consolidated/validate_dashboard.py")
        if refresh_exit_code != 0 and tournament_exit_code == 0:
            tournament_exit_code = refresh_exit_code

    if not args.skip_git:
        commit_message = (
            "Local run: update BTC consolidated artifacts [skip ci]"
            if run_tournament
            else "Local run: update BTC consolidated validation dashboard [skip ci]"
        )
        committed = stage_and_commit_artifacts(commit_message)
        if committed and not args.skip_push:
            if not push_current_head(commit_message):
                return 0, True

    return tournament_exit_code, False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the local consolidated BTC tournament pipeline."
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
        help="Commit local consolidated artifact changes without pushing them to origin/main.",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Do not commit or push consolidated artifacts.",
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
            consolidated_config.ensure_output_dirs()
            max_attempts = 2 if not args.skip_git and not args.skip_push else 1
            for attempt in range(1, max_attempts + 1):
                if attempt > 1:
                    log_step(
                        f"Retry consolidated pipeline on latest origin/main (attempt {attempt}/{max_attempts})"
                    )
                exit_code, should_retry = run_pipeline_once(args)
                if not should_retry:
                    return exit_code
            raise RuntimeError(
                "Failed to push consolidated artifacts because origin/main changed repeatedly during the local run."
            )


if __name__ == "__main__":
    raise SystemExit(main())
