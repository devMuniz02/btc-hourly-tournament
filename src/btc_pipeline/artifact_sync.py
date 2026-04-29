from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PrintFn = Callable[[str], None]


def normalize_git_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def run_git_command_in_dir(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=False,
        text=True,
        capture_output=True,
    )


def print_git_result(result: subprocess.CompletedProcess[str], print_fn: PrintFn = print) -> None:
    if result.stdout.strip():
        print_fn(result.stdout.rstrip("\n"))
    if result.stderr.strip():
        print_fn(result.stderr.rstrip("\n"))


def _run_git_binary(repo_dir: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=False,
        capture_output=True,
    )


def fetch_remote_file_bytes(repo_dir: Path, relative_path: Path, ref: str = "origin/main") -> bytes | None:
    result = _run_git_binary(repo_dir, "show", f"{ref}:{normalize_git_path(str(relative_path))}")
    if result.returncode != 0:
        return None
    return result.stdout


def _parse_timestamp_sort_key(value: str) -> tuple[int, datetime | str]:
    text = str(value or "").strip()
    if not text:
        return (1, "")
    normalized = text.replace(" ", "T")
    try:
        return (0, datetime.fromisoformat(normalized))
    except ValueError:
        return (1, text)


def merge_csv_contents(remote_text: str, local_text: str) -> str:
    remote_reader = csv.DictReader(io.StringIO(remote_text))
    local_reader = csv.DictReader(io.StringIO(local_text))
    remote_rows = list(remote_reader)
    local_rows = list(local_reader)
    fieldnames = list(remote_reader.fieldnames or local_reader.fieldnames or [])
    if not fieldnames:
        return local_text or remote_text

    for name in local_reader.fieldnames or []:
        if name not in fieldnames:
            fieldnames.append(name)

    remote_by_timestamp = {
        str(row.get("timestamp", "")).strip(): row
        for row in remote_rows
        if str(row.get("timestamp", "")).strip()
    }
    merged_rows = list(remote_rows)
    for row in local_rows:
        timestamp = str(row.get("timestamp", "")).strip()
        if timestamp and timestamp not in remote_by_timestamp:
            merged_rows.append(row)

    merged_rows.sort(key=lambda row: _parse_timestamp_sort_key(str(row.get("timestamp", "")).strip()))

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in merged_rows:
        normalized_row = {field: row.get(field, "") for field in fieldnames}
        writer.writerow(normalized_row)
    return output.getvalue()


def _is_empty_json_value(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def merge_json_objects(remote_value: Any, local_value: Any) -> Any:
    if isinstance(remote_value, dict) and isinstance(local_value, dict):
        merged = json.loads(json.dumps(remote_value))
        for key, local_item in local_value.items():
            if key not in merged:
                merged[key] = json.loads(json.dumps(local_item))
                continue
            remote_item = merged[key]
            if isinstance(remote_item, dict) and isinstance(local_item, dict):
                merged[key] = merge_json_objects(remote_item, local_item)
                continue
            if isinstance(remote_item, list) and isinstance(local_item, list):
                if not remote_item and local_item:
                    merged[key] = json.loads(json.dumps(local_item))
                continue
            if _is_empty_json_value(remote_item) and not _is_empty_json_value(local_item):
                merged[key] = json.loads(json.dumps(local_item))
        return merged
    if _is_empty_json_value(remote_value) and not _is_empty_json_value(local_value):
        return json.loads(json.dumps(local_value))
    return json.loads(json.dumps(remote_value))


def merge_json_contents(remote_text: str, local_text: str) -> str:
    remote_obj = json.loads(remote_text)
    local_obj = json.loads(local_text)
    merged = merge_json_objects(remote_obj, local_obj)
    return json.dumps(merged, indent=2) + "\n"


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def merge_artifact_file(local_path: Path, remote_bytes: bytes | None, *, prefer_remote_binary: bool) -> None:
    suffix = local_path.suffix.lower()
    if suffix == ".csv":
        if remote_bytes is None:
            return
        remote_text = remote_bytes.decode("utf-8")
        local_text = local_path.read_text(encoding="utf-8") if local_path.exists() else ""
        merged_text = merge_csv_contents(remote_text, local_text)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(merged_text, encoding="utf-8", newline="\n")
        return
    if suffix == ".json":
        if remote_bytes is None:
            return
        remote_text = remote_bytes.decode("utf-8")
        local_text = local_path.read_text(encoding="utf-8") if local_path.exists() else "{}"
        merged_text = merge_json_contents(remote_text, local_text)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(merged_text, encoding="utf-8", newline="\n")
        return
    if remote_bytes is not None and prefer_remote_binary:
        _write_bytes(local_path, remote_bytes)


def apply_local_artifact_to_repo(destination_path: Path, local_snapshot_path: Path) -> None:
    if not local_snapshot_path.exists():
        return
    suffix = destination_path.suffix.lower()
    if suffix == ".csv":
        remote_text = destination_path.read_text(encoding="utf-8") if destination_path.exists() else ""
        local_text = local_snapshot_path.read_text(encoding="utf-8")
        merged_text = merge_csv_contents(remote_text, local_text)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(merged_text, encoding="utf-8", newline="\n")
        return
    if suffix == ".json":
        remote_text = destination_path.read_text(encoding="utf-8") if destination_path.exists() else "{}"
        local_text = local_snapshot_path.read_text(encoding="utf-8")
        merged_text = merge_json_contents(remote_text, local_text)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_text(merged_text, encoding="utf-8", newline="\n")
        return
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_snapshot_path, destination_path)


def sync_artifacts_from_remote(
    *,
    repo_root: Path,
    artifact_files: tuple[Path, ...] | list[Path],
    print_fn: PrintFn = print,
) -> None:
    fetch_result = run_git_command_in_dir(repo_root, "fetch", "origin", "main")
    if fetch_result.returncode != 0:
        print_git_result(fetch_result, print_fn)
        raise RuntimeError("Failed to fetch origin/main before syncing artifacts.")
    print_git_result(fetch_result, print_fn)

    for artifact_path in artifact_files:
        relative_path = artifact_path.relative_to(repo_root)
        remote_bytes = fetch_remote_file_bytes(repo_root, relative_path)
        merge_artifact_file(artifact_path, remote_bytes, prefer_remote_binary=True)


def stage_and_commit_artifacts(
    *,
    repo_root: Path,
    artifact_files: tuple[Path, ...] | list[Path],
    commit_message: str,
    print_fn: PrintFn = print,
) -> bool:
    existing_files = [path for path in artifact_files if path.exists()]
    if not existing_files:
        print_fn("No artifact files exist yet. Nothing to commit.")
        return False

    run_git_command_in_dir(repo_root, "config", "user.name", "local-btc-bot")
    run_git_command_in_dir(repo_root, "config", "user.email", "local-btc-bot@users.noreply.github.com")

    for path in existing_files:
        add_result = run_git_command_in_dir(repo_root, "add", normalize_git_path(str(path.relative_to(repo_root))))
        if add_result.returncode != 0:
            print_git_result(add_result, print_fn)
            raise RuntimeError(f"Failed to stage artifact '{path.relative_to(repo_root)}'.")

    staged_status = run_git_command_in_dir(repo_root, "diff", "--cached", "--name-only")
    if staged_status.returncode != 0:
        print_git_result(staged_status, print_fn)
        raise RuntimeError("Failed to inspect staged artifacts.")
    if not staged_status.stdout.strip():
        print_fn("No staged artifact changes detected.")
        return False

    commit_result = run_git_command_in_dir(repo_root, "commit", "-m", commit_message)
    if commit_result.returncode != 0:
        print_git_result(commit_result, print_fn)
        raise RuntimeError("Failed to commit local artifacts.")
    print_git_result(commit_result, print_fn)
    return True


def publish_artifacts_to_origin(
    *,
    repo_root: Path,
    artifact_files: tuple[Path, ...] | list[Path],
    commit_message: str,
    print_fn: PrintFn = print,
    max_attempts: int = 3,
) -> bool:
    artifact_files = tuple(artifact_files)
    with tempfile.TemporaryDirectory() as snapshot_root_raw:
        snapshot_root = Path(snapshot_root_raw)
        for path in artifact_files:
            _copy_if_exists(path, snapshot_root / path.relative_to(repo_root))

        for attempt in range(1, max_attempts + 1):
            fetch_result = run_git_command_in_dir(repo_root, "fetch", "origin", "main")
            if fetch_result.returncode != 0:
                print_git_result(fetch_result, print_fn)
                raise RuntimeError("Failed to fetch origin/main before publishing artifacts.")
            print_git_result(fetch_result, print_fn)

            with tempfile.TemporaryDirectory() as worktree_root_raw:
                worktree_root = Path(worktree_root_raw)
                publish_worktree = worktree_root / "publish"
                add_result = run_git_command_in_dir(
                    repo_root,
                    "worktree",
                    "add",
                    "--detach",
                    str(publish_worktree),
                    "origin/main",
                )
                if add_result.returncode != 0:
                    print_git_result(add_result, print_fn)
                    raise RuntimeError("Failed to create temporary publish worktree from origin/main.")
                try:
                    for path in artifact_files:
                        relative_path = path.relative_to(repo_root)
                        apply_local_artifact_to_repo(
                            publish_worktree / relative_path,
                            snapshot_root / relative_path,
                        )

                    changed = stage_and_commit_artifacts(
                        repo_root=publish_worktree,
                        artifact_files=[publish_worktree / path.relative_to(repo_root) for path in artifact_files],
                        commit_message=commit_message,
                        print_fn=print_fn,
                    )
                    if not changed:
                        for path in artifact_files:
                            _copy_if_exists(publish_worktree / path.relative_to(repo_root), path)
                        return True

                    push_result = run_git_command_in_dir(publish_worktree, "push", "origin", "HEAD:main")
                    if push_result.returncode == 0:
                        print_git_result(push_result, print_fn)
                        for path in artifact_files:
                            _copy_if_exists(publish_worktree / path.relative_to(repo_root), path)
                        return True
                    print_git_result(push_result, print_fn)
                finally:
                    remove_result = run_git_command_in_dir(repo_root, "worktree", "remove", "--force", str(publish_worktree))
                    if remove_result.returncode != 0:
                        print_git_result(remove_result, print_fn)
                        raise RuntimeError("Failed to remove the temporary publish worktree.")
            if attempt < max_attempts:
                print_fn("Retrying artifact publish on the latest origin/main.")
        raise RuntimeError("Failed to push local artifacts to origin/main.")
