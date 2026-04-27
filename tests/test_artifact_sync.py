from __future__ import annotations

import json
import shutil
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import artifact_sync


ROOT = Path(__file__).resolve().parents[1]


def temporary_directory_factory(root: Path):
    counter = {"value": 0}

    @contextmanager
    def fake_temporary_directory():
        counter["value"] += 1
        path = root / f"tmpdir-{counter['value']}"
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        try:
            yield str(path)
        finally:
            shutil.rmtree(path, ignore_errors=True)

    return fake_temporary_directory


class ArtifactSyncTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = ROOT / "temp" / f"artifact-sync-tests-{self._testMethodName}"
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        if self.temp_root.exists():
            shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_merge_csv_contents_remote_wins_duplicates_and_appends_local_only(self) -> None:
        remote_text = (
            "timestamp,value\n"
            "2026-04-27T10:00:00+00:00,remote-1\n"
            "2026-04-27T11:00:00+00:00,remote-2\n"
        )
        local_text = (
            "timestamp,value\n"
            "2026-04-27T10:00:00+00:00,local-duplicate\n"
            "2026-04-27T12:00:00+00:00,local-only\n"
        )

        merged = artifact_sync.merge_csv_contents(remote_text, local_text)

        self.assertIn("2026-04-27T10:00:00+00:00,remote-1", merged)
        self.assertNotIn("local-duplicate", merged)
        self.assertIn("2026-04-27T12:00:00+00:00,local-only", merged)
        self.assertLess(
            merged.index("2026-04-27T11:00:00+00:00,remote-2"),
            merged.index("2026-04-27T12:00:00+00:00,local-only"),
        )

    def test_merge_json_contents_preserves_remote_and_fills_missing(self) -> None:
        remote_text = json.dumps(
            {
                "status": "success",
                "probability_up": 0.7,
                "workflow_name": "",
                "nested": {"keep": "remote"},
            }
        )
        local_text = json.dumps(
            {
                "status": "failed",
                "probability_up": 0.2,
                "workflow_name": "hourly24",
                "error": "boom",
                "nested": {"keep": "local", "extra": "value"},
            }
        )

        merged = json.loads(artifact_sync.merge_json_contents(remote_text, local_text))

        self.assertEqual(merged["status"], "success")
        self.assertEqual(merged["probability_up"], 0.7)
        self.assertEqual(merged["workflow_name"], "hourly24")
        self.assertEqual(merged["error"], "boom")
        self.assertEqual(merged["nested"]["keep"], "remote")
        self.assertEqual(merged["nested"]["extra"], "value")

    def test_sync_artifacts_from_remote_merges_artifacts_without_touching_unrelated_files(self) -> None:
        repo = self.temp_root / "sync-repo"
        repo.mkdir(parents=True, exist_ok=True)
        history_path = repo / "history.csv"
        notes_path = repo / "notes.txt"
        history_path.write_text(
            "timestamp,value\n"
            "2026-04-27T10:00:00+00:00,local-duplicate\n"
            "2026-04-27T12:00:00+00:00,local-only\n",
            encoding="utf-8",
            newline="\n",
        )
        notes_path.write_text("keep me changed locally\n", encoding="utf-8")

        with patch.object(
            artifact_sync,
            "run_git_command_in_dir",
            return_value=SimpleNamespace(returncode=0, stdout="", stderr=""),
        ) as git_mock, patch.object(
            artifact_sync,
            "fetch_remote_file_bytes",
            return_value=(
                "timestamp,value\n"
                "2026-04-27T10:00:00+00:00,remote\n"
                "2026-04-27T11:00:00+00:00,remote-new\n"
            ).encode("utf-8"),
        ):
            artifact_sync.sync_artifacts_from_remote(
                repo_root=repo,
                artifact_files=(history_path,),
                print_fn=lambda _: None,
            )

        git_mock.assert_called_once_with(repo, "fetch", "origin", "main")
        merged = history_path.read_text(encoding="utf-8")
        self.assertIn("2026-04-27T10:00:00+00:00,remote", merged)
        self.assertIn("2026-04-27T11:00:00+00:00,remote-new", merged)
        self.assertIn("2026-04-27T12:00:00+00:00,local-only", merged)
        self.assertEqual(notes_path.read_text(encoding="utf-8"), "keep me changed locally\n")

    def test_publish_artifacts_to_origin_merges_remote_and_local_rows(self) -> None:
        repo = self.temp_root / "publish-repo"
        repo.mkdir(parents=True, exist_ok=True)
        history_path = repo / "history.csv"
        code_path = repo / "code.py"
        history_path.write_text(
            "timestamp,value\n"
            "2026-04-27T10:00:00+00:00,local-duplicate\n"
            "2026-04-27T12:00:00+00:00,local-only\n",
            encoding="utf-8",
            newline="\n",
        )
        code_path.write_text("print('local code changed')\n", encoding="utf-8")
        recorded_worktrees: list[Path] = []

        def fake_run_git(repo_dir: Path, *args: str) -> SimpleNamespace:
            if args[:3] == ("worktree", "add", "--detach"):
                publish_worktree = Path(args[3])
                publish_worktree.mkdir(parents=True, exist_ok=True)
                remote_history = publish_worktree / "history.csv"
                remote_history.write_text(
                    "timestamp,value\n"
                    "2026-04-27T10:00:00+00:00,remote\n"
                    "2026-04-27T11:00:00+00:00,remote-new\n",
                    encoding="utf-8",
                    newline="\n",
                )
                recorded_worktrees.append(publish_worktree)
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args[:3] == ("worktree", "remove", "--force"):
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args == ("fetch", "origin", "main"):
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args == ("push", "origin", "HEAD:main"):
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(artifact_sync.tempfile, "TemporaryDirectory", side_effect=temporary_directory_factory(self.temp_root)), \
             patch.object(artifact_sync, "run_git_command_in_dir", side_effect=fake_run_git) as git_mock, patch.object(
            artifact_sync,
            "stage_and_commit_artifacts",
            return_value=True,
        ) as commit_mock:
            published = artifact_sync.publish_artifacts_to_origin(
                repo_root=repo,
                artifact_files=(history_path,),
                commit_message="sync artifacts",
                print_fn=lambda _: None,
            )

        self.assertTrue(published)
        self.assertTrue(recorded_worktrees)
        commit_mock.assert_called_once()
        self.assertEqual(git_mock.call_args_list[0].args[1:], ("fetch", "origin", "main"))
        merged_local = history_path.read_text(encoding="utf-8")
        self.assertIn("2026-04-27T10:00:00+00:00,remote", merged_local)
        self.assertIn("2026-04-27T11:00:00+00:00,remote-new", merged_local)
        self.assertIn("2026-04-27T12:00:00+00:00,local-only", merged_local)
        self.assertEqual(code_path.read_text(encoding="utf-8"), "print('local code changed')\n")

    def test_publish_artifacts_to_origin_retries_after_push_race(self) -> None:
        repo = self.temp_root / "retry-repo"
        repo.mkdir(parents=True, exist_ok=True)
        history_path = repo / "history.csv"
        history_path.write_text(
            "timestamp,value\n2026-04-27T12:00:00+00:00,local-only\n",
            encoding="utf-8",
            newline="\n",
        )
        push_attempts: list[int] = []

        def fake_run_git(repo_dir: Path, *args: str) -> SimpleNamespace:
            if args[:3] == ("worktree", "add", "--detach"):
                publish_worktree = Path(args[3])
                publish_worktree.mkdir(parents=True, exist_ok=True)
                (publish_worktree / "history.csv").write_text(
                    "timestamp,value\n2026-04-27T11:00:00+00:00,remote-new\n",
                    encoding="utf-8",
                    newline="\n",
                )
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args[:3] == ("worktree", "remove", "--force"):
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args == ("fetch", "origin", "main"):
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            if args == ("push", "origin", "HEAD:main"):
                push_attempts.append(1)
                if len(push_attempts) == 1:
                    return SimpleNamespace(returncode=1, stdout="", stderr="race")
                return SimpleNamespace(returncode=0, stdout="", stderr="")
            return SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(artifact_sync.tempfile, "TemporaryDirectory", side_effect=temporary_directory_factory(self.temp_root)), \
             patch.object(artifact_sync, "run_git_command_in_dir", side_effect=fake_run_git), patch.object(
            artifact_sync,
            "stage_and_commit_artifacts",
            return_value=True,
        ):
            published = artifact_sync.publish_artifacts_to_origin(
                repo_root=repo,
                artifact_files=(history_path,),
                commit_message="sync artifacts",
                print_fn=lambda _: None,
            )

        self.assertTrue(published)
        self.assertEqual(len(push_attempts), 2)
        merged_local = history_path.read_text(encoding="utf-8")
        self.assertIn("2026-04-27T11:00:00+00:00,remote-new", merged_local)
        self.assertIn("2026-04-27T12:00:00+00:00,local-only", merged_local)


if __name__ == "__main__":
    unittest.main()
