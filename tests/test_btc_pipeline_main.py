from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from src.btc_pipeline import main as tournament


class ChampionPromotionTests(unittest.TestCase):
    def test_promote_champion_logs_pyfunc_model_with_artifact_path(self) -> None:
        candidate = SimpleNamespace(name="MLPClassifier", family="mlp_sklearn")
        winner = {
            "candidate": candidate,
            "accuracy": 0.75,
            "f1": 0.8,
            "next_probability": 0.6,
        }
        client = MagicMock()
        active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-123"))
        version = SimpleNamespace(version="42")

        with tempfile.TemporaryDirectory() as temp_dir, \
            patch.object(tournament.mlflow, "active_run", return_value=active_run), \
            patch.object(tournament.mlflow, "set_tags"), \
            patch.object(tournament.mlflow, "log_metrics"), \
            patch.object(tournament.mlflow.pyfunc, "log_model") as log_model, \
            patch.object(
                tournament,
                "get_promotion_payload",
                return_value=(
                    {
                        "package_dir": Path(temp_dir),
                        "input_example": pd.DataFrame({"close": [1.0]}),
                        "signature": None,
                        "pip_requirements": ["mlflow==2.22.1"],
                    },
                    False,
                ),
            ), \
            patch.object(tournament, "find_logged_model_version", return_value=version):
            promoted = tournament.promote_champion(
                client=client,
                registered_model_name="btc-test-model",
                winner=winner,
                validation_start="2026-04-27T00:00:00+00:00",
                validation_end="2026-04-27T23:00:00+00:00",
                feature_rows=pd.DataFrame({"close": [1.0]}),
                alias="champion",
            )

        self.assertEqual(promoted, "42")
        log_model.assert_called_once()
        self.assertIn("artifact_path", log_model.call_args.kwargs)
        self.assertNotIn("name", log_model.call_args.kwargs)
        self.assertEqual(
            log_model.call_args.kwargs["artifact_path"],
            f"{tournament.MODEL_ARTIFACT_NAME}_mlp_sklearn",
        )


if __name__ == "__main__":
    unittest.main()
