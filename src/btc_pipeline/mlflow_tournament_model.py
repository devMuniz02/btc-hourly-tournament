from __future__ import annotations

import mlflow

from src.btc_pipeline.main import TournamentPyFuncModel


mlflow.models.set_model(TournamentPyFuncModel())
