from __future__ import annotations

import mlflow

from main import TournamentPyFuncModel


mlflow.models.set_model(TournamentPyFuncModel())
