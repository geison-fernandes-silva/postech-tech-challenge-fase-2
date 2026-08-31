"""Central configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Runtime settings loaded from environment variables."""

    tracking_uri: str
    experiment_name: str
    model_name: str
    random_seed: int


def get_settings() -> Settings:
    """Return application settings with deterministic defaults."""
    load_dotenv()
    return Settings(
        tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"),
        experiment_name=os.getenv("MLFLOW_EXPERIMENT_NAME", "purchase-propensity"),
        model_name=os.getenv("MODEL_NAME", "purchase-propensity-classifier"),
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
    )


def ensure_parent(path: Path) -> None:
    """Create the parent folder for a file path when necessary."""
    path.parent.mkdir(parents=True, exist_ok=True)
