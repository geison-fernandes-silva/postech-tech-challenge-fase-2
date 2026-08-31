"""Train, evaluate, track, and register purchase propensity models."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from purchase_propensity.config import Settings, ensure_parent, get_settings
from purchase_propensity.features import build_models, split_features_target


@dataclass
class CandidateResult:
    """Tracked outcome for a trained model candidate."""

    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    run_id: str


def parse_args() -> argparse.Namespace:
    """Parse paths used by the DVC training stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    return parser.parse_args()


def load_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    """Load training parameters from the versioned YAML file."""
    with path.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def configure_mlflow(settings: Settings) -> None:
    """Configure local or remote MLflow tracking."""
    mlflow.set_tracking_uri(settings.tracking_uri)
    mlflow.set_experiment(settings.experiment_name)


def calculate_metrics(target: pd.Series, probabilities: Any) -> dict[str, float]:
    """Calculate binary classification metrics from predicted probabilities."""
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(target, probabilities),
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }


def log_candidate(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> CandidateResult:
    """Train one candidate and log its artifacts and metrics to MLflow."""
    trained = clone(pipeline).fit(x_train, y_train)
    probabilities = trained.predict_proba(x_test)[:, 1]
    metrics = calculate_metrics(y_test, probabilities)
    input_example = x_train.head(5)
    signature = infer_signature(input_example, trained.predict(input_example))
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("candidate", name)
        mlflow.log_params(trained.named_steps["model"].get_params())
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            trained,
            artifact_path="model",
            input_example=input_example,
            signature=signature,
        )
        return CandidateResult(name, trained, metrics, run.info.run_id)


def register_best_model(result: CandidateResult, settings: Settings) -> str:
    """Register the strongest candidate in the MLflow Model Registry."""
    client = MlflowClient()
    try:
        client.get_registered_model(settings.model_name)
    except mlflow.exceptions.MlflowException:
        client.create_registered_model(settings.model_name)
    version = client.create_model_version(
        settings.model_name, f"runs:/{result.run_id}/model", result.run_id
    )
    if hasattr(client, "set_registered_model_alias"):
        client.set_registered_model_alias(settings.model_name, "champion", version.version)
    return str(version.version)


def select_best(results: list[CandidateResult]) -> CandidateResult:
    """Return the candidate with the highest validation ROC-AUC."""
    return max(results, key=lambda result: result.metrics["roc_auc"])


def save_outputs(
    result: CandidateResult, version: str, model_path: Path, metrics_path: Path
) -> None:
    """Persist the selected model and a DVC-readable metrics report."""
    ensure_parent(model_path)
    ensure_parent(metrics_path)
    joblib.dump(result.pipeline, model_path)
    payload = {"best_model": result.name, "model_registry_version": version, **result.metrics}
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train(input_path: Path, model_path: Path, metrics_path: Path) -> CandidateResult:
    """Execute the complete reproducible training workflow."""
    settings = get_settings()
    params = load_params()
    frame = pd.read_csv(input_path)
    features, target = split_features_target(frame)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=params["base"]["test_size"],
        random_state=settings.random_seed,
        stratify=target,
    )
    configure_mlflow(settings)
    results = [
        log_candidate(name, model, x_train, y_train, x_test, y_test)
        for name, model in build_models(features, params["model"], settings.random_seed).items()
    ]
    best = select_best(results)
    save_outputs(best, register_best_model(best, settings), model_path, metrics_path)
    return best


def main() -> None:
    """Run training from the command line."""
    args = parse_args()
    result = train(args.input, args.model_output, args.metrics_output)
    print(f"Best model: {result.name} | ROC-AUC: {result.metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
