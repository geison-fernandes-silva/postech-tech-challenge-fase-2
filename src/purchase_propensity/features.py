"""Feature preprocessing and model definitions."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from purchase_propensity.preprocess import TARGET_COLUMN


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the purchase target."""
    return frame.drop(columns=[TARGET_COLUMN]), frame[TARGET_COLUMN]


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    """Build numeric and categorical transformations from a feature frame."""
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()
    numeric_pipe = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


def build_models(features: pd.DataFrame, params: dict[str, Any], seed: int) -> dict[str, Pipeline]:
    """Return reproducible candidate classification pipelines."""
    preprocessor = build_preprocessor(features)
    logistic = LogisticRegression(random_state=seed, **params["logistic_regression"])
    forest = RandomForestClassifier(random_state=seed, n_jobs=-1, **params["random_forest"])
    return {
        "logistic_regression": Pipeline([("preprocess", preprocessor), ("model", logistic)]),
        "random_forest": Pipeline([("preprocess", preprocessor), ("model", forest)]),
    }
