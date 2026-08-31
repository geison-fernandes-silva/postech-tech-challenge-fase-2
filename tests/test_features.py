"""Tests for model feature engineering."""

import pandas as pd

from purchase_propensity.features import build_models, split_features_target


def test_candidate_models_fit_and_predict() -> None:
    """Both candidate pipelines accept mixed tabular features."""
    frame = pd.DataFrame(
        {
            "PageValues": [0.0, 2.4, 0.0, 5.1, 0.5, 8.0],
            "VisitorType": ["New_Visitor", "Returning_Visitor"] * 3,
            "Revenue": [0, 1, 0, 1, 0, 1],
        }
    )
    features, target = split_features_target(frame)
    params = {
        "logistic_regression": {"C": 1.0, "max_iter": 100},
        "random_forest": {"n_estimators": 5, "max_depth": 2, "min_samples_leaf": 1},
    }

    models = build_models(features, params, seed=42)

    assert set(models) == {"logistic_regression", "random_forest"}
    assert models["logistic_regression"].fit(features, target).predict(features).shape == (6,)
