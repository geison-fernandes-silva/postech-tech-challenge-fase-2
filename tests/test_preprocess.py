"""Tests for deterministic data preparation."""

import pandas as pd

from purchase_propensity.preprocess import TARGET_COLUMN, clean_dataset


def test_clean_dataset_removes_duplicates_and_casts_target() -> None:
    """Prepared data has one row per observation and an integer target."""
    frame = pd.DataFrame({"Administrative": [0, 0], TARGET_COLUMN: [True, True]})

    clean = clean_dataset(frame)

    assert len(clean) == 1
    assert clean[TARGET_COLUMN].dtype.kind in {"i", "u"}
