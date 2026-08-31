"""Prepare the source dataset for reproducible model training."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from purchase_propensity.config import ensure_parent

TARGET_COLUMN = "Revenue"


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows and normalize the binary target."""
    clean = frame.drop_duplicates().copy()
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype(int)
    return clean


def preprocess_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Read, clean, and persist the prepared dataset."""
    prepared = clean_dataset(pd.read_csv(input_path))
    ensure_parent(output_path)
    prepared.to_csv(output_path, index=False)
    return prepared


def main() -> None:
    """Run preprocessing from the command line."""
    args = parse_args()
    preprocess_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
