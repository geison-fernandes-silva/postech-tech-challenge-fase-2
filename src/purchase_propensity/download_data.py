"""Download the UCI Online Shoppers Purchasing Intention dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd

from purchase_propensity.config import ensure_parent

DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv"
)


def parse_args() -> argparse.Namespace:
    """Parse the destination for the raw dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Create the documented deterministic fallback dataset.",
    )
    return parser.parse_args()


def write_synthetic_fallback(output: Path, seed: int = 42, rows: int = 12_330) -> None:
    """Create a deterministic e-commerce navigation dataset for offline execution."""
    generator = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "Administrative": generator.poisson(2, rows),
            "Administrative_Duration": generator.gamma(2, 25, rows),
            "Informational": generator.poisson(1, rows),
            "Informational_Duration": generator.gamma(1.5, 20, rows),
            "ProductRelated": generator.poisson(30, rows),
            "ProductRelated_Duration": generator.gamma(3, 100, rows),
            "BounceRates": generator.beta(2, 20, rows),
            "ExitRates": generator.beta(3, 16, rows),
            "PageValues": generator.gamma(1.2, 9, rows),
            "SpecialDay": generator.choice([0.0, 0.2, 0.4, 0.6, 0.8], rows),
            "Month": generator.choice(
                ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rows
            ),
            "OperatingSystems": generator.integers(1, 9, rows),
            "Browser": generator.integers(1, 14, rows),
            "Region": generator.integers(1, 10, rows),
            "TrafficType": generator.integers(1, 21, rows),
            "VisitorType": generator.choice(
                ["Returning_Visitor", "New_Visitor", "Other"], rows, p=[0.75, 0.22, 0.03]
            ),
            "Weekend": generator.choice([True, False], rows, p=[0.24, 0.76]),
        }
    )
    signal = (
        -3.0
        + 0.11 * frame["ProductRelated"]
        + 0.09 * frame["PageValues"]
        - 7 * frame["BounceRates"]
    )
    probability = 1 / (1 + np.exp(-signal))
    frame["Revenue"] = generator.binomial(1, probability).astype(bool)
    frame.to_csv(output, index=False)


def download_dataset(output: Path, offline: bool) -> None:
    """Download UCI data or create the documented offline fallback."""
    ensure_parent(output)
    if offline:
        write_synthetic_fallback(output)
        return
    try:
        with urlopen(DATASET_URL, timeout=20) as response:
            output.write_bytes(response.read())
    except OSError:
        write_synthetic_fallback(output)


def main() -> None:
    """Run the dataset download command."""
    args = parse_args()
    download_dataset(args.output, args.offline)


if __name__ == "__main__":
    main()
