"""Helpers to build CSV instruction files for the ember unit tests."""

from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["eventid", "t", "x", "y", "z", "ed", "nestid", "A", "Z", "photons", "electrons"]


def make_instructions(number_of_events: int, seed: int = 0, energy: float = 40.0) -> pd.DataFrame:
    """Single-interaction beta events uniformly distributed in the TPC."""
    rng = np.random.default_rng(seed)
    radius = np.sqrt(rng.uniform(0, 60.0**2, number_of_events))
    angle = rng.uniform(-np.pi, np.pi, number_of_events)
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": radius * np.cos(angle),
            "y": radius * np.sin(angle),
            "z": rng.uniform(-145.0, -5.0, number_of_events),
            "ed": np.full(number_of_events, energy),
            "nestid": np.full(number_of_events, 8),
            "A": np.zeros(number_of_events, dtype=int),
            "Z": np.zeros(number_of_events, dtype=int),
            "photons": np.full(number_of_events, -1),
            "electrons": np.full(number_of_events, -1),
        }
    )


def write_csv(data_frame: pd.DataFrame, folder: Path, name: str = "instructions.csv") -> str:
    path = Path(folder) / name
    data_frame.to_csv(path, index=False)
    return str(path)
