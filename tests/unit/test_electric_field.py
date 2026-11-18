"""Tests for the ElectricField plugin."""

import json

import numpy as np
import strax
from instruction_helpers import make_instructions, write_csv

from ember.plugins.micro_physics import (
    CsvInput,
    DefaultSimulation,
    ElectricField,
    VolumeProperties,
    VolumeSelection,
)

PLUGINS = [CsvInput, VolumeProperties, VolumeSelection, DefaultSimulation, ElectricField]


def electric_field_values(tmp_path, efield_map: str = None) -> np.ndarray:
    config = {"input_file": write_csv(make_instructions(30), tmp_path)}
    if efield_map is not None:
        config["efield_map"] = efield_map
    context = strax.Context(storage=[], register=PLUGINS, config=config)
    return context.get_array("000000", ["interactions_in_roi", "electric_field_values"])


def test_default_map_gives_configured_constant_field(tmp_path):
    drift_field = strax.URLConfig.evaluate_dry("detector://drift_field")
    result = electric_field_values(tmp_path)
    assert len(result) == 30
    np.testing.assert_allclose(result["e_field"], drift_field)


def test_field_follows_map_and_negative_values_are_clipped(tmp_path):
    map_path = tmp_path / "field.json"
    map_data = {
        "coordinate_system": [["r", [0.0, 100.0, 2]], ["z", [-160.0, 0.0, 2]]],
        "map": [[-50.0, 110.0], [-50.0, 110.0]],
    }
    map_path.write_text(json.dumps(map_data))
    result = electric_field_values(tmp_path, f"ember_map://{map_path}")
    expected = np.clip(-50.0 + 160.0 * (result["z"] + 160.0) / 160.0, 0, None)
    np.testing.assert_allclose(result["e_field"], expected, rtol=1e-5, atol=1e-3)
    assert np.all(result["e_field"] >= 0)
