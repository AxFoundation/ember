"""Tests for VolumeProperties and the selection plugins."""

import numpy as np
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import make_instructions, write_csv

from ember.common import VOLUMES_IDS
from ember.plugins.micro_physics import (
    CsvInput,
    DefaultSimulation,
    VolumeProperties,
    VolumeSelection,
)
from ember.plugins.micro_physics.volume_properties import in_cylinder

DETECTOR = {
    name: strax.URLConfig.evaluate_dry(f"detector://{name}")
    for name in [
        "tpc_radius",
        "z_gate",
        "z_cathode",
        "z_bottom_pmts",
        "z_liquid_level",
        "z_top_pmts",
        "xenon_density_liquid",
        "xenon_density_gas",
    ]
}
PLUGINS = [CsvInput, VolumeProperties, VolumeSelection, DefaultSimulation]


def context_for_positions(tmp_path, positions: list) -> strax.Context:
    data_frame = make_instructions(len(positions))
    data_frame[["x", "y", "z"]] = np.array(positions, dtype=float)
    path = write_csv(data_frame, tmp_path)
    return strax.Context(storage=[], register=PLUGINS, config={"input_file": path})


OFFSET = 0.001  # cm, well above the float32 resolution of the positions (~1e-5 cm)
BOUNDARY_CASES = [
    ([0.0, 0.0, DETECTOR["z_cathode"] + OFFSET], "tpc"),
    ([0.0, 0.0, DETECTOR["z_gate"] - OFFSET], "tpc"),
    ([0.0, 0.0, DETECTOR["z_gate"] + OFFSET], "undefined"),
    ([0.0, 0.0, DETECTOR["z_cathode"] - OFFSET], "below_cathode"),
    ([0.0, 0.0, DETECTOR["z_bottom_pmts"] + OFFSET], "below_cathode"),
    ([0.0, 0.0, DETECTOR["z_bottom_pmts"] - OFFSET], "undefined"),
    ([0.0, 0.0, DETECTOR["z_liquid_level"] + OFFSET], "gas_phase"),
    ([0.0, 0.0, DETECTOR["z_top_pmts"] + OFFSET], "undefined"),
    ([DETECTOR["tpc_radius"] + OFFSET, 0.0, -50.0], "undefined"),
    ([DETECTOR["tpc_radius"] - OFFSET, 0.0, -50.0], "tpc"),
]


def test_volume_ids_next_to_boundaries(tmp_path):
    positions = [position for position, _ in BOUNDARY_CASES]
    result = context_for_positions(tmp_path, positions).get_array(
        "000000", ["clustered_interactions", "volume_properties"]
    )
    result_in_input_order = result[np.argsort(result["eventid"])]
    for (position, volume), vol_id in zip(BOUNDARY_CASES, result_in_input_order["vol_id"]):
        assert vol_id == VOLUMES_IDS[volume], f"position {position}"


def test_in_cylinder_boundaries_are_lower_inclusive_upper_exclusive():
    z = np.array([-10.0, 0.0, -5.0, -5.0])
    x = np.array([0.0, 0.0, 3.0, 2.999])
    mask = in_cylinder(x, np.zeros(4), z, -10.0, 0.0, 3.0)
    np.testing.assert_array_equal(mask, [True, False, False, True])


def test_densities(tmp_path):
    positions = [[0.0, 0.0, -50.0], [0.0, 0.0, DETECTOR["z_liquid_level"] + 1.0]]
    result = context_for_positions(tmp_path, positions).get_array("000000", "volume_properties")
    densities = dict(zip(result["vol_id"], result["xe_density"]))
    assert densities[VOLUMES_IDS["tpc"]] == pytest.approx(DETECTOR["xenon_density_liquid"])
    assert densities[VOLUMES_IDS["gas_phase"]] == pytest.approx(DETECTOR["xenon_density_gas"])


def test_default_selection_keeps_tpc_and_below_cathode(tmp_path):
    positions = [position for position, _ in BOUNDARY_CASES]
    result = context_for_positions(tmp_path, positions).get_array("000000", "interactions_in_roi")
    accepted = [volume for _, volume in BOUNDARY_CASES if volume in ("tpc", "below_cathode")]
    assert len(result) == len(accepted)
    assert set(result["vol_id"]) == {VOLUMES_IDS["tpc"], VOLUMES_IDS["below_cathode"]}


def test_selection_logic_expression():
    interactions = np.zeros(4, dtype=[("first", bool), ("second", bool)])
    interactions["first"] = [True, True, False, False]
    interactions["second"] = [True, False, True, False]
    np.testing.assert_array_equal(
        DefaultSimulation.evaluate_selection_logic(interactions, "first & ~second"),
        [False, True, False, False],
    )


@settings(deadline=None, max_examples=200)
@given(
    x=st.floats(-80, 80),
    y=st.floats(-80, 80),
    z=st.floats(-160, 10),
)
def test_volumes_do_not_overlap(x, y, z):
    position = (np.array([x]), np.array([y]), np.array([z]))
    volume_limits = [
        (DETECTOR["z_cathode"], DETECTOR["z_gate"]),
        (DETECTOR["z_bottom_pmts"], DETECTOR["z_cathode"]),
        (DETECTOR["z_liquid_level"], DETECTOR["z_top_pmts"]),
    ]
    number_of_volumes = sum(
        in_cylinder(*position, min_z, max_z, DETECTOR["tpc_radius"])[0]
        for min_z, max_z in volume_limits
    )
    assert number_of_volumes <= 1
