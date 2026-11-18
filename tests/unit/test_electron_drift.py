"""Tests of the ElectronDrift plugin."""

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import COLUMNS, make_instructions, write_csv

import ember
from ember.electron_transport import (
    drift_velocity_liquid,
    longitudinal_diffusion_constant_liquid,
)


def detector_value(key: str):
    return strax.URLConfig.evaluate_dry(f"detector://{key}")


Z_GATE = detector_value("z_gate")
Z_CATHODE = detector_value("z_cathode")
DRIFT_TIME_GATE = detector_value("drift_time_gate")
DRIFT_VELOCITY = float(
    drift_velocity_liquid(
        detector_value("drift_field"),
        detector_value("liquid_temperature"),
        detector_value("gas_pressure"),
        detector_value("xenon_density_liquid"),
    )
)
DIFFUSION_LONGITUDINAL = float(
    longitudinal_diffusion_constant_liquid(detector_value("drift_field"))
)


def fixed_electron_instructions(
    number_of_events: int, electrons: int, position: tuple
) -> pd.DataFrame:
    """Events with a given number of electrons at one position (no photons)."""
    x_value, y_value, z_value = position
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": np.full(number_of_events, x_value),
            "y": np.full(number_of_events, y_value),
            "z": np.full(number_of_events, z_value),
            "ed": np.full(number_of_events, -1.0),
            "nestid": np.full(number_of_events, 8),
            "A": np.zeros(number_of_events, dtype=int),
            "Z": np.zeros(number_of_events, dtype=int),
            "photons": np.zeros(number_of_events, dtype=int),
            "electrons": np.full(number_of_events, electrons),
        }
    )[COLUMNS]


def make_context(folder, instructions: pd.DataFrame, **config) -> strax.Context:
    context = ember.ember_context(output_folder=folder / "data")
    context.set_config({"input_file": write_csv(instructions, folder), **config})
    return context


def expected_drift_time(z: np.ndarray) -> np.ndarray:
    return (Z_GATE - z) / DRIFT_VELOCITY + DRIFT_TIME_GATE


def expected_drift_time_spread(z: np.ndarray) -> np.ndarray:
    drift_time_below_gate = (Z_GATE - z) / DRIFT_VELOCITY
    velocity_gate = (detector_value("z_liquid_level") - Z_GATE) / DRIFT_TIME_GATE
    spread_squared = 2 * DIFFUSION_LONGITUDINAL * drift_time_below_gate / DRIFT_VELOCITY**2
    spread_squared += (
        2
        * detector_value("diffusion_constant_longitudinal_gate")
        * DRIFT_TIME_GATE
        / velocity_gate**2
    )
    return np.sqrt(spread_squared)


@pytest.fixture(scope="module")
def drift_plugin(tmp_path_factory):
    """A set-up ElectronDrift plugin."""
    folder = tmp_path_factory.mktemp("plugin")
    context = make_context(folder, make_instructions(5))
    return context.get_single_plugin("0", "drifted_electrons")


def interactions(x, y, z, electrons) -> np.ndarray:
    """Minimal ``interactions_in_roi`` array with the fields used by ElectronDrift."""
    data = np.zeros(
        len(x),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("electrons", np.int32),
        ]
        + strax.time_fields,
    )
    data["x"], data["y"], data["z"], data["electrons"] = x, y, z, electrons
    data["time"] = np.arange(len(x)) * 1000
    data["endtime"] = data["time"]
    return data


def test_zero_electrons_stay_zero(tmp_path):
    instructions = fixed_electron_instructions(10, electrons=0, position=(0.0, 0.0, -50.0))
    instructions["photons"] = 100
    drifted = make_context(tmp_path, instructions).get_array("0", "drifted_electrons")
    assert len(drifted) == 10
    assert np.all(drifted["n_electron_interface"] == 0)
    assert np.all(drifted["drift_time_mean"] == 0)


def test_no_electrons_survive_below_the_cathode(tmp_path):
    position = (10.0, 0.0, Z_CATHODE - 2.0)
    instructions = fixed_electron_instructions(10, electrons=500, position=position)
    drifted = make_context(tmp_path, instructions).get_array("0", "drifted_electrons")
    assert np.all(drifted["n_electron_interface"] == 0)


def test_without_lifetime_all_electrons_reach_the_surface(tmp_path):
    instructions = fixed_electron_instructions(10, electrons=500, position=(20.0, 10.0, -100.0))
    context = make_context(tmp_path, instructions, electron_lifetime_liquid=0)
    drifted = context.get_array("0", "drifted_electrons")
    assert np.all(drifted["n_electron_interface"] == 500)


def test_lifetime_losses_follow_the_exponential(tmp_path):
    number_of_events, electrons, lifetime = 200, 1000, 1e6
    position = (0.0, 0.0, -100.0)
    instructions = fixed_electron_instructions(number_of_events, electrons, position)
    context = make_context(tmp_path, instructions, electron_lifetime_liquid=lifetime)
    drifted = context.get_array("0", "drifted_electrons")
    probability = np.exp(-expected_drift_time(position[2]) / lifetime)
    expected = electrons * probability
    sigma_of_mean = np.sqrt(electrons * probability * (1 - probability) / number_of_events)
    assert abs(drifted["n_electron_interface"].mean() - expected) < 5 * sigma_of_mean


def test_observed_position_is_the_true_position(tmp_path):
    instructions = fixed_electron_instructions(5, electrons=100, position=(12.5, -30.0, -60.0))
    drifted = make_context(tmp_path, instructions).get_array("0", "drifted_electrons")
    np.testing.assert_allclose(drifted["x_obs"], 12.5)
    np.testing.assert_allclose(drifted["y_obs"], -30.0)
    np.testing.assert_allclose(drifted["z_obs"], -60.0)


def test_beta_events_run_end_to_end(tmp_path):
    context = make_context(tmp_path, make_instructions(50))
    data = context.get_array("0", ["microphysics_summary", "drifted_electrons"])
    assert np.all(data["n_electron_interface"] <= data["electrons"])
    np.testing.assert_allclose(data["drift_time_mean"], expected_drift_time(data["z"]), atol=1)


@settings(deadline=None, max_examples=50)
@given(
    radius=st.floats(min_value=0.0, max_value=66.0),
    z=st.floats(min_value=-148.0, max_value=-0.01),
)
def test_drift_time_parameters_follow_the_drift_map(drift_plugin, radius, z):
    mean, spread = drift_plugin.get_s2_drift_time_params(np.array([[radius, z]]))
    assert mean[0] == pytest.approx(expected_drift_time(z), rel=1e-6)
    assert spread[0] == pytest.approx(expected_drift_time_spread(z), rel=1e-6)


@settings(deadline=None, max_examples=30)
@given(z_values=st.lists(st.floats(-148.0, -0.01), min_size=2, max_size=20))
def test_deeper_interactions_drift_longer_and_spread_more(drift_plugin, z_values):
    z_values = np.sort(np.array(z_values))[::-1]
    positions = np.column_stack([np.full(len(z_values), 30.0), z_values])
    mean, spread = drift_plugin.get_s2_drift_time_params(positions)
    assert np.all(np.diff(mean) >= 0)
    assert np.all(np.diff(spread) >= 0)
    assert np.all(mean >= DRIFT_TIME_GATE)


@settings(deadline=None, max_examples=50)
@given(
    electrons=st.lists(st.integers(0, 5000), min_size=1, max_size=20),
    z=st.floats(-154.0, -0.01),
    radius=st.floats(0.0, 66.0),
)
def test_electrons_at_the_interface_are_between_zero_and_the_produced_electrons(
    drift_plugin, electrons, z, radius
):
    number = len(electrons)
    data = interactions(np.full(number, radius), np.zeros(number), np.full(number, z), electrons)
    result = drift_plugin.compute(data)
    assert np.all(result["n_electron_interface"] >= 0)
    assert np.all(result["n_electron_interface"] <= np.array(electrons))
    np.testing.assert_array_equal(result["time"], data["time"])


@pytest.mark.parametrize(
    "config, message",
    [({"drift_time_gate": 0.0}, "drift_time_gate"), ({"z_liquid_level": -1.0}, "liquid surface")],
)
def test_wrong_gate_configuration_raises(tmp_path, config, message):
    context = make_context(tmp_path, make_instructions(5), **config)
    with pytest.raises(ValueError, match=message):
        context.get_single_plugin("0", "drifted_electrons")
