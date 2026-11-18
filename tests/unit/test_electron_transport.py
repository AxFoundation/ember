"""Tests of the electron transport helpers (NEST drift velocities, diffusion, path averages)."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.integrate import quad
from strax import URLConfig

import ember  # noqa: F401  (registers the protocols)
from ember.electron_transport import (
    DRIFT_PROPERTY_MAP_NAMES,
    build_drift_property_map,
    anode_field_profile,
    build_luminescence_timing_map,
    drift_time_above_gate,
    drift_velocity_gas,
    drift_velocity_liquid,
    electroluminescence_duration,
    electroluminescence_yield_per_length,
    longitudinal_diffusion_constant_liquid,
    path_averaged_drift_properties,
    transverse_diffusion_constant_liquid,
    xenon_number_density,
)
from ember.maps import RegularGridMap

TEMPERATURE = 177.15
PRESSURE = 1.89
DENSITY = 2.81


def detector_value(key: str):
    return URLConfig.evaluate_dry(f"detector://{key}")


def nest_conditions() -> dict:
    return dict(temperature=TEMPERATURE, pressure=PRESSURE, density=DENSITY)


def test_drift_velocity_liquid_at_23_v_per_cm():
    assert drift_velocity_liquid(23.0, **nest_conditions()) == pytest.approx(6.261e-5, rel=1e-3)


def test_drift_velocity_keeps_the_shape_of_the_field():
    field = np.full((3, 4), 100.0)
    velocity = drift_velocity_liquid(field, **nest_conditions())
    assert velocity.shape == (3, 4)
    assert np.all(velocity == velocity[0, 0])


def test_drift_velocity_gas_is_faster_than_liquid():
    velocity_gas = drift_velocity_gas(5000.0, TEMPERATURE, PRESSURE, 0.0177)
    velocity_liquid = drift_velocity_liquid(2900.0, **nest_conditions())
    assert velocity_gas > velocity_liquid > 0


@pytest.mark.parametrize(
    "field, transverse, longitudinal",
    [(23.0, 49.9966, 49.7703), (2900.0, 62.1201, 9.14665)],
)
def test_diffusion_constants_at_reference_fields(field, transverse, longitudinal):
    assert transverse_diffusion_constant_liquid(field) == pytest.approx(transverse * 1e-9, rel=1e-5)
    assert longitudinal_diffusion_constant_liquid(field) == pytest.approx(
        longitudinal * 1e-9, rel=1e-5
    )


@settings(deadline=None, max_examples=30)
@given(
    field=st.floats(min_value=10.0, max_value=5000.0),
    number_of_points=st.integers(min_value=2, max_value=50),
)
def test_constant_field_gives_local_values(field, number_of_points):
    z_values = np.linspace(-150.0, 0.5, number_of_points)
    properties = path_averaged_drift_properties(
        z_values, np.full(number_of_points, field), 0.0, **nest_conditions()
    )
    np.testing.assert_allclose(
        properties["drift_speed_map"], drift_velocity_liquid(field, **nest_conditions()), rtol=1e-9
    )
    np.testing.assert_allclose(
        properties["diffusion_longitudinal_map"],
        longitudinal_diffusion_constant_liquid(field),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        properties["diffusion_transverse_map"],
        transverse_diffusion_constant_liquid(field),
        rtol=1e-9,
    )


def test_linear_field_matches_numerical_integration():
    z_values = np.linspace(-150.0, 0.0, 3001)
    field_values = 20.0 + 0.5 * (z_values + 150.0)
    properties = path_averaged_drift_properties(z_values, field_values, 0.0, **nest_conditions())

    def field_at(z: float) -> float:
        return 20.0 + 0.5 * (z + 150.0)

    def inverse_velocity(z: float) -> float:
        return 1.0 / drift_velocity_liquid(field_at(z), **nest_conditions())

    def longitudinal_integrand(z: float) -> float:
        velocity = drift_velocity_liquid(field_at(z), **nest_conditions())
        return longitudinal_diffusion_constant_liquid(field_at(z)) / velocity**3

    z_start = -100.0
    index = np.argmin(np.abs(z_values - z_start))
    drift_time = quad(inverse_velocity, z_start, 0.0)[0]
    time_spread_squared = 2 * quad(longitudinal_integrand, z_start, 0.0)[0]

    drift_speed = properties["drift_speed_map"][index]
    diffusion_longitudinal = properties["diffusion_longitudinal_map"][index]
    assert -z_start / drift_speed == pytest.approx(drift_time, rel=1e-5)
    assert 2 * diffusion_longitudinal * drift_time / drift_speed**2 == pytest.approx(
        time_spread_squared, rel=1e-4
    )


@settings(deadline=None, max_examples=30)
@given(
    field_values=arrays(
        np.float64, st.integers(min_value=2, max_value=30), elements=st.floats(10.0, 5000.0)
    )
)
def test_drift_time_and_spreads_grow_with_depth(field_values):
    z_values = np.linspace(-150.0, 0.0, len(field_values))
    properties = path_averaged_drift_properties(z_values, field_values, 0.0, **nest_conditions())
    drift_length = -z_values
    drift_time = drift_length / properties["drift_speed_map"]
    time_spread_squared = 2 * properties["diffusion_longitudinal_map"] * drift_time
    time_spread_squared /= properties["drift_speed_map"] ** 2
    transverse_spread_squared = 2 * properties["diffusion_transverse_map"] * drift_time

    for quantity in (drift_time, time_spread_squared, transverse_spread_squared):
        assert np.all(np.diff(quantity) <= 1e-9 * np.max(quantity))
        assert quantity[-1] == 0


def test_positions_above_the_gate_get_the_gate_values():
    z_values = np.linspace(-10.0, 0.5, 22)
    field_values = np.linspace(20.0, 200.0, 22)
    properties = path_averaged_drift_properties(z_values, field_values, 0.0, **nest_conditions())
    gate_field = np.interp(0.0, z_values, field_values)
    is_above_gate = z_values >= 0.0
    np.testing.assert_allclose(
        properties["drift_speed_map"][is_above_gate],
        drift_velocity_liquid(gate_field, **nest_conditions()),
    )


def test_gate_outside_the_z_range_raises():
    with pytest.raises(ValueError, match="gate"):
        path_averaged_drift_properties(
            np.linspace(-10.0, -1.0, 5), np.full(5, 23.0), 0.0, **nest_conditions()
        )


def test_decreasing_z_values_raise():
    with pytest.raises(ValueError, match="increasing"):
        path_averaged_drift_properties(
            np.linspace(1.0, -10.0, 5), np.full(5, 23.0), 0.0, **nest_conditions()
        )


def test_build_drift_property_map_shape_and_values():
    field_map = RegularGridMap(
        {
            "coordinate_system": [["r", [0.0, 66.4, 2]], ["z", [-155.0, 0.5, 2]]],
            "map": np.full((2, 2), 23.0),
        }
    )
    r_values = np.linspace(0.0, 66.4, 3)
    z_values = np.linspace(-155.0, 0.5, 40)
    map_data = build_drift_property_map(field_map, r_values, z_values, 0.0, **nest_conditions())

    assert map_data["coordinate_system"] == [["r", [0.0, 66.4, 3]], ["z", [-155.0, 0.5, 40]]]
    for name in DRIFT_PROPERTY_MAP_NAMES:
        assert map_data[name].shape == (3, 40)
    loaded = RegularGridMap(map_data)
    assert loaded([[30.0, -50.0]], map_name="drift_speed_map")[0] == pytest.approx(
        drift_velocity_liquid(23.0, **nest_conditions())
    )


def test_uneven_grid_raises():
    field_map = RegularGridMap(
        {"coordinate_system": [["r", [0.0, 1.0, 2]], ["z", [-1.0, 1.0, 2]]], "map": np.ones((2, 2))}
    )
    with pytest.raises(ValueError, match="evenly spaced"):
        build_drift_property_map(
            field_map, np.array([0.0, 0.1, 1.0]), np.linspace(-1, 1, 5), 0.0, **nest_conditions()
        )


def test_detector_gate_values_agree_with_nest():
    conditions = dict(
        temperature=detector_value("liquid_temperature"),
        pressure=detector_value("gas_pressure"),
        density=detector_value("xenon_density_liquid"),
    )
    extraction_field = detector_value("extraction_field_liquid")
    drift_time_gate = drift_time_above_gate(
        detector_value("z_liquid_level"), detector_value("z_gate"), extraction_field, **conditions
    )
    assert detector_value("drift_time_gate") == pytest.approx(drift_time_gate, rel=1e-4)
    assert detector_value("diffusion_constant_longitudinal_gate") == pytest.approx(
        longitudinal_diffusion_constant_liquid(extraction_field), rel=1e-3
    )


def test_electroluminescence_duration_is_gap_over_gas_velocity():
    velocity = drift_velocity_gas(5684.0, TEMPERATURE, PRESSURE, 0.0177)
    duration = electroluminescence_duration(0.3, 5684.0, TEMPERATURE, PRESSURE, 0.0177)
    assert duration == pytest.approx(0.3 / velocity)


GAS_DENSITY = 0.0177
GAS_GAP = 0.2988
FIELD_GAS = 5684.0
WIRE_PITCH = 2 * np.pi * 0.089
WIRE_RADIUS = 0.0108


def gas_conditions() -> dict:
    return dict(temperature=TEMPERATURE, pressure=PRESSURE, density_gas=GAS_DENSITY)


def u_values(timing_map: dict) -> np.ndarray:
    lower, upper, number_of_points = timing_map["coordinate_system"][0][1]
    return np.linspace(lower, upper, number_of_points)


def test_xenon_number_density():
    assert xenon_number_density(0.0177) == pytest.approx(8.119e19, rel=1e-3)


def test_electroluminescence_yield_follows_the_monteiro_fit():
    number_density = xenon_number_density(GAS_DENSITY)
    field_at_10_td = 10 * 1e-17 * number_density
    expected = (0.140 * 10 - 0.474) * 1e-17 * number_density
    assert electroluminescence_yield_per_length(field_at_10_td, number_density) == pytest.approx(
        expected
    )
    field_below_threshold = 3.3 * 1e-17 * number_density
    assert electroluminescence_yield_per_length(field_below_threshold, number_density) == 0


def test_mesh_field_is_uniform():
    np.testing.assert_allclose(anode_field_profile(np.linspace(0.01, 0.3, 7), 5000.0), 5000.0)


@settings(deadline=None, max_examples=30)
@given(
    distance=st.floats(min_value=1e-3, max_value=1.0),
    pitch=st.floats(min_value=0.05, max_value=1.0),
)
def test_wire_field_is_at_least_the_far_field_and_continuous(distance, pitch):
    field = anode_field_profile(np.array([distance]), 5000.0, anode_wire_pitch=pitch)[0]
    assert field >= 5000.0
    domination_distance = pitch / (2 * np.pi)
    if distance < domination_distance:
        assert field == pytest.approx(5000.0 * domination_distance / distance)
    at_domination_distance = anode_field_profile(
        np.array([domination_distance]), 5000.0, anode_wire_pitch=pitch
    )
    assert at_domination_distance[0] == pytest.approx(5000.0)


def test_mesh_luminescence_is_uniform_until_the_crossing_time():
    timing_map = build_luminescence_timing_map(GAS_GAP, FIELD_GAS, **gas_conditions())
    duration = electroluminescence_duration(GAS_GAP, FIELD_GAS, **gas_conditions())
    np.testing.assert_allclose(timing_map["map"], u_values(timing_map) * duration, rtol=1e-6)


def test_wire_luminescence_is_shorter_and_concentrated_at_the_end():
    mesh_map = build_luminescence_timing_map(GAS_GAP, FIELD_GAS, **gas_conditions())
    wire_map = build_luminescence_timing_map(
        GAS_GAP,
        FIELD_GAS,
        **gas_conditions(),
        anode_wire_pitch=WIRE_PITCH,
        anode_wire_radius=WIRE_RADIUS,
    )
    end_time = wire_map["map"][-1]
    assert end_time < mesh_map["map"][-1]
    median = np.interp(0.5, u_values(wire_map), wire_map["map"])
    assert median > 0.8 * end_time


@settings(deadline=None, max_examples=20)
@given(
    field_gas=st.floats(min_value=3000.0, max_value=15000.0),
    pitch=st.floats(min_value=0.1, max_value=1.0),
)
def test_luminescence_map_is_non_decreasing_and_starts_at_zero(field_gas, pitch):
    timing_map = build_luminescence_timing_map(
        GAS_GAP,
        field_gas,
        **gas_conditions(),
        anode_wire_pitch=pitch,
        anode_wire_radius=WIRE_RADIUS,
        number_of_steps=2000,
    )
    assert np.all(np.diff(timing_map["map"]) >= 0)
    assert timing_map["map"][0] >= 0
    assert RegularGridMap(timing_map)([[1.0]])[0] == pytest.approx(timing_map["map"][-1])


def test_luminescence_below_threshold_raises():
    with pytest.raises(ValueError, match="threshold"):
        build_luminescence_timing_map(GAS_GAP, 1000.0, **gas_conditions())


@pytest.mark.parametrize(
    "wire_parameters, message",
    [
        (dict(anode_wire_pitch=WIRE_PITCH), "both"),
        (dict(anode_wire_pitch=0.05, anode_wire_radius=0.01), "radius"),
        (dict(anode_wire_pitch=WIRE_PITCH, anode_wire_radius=0.0), "radius"),
    ],
)
def test_wrong_wire_parameters_raise(wire_parameters, message):
    with pytest.raises(ValueError, match=message):
        build_luminescence_timing_map(GAS_GAP, FIELD_GAS, **gas_conditions(), **wire_parameters)
