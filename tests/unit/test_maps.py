"""Tests for ember.maps and the ember_map:// protocol."""

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strax import URLConfig

import ember  # noqa: F401  (registers the protocols)
from ember.electron_transport import (
    DRIFT_PROPERTY_MAP_NAMES,
    build_drift_property_map,
    electroluminescence_duration,
)
from ember.maps import (
    RegularGridMap,
    encode_map_values,
    load_map,
    load_map_data,
    save_map_data,
)


def linear_map_data(slope_r: float, slope_z: float, offset: float) -> dict:
    r_points = np.linspace(0.0, 10.0, 5)
    z_points = np.linspace(-20.0, 0.0, 9)
    values = offset + slope_r * r_points[:, None] + slope_z * z_points[None, :]
    return {
        "name": "linear",
        "coordinate_system": [["r", [0.0, 10.0, 5]], ["z", [-20.0, 0.0, 9]]],
        "map": values.tolist(),
    }


def test_linear_interpolation_known_values():
    itp_map = RegularGridMap(linear_map_data(2.0, 1.0, 3.0))
    positions = np.array([[0.0, 0.0], [5.0, -10.0], [1.3, -7.7]])
    np.testing.assert_allclose(itp_map(positions), 3.0 + 2.0 * positions[:, 0] + positions[:, 1])


def test_positions_outside_grid_use_boundary_values():
    itp_map = RegularGridMap(linear_map_data(2.0, 1.0, 3.0))
    np.testing.assert_allclose(itp_map(np.array([[20.0, 5.0], [-1.0, -50.0]])), [23.0, -17.0])


def test_vector_map_returns_one_value_per_channel():
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    map_data = {"coordinate_system": [["x", [0, 1, 2]], ["y", [0, 2, 3]]], "map": values.tolist()}
    result = RegularGridMap(map_data)(np.array([[0.0, 0.0], [1.0, 2.0]]))
    assert result.shape == (2, 4)
    np.testing.assert_allclose(result, [values[0, 0], values[1, 2]])


def test_nearest_method_gives_a_step():
    map_data = {"coordinate_system": [["z", [-1.5, 1.5, 4]]], "map": [0.0, 0.0, 1.0, 1.0]}
    itp_map = RegularGridMap(map_data, method="nearest")
    np.testing.assert_allclose(itp_map(np.array([[-0.01], [0.01]])), [0.0, 1.0])


def test_wrong_map_shape_raises():
    map_data = {"coordinate_system": [["x", [0, 1, 3]]], "map": [1.0, 2.0]}
    with pytest.raises(ValueError, match="does not match"):
        RegularGridMap(map_data)


def test_wrong_number_of_coordinates_raises():
    itp_map = RegularGridMap(linear_map_data(1.0, 1.0, 0.0))
    with pytest.raises(ValueError, match="expects positions with 2 coordinates"):
        itp_map(np.zeros((3, 3)))


def test_load_map_from_file(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(linear_map_data(1.0, 0.0, 0.0)))
    assert load_map(path)(np.array([[4.0, -1.0]]))[0] == pytest.approx(4.0)


def test_dummy_drift_field_map_is_constant():
    drift_field = URLConfig.evaluate_dry("detector://drift_field")
    itp_map = URLConfig.evaluate_dry("ember_map://drift_field_constant.json")
    positions = np.array([[0.0, -150.0], [66.4, 0.0], [30.0, -75.0]])
    np.testing.assert_allclose(itp_map(positions), drift_field)


def test_dummy_survival_map_is_zero_below_cathode():
    z_cathode = URLConfig.evaluate_dry("detector://z_cathode")
    itp_map = URLConfig.evaluate_dry(
        "ember_map://electron_survival_below_cathode.json?method=nearest"
    )
    positions = np.array([[10.0, z_cathode - 0.001], [10.0, z_cathode + 0.001], [60.0, -10.0]])
    np.testing.assert_allclose(itp_map(positions), [0.0, 1.0, 1.0])


def test_dummy_drift_map_matches_the_field_map():
    """The drift map is derived from the field map; regenerate it when the field map changes."""
    drift_map = URLConfig.evaluate_dry("ember_map://field_dependencies_dummy.json")
    field_map = URLConfig.evaluate_dry("ember_map://drift_field_constant.json")
    expected = build_drift_property_map(
        field_map,
        drift_map.grid[0],
        drift_map.grid[1],
        z_gate=detector_value("z_gate"),
        temperature=detector_value("liquid_temperature"),
        pressure=detector_value("gas_pressure"),
        density=detector_value("xenon_density_liquid"),
    )
    assert drift_map.map_names == sorted(DRIFT_PROPERTY_MAP_NAMES)
    for name in DRIFT_PROPERTY_MAP_NAMES:
        np.testing.assert_allclose(drift_map.maps[name], expected[name], rtol=1e-6)
    assert drift_map.lower_bounds[1] <= detector_value("z_cathode")
    assert drift_map.upper_bounds[1] >= detector_value("z_liquid_level")


@settings(deadline=None, max_examples=50)
@given(
    slope_r=st.floats(-10, 10),
    slope_z=st.floats(-10, 10),
    offset=st.floats(-100, 100),
    r_value=st.floats(-5, 15),
    z_value=st.floats(-30, 10),
)
def test_linear_function_is_reproduced(slope_r, slope_z, offset, r_value, z_value):
    itp_map = RegularGridMap(linear_map_data(slope_r, slope_z, offset))
    clipped_r = np.clip(r_value, 0.0, 10.0)
    clipped_z = np.clip(z_value, -20.0, 0.0)
    expected = offset + slope_r * clipped_r + slope_z * clipped_z
    result = itp_map(np.array([[r_value, z_value]]))[0]
    assert result == pytest.approx(expected, abs=1e-9 * (1 + abs(expected)) + 1e-9)


def per_channel_map_data(n_channels: int = 4) -> dict:
    x_points = np.linspace(-1.0, 1.0, 3)
    z_points = np.linspace(-2.0, 0.0, 5)
    channel_weights = np.arange(1, n_channels + 1, dtype=float)
    values = (1.0 + x_points[:, None, None] ** 2 - 0.1 * z_points[None, :, None]) * channel_weights
    return {
        "name": "per channel",
        "coordinate_system": [["x", [-1.0, 1.0, 3]], ["z", [-2.0, 0.0, 5]]],
        "map": values * 1e-3,
    }


def test_compressed_and_quantized_map_round_trip(tmp_path):
    map_data = per_channel_map_data()
    path = tmp_path / "map.pkl"
    save_map_data(encode_map_values(map_data, quantized=1e-7), path)
    loaded = load_map(path)
    np.testing.assert_allclose(loaded.values, map_data["map"], atol=1e-7)


def test_json_gz_round_trip(tmp_path):
    path = tmp_path / "map.json.gz"
    save_map_data(linear_map_data(1.0, 2.0, 3.0), path)
    assert load_map(path)(np.array([[2.0, -1.0]]))[0] == pytest.approx(3.0)


def test_unsupported_suffix_raises(tmp_path):
    with pytest.raises(ValueError, match="Unsupported map file format"):
        load_map_data(tmp_path / "map.txt")


def test_quantizing_negative_values_raises():
    with pytest.raises(ValueError, match="non-negative"):
        encode_map_values({"coordinate_system": [["x", [0, 1, 2]]], "map": [-1.0, 1.0]})


def test_named_sub_maps():
    map_data = {
        "coordinate_system": [["z", [0.0, 1.0, 2]]],
        "top": [1.0, 3.0],
        "bottom": [10.0, 30.0],
        "description": "two maps",
    }
    itp_map = RegularGridMap(map_data)
    assert itp_map.map_names == ["bottom", "top"]
    assert itp_map(np.array([[0.5]]), map_name="top")[0] == pytest.approx(2.0)
    assert itp_map(np.array([[0.5]]), map_name="bottom")[0] == pytest.approx(20.0)
    with pytest.raises(KeyError, match="no map 'map'"):
        itp_map(np.array([[0.5]]))


def test_masked_channels_are_zero():
    itp_map = RegularGridMap(per_channel_map_data())
    masked = itp_map.with_masked_channels(np.array([True, False, True, False]))
    result = masked(np.array([[0.3, -0.7]]))[0]
    assert result[1] == 0.0 and result[3] == 0.0
    assert result[0] > 0 and result[2] > 0


def test_mask_with_wrong_length_raises():
    with pytest.raises(ValueError, match="channel mask"):
        RegularGridMap(per_channel_map_data()).with_masked_channels(np.ones(3, dtype=bool))


@settings(deadline=None, max_examples=50)
@given(x_value=st.floats(-2, 2), z_value=st.floats(-3, 1))
def test_summed_map_equals_sum_of_channels(x_value, z_value):
    itp_map = RegularGridMap(per_channel_map_data())
    position = np.array([[x_value, z_value]])
    expected = itp_map(position).sum(axis=-1)
    np.testing.assert_allclose(itp_map.summed_over_channels()(position), expected, rtol=1e-12)


def test_dummy_s1_pattern_map_matches_detector():
    itp_map = URLConfig.evaluate_dry("ember_map://s1_pattern_map_dummy.pkl")
    n_tpc_pmts = URLConfig.evaluate_dry("detector://n_tpc_pmts")
    n_top_pmts = URLConfig.evaluate_dry("detector://n_top_pmts")
    g1 = URLConfig.evaluate_dry("detector://g1")
    z_cathode = URLConfig.evaluate_dry("detector://z_cathode")
    assert itp_map.axis_names == ["x", "y", "z"]
    assert itp_map.values.shape[-1] == n_tpc_pmts

    rng = np.random.default_rng(1)
    radius = 60.0 * np.sqrt(rng.uniform(0, 1, 20000))
    angle = rng.uniform(-np.pi, np.pi, 20000)
    positions = np.column_stack(
        [radius * np.cos(angle), radius * np.sin(angle), rng.uniform(z_cathode, 0, 20000)]
    )
    total_lce = itp_map.summed_over_channels()(positions)
    assert total_lce.mean() == pytest.approx(g1, rel=0.05)

    values_at_gate = itp_map(np.array([[0.0, 0.0, -0.5]]))[0]
    assert values_at_gate[:n_top_pmts].sum() / values_at_gate.sum() == pytest.approx(0.65, abs=0.01)


def test_dummy_s1_pattern_map_covers_region_below_cathode():
    itp_map = URLConfig.evaluate_dry("ember_map://s1_pattern_map_dummy.pkl")
    z_bottom_pmts = URLConfig.evaluate_dry("detector://z_bottom_pmts")
    assert itp_map.lower_bounds[2] == pytest.approx(z_bottom_pmts)
    assert np.all(itp_map(np.array([[0.0, 0.0, z_bottom_pmts + 1.0]])) > 0)


def test_dummy_s1_timing_map_covers_the_liquid_and_is_non_decreasing_in_u():
    itp_map = URLConfig.evaluate_dry("ember_map://s1_optical_propagation_dummy.json")
    assert itp_map.axis_names == ["z", "u"]
    z_bottom_pmts = URLConfig.evaluate_dry("detector://z_bottom_pmts")
    z_liquid_level = URLConfig.evaluate_dry("detector://z_liquid_level")
    assert itp_map.lower_bounds[0] == pytest.approx(z_bottom_pmts)
    assert itp_map.upper_bounds[0] == pytest.approx(z_liquid_level)
    u_values = np.linspace(0, 1, 101)
    for map_name in ["top", "bottom"]:
        delays = itp_map(np.column_stack([np.full(101, -70.0), u_values]), map_name=map_name)
        assert delays[0] > 0
        assert np.all(np.diff(delays) >= 0)


def test_dummy_spe_table_has_one_column_per_channel():
    table = URLConfig.evaluate_dry("ember_table://spe_distributions_dummy.csv")
    n_tpc_pmts = URLConfig.evaluate_dry("detector://n_tpc_pmts")
    assert table.columns[0] == "charge"
    assert set(table.columns[1:]) == {str(channel) for channel in range(n_tpc_pmts)}
    mean_charge = np.sum(table["charge"] * table["0"]) / table["0"].sum()
    assert mean_charge == pytest.approx(1.0, abs=0.01)


def detector_value(key: str):
    return URLConfig.evaluate_dry(f"detector://{key}")


@pytest.fixture(scope="module")
def s2_pattern_map():
    return URLConfig.evaluate_dry("ember_map://s2_pattern_map_dummy.pkl")


@pytest.fixture(scope="module")
def pmt_positions():
    return URLConfig.evaluate_dry("ember_table://pmt_positions_dummy.csv")


def test_dummy_s2_pattern_map_has_one_entry_per_channel(s2_pattern_map):
    assert s2_pattern_map.axis_names == ["x", "y"]
    assert s2_pattern_map.values.shape[-1] == detector_value("n_tpc_pmts")


@settings(deadline=None, max_examples=30)
@given(
    radius_fraction=st.floats(min_value=0.0, max_value=1.2),
    angle=st.floats(min_value=0.0, max_value=2 * np.pi),
)
def test_dummy_s2_pattern_sums_to_one_with_the_configured_aft(
    s2_pattern_map, radius_fraction, angle
):
    radius = radius_fraction * detector_value("tpc_radius")
    position = [[radius * np.cos(angle), radius * np.sin(angle)]]
    pattern = s2_pattern_map(position)[0]
    n_top_pmts = detector_value("n_top_pmts")
    assert pattern.sum() == pytest.approx(1.0, abs=1e-5)
    assert pattern[:n_top_pmts].sum() == pytest.approx(
        detector_value("s2_mean_area_fraction_top"), abs=1e-5
    )


def test_dummy_s2_pattern_is_brightest_in_the_pmt_above_the_extraction(
    s2_pattern_map, pmt_positions
):
    n_top_pmts = detector_value("n_top_pmts")
    top_positions = pmt_positions.loc[pmt_positions["array"] == "top", ["x", "y"]].to_numpy()
    patterns = s2_pattern_map(top_positions)
    np.testing.assert_array_equal(
        np.argmax(patterns[:, :n_top_pmts], axis=1), np.arange(n_top_pmts)
    )


def test_dummy_pmt_positions_fit_the_detector(pmt_positions):
    n_top_pmts = detector_value("n_top_pmts")
    pmt_diameter = detector_value("pmt_diameter")
    assert len(pmt_positions) == detector_value("n_tpc_pmts")
    np.testing.assert_array_equal(pmt_positions["i"], np.arange(len(pmt_positions)))
    assert list(pmt_positions["array"]) == ["top"] * n_top_pmts + ["bottom"] * (
        len(pmt_positions) - n_top_pmts
    )
    radii = np.hypot(pmt_positions["x"], pmt_positions["y"])
    assert np.all(radii + pmt_diameter / 2 <= detector_value("tpc_radius") + 1e-9)
    for array_name in ("top", "bottom"):
        array = pmt_positions.loc[pmt_positions["array"] == array_name, ["x", "y"]].to_numpy()
        distances = np.hypot(*(array[:, np.newaxis, :] - array[np.newaxis, :, :]).T)
        np.fill_diagonal(distances, np.inf)
        assert distances.min() >= pmt_diameter


def test_dummy_s2_correction_map_is_constant_one():
    correction_map = URLConfig.evaluate_dry("ember_map://s2_correction_map_dummy.json")
    radius = detector_value("tpc_radius")
    positions = np.array([[0.0, 0.0], [radius, 0.0], [-radius / 2, radius / 3]])
    np.testing.assert_allclose(correction_map(positions), 1.0)


def test_dummy_luminescence_timing_map_matches_the_detector():
    """Regenerate dummy_s2_timing_maps.ipynb when the gas gap or the extraction field changes."""
    timing_map = URLConfig.evaluate_dry("ember_map://s2_luminescence_timing_dummy.json")
    duration = electroluminescence_duration(
        detector_value("z_anode") - detector_value("z_liquid_level"),
        detector_value("lxe_dielectric_constant") * detector_value("extraction_field_liquid"),
        detector_value("liquid_temperature"),
        detector_value("gas_pressure"),
        detector_value("xenon_density_gas"),
    )
    assert timing_map.axis_names == ["u"]
    np.testing.assert_allclose(timing_map([[0.0], [0.5], [1.0]]), [0.0, duration / 2, duration])


def test_dummy_s2_optical_timing_map_format():
    timing_map = URLConfig.evaluate_dry("ember_map://s2_optical_propagation_dummy.json")
    assert timing_map.axis_names == ["u"]
    assert timing_map.map_names == ["bottom", "top"]
    for array_name in ["top", "bottom"]:
        assert np.all(np.diff(timing_map.maps[array_name]) >= 0)
        assert timing_map.maps[array_name][0] >= 0
    median = [[0.5]]
    assert timing_map(median, map_name="top")[0] < timing_map(median, map_name="bottom")[0]


def channel_map(
    values: np.ndarray,
    x_range: tuple = (-10.0, 10.0),
    y_range: tuple = (-10.0, 10.0),
    method: str = "linear",
) -> RegularGridMap:
    """Per-channel 2D map with values of shape (n_x, n_y, n_channels)."""
    n_x, n_y = values.shape[:2]
    map_data = {
        "coordinate_system": [["x", [*x_range, n_x]], ["y", [*y_range, n_y]]],
        "map": values,
    }
    return RegularGridMap(map_data, method=method)


def reference_draw_channels(
    itp_map: RegularGridMap, positions: np.ndarray, n_photons: np.ndarray, seed: int
) -> np.ndarray:
    """Straightforward version: interpolate with scipy, normalize, draw row by row."""
    probabilities = itp_map(positions)
    uniforms = np.random.default_rng(seed).random(n_photons.sum())
    channels = []
    photon_offset = 0
    for row, row_photons in enumerate(n_photons):
        if row_photons == 0:
            continue
        cdf = np.cumsum(probabilities[row]) / probabilities[row].sum()
        last_active_channel = np.flatnonzero(probabilities[row] > 0)[-1]
        row_uniforms = uniforms[photon_offset : photon_offset + row_photons]
        channels.append(
            np.minimum(np.searchsorted(cdf, row_uniforms, side="right"), last_active_channel)
        )
        photon_offset += row_photons
    return np.concatenate(channels or [np.zeros(0, dtype=np.int64)]).astype(np.int16)


def test_grid_cells_known_values():
    itp_map = channel_map(np.ones((5, 3, 1)), x_range=(0.0, 4.0), y_range=(0.0, 1.0))
    lower_indices, fractions = itp_map.grid_cells(np.array([[1.25, 0.5], [4.0, 0.0], [-3.0, 9.0]]))
    np.testing.assert_array_equal(lower_indices, [[1, 1], [3, 0], [0, 1]])
    np.testing.assert_allclose(fractions, [[0.25, 0.0], [1.0, 0.0], [0.0, 1.0]])


def test_grid_cells_of_a_single_point_axis():
    itp_map = channel_map(np.ones((1, 4, 2)), x_range=(2.0, 2.0), y_range=(0.0, 3.0))
    lower_indices, fractions = itp_map.grid_cells(np.array([[7.0, 1.5]]))
    np.testing.assert_array_equal(lower_indices, [[0, 1]])
    np.testing.assert_allclose(fractions, [[0.0, 0.5]])


def test_grid_cells_nearest_rounds_like_scipy():
    itp_map = channel_map(
        np.ones((3, 2, 1)), x_range=(0.0, 2.0), y_range=(0.0, 1.0), method="nearest"
    )
    _, fractions = itp_map.grid_cells(np.array([[0.5, 0.51], [1.49, 0.2]]))
    np.testing.assert_array_equal(fractions, [[0.0, 1.0], [0.0, 0.0]])


def test_draw_channels_follows_the_probabilities():
    values = np.tile([0.2, 0.0, 0.8], (2, 2, 1))
    itp_map = channel_map(values)
    channels = itp_map.draw_channels(
        np.array([[0.0, 0.0]]), np.array([100_000]), np.random.default_rng(1)
    )
    assert channels.dtype == np.int16
    assert np.mean(channels == 0) == pytest.approx(0.2, abs=0.01)
    assert not np.any(channels == 1)


def test_draw_channels_interpolates_between_grid_points():
    values = np.array([[[1.0, 0.0]], [[0.0, 1.0]]])
    itp_map = channel_map(values, x_range=(0.0, 1.0), y_range=(0.0, 0.0))
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.25, 0.0], [5.0, 0.0]])
    channels = itp_map.draw_channels(
        positions, np.array([100, 100, 100_000, 100]), np.random.default_rng(2)
    )
    assert np.all(channels[:100] == 0)
    assert np.all(channels[100:200] == 1)
    assert np.mean(channels[200:100_200]) == pytest.approx(0.25, abs=0.01)
    assert np.all(channels[100_200:] == 1)


def test_draw_channels_uses_the_same_random_numbers_as_drawing_by_hand():
    rng_values = np.random.default_rng(3)
    itp_map = channel_map(rng_values.random((6, 5, 30)))
    positions = rng_values.uniform(-12, 12, (40, 2))
    n_photons = rng_values.integers(0, 20, 40)
    np.testing.assert_array_equal(
        itp_map.draw_channels(positions, n_photons, np.random.default_rng(4)),
        reference_draw_channels(itp_map, positions, n_photons, seed=4),
    )


def test_draw_channels_with_many_channels():
    rng_values = np.random.default_rng(5)
    values = rng_values.random((3, 4, 5000))
    values[..., ::7] = 0.0
    itp_map = channel_map(values)
    positions = rng_values.uniform(-10, 10, (20, 2))
    n_photons = np.full(20, 300)
    channels = itp_map.draw_channels(positions, n_photons, np.random.default_rng(6))
    np.testing.assert_array_equal(
        channels, reference_draw_channels(itp_map, positions, n_photons, seed=6)
    )
    assert not np.any(channels % 7 == 0)


def test_draw_channels_without_photons():
    itp_map = channel_map(np.ones((2, 2, 3)))
    channels = itp_map.draw_channels(np.zeros((0, 2)), np.zeros(0), np.random.default_rng(0))
    assert channels.shape == (0,) and channels.dtype == np.int16


def test_dark_positions_without_photons_are_skipped():
    values = np.zeros((2, 1, 2))
    values[1, 0] = [0.0, 1.0]
    itp_map = channel_map(values, x_range=(0.0, 1.0), y_range=(0.0, 0.0))
    channels = itp_map.draw_channels(
        np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([0, 5]), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(channels, [1] * 5)


@pytest.mark.parametrize("dark_values", [np.zeros((2, 2, 3)), -np.ones((2, 2, 3))])
def test_draw_channels_raises_without_light(dark_values):
    with pytest.raises(ValueError, match="no light or negative values"):
        channel_map(dark_values).draw_channels(
            np.array([[0.0, 0.0]]), np.array([5]), np.random.default_rng(0)
        )


@pytest.mark.parametrize(
    "map_data",
    [
        {"coordinate_system": [["x", [0, 1, 2]]], "map": np.ones((2, 3))},
        {"coordinate_system": [["x", [0, 1, 2]], ["y", [0, 1, 2]]], "map": np.ones((2, 2))},
        {
            "coordinate_system": [["x", [0, 1, 2]], ["y", [0, 1, 2]], ["z", [0, 1, 2]]],
            "map": np.ones((2, 2, 2, 3)),
        },
    ],
)
def test_draw_channels_needs_a_2d_channel_map(map_data):
    itp_map = RegularGridMap(map_data)
    positions = np.zeros((1, itp_map.n_dimensions))
    with pytest.raises(ValueError, match="2D map with one value per channel"):
        itp_map.draw_channels(positions, np.array([1]), np.random.default_rng(0))


def test_draw_channels_checks_the_photon_numbers():
    itp_map = channel_map(np.ones((2, 2, 3)))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positions but"):
        itp_map.draw_channels(np.zeros((2, 2)), np.array([1]), rng)
    with pytest.raises(ValueError, match="must not be negative"):
        itp_map.draw_channels(np.zeros((1, 2)), np.array([-1]), rng)
    with pytest.raises(ValueError, match="must be 1D"):
        itp_map.draw_channels(np.zeros((1, 2)), np.array([[1]]), rng)


def test_draw_channels_refuses_more_channels_than_int16():
    itp_map = channel_map(np.ones((1, 1, 2**15 + 1)))
    with pytest.raises(ValueError, match="int16"):
        itp_map.draw_channels(np.zeros((1, 2)), np.array([1]), np.random.default_rng(0))


@settings(deadline=None, max_examples=100)
@given(
    n_x=st.integers(1, 12),
    n_y=st.integers(1, 12),
    n_channels=st.integers(1, 700),
    n_positions=st.integers(0, 30),
    x_lower=st.floats(-100, 100),
    x_width=st.floats(0.0, 200),
    y_lower=st.floats(-100, 100),
    y_width=st.floats(0.0, 200),
    method=st.sampled_from(["linear", "nearest"]),
    seed=st.integers(0, 2**32 - 1),
)
def test_draw_channels_matches_the_reference_for_any_grid_and_channel_number(
    n_x, n_y, n_channels, n_positions, x_lower, x_width, y_lower, y_width, method, seed
):
    rng_values = np.random.default_rng(seed)
    x_upper = x_lower + (x_width if n_x > 1 else 0.0)
    y_upper = y_lower + (y_width if n_y > 1 else 0.0)
    if (n_x > 1 and x_upper <= x_lower) or (n_y > 1 and y_upper <= y_lower):
        return
    values = rng_values.random((n_x, n_y, n_channels))
    channel_is_dark = rng_values.random(n_channels) < 0.3
    channel_is_dark[rng_values.integers(n_channels)] = False
    values[..., channel_is_dark] = 0.0
    itp_map = channel_map(values, (x_lower, x_upper), (y_lower, y_upper), method=method)
    positions = np.column_stack(
        [
            rng_values.uniform(x_lower - 10, x_upper + 10, n_positions),
            rng_values.uniform(y_lower - 10, y_upper + 10, n_positions),
        ]
    )
    n_photons = rng_values.integers(0, 40, n_positions)

    channels = itp_map.draw_channels(positions, n_photons, np.random.default_rng(seed + 1))

    assert len(channels) == n_photons.sum()
    assert not np.any(channel_is_dark[channels])
    np.testing.assert_array_equal(
        channels, reference_draw_channels(itp_map, positions, n_photons, seed=seed + 1)
    )
