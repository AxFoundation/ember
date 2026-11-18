import dataclasses

import numba
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ember.optical_toy_mc import (
    OUTCOME_ABSORBED,
    OUTCOME_BOTTOM,
    OUTCOME_TOP,
    TpcOptics,
    SPEED_OF_LIGHT_IN_VACUUM,
    build_s1_timing_map,
    build_s2_timing_map,
    draw_positions_in_disk,
    draw_positions_in_layer,
    fresnel_reflectance,
    lambertian_direction,
    rayleigh_direction,
    seed_random_generator,
    simulate_photons,
    simulate_s2_photons,
)

NEVER = 1e12

SMALL_TPC = TpcOptics(
    tpc_radius=20.0,
    z_bottom_pmts=-50.0,
    z_liquid_level=0.5,
    z_top_pmts=5.0,
    lxe_refractive_index=1.63,
    lxe_rayleigh_scattering_length=32.0,
    lxe_absorption_length=1245.0,
    gxe_absorption_length=1245.0,
    ptfe_reflectivity_liquid=1.0,
    ptfe_reflectivity_gas=0.8,
    photocathode_coverage_top=0.6,
    photocathode_coverage_bottom=0.6,
)

BLACK_TPC_WITHOUT_SCATTERING = dataclasses.replace(
    SMALL_TPC,
    lxe_rayleigh_scattering_length=NEVER,
    lxe_absorption_length=NEVER,
    gxe_absorption_length=NEVER,
    ptfe_reflectivity_liquid=0.0,
    ptfe_reflectivity_gas=0.0,
    photocathode_coverage_top=1.0,
    photocathode_coverage_bottom=1.0,
)


@numba.njit
def _sample_lambertian_cosines(number_of_samples):
    cosines = np.empty(number_of_samples)
    for index in range(number_of_samples):
        cosines[index] = lambertian_direction(0.0, 0.0, 1.0)[2]
    return cosines


@numba.njit
def _sample_rayleigh_cosines(number_of_samples):
    cosines = np.empty(number_of_samples)
    for index in range(number_of_samples):
        cosines[index] = rayleigh_direction(0.0, 0.0, 1.0)[2]
    return cosines


def test_fresnel_reflectance_at_normal_incidence():
    reflectance, cos_transmitted = fresnel_reflectance(1.0, 1.63, 1.0)
    assert reflectance == pytest.approx(((1.63 - 1) / (1.63 + 1)) ** 2)
    assert cos_transmitted == pytest.approx(1.0)


def test_total_internal_reflection_above_the_critical_angle():
    critical_angle = np.arcsin(1 / 1.63)
    assert fresnel_reflectance(np.cos(critical_angle + 0.01), 1.63, 1.0)[0] == 1.0
    assert fresnel_reflectance(np.cos(critical_angle - 0.01), 1.63, 1.0)[0] < 1.0


@given(st.floats(0.0, 1.5), st.floats(1.1, 2.0))
def test_fresnel_reflectance_is_the_same_in_both_directions(angle_in_gas, n_liquid):
    reflectance_from_gas, cos_in_liquid = fresnel_reflectance(np.cos(angle_in_gas), 1.0, n_liquid)
    reflectance_from_liquid, cos_in_gas = fresnel_reflectance(cos_in_liquid, n_liquid, 1.0)
    assert 0 <= reflectance_from_gas <= 1
    assert reflectance_from_liquid == pytest.approx(reflectance_from_gas, abs=1e-6)
    assert cos_in_gas == pytest.approx(np.cos(angle_in_gas), abs=1e-6)


def test_lambertian_reflection_has_mean_cosine_two_thirds():
    seed_random_generator(1)
    cosines = _sample_lambertian_cosines(200_000)
    assert np.all(cosines > 0)
    assert cosines.mean() == pytest.approx(2 / 3, abs=0.005)


def test_rayleigh_scattering_has_mean_squared_cosine_two_fifths():
    seed_random_generator(2)
    cosines = _sample_rayleigh_cosines(200_000)
    assert cosines.mean() == pytest.approx(0.0, abs=0.005)
    assert np.mean(cosines**2) == pytest.approx(0.4, abs=0.005)


def test_on_axis_photons_reach_the_bottom_with_the_solid_angle_fraction():
    z_source = -30.0
    positions = np.tile([0.0, 0.0, z_source], (200_000, 1))
    photons = simulate_photons(positions, BLACK_TPC_WITHOUT_SCATTERING, seed=3)
    distance = z_source - BLACK_TPC_WITHOUT_SCATTERING.z_bottom_pmts
    radius = BLACK_TPC_WITHOUT_SCATTERING.tpc_radius
    solid_angle_fraction = 0.5 * (1 - distance / np.hypot(distance, radius))
    assert np.mean(photons["outcome"] == OUTCOME_BOTTOM) == pytest.approx(
        solid_angle_fraction, abs=0.003
    )
    assert np.all(photons["ptfe_reflections_liquid"] == 0)


def test_all_photons_are_detected_without_losses():
    lossless_tpc = dataclasses.replace(
        SMALL_TPC,
        lxe_absorption_length=NEVER,
        gxe_absorption_length=NEVER,
        ptfe_reflectivity_gas=1.0,
    )
    positions = draw_positions_in_disk(-20.0, 5000, 19.0, np.random.default_rng(4))
    photons = simulate_photons(positions, lossless_tpc, seed=4)
    assert not np.any(photons["outcome"] == OUTCOME_ABSORBED)


def test_simulation_is_reproducible_with_the_same_seed():
    positions = draw_positions_in_disk(-20.0, 1000, 19.0, np.random.default_rng(5))
    first = simulate_photons(positions, SMALL_TPC, seed=5)
    second = simulate_photons(positions, SMALL_TPC, seed=5)
    np.testing.assert_array_equal(first, second)


def test_start_positions_outside_the_tpc_raise():
    with pytest.raises(ValueError, match="inside the TPC"):
        simulate_photons(np.array([[0.0, 0.0, -60.0]]), SMALL_TPC, seed=0)
    with pytest.raises(ValueError, match="shape"):
        simulate_photons(np.zeros((3, 2)), SMALL_TPC, seed=0)


@settings(deadline=None, max_examples=25)
@given(
    z_source=st.floats(-49.0, 0.0),
    scattering_length=st.floats(5.0, 100.0),
    reflectivity=st.floats(0.0, 1.0),
    seed=st.integers(0, 1000),
)
def test_no_photon_is_faster_than_the_direct_flight(
    z_source, scattering_length, reflectivity, seed
):
    optics = dataclasses.replace(
        SMALL_TPC,
        lxe_rayleigh_scattering_length=scattering_length,
        ptfe_reflectivity_liquid=reflectivity,
    )
    positions = draw_positions_in_disk(z_source, 500, 19.0, np.random.default_rng(seed))
    photons = simulate_photons(positions, optics, seed=seed)
    for array_name, outcome in [("top", OUTCOME_TOP), ("bottom", OUTCOME_BOTTOM)]:
        arrival_times = photons["arrival_time"][photons["outcome"] == outcome]
        assert np.all(arrival_times >= optics.direct_flight_time(z_source, array_name) - 1e-6)
    detected = photons["outcome"] != OUTCOME_ABSORBED
    assert np.all(photons["path_liquid"][detected] > 0)


def test_timing_map_has_the_expected_format_and_is_non_decreasing():
    timing_map = build_s1_timing_map(
        SMALL_TPC, number_of_z_points=4, number_of_u_points=11, photons_per_slice=5000, seed=6
    )
    assert timing_map["coordinate_system"] == [
        ["z", (SMALL_TPC.z_bottom_pmts, SMALL_TPC.z_liquid_level, 4)],
        ["u", (0.0, 1.0, 11)],
    ]
    for array_name in ["top", "bottom"]:
        delays = timing_map[array_name]
        assert delays.shape == (4, 11)
        assert np.all(np.diff(delays, axis=1) >= 0)
    assert timing_map["top"][0, 5] > timing_map["bottom"][0, 5]
    assert timing_map["top"][-1, 5] < timing_map["bottom"][-1, 5]


def test_timing_map_raises_if_an_array_gets_no_light():
    dark_top_tpc = dataclasses.replace(SMALL_TPC, photocathode_coverage_top=0.0)
    with pytest.raises(ValueError, match="top array"):
        build_s1_timing_map(dark_top_tpc, 2, 5, photons_per_slice=100, seed=7)


def test_optics_from_detector_config():
    optics = TpcOptics.from_detector_config()
    assert optics.lxe_refractive_index == pytest.approx(1.63)
    assert optics.lxe_rayleigh_scattering_length == pytest.approx(32.0)
    assert optics.photocathode_coverage_top == pytest.approx(253 * 3.2**2 / 66.4**2)
    assert optics.photocathode_coverage_bottom == pytest.approx(241 * 3.2**2 / 66.4**2)
    assert optics.direct_flight_time(optics.z_bottom_pmts, "bottom") == pytest.approx(0.0)


Z_ANODE_SMALL_TPC = 0.8


@settings(deadline=None, max_examples=25)
@given(
    z_lower=st.floats(-10.0, 4.0),
    thickness=st.floats(0.01, 1.0),
    seed=st.integers(0, 1000),
)
def test_positions_in_layer_lie_inside_the_layer(z_lower, thickness, seed):
    positions = draw_positions_in_layer(
        z_lower, z_lower + thickness, 1000, 19.0, np.random.default_rng(seed)
    )
    assert positions.shape == (1000, 3)
    assert np.all(np.hypot(positions[:, 0], positions[:, 1]) < 19.0)
    assert np.all((positions[:, 2] >= z_lower) & (positions[:, 2] <= z_lower + thickness))


def test_positions_in_layer_with_wrong_order_raise():
    with pytest.raises(ValueError, match="below"):
        draw_positions_in_layer(1.0, 0.5, 10, 19.0, np.random.default_rng(0))


def test_s2_photons_mostly_reach_the_top_array():
    photons = simulate_s2_photons(SMALL_TPC, Z_ANODE_SMALL_TPC, 20000, seed=8)
    n_top = np.sum(photons["outcome"] == OUTCOME_TOP)
    n_bottom = np.sum(photons["outcome"] == OUTCOME_BOTTOM)
    assert n_top / (n_top + n_bottom) > 0.5
    assert np.all(photons["path_gas"][photons["outcome"] != OUTCOME_ABSORBED] > 0)


def test_s2_photons_are_reproducible_with_the_same_seed():
    first = simulate_s2_photons(SMALL_TPC, Z_ANODE_SMALL_TPC, 1000, seed=9)
    second = simulate_s2_photons(SMALL_TPC, Z_ANODE_SMALL_TPC, 1000, seed=9)
    np.testing.assert_array_equal(first, second)


def test_no_s2_photon_is_faster_than_the_direct_flight():
    photons = simulate_s2_photons(SMALL_TPC, Z_ANODE_SMALL_TPC, 20000, seed=10)
    fastest_top = (SMALL_TPC.z_top_pmts - Z_ANODE_SMALL_TPC) / SPEED_OF_LIGHT_IN_VACUUM
    fastest_bottom = (
        (SMALL_TPC.z_liquid_level - SMALL_TPC.z_bottom_pmts) * SMALL_TPC.lxe_refractive_index
    ) / SPEED_OF_LIGHT_IN_VACUUM
    top_times = photons["arrival_time"][photons["outcome"] == OUTCOME_TOP]
    bottom_times = photons["arrival_time"][photons["outcome"] == OUTCOME_BOTTOM]
    assert np.all(top_times >= fastest_top - 1e-9)
    assert np.all(bottom_times >= fastest_bottom - 1e-9)


def test_s2_timing_map_has_the_expected_format_and_is_non_decreasing():
    photons = simulate_s2_photons(SMALL_TPC, Z_ANODE_SMALL_TPC, 20000, seed=11)
    timing_map = build_s2_timing_map(photons, number_of_u_points=21)
    assert timing_map["coordinate_system"] == [["u", (0.0, 1.0, 21)]]
    for array_name in ["top", "bottom"]:
        delays = timing_map[array_name]
        assert delays.shape == (21,)
        assert np.all(np.diff(delays) >= 0)
    assert timing_map["top"][10] < timing_map["bottom"][10]


def test_s2_timing_map_raises_if_an_array_gets_no_light():
    dark_bottom_tpc = dataclasses.replace(SMALL_TPC, photocathode_coverage_bottom=0.0)
    photons = simulate_s2_photons(dark_bottom_tpc, Z_ANODE_SMALL_TPC, 100, seed=12)
    with pytest.raises(ValueError, match="bottom array"):
        build_s2_timing_map(photons, number_of_u_points=5)
