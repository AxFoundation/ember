"""Tests of the S2PhotonPropagation plugin and its helpers."""

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import COLUMNS, make_instructions, write_csv

import ember
from ember.maps import save_map_data
from ember.plugins.detector_physics.s2_photon_propagation import find_electron_split_index


def detector_value(key: str):
    return strax.URLConfig.evaluate_dry(f"detector://{key}")


N_TPC_PMTS = detector_value("n_tpc_pmts")
N_TOP_PMTS = detector_value("n_top_pmts")
P_DOUBLE_PE = detector_value("p_double_pe_emission")
SINGLET_FRACTION = detector_value("singlet_fraction_gas")


def electron_instructions(number_of_events: int, electrons: int) -> pd.DataFrame:
    """Events with a given number of electrons close to the gate (no photons)."""
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": np.linspace(-30.0, 30.0, number_of_events),
            "y": np.zeros(number_of_events),
            "z": np.full(number_of_events, -5.0),
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


def electrons(times, n_photons, x=0.0, y=0.0) -> np.ndarray:
    """Minimal merged ``extracted_electrons`` + ``s2_photons`` array."""
    data = np.zeros(
        len(times),
        dtype=[
            ("x_interface", np.float32),
            ("y_interface", np.float32),
            ("cluster_id", np.int32),
            ("n_s2_photons", np.int32),
        ]
        + strax.time_fields,
    )
    data["time"] = times
    data["endtime"] = times
    data["n_s2_photons"] = n_photons
    data["x_interface"] = x
    data["y_interface"] = y
    data["cluster_id"] = np.arange(len(times))
    return data


@pytest.fixture(scope="module")
def beta_s2_photons(tmp_path_factory):
    """Extracted electrons and propagated S2 photons of 30 betas of 40 keV."""
    folder = tmp_path_factory.mktemp("beta")
    context = make_context(folder, make_instructions(30))
    return (
        context.get_array("0", ["extracted_electrons", "s2_photons"]),
        context.get_array("0", "propagated_s2_photons"),
    )


@pytest.fixture(scope="module")
def propagation_plugin(tmp_path_factory):
    folder = tmp_path_factory.mktemp("plugin")
    context = make_context(folder, make_instructions(5))
    return context.get_single_plugin("0", "propagated_s2_photons")


def test_one_photon_row_per_s2_photon(beta_s2_photons):
    extracted, photons = beta_s2_photons
    assert len(photons) == extracted["n_s2_photons"].sum()
    assert np.all(photons["photon_type"] == 2)
    assert np.all(np.diff(photons["time"]) >= 0)


def test_photons_keep_their_cluster(beta_s2_photons):
    extracted, photons = beta_s2_photons
    cluster_ids, counts = np.unique(photons["cluster_id"], return_counts=True)
    expected = pd.Series(extracted["n_s2_photons"]).groupby(extracted["cluster_id"]).sum()
    np.testing.assert_array_equal(counts, expected.loc[cluster_ids].to_numpy())


def test_photon_aft_follows_the_pattern_map(beta_s2_photons):
    _, photons = beta_s2_photons
    expected = detector_value("s2_mean_area_fraction_top")
    sigma = np.sqrt(expected * (1 - expected) / len(photons))
    assert abs(np.mean(photons["channel"] < N_TOP_PMTS) - expected) < 5 * sigma


def test_double_pe_fraction(beta_s2_photons):
    _, photons = beta_s2_photons
    sigma = np.sqrt(P_DOUBLE_PE * (1 - P_DOUBLE_PE) / len(photons))
    assert abs(photons["dpe"].mean() - P_DOUBLE_PE) < 5 * sigma


def test_photons_never_arrive_before_their_electron(propagation_plugin):
    data = electrons(np.array([1000, 5000]), np.array([2000, 2000]))
    photons = propagation_plugin.propagate_photons(data)
    times_by_cluster = dict(zip(data["cluster_id"], data["time"]))
    electron_times = np.array([times_by_cluster[cid] for cid in photons["cluster_id"]])
    assert np.all(photons["time"] >= electron_times)


def test_disabled_channels_are_never_hit(tmp_path):
    to_pe = np.full(N_TPC_PMTS, 0.005)
    disabled_channels = np.arange(0, N_TPC_PMTS, 3)
    to_pe[disabled_channels] = 0.0
    context = make_context(
        tmp_path, electron_instructions(5, electrons=200), gain_model_mc=to_pe.tolist()
    )
    photons = context.get_array("0", "propagated_s2_photons")
    extracted = context.get_array("0", "s2_photons")
    assert len(photons) == extracted["n_s2_photons"].sum()
    assert not np.any(np.isin(photons["channel"], disabled_channels))


def test_all_channels_switched_off_raises(tmp_path):
    context = make_context(
        tmp_path, make_instructions(5), gain_model_mc=np.zeros(N_TPC_PMTS).tolist()
    )
    with pytest.raises(ValueError, match="switched off"):
        context.get_single_plugin("0", "propagated_s2_photons")


def test_custom_constant_timing_maps(tmp_path):
    luminescence_file = tmp_path / "luminescence.json"
    save_map_data(
        {"coordinate_system": [["u", [0, 1, 2]]], "map": [500.0, 500.0]}, luminescence_file
    )
    optical_file = tmp_path / "optical.json"
    save_map_data(
        {"coordinate_system": [["u", [0, 1, 2]]], "top": [10.0, 10.0], "bottom": [20.0, 20.0]},
        optical_file,
    )
    context = make_context(
        tmp_path,
        make_instructions(5),
        s2_luminescence_timing_map=f"ember_map://{luminescence_file}",
        s2_optical_propagation_spline=f"ember_map://{optical_file}",
        singlet_lifetime_gas=0,
        triplet_lifetime_gas=0,
    )
    plugin = context.get_single_plugin("0", "propagated_s2_photons")
    channels = np.array([0, N_TOP_PMTS - 1, N_TOP_PMTS, N_TPC_PMTS - 1])
    np.testing.assert_allclose(
        plugin.photon_timings(np.zeros((1, 2)), np.array([4]), channels), [510, 510, 520, 520]
    )


def test_default_luminescence_delay_is_within_the_gas_gap_drift(propagation_plugin):
    delays = propagation_plugin.luminescence_timings(10_000)
    maximum = propagation_plugin.s2_luminescence_timing_map([[1.0]])[0]
    assert delays.min() >= 0
    assert delays.max() <= maximum
    assert delays.mean() == pytest.approx(maximum / 2, rel=0.05)


def test_optical_delays_are_not_negative(propagation_plugin):
    channels = np.repeat([0, N_TPC_PMTS - 1], 1000)
    delays = propagation_plugin.optical_propagation(channels)
    assert np.all(delays >= 0)
    assert np.median(delays[channels >= N_TOP_PMTS]) > np.median(delays[channels < N_TOP_PMTS])


def test_singlet_triplet_delays_follow_the_singlet_fraction(propagation_plugin):
    number_of_photons = 100_000
    delays = propagation_plugin.singlet_triplet_delays(number_of_photons)
    expected_mean = SINGLET_FRACTION * detector_value("singlet_lifetime_gas") + (
        1 - SINGLET_FRACTION
    ) * detector_value("triplet_lifetime_gas")
    assert np.all(delays >= 0)
    assert delays.mean() == pytest.approx(expected_mean, rel=0.02)


def test_wrong_singlet_fraction_raises(tmp_path):
    context = make_context(tmp_path, make_instructions(5), singlet_fraction_gas=1.5)
    with pytest.raises(ValueError, match="singlet_fraction_gas"):
        context.get_single_plugin("0", "propagated_s2_photons")


def test_propagation_is_reproducible(tmp_path):
    first = make_context(tmp_path / "a", make_instructions(5)).get_array(
        "0", "propagated_s2_photons"
    )
    second = make_context(tmp_path / "b", make_instructions(5)).get_array(
        "0", "propagated_s2_photons"
    )
    np.testing.assert_array_equal(first, second)


def test_downchunking_keeps_all_photons(tmp_path):
    context = make_context(
        tmp_path,
        electron_instructions(6, electrons=100),
        propagated_s2_photons_file_size_target=0.01,
    )
    chunks = list(context.get_iter("0", "propagated_s2_photons"))
    photons = np.concatenate([chunk.data for chunk in chunks])
    extracted = context.get_array("0", "s2_photons")
    assert len(chunks) > 1
    assert len(photons) == extracted["n_s2_photons"].sum()
    for chunk in chunks:
        assert np.all(chunk.data["time"] >= chunk.start)
        assert np.all(chunk.data["time"] < chunk.end)
    for previous, following in zip(chunks[:-1], chunks[1:]):
        assert previous.end == following.start


def test_photons_behind_the_chunk_end_raise(propagation_plugin):
    data = electrons(np.array([1000]), np.array([100]))
    with pytest.raises(ValueError, match="behind the chunk end"):
        list(propagation_plugin.compute(data, start=0, end=1001))


def test_split_index_only_at_large_gaps():
    times = np.array([0, 10, 3_000_000, 3_000_010, 6_000_000])
    split_index = find_electron_split_index(
        times,
        np.full(5, 10**6),
        bytes_per_photon=1,
        file_size_target=1.0,
        min_gap_length=2e6,
    )
    np.testing.assert_array_equal(split_index, [2, 4])


@settings(deadline=None, max_examples=100)
@given(
    gaps=st.lists(st.integers(0, 10**7), min_size=0, max_size=50),
    photons=st.integers(0, 10**6),
    target=st.floats(0.001, 100.0),
)
def test_split_index_is_increasing_and_at_large_gaps(gaps, photons, target):
    times = np.concatenate([[0], np.cumsum(gaps)]).astype(np.int64)
    min_gap_length = 2e6
    split_index = find_electron_split_index(
        times, np.full(len(times), photons), 28, target, min_gap_length
    )
    assert np.all(np.diff(split_index) > 0)
    assert np.all((split_index > 0) & (split_index < len(times)))
    assert np.all(times[split_index] - times[split_index - 1] >= min_gap_length)
