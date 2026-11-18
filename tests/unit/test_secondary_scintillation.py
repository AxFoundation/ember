"""Tests of the SecondaryScintillation plugin."""

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import COLUMNS, make_instructions, write_csv

import ember
from ember.maps import save_map_data
from ember.plugins.detector_physics.secondary_scintillation import sum_per_cluster


def detector_value(key: str):
    return strax.URLConfig.evaluate_dry(f"detector://{key}")


G2 = detector_value("g2")
P_DOUBLE_PE = detector_value("p_double_pe_emission")
EXTRACTION_YIELD = detector_value("electron_extraction_efficiency")


def electron_instructions(number_of_events: int, electrons: int) -> pd.DataFrame:
    """Events with a given number of electrons close to the gate (no photons)."""
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": np.zeros(number_of_events),
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


@pytest.fixture(scope="module")
def beta_s2(tmp_path_factory):
    folder = tmp_path_factory.mktemp("beta")
    context = make_context(folder, make_instructions(50))
    return (
        context.get_array("0", ["microphysics_summary", "drifted_electrons", "s2_photons_sum"]),
        context.get_array("0", "extracted_electrons"),
        context.get_array("0", "s2_photons"),
    )


def test_s2_photons_are_aligned_with_the_extracted_electrons(beta_s2):
    _, extracted, photons = beta_s2
    assert len(photons) == len(extracted)
    np.testing.assert_array_equal(photons["time"], extracted["time"])
    np.testing.assert_array_equal(photons["endtime"], extracted["endtime"])


def test_mean_photons_per_electron_follow_the_gain(beta_s2):
    _, _, photons = beta_s2
    expected = G2 / EXTRACTION_YIELD / (1 + P_DOUBLE_PE)
    sigma_of_mean = np.sqrt(expected / len(photons))
    assert abs(photons["n_s2_photons"].mean() - expected) < 5 * sigma_of_mean


def test_sum_per_interaction_is_the_sum_over_its_electrons(beta_s2):
    clusters, extracted, photons = beta_s2
    expected = defaultdict(int)
    for cluster_id, n_photons in zip(extracted["cluster_id"], photons["n_s2_photons"]):
        expected[cluster_id] += n_photons
    for cluster_id, total in zip(clusters["cluster_id"], clusters["sum_s2_photons"]):
        assert total == expected[cluster_id]


@pytest.mark.parametrize("extraction_yield", [0.4, 1.0])
def test_extraction_yield_does_not_change_the_s2_per_electron_at_the_surface(
    tmp_path, extraction_yield
):
    context = make_context(
        tmp_path, electron_instructions(200, 100), electron_extraction_yield=extraction_yield
    )
    clusters = context.get_array("0", ["drifted_electrons", "s2_photons_sum"])
    electrons_at_surface = clusters["n_electron_interface"].sum()
    photoelectrons_per_electron = (
        clusters["sum_s2_photons"].sum() * (1 + P_DOUBLE_PE) / electrons_at_surface
    )
    extracted = electrons_at_surface * extraction_yield
    photons_per_extracted_electron = G2 / extraction_yield / (1 + P_DOUBLE_PE)
    relative_sigma = np.sqrt(
        (1 - extraction_yield) / extracted + 1 / (extracted * photons_per_extracted_electron)
    )
    assert abs(photoelectrons_per_electron / G2 - 1) < 5 * relative_sigma


def test_interactions_without_electrons_get_no_s2_photons(tmp_path):
    context = make_context(tmp_path, electron_instructions(5, 0))
    assert np.all(context.get_array("0", "s2_photons_sum")["sum_s2_photons"] == 0)
    assert len(context.get_array("0", "s2_photons")) == 0


def test_nan_in_the_correction_map_gives_no_photons(tmp_path):
    radius = detector_value("tpc_radius")
    nan_map = {
        "coordinate_system": [["x", [-radius, radius, 2]], ["y", [-radius, radius, 2]]],
        "map": [[float("nan")] * 2] * 2,
    }
    map_file = tmp_path / "nan_map.json"
    save_map_data(nan_map, map_file)
    context = make_context(
        tmp_path, electron_instructions(5, 100), s2_correction_map=f"ember_map://{map_file}"
    )
    assert np.all(context.get_array("0", "s2_photons")["n_s2_photons"] == 0)


def test_zero_extraction_yield_raises(tmp_path):
    context = make_context(tmp_path, make_instructions(5), electron_extraction_yield=0.0)
    with pytest.raises(ValueError, match="electron_extraction_yield"):
        context.get_single_plugin("0", "s2_photons")


@settings(deadline=None, max_examples=100)
@given(
    electron_cluster_ids=st.lists(st.integers(0, 30), max_size=200),
    interaction_cluster_ids=st.lists(st.integers(0, 40), unique=True, max_size=40),
    seed=st.integers(0, 1000),
)
def test_sum_per_cluster_equals_the_naive_sum(electron_cluster_ids, interaction_cluster_ids, seed):
    values = np.random.default_rng(seed).poisson(30, len(electron_cluster_ids))
    expected = defaultdict(int)
    for cluster_id, value in zip(electron_cluster_ids, values):
        expected[cluster_id] += value
    result = sum_per_cluster(
        np.array(interaction_cluster_ids, dtype=np.int32),
        np.array(electron_cluster_ids, dtype=np.int32),
        values,
    )
    assert result.tolist() == [expected[cluster_id] for cluster_id in interaction_cluster_ids]
