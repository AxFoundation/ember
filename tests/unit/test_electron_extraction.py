"""Tests of the ElectronExtraction plugin."""

import numpy as np
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import make_instructions, write_csv

import ember
from ember.dtypes import individual_electron_fields


def make_context(folder, **config) -> strax.Context:
    context = ember.ember_context(output_folder=folder / "data")
    context.set_config({"input_file": write_csv(make_instructions(20), folder), **config})
    return context


def electrons_at_interface(number: int, time: int = 1000) -> np.ndarray:
    """Electrons arriving at the same time at the liquid surface."""
    electrons = np.zeros(number, dtype=individual_electron_fields + strax.time_fields)
    electrons["time"] = time
    electrons["endtime"] = time
    electrons["x_interface"] = np.linspace(-10, 10, number)
    electrons["y_interface"] = np.linspace(5, -5, number)
    electrons["cluster_id"] = np.arange(number) % 7
    return electrons


@pytest.fixture(scope="module")
def extraction_plugin(tmp_path_factory):
    return make_context(tmp_path_factory.mktemp("plugin")).get_single_plugin(
        "0", "extracted_electrons"
    )


@pytest.fixture(scope="module")
def full_extraction_plugin(tmp_path_factory):
    """Plugin that extracts every electron."""
    context = make_context(tmp_path_factory.mktemp("full"), electron_extraction_yield=1.0)
    return context.get_single_plugin("0", "extracted_electrons")


@pytest.fixture(scope="module")
def beta_extraction(tmp_path_factory):
    context = make_context(tmp_path_factory.mktemp("beta"))
    return (
        context.get_array("0", "electrons_at_interface"),
        context.get_array("0", "extracted_electrons"),
    )


def test_extracted_fraction_follows_the_extraction_yield(beta_extraction):
    at_interface, extracted = beta_extraction
    probability = strax.URLConfig.evaluate_dry("detector://electron_extraction_efficiency")
    number = len(at_interface)
    sigma = np.sqrt(number * probability * (1 - probability))
    assert abs(len(extracted) - number * probability) < 5 * sigma


def test_extracted_electrons_are_sorted_and_keep_their_clusters(beta_extraction):
    at_interface, extracted = beta_extraction
    assert np.all(np.diff(extracted["time"]) >= 0)
    np.testing.assert_array_equal(extracted["time"], extracted["endtime"])
    assert set(extracted["cluster_id"]) <= set(at_interface["cluster_id"])


@pytest.mark.parametrize("probability, expected_fraction", [(0.0, 0.0), (1.0, 1.0)])
def test_extraction_yield_limits(tmp_path, probability, expected_fraction):
    context = make_context(tmp_path, electron_extraction_yield=probability)
    at_interface = context.get_array("0", "electrons_at_interface")
    extracted = context.get_array("0", "extracted_electrons")
    assert len(extracted) == expected_fraction * len(at_interface)


def test_trapping_delay_is_exponential(full_extraction_plugin):
    electrons = electrons_at_interface(200_000)
    extracted = full_extraction_plugin.compute(electrons, start=0, end=10**9)
    delays = extracted["time"] - 1000
    trapping_time = strax.URLConfig.evaluate_dry("detector://electron_trapping_time")
    assert np.all(delays >= 0)
    assert delays.mean() == pytest.approx(trapping_time - 0.5, rel=0.01)
    assert np.median(delays) == pytest.approx(trapping_time * np.log(2), rel=0.02)


def test_positions_are_carried_over(full_extraction_plugin):
    electrons = electrons_at_interface(50)
    extracted = full_extraction_plugin.compute(electrons, start=0, end=10**9)
    by_position = np.lexsort((extracted["y_interface"], extracted["x_interface"]))
    expected = np.lexsort((electrons["y_interface"], electrons["x_interface"]))
    for field in ["x_interface", "y_interface", "cluster_id"]:
        np.testing.assert_array_equal(extracted[field][by_position], electrons[field][expected])


@settings(deadline=None, max_examples=50)
@given(
    times=st.lists(st.integers(0, 10**7), min_size=0, max_size=200),
)
def test_output_is_a_sorted_subset_never_earlier_than_the_input(extraction_plugin, times):
    electrons = np.zeros(len(times), dtype=individual_electron_fields + strax.time_fields)
    electrons["time"] = np.sort(times)
    electrons["endtime"] = electrons["time"]
    electrons["cluster_id"] = np.arange(len(times))
    extracted = extraction_plugin.compute(electrons, start=0, end=10**9)
    assert len(extracted) <= len(electrons)
    assert np.all(np.diff(extracted["time"]) >= 0)
    original_times = dict(zip(electrons["cluster_id"], electrons["time"]))
    assert all(row["time"] >= original_times[row["cluster_id"]] for row in extracted)


def test_electrons_behind_the_chunk_end_raise(extraction_plugin):
    electrons = electrons_at_interface(100, time=999_990)
    with pytest.raises(ValueError, match="behind the chunk end"):
        extraction_plugin.compute(electrons, start=0, end=1_000_000)


@pytest.mark.parametrize(
    "config, message",
    [
        ({"electron_extraction_yield": 1.2}, "electron_extraction_yield"),
        ({"electron_trapping_time": -1.0}, "electron_trapping_time"),
    ],
)
def test_invalid_configuration_raises(tmp_path, config, message):
    with pytest.raises(ValueError, match=message):
        make_context(tmp_path, **config).get_single_plugin("0", "extracted_electrons")
