"""Tests of the PMTAfterPulses plugin and its helpers."""

import numpy as np
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hypothesis_numpy

import ember
from ember.dtypes import propagated_photons_fields
from ember.plugins.pmt_and_daq.pmt_afterpulses import (
    AfterpulseElement,
    draw_afterpulses,
    draw_from_cumulative_rows,
    read_afterpulse_elements,
)

N_TPC_PMTS = strax.URLConfig.evaluate_dry("detector://n_tpc_pmts")
PHOTON_DTYPE = propagated_photons_fields + strax.time_fields


def photons(times, channels, is_dpe=False) -> np.ndarray:
    data = np.zeros(len(times), dtype=PHOTON_DTYPE)
    data["time"] = times
    data["endtime"] = times
    data["channel"] = channels
    data["dpe"] = is_dpe
    data["photon_type"] = 1
    return data


def single_bin_element(probability: float, n_channels: int = 3) -> AfterpulseElement:
    """Afterpulse with a fixed delay of 2 bins (20 ns) and a fixed size of 3 bins (0.3 PE)."""
    delaytime_cdf = np.tile([0.0, 0.0, probability, probability], (n_channels, 1))
    amplitude_cdf = np.tile([0.0, 0.0, 0.0, 1.0], (n_channels, 1))
    return AfterpulseElement("test", delaytime_cdf, 10.0, amplitude_cdf, 0.1)


def test_inverse_cdf_known_values():
    cdf = np.array([[0.0, 0.2, 0.5, 1.0], [0.1, 0.1, 0.3, 0.3]])
    rows = np.array([0, 0, 0, 0, 1, 1, 1])
    random_numbers = np.array([0.1, 0.2, 0.21, 1.0, 0.05, 0.1, 0.3])
    np.testing.assert_array_equal(
        draw_from_cumulative_rows(cdf, rows, random_numbers), [1, 1, 2, 3, 0, 0, 2]
    )


@settings(deadline=None, max_examples=50)
@given(
    probabilities=hypothesis_numpy.arrays(
        np.float64, (4, 6), elements=st.floats(0.0, 1.0, allow_subnormal=False)
    ),
    data=st.data(),
)
def test_inverse_cdf_returns_the_smallest_bin_reaching_the_number(probabilities, data):
    probabilities[:, 0] += 1e-3
    cdf = np.cumsum(probabilities, axis=1)
    cdf /= cdf[:, -1:]
    rows = np.array(data.draw(st.lists(st.integers(0, 3), min_size=1, max_size=20)))
    random_numbers = np.array(
        [data.draw(st.floats(1e-6, 1.0)) * cdf[row, -1] for row in rows], dtype=np.float64
    )
    bins = draw_from_cumulative_rows(cdf, rows, random_numbers)
    assert np.all(cdf[rows, bins] >= random_numbers - 1e-12)
    earlier = bins > 0
    assert np.all(cdf[rows[earlier], bins[earlier] - 1] < random_numbers[earlier] + 1e-12)


def test_no_photons_give_no_afterpulses():
    times, channels, gains = draw_afterpulses(
        np.zeros(0, np.int64),
        np.zeros(0, np.int16),
        np.zeros(0, bool),
        [single_bin_element(0.5)],
        np.ones(3),
        1.0,
        0.0,
        np.random.default_rng(0),
    )
    assert len(times) == len(channels) == len(gains) == 0


@pytest.mark.parametrize(
    "is_dpe, modifier, expected_factor", [(False, 1.0, 1), (True, 1.0, 2), (False, 2.0, 2)]
)
def test_afterpulse_rate(is_dpe, modifier, expected_factor):
    probability = 0.01
    n_photons = 1_000_000
    times, _, _ = draw_afterpulses(
        np.zeros(n_photons, np.int64),
        np.ones(n_photons, np.int16),
        np.full(n_photons, is_dpe),
        [single_bin_element(probability)],
        np.ones(3),
        modifier,
        0.0,
        np.random.default_rng(1),
    )
    expected = expected_factor * probability * n_photons
    assert abs(len(times) - expected) < 5 * np.sqrt(expected)


def test_delay_size_gain_and_time_modifier():
    gains = np.array([1e6, 2e6, 3e6])
    times, channels, afterpulse_gains = draw_afterpulses(
        np.array([1000, 5000]),
        np.array([2, 1], np.int16),
        np.zeros(2, bool),
        [single_bin_element(1.0)],
        gains,
        1.0,
        5.0,
        np.random.default_rng(2),
    )
    np.testing.assert_array_equal(times, [1015.0, 5015.0])
    np.testing.assert_array_equal(channels, [2, 1])
    np.testing.assert_allclose(afterpulse_gains, 0.3 * gains[[2, 1]])


def test_channels_without_afterpulse_probability_get_none():
    element = single_bin_element(0.5)
    element.delaytime_cdf[0] = 0.0
    _, channels, _ = draw_afterpulses(
        np.zeros(30_000, np.int64),
        np.arange(30_000, dtype=np.int16) % 3,
        np.zeros(30_000, bool),
        [element],
        np.ones(3),
        1.0,
        0.0,
        np.random.default_rng(3),
    )
    assert len(channels) > 0
    assert not np.any(channels == 0)


def test_delay_distribution_follows_the_cdf():
    delay_probabilities = np.array([0.0, 0.1, 0.3, 0.4, 0.2])
    probability = 0.2
    element = AfterpulseElement(
        "test",
        np.tile(probability * np.cumsum(delay_probabilities), (1, 1)),
        10.0,
        np.array([[0.0, 1.0]]),
        1.0,
    )
    n_photons = 500_000
    times, _, _ = draw_afterpulses(
        np.zeros(n_photons, np.int64),
        np.zeros(n_photons, np.int16),
        np.zeros(n_photons, bool),
        [element],
        np.ones(1),
        1.0,
        0.0,
        np.random.default_rng(4),
    )
    counts = np.bincount((times / 10).astype(int), minlength=5)
    expected = delay_probabilities * len(times)
    nonzero = expected > 0
    chi2 = np.sum((counts[nonzero] - expected[nonzero]) ** 2 / expected[nonzero])
    assert counts[0] == 0
    assert chi2 < 20


def test_one_dimensional_amplitude_cdf_is_used_for_every_channel():
    config = {
        "Xe+": {
            "delaytime_cdf": np.tile([0.0, 0.01], (3, 1)).tolist(),
            "delaytime_bin_size": 10,
            "amplitude_cdf": [0.0, 1.0],
            "amplitude_bin_size": 0.1,
        }
    }
    (element,) = read_afterpulse_elements(config, n_channels=3)
    assert element.amplitude_cdf.shape == (3, 2)
    assert element.name == "Xe+"


@pytest.mark.parametrize(
    "name, delaytime_cdf, n_channels",
    [
        ("Uniform", [[0.0, 0.01]] * 3, 3),
        ("Xe+", [[0.0, np.nan]] * 3, 3),
        ("Xe+", [[0.02, 0.01]] * 3, 3),
        ("Xe+", [[0.0, 1.5]] * 3, 3),
        ("Xe+", [[0.0, 0.01]] * 2, 3),
    ],
)
def test_invalid_afterpulse_configuration_raises(name, delaytime_cdf, n_channels):
    config = {
        name: {
            "delaytime_cdf": delaytime_cdf,
            "delaytime_bin_size": 10,
            "amplitude_cdf": [0.0, 1.0],
            "amplitude_bin_size": 0.1,
        }
    }
    with pytest.raises(ValueError):
        read_afterpulse_elements(config, n_channels=n_channels)


def make_context(folder, **config) -> strax.Context:
    context = ember.ember_context(output_folder=folder / "data")
    context.set_config({"input_file": str(folder / "unused.csv"), **config})
    return context


@pytest.fixture(scope="module")
def afterpulse_plugin(tmp_path_factory):
    return make_context(tmp_path_factory.mktemp("ap")).get_single_plugin("0", "pmt_afterpulses")


def test_plugin_output_with_the_dummy_configuration(afterpulse_plugin):
    n_photons = 1_000_000
    s1_photons = photons(np.arange(n_photons) % 1000, np.arange(n_photons) % N_TPC_PMTS)
    s2_photons = photons([0], [0])
    result = afterpulse_plugin.compute(s1_photons, s2_photons, start=0, end=10**7)

    expected = 0.0012 * (n_photons + 1)
    assert abs(len(result) - expected) < 5 * np.sqrt(expected)
    assert np.all(result["photon_type"] == 0)
    assert np.all(result["cluster_id"] == -1)
    assert not np.any(result["dpe"])
    assert np.all(np.diff(result["time"]) >= 0)
    assert np.all(result["time"] >= 400)
    assert np.all(result["photon_gain"] > 0)


def test_plugin_without_photons_returns_nothing(afterpulse_plugin):
    empty = photons([], [])
    assert len(afterpulse_plugin.compute(empty, empty, start=0, end=100)) == 0


def test_plugin_raises_for_afterpulses_behind_the_chunk_end(afterpulse_plugin):
    s1_photons = photons(np.full(100_000, 900), np.zeros(100_000))
    with pytest.raises(ValueError, match="behind the chunk end"):
        afterpulse_plugin.compute(s1_photons, photons([], []), start=0, end=1000)


def test_disabled_afterpulses_give_nothing(tmp_path):
    plugin = make_context(tmp_path, enable_pmt_afterpulses=False).get_single_plugin(
        "0", "pmt_afterpulses"
    )
    s1_photons = photons(np.zeros(10_000), np.zeros(10_000))
    assert len(plugin.compute(s1_photons, photons([], []), start=0, end=10**7)) == 0


def test_gain_model_with_wrong_length_raises(tmp_path):
    with pytest.raises(ValueError, match="channels"):
        make_context(tmp_path, gain_model_mc=np.ones(3)).get_single_plugin("0", "pmt_afterpulses")


def test_photon_summary_merges_and_sorts(tmp_path):
    plugin = make_context(tmp_path).get_single_plugin("0", "photon_summary")
    s1_photons = photons([30, 10], [1, 2])
    s2_photons = photons([20], [3])
    afterpulses = photons([5], [4])
    merged = plugin.compute(s1_photons=s1_photons, s2_photons=s2_photons, ap_photons=afterpulses)
    np.testing.assert_array_equal(merged["time"], [5, 10, 20, 30])
    np.testing.assert_array_equal(merged["channel"], [4, 2, 3, 1])
    assert plugin.dtype_for("photon_summary") == np.dtype(PHOTON_DTYPE)
