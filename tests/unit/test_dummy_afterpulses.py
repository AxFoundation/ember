"""Tests for the dummy afterpulse configuration (map_generators/dummy_pmt_afterpulses.ipynb)."""

import numpy as np
import pytest
from strax import URLConfig

import ember  # noqa: F401  (registers the protocols)

LITERATURE_IONS = {
    # name: (M/Q, mean delay [ns], probability per photoelectron), Barrow et al. 2017
    "He+": (4, 520.0, 0.02e-2),
    "CH4+": (16, 1010.0, 0.02e-2),
    "Ar+": (40, 1580.0, 0.08e-2),
}


@pytest.fixture(scope="module")
def afterpulse_config():
    config = URLConfig.evaluate_dry("ember_file://pmt_afterpulse_config_dummy.json.gz")
    return {
        name: {key: np.asarray(value) for key, value in entry.items()}
        for name, entry in config.items()
    }


def test_config_contains_the_literature_ions(afterpulse_config):
    assert set(afterpulse_config) == set(LITERATURE_IONS)


def test_entries_have_fuse_keys(afterpulse_config):
    for entry in afterpulse_config.values():
        assert set(entry) == {
            "delaytime_cdf",
            "delaytime_bin_size",
            "amplitude_cdf",
            "amplitude_bin_size",
        }


def test_delay_cdf_has_one_row_per_channel(afterpulse_config):
    n_tpc_pmts = URLConfig.evaluate_dry("detector://n_tpc_pmts")
    for entry in afterpulse_config.values():
        assert entry["delaytime_cdf"].shape[0] == n_tpc_pmts


@pytest.mark.parametrize("key", ["delaytime_cdf", "amplitude_cdf"])
def test_cdfs_are_non_negative_and_non_decreasing(afterpulse_config, key):
    for entry in afterpulse_config.values():
        cdf = np.atleast_2d(entry[key])
        assert np.all(cdf >= 0)
        assert np.all(np.diff(cdf, axis=-1) >= 0)


def test_last_delay_cdf_value_is_the_literature_probability(afterpulse_config):
    for name, (_, _, probability) in LITERATURE_IONS.items():
        np.testing.assert_allclose(
            afterpulse_config[name]["delaytime_cdf"][:, -1], probability, rtol=1e-6
        )


def test_amplitude_cdf_is_normalized_and_size_zero_is_impossible(afterpulse_config):
    for entry in afterpulse_config.values():
        assert entry["amplitude_cdf"][-1] == pytest.approx(1.0)
        assert entry["amplitude_cdf"][0] == 0.0


def test_mean_delay_matches_the_literature(afterpulse_config):
    for name, (_, mean_delay, _) in LITERATURE_IONS.items():
        entry = afterpulse_config[name]
        delay_cdf = entry["delaytime_cdf"][0]
        delays = np.arange(len(delay_cdf)) * entry["delaytime_bin_size"]
        probabilities = np.diff(delay_cdf, prepend=0)
        measured_mean = np.sum(delays * probabilities) / delay_cdf[-1]
        assert measured_mean == pytest.approx(mean_delay, abs=entry["delaytime_bin_size"] / 2)


def test_mean_size_follows_the_power_law_through_h_and_xe(afterpulse_config):
    size_exponent = np.log(18.0 / 2.0) / np.log(131)
    for name, (mass_to_charge, _, _) in LITERATURE_IONS.items():
        entry = afterpulse_config[name]
        sizes = np.arange(len(entry["amplitude_cdf"])) * entry["amplitude_bin_size"]
        probabilities = np.diff(entry["amplitude_cdf"], prepend=0)
        expected_mean = 18.0 * mass_to_charge**-size_exponent
        assert np.sum(sizes * probabilities) == pytest.approx(expected_mean, rel=0.02)
