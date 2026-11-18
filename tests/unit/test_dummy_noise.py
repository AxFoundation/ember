"""Tests for the dummy noise file (map_generators/dummy_noise.ipynb)."""

import numpy as np
import pytest
from strax import URLConfig

import ember  # noqa: F401  (registers the protocols)


@pytest.fixture(scope="module")
def noise():
    return URLConfig.evaluate_dry("ember_file://noise_dummy.npz")["arr_0"]


def test_noise_has_one_column_per_channel(noise):
    assert noise.ndim == 2
    assert noise.shape[1] == URLConfig.evaluate_dry("detector://n_tpc_pmts")


def test_noise_is_longer_than_a_template(noise):
    samples_before = URLConfig.evaluate_dry("detector://samples_before_pulse_center")
    samples_after = URLConfig.evaluate_dry("detector://samples_after_pulse_center")
    assert noise.shape[0] > 10 * (samples_before + samples_after)


def test_noise_is_in_whole_adc_counts(noise):
    assert np.issubdtype(noise.dtype, np.integer)


def test_noise_has_mean_zero_and_the_configured_sigma(noise):
    noise_sigma = URLConfig.evaluate_dry("detector://noise_sigma")
    expected_sigma = np.sqrt(noise_sigma**2 + 1 / 12)
    standard_error_of_mean = expected_sigma / np.sqrt(noise.size)
    assert abs(noise.mean()) < 5 * standard_error_of_mean
    assert noise.std() == pytest.approx(expected_sigma, rel=0.01)


def test_noise_of_two_channels_is_uncorrelated(noise):
    correlation = np.corrcoef(noise[:, 0], noise[:, 1])[0, 1]
    assert abs(correlation) < 5 / np.sqrt(noise.shape[0])
