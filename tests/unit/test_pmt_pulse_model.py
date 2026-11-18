"""Tests for the analytic SPE pulse model and the dummy pulse shape file."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strax import URLConfig

import ember  # noqa: F401  (registers the protocols)
from ember.pmt_pulse_model import (
    fit_lognormal_pulse,
    lognormal_pulse_shape,
    measure_pulse_timing,
    smear_pulse_shape,
)

LITERATURE_TIMING = (2.6, 9.7, 7.2)  # ns, Lung et al. 2012 (rise, fall, FWHM)
FINE_TIMES = np.arange(-50, 300, 0.01)


def test_lognormal_pulse_has_its_maximum_one_at_zero():
    times = np.array([-1.0, 0.0, 1.0])
    amplitudes = lognormal_pulse_shape(times, scale=5.0, sigma=0.5)
    assert amplitudes[1] == pytest.approx(1.0)
    assert np.all(amplitudes[[0, 2]] < 1.0)


def test_lognormal_pulse_is_zero_before_its_start():
    times = np.array([-10.0, -5.0, -4.9])
    amplitudes = lognormal_pulse_shape(times, scale=5.0, sigma=0.5)
    np.testing.assert_array_equal(amplitudes[:2], 0.0)
    assert amplitudes[2] > 0.0


@pytest.mark.parametrize("scale, sigma", [(0.0, 0.5), (5.0, 0.0), (-1.0, 0.5)])
def test_lognormal_pulse_rejects_non_positive_parameters(scale, sigma):
    with pytest.raises(ValueError):
        lognormal_pulse_shape(np.zeros(3), scale, sigma)


def test_timing_of_a_triangular_pulse():
    times = np.arange(-10, 40, 0.5)
    amplitudes = np.interp(times, [0, 10, 30], [0, 1, 0], left=0, right=0)
    rise_time, fall_time, fwhm = measure_pulse_timing(times, amplitudes)
    assert rise_time == pytest.approx(8.0)
    assert fall_time == pytest.approx(16.0)
    assert fwhm == pytest.approx(15.0)


def test_timing_of_a_truncated_pulse_raises():
    times = np.arange(0, 10, 0.5)
    with pytest.raises(ValueError):
        measure_pulse_timing(times, lognormal_pulse_shape(times, scale=5.0, sigma=0.5))


def test_timing_of_a_pulse_without_positive_maximum_raises():
    with pytest.raises(ValueError):
        measure_pulse_timing(np.arange(5.0), np.zeros(5))


def test_fit_reproduces_the_literature_timing():
    scale, sigma = fit_lognormal_pulse(*LITERATURE_TIMING)
    timing = measure_pulse_timing(FINE_TIMES, lognormal_pulse_shape(FINE_TIMES, scale, sigma))
    np.testing.assert_allclose(timing, LITERATURE_TIMING, atol=0.05)


def test_fit_rejects_non_positive_timing():
    with pytest.raises(ValueError):
        fit_lognormal_pulse(2.6, -1.0, 7.2)


@settings(deadline=None, max_examples=30)
@given(
    scale=st.floats(1.0, 20.0),
    sigma=st.floats(0.2, 0.8),
    amplitude_factor=st.floats(1e-3, 1e3),
    time_shift=st.floats(-20.0, 20.0),
)
def test_timing_does_not_depend_on_amplitude_and_time_shift(
    scale, sigma, amplitude_factor, time_shift
):
    reference = measure_pulse_timing(FINE_TIMES, lognormal_pulse_shape(FINE_TIMES, scale, sigma))
    shifted_and_scaled = measure_pulse_timing(
        FINE_TIMES + time_shift, amplitude_factor * lognormal_pulse_shape(FINE_TIMES, scale, sigma)
    )
    np.testing.assert_allclose(shifted_and_scaled, reference, rtol=1e-9, atol=1e-9)


@settings(deadline=None, max_examples=8)
@given(scale=st.floats(2.0, 15.0), sigma=st.floats(0.3, 0.7))
def test_fit_recovers_the_parameters_of_a_lognormal_pulse(scale, sigma):
    timing = measure_pulse_timing(FINE_TIMES, lognormal_pulse_shape(FINE_TIMES, scale, sigma))
    fitted_scale, fitted_sigma = fit_lognormal_pulse(*timing)
    assert fitted_scale == pytest.approx(scale, rel=1e-3)
    assert fitted_sigma == pytest.approx(sigma, rel=1e-3)


@pytest.fixture(scope="module")
def dummy_pulse_shape():
    return URLConfig.evaluate_dry("ember_file://pmt_pulse_shape_dummy.json")


def test_dummy_pulse_shape_has_fuse_keys_on_a_1ns_grid(dummy_pulse_shape):
    times = np.array(dummy_pulse_shape["pe_pulse_ts"])
    amplitudes = np.array(dummy_pulse_shape["pe_pulse_ys"])
    assert times.shape == amplitudes.shape
    np.testing.assert_array_equal(np.diff(times), 1)
    assert np.all(amplitudes >= 0)
    assert abs(times[np.argmax(amplitudes)]) <= 2


def test_dummy_pulse_shape_covers_the_template_window(dummy_pulse_shape):
    sample_duration = URLConfig.evaluate_dry("detector://sample_duration")
    samples_before = URLConfig.evaluate_dry("detector://samples_before_pulse_center")
    samples_after = URLConfig.evaluate_dry("detector://samples_after_pulse_center")
    times = dummy_pulse_shape["pe_pulse_ts"]
    assert times[0] <= -samples_before * sample_duration
    assert times[-1] >= samples_after * sample_duration


def test_dummy_pulse_shape_is_the_smeared_literature_pulse(dummy_pulse_shape):
    smearing_sigma = URLConfig.evaluate_dry("detector://pmt_pulse_smearing_sigma")
    scale, sigma = fit_lognormal_pulse(*LITERATURE_TIMING)
    fine_times = np.arange(-60, 260, 0.1)
    expected = smear_pulse_shape(
        fine_times, lognormal_pulse_shape(fine_times, scale, sigma), smearing_sigma
    )
    expected_timing = measure_pulse_timing(fine_times, expected)
    timing = measure_pulse_timing(
        dummy_pulse_shape["pe_pulse_ts"], dummy_pulse_shape["pe_pulse_ys"]
    )
    np.testing.assert_allclose(timing, expected_timing, atol=0.5)
    assert timing[2] >= LITERATURE_TIMING[2]


def gaussian(times, mean, sigma):
    return np.exp(-0.5 * ((times - mean) / sigma) ** 2)


def test_smearing_two_gaussians_adds_their_widths_quadratically():
    times = np.arange(-100, 100, 0.01)
    smeared = smear_pulse_shape(times, gaussian(times, 0.0, 3.0), 4.0)
    _, _, fwhm = measure_pulse_timing(times, smeared)
    assert fwhm == pytest.approx(2.35482 * 5.0, rel=1e-3)


def test_smearing_with_zero_sigma_returns_the_pulse():
    times = np.arange(-10, 10, 0.5)
    pulse = gaussian(times, 0.0, 2.0)
    smeared = smear_pulse_shape(times, pulse, 0.0)
    np.testing.assert_array_equal(smeared, pulse)
    assert smeared is not pulse


def test_smearing_rejects_negative_sigma_and_uneven_times():
    with pytest.raises(ValueError):
        smear_pulse_shape(np.arange(5.0), np.ones(5), -1.0)
    with pytest.raises(ValueError):
        smear_pulse_shape(np.array([0.0, 1.0, 3.0]), np.ones(3), 1.0)


@settings(deadline=None, max_examples=30)
@given(scale=st.floats(2.0, 15.0), sigma=st.floats(0.3, 0.7), smearing=st.floats(0.5, 6.0))
def test_smearing_keeps_area_and_mean_time(scale, sigma, smearing):
    times = np.arange(-100, 300, 0.05)
    pulse = lognormal_pulse_shape(times, scale, sigma)
    smeared = smear_pulse_shape(times, pulse, smearing)
    assert smeared.sum() == pytest.approx(pulse.sum(), rel=1e-6)
    mean_time = np.sum(times * pulse) / pulse.sum()
    assert np.sum(times * smeared) / smeared.sum() == pytest.approx(mean_time, abs=0.01)
    assert measure_pulse_timing(times, smeared)[2] > measure_pulse_timing(times, pulse)[2]
