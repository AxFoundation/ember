"""Analytic single photoelectron (SPE) pulse shape for the dummy PMT pulse template.

The pulse is a log-normal function of time. With its two parameters it reproduces the measured
rise time (10 % → 90 %), fall time (90 % → 10 %) and FWHM of the Hamamatsu R11410-10
(Lung et al. 2012, arXiv:1202.2628, section 4.3: 2.6 ns, 9.7 ns and 7.2 ns at -1750 V).

The cables, the amplifier and the limited bandwidth of the digitizer widen the pulse that the DAQ
records. :func:`smear_pulse_shape` models this by a convolution with a normal distribution.

Used by ``map_generators/dummy_pmt_pulse.ipynb`` and the tests, not by the plugins.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

FIT_TIME_STEP = 0.01  # ns
FIT_TIME_RANGE = 300.0  # ns


def lognormal_pulse_shape(times: np.ndarray, scale: float, sigma: float) -> np.ndarray:
    """Return a log-normal pulse with its maximum (1) at ``times = 0``.

    The pulse is ``exp(-0.5 * (ln((t + scale) / scale) / sigma)²)`` for ``t > -scale`` and 0
    before.

    :param times: times relative to the pulse maximum [ns].
    :param scale: time between the start and the maximum of the pulse [ns].
    :param sigma: width of the log-normal (dimensionless).
    :return: pulse amplitudes, same shape as ``times``.
    """
    if scale <= 0 or sigma <= 0:
        raise ValueError(f"scale and sigma must be positive, got {scale} and {sigma}.")
    times = np.asarray(times, dtype=np.float64)
    shifted_times = times + scale
    amplitudes = np.zeros_like(times)
    after_start = shifted_times > 0
    log_ratio = np.log(shifted_times[after_start] / scale) / sigma
    amplitudes[after_start] = np.exp(-0.5 * log_ratio**2)
    return amplitudes


def measure_pulse_timing(times: np.ndarray, amplitudes: np.ndarray) -> tuple[float, float, float]:
    """Measure rise time (10 → 90 %), fall time (90 → 10 %) and FWHM of a single-peaked pulse.

    The level crossings are linearly interpolated between the samples.

    :param times: increasing sample times [ns].
    :param amplitudes: pulse amplitudes (positive pulse with one maximum).
    :return: ``(rise_time, fall_time, fwhm)`` in the units of ``times``.
    """
    times = np.asarray(times, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    peak_index = int(np.argmax(amplitudes))
    peak_amplitude = amplitudes[peak_index]
    if peak_amplitude <= 0:
        raise ValueError("The pulse has no positive maximum.")
    rising_times, rising_amplitudes = times[: peak_index + 1], amplitudes[: peak_index + 1]
    falling_times, falling_amplitudes = times[peak_index:], amplitudes[peak_index:]
    if rising_amplitudes[0] > 0.1 * peak_amplitude or falling_amplitudes[-1] > 0.1 * peak_amplitude:
        raise ValueError("The time range does not contain the full pulse (10 % levels).")

    def rising_crossing(fraction: float) -> float:
        return float(np.interp(fraction * peak_amplitude, rising_amplitudes, rising_times))

    def falling_crossing(fraction: float) -> float:
        return float(np.interp(-fraction * peak_amplitude, -falling_amplitudes, falling_times))

    rise_time = rising_crossing(0.9) - rising_crossing(0.1)
    fall_time = falling_crossing(0.1) - falling_crossing(0.9)
    fwhm = falling_crossing(0.5) - rising_crossing(0.5)
    return rise_time, fall_time, fwhm


def fit_lognormal_pulse(rise_time: float, fall_time: float, fwhm: float) -> tuple[float, float]:
    """Find the log-normal pulse parameters that best reproduce the given timing values.

    Minimizes the squared differences of the three timing values (least squares in ns²).

    :param rise_time: 10 % → 90 % rise time [ns].
    :param fall_time: 90 % → 10 % fall time [ns].
    :param fwhm: full width at half maximum [ns].
    :return: ``(scale, sigma)`` of :func:`lognormal_pulse_shape`.
    """
    target = np.array([rise_time, fall_time, fwhm], dtype=np.float64)
    if np.any(target <= 0):
        raise ValueError(f"Timing values must be positive, got {target}.")
    fit_times = np.arange(-FIT_TIME_RANGE, FIT_TIME_RANGE, FIT_TIME_STEP)

    def squared_deviation(log_parameters: np.ndarray) -> float:
        scale, sigma = np.exp(log_parameters)
        try:
            timing = measure_pulse_timing(fit_times, lognormal_pulse_shape(fit_times, scale, sigma))
        except ValueError:
            return np.inf
        return float(np.sum((np.array(timing) - target) ** 2))

    start = np.log([fwhm, 0.5])
    result = minimize(
        squared_deviation,
        start,
        method="Nelder-Mead",
        options=dict(xatol=1e-8, fatol=1e-12, maxiter=4000),
    )
    scale, sigma = np.exp(result.x)
    return float(scale), float(sigma)


def smear_pulse_shape(times: np.ndarray, amplitudes: np.ndarray, sigma: float) -> np.ndarray:
    """Convolve a pulse with a normal distribution (models the bandwidth of the readout chain).

    The area and the mean time of the pulse stay the same. The pulse must be zero at both ends of
    the time range, otherwise area is lost at the edges.

    :param times: equally spaced times [ns].
    :param amplitudes: pulse amplitudes at ``times``.
    :param sigma: standard deviation of the normal distribution [ns]; 0 returns a copy.
    :return: smeared amplitudes at ``times``.
    """
    times = np.asarray(times, dtype=np.float64)
    amplitudes = np.asarray(amplitudes, dtype=np.float64)
    if sigma < 0:
        raise ValueError(f"sigma must not be negative, got {sigma}.")
    if sigma == 0:
        return amplitudes.copy()
    time_steps = np.diff(times)
    time_step = time_steps[0]
    if not np.allclose(time_steps, time_step):
        raise ValueError("The times must be equally spaced.")
    half_width = int(np.ceil(6 * sigma / time_step))
    kernel_times = np.arange(-half_width, half_width + 1) * time_step
    kernel = np.exp(-0.5 * (kernel_times / sigma) ** 2)
    return np.convolve(amplitudes, kernel / kernel.sum(), mode="same")
