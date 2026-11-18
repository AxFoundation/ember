"""Helper functions shared by ember plugins (ported from ``fuse.common``)."""

import numba
import numpy as np
import pandas as pd
from numba.extending import register_jitable
from scipy.interpolate import interp1d

VOLUMES_IDS = {
    "undefined": 0,
    "tpc": 1,
    "below_cathode": 2,
    "gas_phase": 3,
}


def stable_sort(array: np.ndarray, **kwargs) -> np.ndarray:
    """Sort with the stable mergesort algorithm.

    :param array: array to sort.
    :param kwargs: forwarded to ``np.sort`` (e.g. ``order``).
    :return: sorted copy of the array.
    """
    return np.sort(array, kind="mergesort", **kwargs)


@register_jitable
def stable_argsort(array: np.ndarray) -> np.ndarray:
    """Return the indices that sort the array, using the stable mergesort algorithm.

    Can be called from numba-jitted functions.

    :param array: array to sort.
    :return: sort indices.
    """
    return np.argsort(array, kind="mergesort")


@numba.njit()
def dynamic_chunking(data: np.ndarray, scale: float, n_min: int) -> np.ndarray:
    """Group values into chunks separated by gaps larger than ``scale``.

    A new chunk is only started at a gap larger than ``scale`` once the current chunk holds at
    least ``n_min`` values. Hence all chunks except the last contain at least ``n_min`` values.

    :param data: one dimensional array, e.g. interaction times.
    :param scale: minimal gap between two chunks.
    :param n_min: minimal number of values per chunk.
    :return: chunk index for each value, in the order of ``data``.
    """
    idx_sort = stable_argsort(data)
    idx_undo_sort = stable_argsort(idx_sort)

    data_sorted = data[idx_sort]
    diff = data_sorted[1:] - data_sorted[:-1]

    clusters = [0]
    cluster_index = 0
    n_cluster = 0
    for value in diff:
        if value <= scale:
            clusters.append(cluster_index)
            n_cluster += 1
        elif n_cluster + 1 < n_min:
            clusters.append(cluster_index)
            n_cluster += 1
        else:
            cluster_index = cluster_index + 1
            clusters.append(cluster_index)
            n_cluster = 0

    return np.array(clusters)[idx_undo_sort]


# Photon helpers shared by the S1 and S2 photon propagation

SPE_CDF_GRID_POINTS = 2001
FWHM_TO_SIGMA = 1 / 2.35482


def pmt_gains(
    to_pe: np.ndarray,
    digitizer_voltage_range: float,
    digitizer_bits: int,
    pmt_circuit_load_resistor: float,
) -> np.ndarray:
    """Convert per-channel ADC-to-PE factors into PMT gains (as in fuse).

    :param to_pe: ADC-to-PE factor per channel; 0 marks a switched-off channel.
    :param digitizer_voltage_range: voltage range of the digitizer [V].
    :param digitizer_bits: resolution of the digitizer.
    :param pmt_circuit_load_resistor: fuse's conversion constant from ADC counts to PMT gain.
    :return: gain per channel, 0 for switched-off channels.
    """
    to_pe = np.asarray(to_pe, dtype=np.float64)
    adc_to_current = digitizer_voltage_range / 2**digitizer_bits / pmt_circuit_load_resistor
    return np.divide(adc_to_current, to_pe, out=np.zeros_like(to_pe), where=to_pe != 0)


def init_spe_scaling_factor_distributions(spe_shapes: pd.DataFrame, n_channels: int) -> np.ndarray:
    """Build a lookup table from uniform random numbers to SPE scaling factors, per channel.

    Unlike fuse, the columns are selected by name (``charge`` and ``"0"``, ``"1"``, ...), so an
    additional index column in the CSV file does not shift the channels.

    :param spe_shapes: table with a ``charge`` column and one column per channel.
    :param n_channels: number of channels.
    :return: array of shape ``(n_channels, SPE_CDF_GRID_POINTS)``: the inverse CDF of each channel
        on an even grid of cumulative probabilities from 0 to 1.
    """
    missing_channels = [str(ch) for ch in range(n_channels) if str(ch) not in spe_shapes.columns]
    if "charge" not in spe_shapes.columns or missing_channels:
        raise ValueError(
            "The SPE table needs a 'charge' column and one column per channel. "
            f"Missing channels: {missing_channels[:10]}"
        )
    charge = spe_shapes["charge"].to_numpy(dtype=np.float64)
    grid_cdf = np.linspace(0, 1, SPE_CDF_GRID_POINTS)
    inverse_cdfs = np.zeros((n_channels, SPE_CDF_GRID_POINTS))
    for channel in range(n_channels):
        spectrum = spe_shapes[str(channel)].to_numpy(dtype=np.float64)
        if spectrum.sum() > 0:
            cdf = np.cumsum(spectrum) / np.sum(spectrum)
            scaled_bins = charge
        else:
            cdf = np.linspace(0, 1, 10)
            scaled_bins = np.zeros_like(cdf)
        inverse_cdfs[channel] = interp1d(
            cdf,
            scaled_bins,
            kind="next",
            bounds_error=False,
            fill_value=(scaled_bins[0], scaled_bins[-1]),
        )(grid_cdf)
    return inverse_cdfs


def sample_spe_scaling_factors(
    uniform_numbers: np.ndarray, channels: np.ndarray, spe_scaling_factor_distributions: np.ndarray
) -> np.ndarray:
    """Draw SPE scaling factors from the per-channel lookup table (as in fuse, vectorized).

    :param uniform_numbers: random numbers in [0, 1), one per photon.
    :param channels: channel of each photon.
    :param spe_scaling_factor_distributions: lookup table from
        :func:`init_spe_scaling_factor_distributions`.
    :return: SPE scaling factor of each photon.
    """
    n_grid_steps = spe_scaling_factor_distributions.shape[1] - 1
    indices = np.int64(uniform_numbers * n_grid_steps) + 1
    indices = np.minimum(indices, n_grid_steps)
    return spe_scaling_factor_distributions[channels, indices]


def pmt_transit_time_spread(
    photon_timings: np.ndarray,
    pmt_transit_time_mean: float,
    pmt_transit_time_spread: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add the PMT transit time to the photon times.

    :param photon_timings: photon times [ns], int64.
    :param pmt_transit_time_mean: mean transit time [ns].
    :param pmt_transit_time_spread: FWHM of the transit time [ns], as given in PMT datasheets.
    :param rng: random number generator.
    :return: photon times including the transit time [ns].
    """
    return photon_timings + rng.normal(
        pmt_transit_time_mean, pmt_transit_time_spread * FWHM_TO_SIGMA, len(photon_timings)
    ).astype(np.int64)


def photon_gain_calculation(
    photon_channels: np.ndarray,
    p_double_pe_emission: float,
    gains: np.ndarray,
    spe_scaling_factor_distributions: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw the gain of each photon, including double photoelectron emission (as in fuse).

    :param photon_channels: channel of each photon.
    :param p_double_pe_emission: probability that a photon releases two photoelectrons.
    :param gains: PMT gain per channel.
    :param spe_scaling_factor_distributions: lookup table from
        :func:`init_spe_scaling_factor_distributions`.
    :param rng: random number generator.
    :return: gain of each photon and a boolean array marking double photoelectron emission.
    """
    photon_is_dpe = rng.binomial(n=1, p=p_double_pe_emission, size=len(photon_channels)).astype(
        np.bool_
    )
    spe_scaling_factors = sample_spe_scaling_factors(
        rng.random(len(photon_channels)), photon_channels, spe_scaling_factor_distributions
    )
    photon_gains = gains[photon_channels] * spe_scaling_factors

    dpe_channels = photon_channels[photon_is_dpe]
    spe_scaling_factors_dpe = sample_spe_scaling_factors(
        rng.random(len(dpe_channels)), dpe_channels, spe_scaling_factor_distributions
    )
    photon_gains[photon_is_dpe] += gains[dpe_channels] * spe_scaling_factors_dpe
    return photon_gains, photon_is_dpe


def build_photon_propagation_output(
    dtype: np.dtype,
    photon_timings: np.ndarray,
    photon_channels: np.ndarray,
    photon_gains: np.ndarray,
    photon_is_dpe: np.ndarray,
    cluster_ids: np.ndarray,
    photon_type: int,
) -> np.ndarray:
    """Fill the photon output array (one row per photon, ``endtime = time``).

    :param dtype: output dtype (``propagated_photons_fields`` + time fields).
    :param photon_timings: arrival times [ns].
    :param photon_channels: channels.
    :param photon_gains: gains.
    :param photon_is_dpe: double photoelectron emission flags.
    :param cluster_ids: cluster id of the interaction that produced each photon.
    :param photon_type: 1 for S1, 2 for S2, 0 for PMT afterpulses.
    :return: the structured output array.
    """
    result = np.zeros(len(photon_channels), dtype=dtype)
    result["time"] = photon_timings
    result["endtime"] = photon_timings
    result["channel"] = photon_channels
    result["photon_gain"] = photon_gains
    result["dpe"] = photon_is_dpe
    result["cluster_id"] = cluster_ids
    result["photon_type"] = photon_type
    return result
