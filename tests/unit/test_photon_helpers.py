"""Tests for the photon helpers in ember.common."""

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st

from ember.common import (
    build_photon_propagation_output,
    init_spe_scaling_factor_distributions,
    photon_gain_calculation,
    pmt_gains,
    pmt_transit_time_spread,
    sample_spe_scaling_factors,
)
from ember.dtypes import propagated_photons_fields

PHOTON_DTYPE = propagated_photons_fields + strax.time_fields


def gaussian_spe_table(n_channels: int, mean: float = 1.0, width: float = 0.3) -> pd.DataFrame:
    charge = np.round(np.arange(-1.0, 5.0, 0.01), 2)
    spectrum = np.exp(-0.5 * ((charge - mean) / width) ** 2)
    columns = {"charge": charge}
    for channel in range(n_channels):
        columns[str(channel)] = spectrum
    return pd.DataFrame(columns)


def test_pmt_gains_known_values():
    gains = pmt_gains(
        np.array([0.005, 0.0, 0.01]),
        digitizer_voltage_range=2.25,
        digitizer_bits=14,
        pmt_circuit_load_resistor=8.010882825e-09,
    )
    adc_to_current = 2.25 / 2**14 / 8.010882825e-09
    np.testing.assert_allclose(gains, [adc_to_current / 0.005, 0.0, adc_to_current / 0.01])


def test_spe_sampling_reproduces_mean_and_width():
    distributions = init_spe_scaling_factor_distributions(gaussian_spe_table(2), n_channels=2)
    rng = np.random.default_rng(0)
    factors = sample_spe_scaling_factors(
        rng.random(200000), np.zeros(200000, dtype=int), distributions
    )
    assert np.median(factors) == pytest.approx(1.0, abs=0.01)
    central_width = (np.quantile(factors, 0.8413) - np.quantile(factors, 0.1587)) / 2
    assert central_width == pytest.approx(0.3, abs=0.01)


def test_last_lookup_entry_is_the_largest_charge_as_in_fuse():
    """fuse's lookup maps the cumulative probability 1.0 to the last charge bin (4.99 PE).

    Hence a fraction 1/2000 of the photons gets this charge. Kept for compatibility with fuse.
    """
    distributions = init_spe_scaling_factor_distributions(gaussian_spe_table(1), n_channels=1)
    assert distributions[0, -1] == pytest.approx(4.99)
    assert distributions[0, -2] < 2.5


def test_spe_table_with_index_column_gives_same_result():
    table = gaussian_spe_table(3)
    table_with_index = table.reset_index().rename(columns={"index": "Unnamed: 0"})
    np.testing.assert_array_equal(
        init_spe_scaling_factor_distributions(table, 3),
        init_spe_scaling_factor_distributions(table_with_index, 3),
    )


def test_spe_channels_are_selected_by_name():
    table = gaussian_spe_table(2)
    table["1"] = np.exp(-0.5 * ((table["charge"] - 2.0) / 0.3) ** 2)
    distributions = init_spe_scaling_factor_distributions(table[["1", "charge", "0"]], 2)
    assert distributions[0, 1000] == pytest.approx(1.0, abs=0.02)
    assert distributions[1, 1000] == pytest.approx(2.0, abs=0.02)


def test_spe_table_without_channel_raises():
    with pytest.raises(ValueError, match="Missing channels"):
        init_spe_scaling_factor_distributions(gaussian_spe_table(2), n_channels=3)


def test_empty_spe_spectrum_gives_zero_factors():
    table = gaussian_spe_table(2)
    table["1"] = 0.0
    distributions = init_spe_scaling_factor_distributions(table, 2)
    np.testing.assert_array_equal(distributions[1], 0.0)


def test_transit_time_mean_and_spread():
    rng = np.random.default_rng(1)
    times = pmt_transit_time_spread(np.zeros(200000, dtype=np.int64), 46.0, 9.0, rng)
    assert times.dtype == np.int64
    assert times.mean() == pytest.approx(45.5, abs=0.1)  # int64 truncation lowers the mean
    assert times.std() == pytest.approx(9.0 / 2.35482, abs=0.1)


def test_dpe_fraction_and_gains():
    n_channels = 4
    distributions = init_spe_scaling_factor_distributions(gaussian_spe_table(n_channels), 4)
    gains = np.array([1e6, 2e6, 3e6, 4e6])
    rng = np.random.default_rng(2)
    channels = rng.integers(0, n_channels, 100000)
    photon_gains, photon_is_dpe = photon_gain_calculation(channels, 0.2, gains, distributions, rng)
    assert photon_is_dpe.mean() == pytest.approx(0.2, abs=0.01)
    single_pe_ratio = photon_gains[~photon_is_dpe] / gains[channels[~photon_is_dpe]]
    double_pe_ratio = photon_gains[photon_is_dpe] / gains[channels[photon_is_dpe]]
    assert single_pe_ratio.mean() == pytest.approx(1.0, abs=0.01)
    assert double_pe_ratio.mean() == pytest.approx(2.0, abs=0.02)


@settings(deadline=None, max_examples=50)
@given(
    times=st.lists(st.integers(0, 10**12), min_size=0, max_size=50),
    photon_type=st.sampled_from([0, 1, 2]),
)
def test_photon_output_is_consistent(times, photon_type):
    n_photons = len(times)
    result = build_photon_propagation_output(
        dtype=PHOTON_DTYPE,
        photon_timings=np.array(times, dtype=np.int64),
        photon_channels=np.arange(n_photons) % 7,
        photon_gains=np.full(n_photons, 1e6),
        photon_is_dpe=np.zeros(n_photons, dtype=bool),
        cluster_ids=np.arange(n_photons),
        photon_type=photon_type,
    )
    assert len(result) == n_photons
    np.testing.assert_array_equal(result["time"], result["endtime"])
    np.testing.assert_array_equal(result["time"], times)
    assert np.all(result["photon_type"] == photon_type)
