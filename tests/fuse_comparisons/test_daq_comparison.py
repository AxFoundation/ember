"""Compare ember's and fuse's PMT response and DAQ for the same photons.

Exact: fuse's PulseWindow and PMTResponseAndDAQ on ember's photon_summary (noise off) must give
the same raw_records as ember. Statistical: fuse also simulates the afterpulses (with ember's
afterpulse file, but fuse's nearest-bin lookup and its own random numbers) and adds noise from
ember's noise file at its own random offsets.
"""

import numpy as np
import pytest
import strax

MAXIMAL_PULL = 5
BASELINE = strax.URLConfig.evaluate_dry("detector://digitizer_reference_baseline")
PMT_TO_PE = strax.URLConfig.evaluate_dry("detector://pmt_to_pe")


def raw_records_pe(raw_records: np.ndarray) -> float:
    in_record = np.arange(raw_records["data"].shape[1])[None, :] < raw_records["length"][:, None]
    return np.sum((BASELINE - raw_records["data"].astype(np.int64)) * in_record) * PMT_TO_PE


def test_raw_records_are_identical_without_noise(daq_simulations):
    ember_records = daq_simulations["exact"]["ember"]["raw_records"]
    fuse_records = daq_simulations["exact"]["fuse"]["raw_records"]
    assert len(ember_records) > 100_000
    np.testing.assert_array_equal(ember_records, fuse_records)


def test_pulse_windows_are_identical(daq_simulations):
    ember_windows = daq_simulations["exact"]["ember"]["pulse_windows"]
    fuse_windows = daq_simulations["exact"]["fuse"]["pulse_windows"]
    for field in ["time", "length", "channel", "dt"]:
        np.testing.assert_array_equal(ember_windows[field], fuse_windows[field])


def test_number_of_afterpulses_agrees(daq_simulations):
    ember_count = len(daq_simulations["statistical"]["ember"]["pmt_afterpulses"])
    fuse_count = len(daq_simulations["statistical"]["fuse"]["pmt_afterpulses"])
    photons = daq_simulations["photons"]
    s1_s2_photons = photons[photons["photon_type"] > 0]
    expected = 0.0012 * (len(s1_s2_photons) + s1_s2_photons["dpe"].sum())
    for count in [ember_count, fuse_count]:
        assert abs(count - expected) / np.sqrt(expected) < MAXIMAL_PULL
    assert abs(ember_count - fuse_count) / np.sqrt(ember_count + fuse_count) < MAXIMAL_PULL


def test_afterpulse_sizes_agree(daq_simulations):
    spe_gain = daq_simulations["spe_gain"]
    sizes = {
        name: daq_simulations["statistical"][name]["pmt_afterpulses"]["photon_gain"] / spe_gain
        for name in ["ember", "fuse"]
    }
    uncertainty = np.sqrt(
        np.var(sizes["ember"]) / len(sizes["ember"]) + np.var(sizes["fuse"]) / len(sizes["fuse"])
    )
    assert abs(sizes["ember"].mean() - sizes["fuse"].mean()) / uncertainty < MAXIMAL_PULL


def test_ember_has_no_afterpulses_without_charge(daq_simulations):
    """fuse's nearest-bin lookup can pick the size bin 0 (0 PE); the inverse CDF cannot."""
    ember_afterpulses = daq_simulations["statistical"]["ember"]["pmt_afterpulses"]
    fuse_afterpulses = daq_simulations["statistical"]["fuse"]["pmt_afterpulses"]
    assert np.all(ember_afterpulses["photon_gain"] > 0)
    print("fuse afterpulses with 0 PE:", np.mean(fuse_afterpulses["photon_gain"] == 0))


def test_number_of_raw_records_agrees(daq_simulations):
    ember_count = len(daq_simulations["statistical"]["ember"]["raw_records"])
    fuse_count = len(daq_simulations["statistical"]["fuse"]["raw_records"])
    assert ember_count == pytest.approx(fuse_count, rel=0.005)


def test_total_area_agrees(daq_simulations):
    ember_pe = raw_records_pe(daq_simulations["statistical"]["ember"]["raw_records"])
    fuse_pe = raw_records_pe(daq_simulations["statistical"]["fuse"]["raw_records"])
    assert ember_pe == pytest.approx(fuse_pe, rel=1e-3)


def test_noise_level_agrees(daq_simulations):
    noise_sigma = {}
    for name in ["ember", "fuse"]:
        first_records = daq_simulations["statistical"][name]["raw_records"]
        first_records = first_records[first_records["record_i"] == 0]
        noise_sigma[name] = np.std(first_records["data"][:, :20].astype(np.float64))
    assert noise_sigma["ember"] == pytest.approx(noise_sigma["fuse"], rel=0.02)
