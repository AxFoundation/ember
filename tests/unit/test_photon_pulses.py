"""Tests of the PulseWindow plugin and concat_overlapping_hits."""

import numpy as np
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st

import ember
from ember.dtypes import propagated_photons_fields
from ember.plugins.pmt_and_daq.photon_pulses import concat_overlapping_hits


def detector_value(key: str):
    return strax.URLConfig.evaluate_dry(f"detector://{key}")


DT = detector_value("sample_duration")
N_TPC_PMTS = detector_value("n_tpc_pmts")
TEMPLATE_SAMPLES = detector_value("samples_before_pulse_center") + detector_value(
    "samples_after_pulse_center"
)
LEFT_EXTENSION = 2 * detector_value("samples_to_store_before") + detector_value(
    "samples_before_pulse_center"
)
RIGHT_EXTENSION = 2 * detector_value("samples_to_store_after") + detector_value(
    "samples_after_pulse_center"
)
PHOTON_DTYPE = propagated_photons_fields + strax.time_fields


def single_photon_pulses(times, channels) -> np.ndarray:
    hits = np.zeros(len(times), dtype=strax.interval_dtype)
    hits["time"] = times
    hits["channel"] = channels
    hits["dt"] = DT
    hits["length"] = TEMPLATE_SAMPLES
    return hits


def windows_of(times, channels, start=0, end=10**9, extensions=(LEFT_EXTENSION, RIGHT_EXTENSION)):
    return concat_overlapping_hits(
        single_photon_pulses(times, channels), extensions, N_TPC_PMTS, start, end
    )


def photons(times, channels) -> np.ndarray:
    data = np.zeros(len(times), dtype=PHOTON_DTYPE)
    data["time"] = times
    data["endtime"] = times
    data["channel"] = channels
    return data


def test_two_close_photons_in_one_channel_share_a_window():
    windows, pulse_ids = windows_of([100_000, 100_500], [3, 3])
    assert len(windows) == 1
    np.testing.assert_array_equal(pulse_ids, [0, 0])
    assert windows["time"][0] == 100_000 - LEFT_EXTENSION * DT
    expected_end = 100_500 + (TEMPLATE_SAMPLES + RIGHT_EXTENSION) * DT
    assert strax.endtime(windows)[0] == expected_end


def test_distant_photons_in_one_channel_get_two_windows():
    gap = (TEMPLATE_SAMPLES + LEFT_EXTENSION + RIGHT_EXTENSION) * DT + 1
    windows, pulse_ids = windows_of([100_000, 100_000 + gap], [3, 3])
    assert len(windows) == 2
    np.testing.assert_array_equal(pulse_ids, [0, 1])


def test_photons_in_different_channels_get_different_windows():
    windows, pulse_ids = windows_of([100_000, 100_000], [1, 2])
    assert len(windows) == 2
    assert pulse_ids[0] != pulse_ids[1]
    assert set(windows["channel"]) == {1, 2}


def test_window_starting_at_zero_keeps_all_its_photons():
    """fuse used time == 0 as 'no window yet' and lost such windows."""
    windows, pulse_ids = windows_of([50, 60, 70], [0, 0, 0], start=0)
    assert len(windows) == 1
    assert windows["time"][0] == 0
    np.testing.assert_array_equal(pulse_ids, [0, 0, 0])


def test_windows_are_clipped_to_the_chunk():
    windows, _ = windows_of([1_000, 5_000, 5_100, 99_900], [0, 1, 1, 2], start=500, end=100_000)
    assert windows["time"].min() >= 500
    assert strax.endtime(windows).max() <= 100_000


def test_channel_out_of_range_raises():
    with pytest.raises(ValueError):
        windows_of([100], [N_TPC_PMTS])


def test_no_photons_give_no_windows():
    windows, pulse_ids = windows_of([], [])
    assert len(windows) == len(pulse_ids) == 0


@settings(deadline=None, max_examples=100)
@given(
    times=st.lists(st.integers(0, 50_000), min_size=1, max_size=60),
    channels=st.lists(st.integers(0, 3), min_size=60, max_size=60),
)
def test_windows_contain_their_photons_and_do_not_overlap(times, channels):
    times = np.sort(np.array(times)) + 20_000
    channels = np.array(channels[: len(times)])
    windows, pulse_ids = windows_of(times, channels)

    assert len(np.unique(windows["pulse_id"])) == len(windows)
    assert set(pulse_ids) == set(windows["pulse_id"])
    window_of_photon = np.sort(windows, order="pulse_id")[pulse_ids]
    assert np.all(window_of_photon["channel"] == channels)
    assert np.all(window_of_photon["time"] <= times)
    assert np.all(strax.endtime(window_of_photon) >= times + TEMPLATE_SAMPLES * DT)

    for channel in np.unique(channels):
        in_channel = np.sort(windows[windows["channel"] == channel], order="time")
        assert np.all(in_channel["time"][1:] >= strax.endtime(in_channel)[:-1])


@pytest.fixture
def pulse_window_plugin(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    context.set_config({"input_file": str(tmp_path / "unused.csv")})
    return context.get_single_plugin("0", "pulse_windows")


def test_plugin_pulse_ids_are_unique_across_chunks(pulse_window_plugin):
    first = pulse_window_plugin.compute(photons([10_000, 10_100], [0, 1]), start=0, end=50_000)
    second = pulse_window_plugin.compute(
        photons([60_000, 60_100], [0, 1]), start=50_000, end=100_000
    )
    first_ids = set(first["pulse_windows"]["pulse_id"])
    second_ids = set(second["pulse_windows"]["pulse_id"])
    assert first_ids == {0, 1}
    assert second_ids == {2, 3}
    assert set(second["pulse_ids"]["pulse_id"]) == second_ids


def test_plugin_keeps_the_photon_order_in_pulse_ids(pulse_window_plugin):
    photon_times = [10_000, 10_000, 10_000, 30_000]
    result = pulse_window_plugin.compute(photons(photon_times, [5, 1, 5, 1]), start=0, end=50_000)
    np.testing.assert_array_equal(result["pulse_ids"]["time"], photon_times)
    windows = np.sort(result["pulse_windows"], order="pulse_id")
    np.testing.assert_array_equal(windows["channel"][result["pulse_ids"]["pulse_id"]], [5, 1, 5, 1])
    assert np.all(np.diff(result["pulse_windows"]["time"]) >= 0)


def test_plugin_without_photons(pulse_window_plugin):
    result = pulse_window_plugin.compute(photons([], []), start=0, end=100)
    assert len(result["pulse_windows"]) == len(result["pulse_ids"]) == 0


def test_more_windows_than_the_result_buffer_holds():
    """The numba generator hands over its buffer every 10 000 windows; none may be lost."""
    n_photons = 25_000
    times = 10_000 + np.arange(n_photons) * 100_000
    windows, pulse_ids = windows_of(times, np.zeros(n_photons, dtype=int), end=10**12)
    np.testing.assert_array_equal(np.sort(windows["pulse_id"]), np.arange(n_photons))
    np.testing.assert_array_equal(pulse_ids, np.arange(n_photons))
    np.testing.assert_array_equal(np.sort(windows["time"]), times - LEFT_EXTENSION * DT)
