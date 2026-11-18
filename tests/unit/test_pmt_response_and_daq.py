"""Tests of the PMTResponseAndDAQ plugin and its helpers."""

import numpy as np
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st

import ember
from ember.common import pmt_gains
from ember.plugins.pmt_and_daq.photon_pulses import concat_overlapping_hits
from ember.plugins.pmt_and_daq.pmt_response_and_daq import (
    RECORD_LENGTH,
    add_current,
    build_waveform,
    find_intervals_below_threshold,
    find_split_index,
    init_pmt_current_templates,
    photon_range_per_window,
    zle_thresholds,
)


def detector_value(key: str):
    return strax.URLConfig.evaluate_dry(f"detector://{key}")


DT = detector_value("sample_duration")
N_TPC_PMTS = detector_value("n_tpc_pmts")
BASELINE = detector_value("digitizer_reference_baseline")
SAMPLES_BEFORE = detector_value("samples_before_pulse_center")
SAMPLES_AFTER = detector_value("samples_after_pulse_center")
TEMPLATE_SAMPLES = SAMPLES_BEFORE + SAMPLES_AFTER
EXTENSIONS = (
    2 * detector_value("samples_to_store_before") + SAMPLES_BEFORE,
    2 * detector_value("samples_to_store_after") + SAMPLES_AFTER,
)
PMT_TO_PE = detector_value("pmt_to_pe")
SPE_GAIN = pmt_gains(
    np.array([PMT_TO_PE]),
    digitizer_voltage_range=detector_value("digitizer_voltage_range"),
    digitizer_bits=detector_value("digitizer_bits"),
    pmt_circuit_load_resistor=detector_value("pmt_circuit_load_resistor"),
)[0]
AREA_PER_PE = detector_value("external_amplification") / DT / PMT_TO_PE  # ADC samples
PULSE_SHAPE = strax.URLConfig.evaluate_dry("ember_file://pmt_pulse_shape_dummy.json")
ADC_MAXIMUM = 2 ** int(detector_value("digitizer_bits")) - 1


def merged_photons(times, channels, gains, start=0, end=10**9):
    """``photon_summary`` + ``pulse_ids`` rows and the matching pulse windows."""
    times = np.asarray(times, dtype=np.int64)
    order = np.argsort(times, kind="stable")
    times, channels, gains = times[order], np.asarray(channels)[order], np.asarray(gains)[order]
    hits = np.zeros(len(times), dtype=strax.interval_dtype)
    hits["time"] = times
    hits["channel"] = channels
    hits["dt"] = DT
    hits["length"] = TEMPLATE_SAMPLES
    windows, pulse_ids = concat_overlapping_hits(hits, EXTENSIONS, N_TPC_PMTS, start, end)
    photons = np.zeros(
        len(times),
        dtype=strax.time_fields
        + [("channel", np.int16), ("photon_gain", np.int32), ("pulse_id", np.int64)],
    )
    photons["time"] = times
    photons["endtime"] = times
    photons["channel"] = channels
    photons["photon_gain"] = gains
    photons["pulse_id"] = pulse_ids
    return photons, strax.sort_by_time(windows)


def make_plugin(folder, **config):
    context = ember.ember_context(output_folder=folder / "data")
    context.set_config({"input_file": str(folder / "unused.csv"), **config})
    return context.get_single_plugin("0", "raw_records")


@pytest.fixture(scope="module")
def plugin_without_noise(tmp_path_factory):
    return make_plugin(tmp_path_factory.mktemp("daq"), enable_noise=False)


def run(plugin, times, channels, gains, start=0, end=10**9):
    photons, windows = merged_photons(times, channels, gains, start, end)
    return list(plugin.compute(photons, windows, start=start, end=end))


def baseline_subtracted_area(records):
    area = 0
    for record in records:
        area += np.sum(BASELINE - record["data"][: record["length"]].astype(np.int64))
    return area


# --- helpers -----------------------------------------------------------------------------


def test_templates_sum_to_one_pe_per_sample_duration():
    templates = init_pmt_current_templates(
        PULSE_SHAPE["pe_pulse_ts"], PULSE_SHAPE["pe_pulse_ys"], DT, SAMPLES_BEFORE, SAMPLES_AFTER, 1
    )
    assert templates.shape == (DT, TEMPLATE_SAMPLES)
    np.testing.assert_allclose(templates.sum(axis=1), 1 / DT)


def test_templates_need_1ns_rounding():
    with pytest.raises(ValueError):
        init_pmt_current_templates([0, 1], [1, 1], DT, SAMPLES_BEFORE, SAMPLES_AFTER, 2)


def test_zle_thresholds_with_special_channels():
    thresholds = zle_thresholds(4, 16000, 15, {"2": 30, "9": 50})
    np.testing.assert_array_equal(thresholds, [15984, 15984, 15969, 15984])


def test_intervals_below_threshold_known_waveform():
    waveform = np.array([10, 10, 1, 10, 10, 10, 10, 1, 1, 10, 10, 10, 10, 10, 1])
    result = np.zeros((7, 2), dtype=np.int64)
    n_intervals = find_intervals_below_threshold(waveform, 5, 3, result)
    np.testing.assert_array_equal(result[:n_intervals], [[2, 2], [7, 8], [14, 14]])


def test_intervals_closer_than_the_holdoff_are_merged():
    waveform = np.array([10, 1, 10, 10, 1, 10, 10, 10, 10, 10])
    result = np.zeros((5, 2), dtype=np.int64)
    n_intervals = find_intervals_below_threshold(waveform, 5, 4, result)
    np.testing.assert_array_equal(result[:n_intervals], [[1, 4]])


def test_intervals_with_an_empty_buffer():
    waveform = np.array([10, 0, 10])
    assert find_intervals_below_threshold(waveform, 5, 1, np.zeros((0, 2), dtype=np.int64)) == 0


def test_intervals_stop_when_the_buffer_is_full():
    waveform = np.array([1, 10, 1, 10, 1, 10])
    result = np.zeros((2, 2), dtype=np.int64)
    assert find_intervals_below_threshold(waveform, 5, 1, result) == 2


def windows(times, lengths):
    data = np.zeros(len(times), dtype=strax.interval_dtype)
    data["time"] = times
    data["length"] = lengths
    data["dt"] = 1
    return data


def test_split_index_known_case():
    pulses = windows([0, 100, 10_000, 20_000], [50, 50, 50, 50])
    np.testing.assert_array_equal(find_split_index(pulses, 1e-4, 1000), [2, 3])


def test_split_index_respects_windows_that_are_still_open():
    """The long first window ends after the second window: no split between 2 and 3."""
    pulses = windows([0, 100, 10_000], [20_000, 50, 50])
    np.testing.assert_array_equal(find_split_index(pulses, 1e-6, 1000), [])


@settings(deadline=None, max_examples=100)
@given(
    starts=st.lists(st.integers(0, 10**6), min_size=1, max_size=40),
    lengths=st.lists(st.integers(1, 10**5), min_size=40, max_size=40),
    min_gap=st.integers(1, 10**5),
)
def test_splits_are_only_made_at_large_gaps(starts, lengths, min_gap):
    pulses = windows(np.sort(starts), lengths[: len(starts)])
    split_index = find_split_index(pulses, 1e-6, min_gap)
    latest_end = np.maximum.accumulate(strax.endtime(pulses))
    for index in split_index:
        assert 0 < index < len(pulses)
        assert pulses["time"][index] - latest_end[index - 1] >= min_gap


def test_window_without_photons_raises():
    with pytest.raises(ValueError):
        photon_range_per_window(np.array([0, 0, 2]), windows([0, 10, 20], [1, 1, 1]))


def test_photons_with_equal_times_are_summed():
    templates = np.tile(np.arange(1.0, 4.0), (10, 1))
    summed = np.zeros(10)
    add_current(np.array([25, 25]), np.array([2.0, 3.0]), 0, 10, templates, summed)
    np.testing.assert_allclose(summed[2:5], 5.0 * np.arange(1.0, 4.0))
    np.testing.assert_allclose(np.delete(summed, [2, 3, 4]), 0.0)


def test_template_reaching_beyond_the_window_is_cut():
    templates = np.ones((10, 3))
    current = np.zeros(4)
    add_current(np.array([30]), np.array([1.0]), 0, 10, templates, current)
    np.testing.assert_allclose(current, [0, 0, 0, 1])


# --- plugin ----------------------------------------------------------------------------


def test_single_photon_gives_one_pe_of_area(plugin_without_noise):
    (chunk,) = run(plugin_without_noise, [1_000_000], [7], [SPE_GAIN])
    records = chunk.data
    assert len(records) == 1
    assert records["channel"][0] == 7
    assert baseline_subtracted_area(records) == pytest.approx(AREA_PER_PE, rel=0.02)


def test_pulse_maximum_is_two_samples_after_the_photon(plugin_without_noise):
    """fuse's template placement shifts the pulse by samples_before_pulse_center."""
    photon_time = 1_000_000
    (chunk,) = run(plugin_without_noise, [photon_time], [7], [SPE_GAIN])
    record = chunk.data[0]
    maximum_time = record["time"] + DT * np.argmin(record["data"][: record["length"]])
    assert maximum_time == photon_time + SAMPLES_BEFORE * DT


def test_records_of_one_pulse_are_consistent(plugin_without_noise):
    times = 1_000_000 + np.arange(0, 3000, 10)
    (chunk,) = run(plugin_without_noise, times, np.zeros(len(times)), np.full(len(times), SPE_GAIN))
    records = chunk.data
    assert len(records) > 1
    np.testing.assert_array_equal(records["record_i"], np.arange(len(records)))
    assert np.all(records["pulse_length"] == records["pulse_length"][0])
    assert records["length"].sum() == records["pulse_length"][0]
    np.testing.assert_array_equal(
        records["time"][1:], records["time"][:-1] + strax.DEFAULT_RECORD_LENGTH * DT
    )
    assert np.all(records["time"] % (2 * DT) == records["time"][0] % (2 * DT))
    assert np.all(records["dt"] == DT)
    last = records[-1]
    assert np.all(last["data"][last["length"] :] == 0)


def test_tiny_pulse_below_threshold_gives_no_records(plugin_without_noise):
    (chunk,) = run(plugin_without_noise, [1_000_000], [3], [SPE_GAIN // 100])
    assert len(chunk.data) == 0


def test_no_photons_give_an_empty_chunk(plugin_without_noise):
    empty_photons, empty_windows = merged_photons([], [], [])
    (chunk,) = plugin_without_noise.compute(empty_photons, empty_windows, start=0, end=100)
    assert len(chunk.data) == 0
    assert (chunk.start, chunk.end) == (0, 100)


def test_records_can_be_processed_by_strax(plugin_without_noise):
    (chunk,) = run(plugin_without_noise, [1_000_000, 1_000_050], [1, 1], [SPE_GAIN, SPE_GAIN])
    records = strax.raw_to_records(chunk.data)
    strax.baseline(records, baseline_samples=40, flip=True)
    strax.integrate(records)
    assert records["area"].sum() == pytest.approx(2 * AREA_PER_PE, rel=0.02)


def test_down_chunking_gives_contiguous_chunks(tmp_path):
    plugin = make_plugin(
        tmp_path,
        enable_noise=False,
        raw_records_file_size_target=1e-4,
        min_records_gap_length_for_splitting=10_000,
    )
    times = np.array([1_000_000, 1_000_100, 2_000_000, 3_000_000, 3_000_010])
    chunks = run(plugin, times, [1, 2, 1, 3, 3], np.full(5, SPE_GAIN), start=0, end=10**7)
    assert len(chunks) > 1
    assert chunks[0].start == 0 and chunks[-1].end == 10**7
    for previous, following in zip(chunks[:-1], chunks[1:]):
        assert previous.end == following.start
    for chunk in chunks:
        if len(chunk.data):
            assert chunk.data["time"].min() >= chunk.start
            assert strax.endtime(chunk.data).max() <= chunk.end
    assert sum(len(chunk.data) for chunk in chunks) == 4


def test_noise_level_in_the_records(tmp_path):
    plugin = make_plugin(tmp_path)
    (chunk,) = run(plugin, [1_000_000], [5], [SPE_GAIN])
    record = chunk.data[0]
    pre_pulse_samples = record["data"][:40].astype(np.float64)
    noise_sigma = detector_value("noise_sigma")
    assert pre_pulse_samples.mean() == pytest.approx(BASELINE, abs=2)
    assert 0.5 * noise_sigma < pre_pulse_samples.std() < 2 * noise_sigma


def test_wrong_noise_shape_raises(tmp_path):
    with pytest.raises(ValueError, match="Noise data"):
        make_plugin(tmp_path, noise_data={"arr_0": np.zeros((100, 3))})


@settings(deadline=None, max_examples=25)
@given(
    times=st.lists(st.integers(0, 200_000), min_size=1, max_size=40),
    channels=st.lists(st.integers(0, 3), min_size=40, max_size=40),
    gain_factors=st.lists(st.floats(0.5, 3.0), min_size=40, max_size=40),
)
def test_records_do_not_overlap_and_keep_the_area(
    plugin_without_noise, times, channels, gain_factors
):
    times = np.array(times) + 1_000_000
    gains = (SPE_GAIN * np.array(gain_factors[: len(times)])).astype(np.int32)
    chunks = run(plugin_without_noise, times, channels[: len(times)], gains, start=0, end=2 * 10**6)
    records = np.concatenate([chunk.data for chunk in chunks])
    for channel in np.unique(records["channel"]):
        in_channel = records[records["channel"] == channel]
        assert np.all(in_channel["time"][1:] >= strax.endtime(in_channel)[:-1])
    expected_area = gains.sum() / SPE_GAIN * AREA_PER_PE
    assert baseline_subtracted_area(records) == pytest.approx(expected_area, rel=0.02, abs=20)


# --- waveform loop against a straightforward reference ------------------------------------


def reference_window_records(plugin, window, times, gains, enable_noise, noise_offset):
    """Records of one window, built step by step with numpy (the pre-0.0.3 algorithm, plus the
    clipping to the digitizer range before the int16 cast)."""
    current = np.zeros(window["length"])
    add_current(times, gains, window["time"] // DT, DT, plugin.pmt_current_templates, current)
    waveform = -np.around(current * plugin.current_2_adc).astype(np.int64)
    if enable_noise:
        noise = plugin.noise_per_channel[window["channel"]]
        first_index = window["time"] // DT + noise_offset + 1
        waveform += noise[(first_index + np.arange(len(waveform))) % len(noise)]
    waveform = np.clip(waveform + BASELINE, 0, ADC_MAXIMUM)

    intervals = np.zeros((len(waveform) // 2, 2), dtype=np.int64)
    n_intervals = find_intervals_below_threshold(
        waveform,
        plugin.thresholds[window["channel"]],
        2 * plugin.trigger_window + 1,
        intervals,
    )
    records = []
    for left, right in intervals[:n_intervals]:
        left = int(np.ceil(np.clip(left - plugin.trigger_window, 0, len(waveform) - 1) / 2) * 2)
        right = int(np.floor(np.clip(right + plugin.trigger_window, 0, len(waveform) - 1) / 2) * 2)
        pulse_length = right - left + 1
        for record_i, first_sample in enumerate(range(left, right + 1, RECORD_LENGTH)):
            record = np.zeros(1, dtype=plugin.dtype)[0]
            n_samples = min(RECORD_LENGTH, right + 1 - first_sample)
            record["channel"] = window["channel"]
            record["time"] = DT * (window["time"] // DT + first_sample)
            record["dt"] = DT
            record["pulse_length"] = pulse_length
            record["record_i"] = record_i
            record["length"] = n_samples
            record["data"][:n_samples] = waveform[first_sample : first_sample + n_samples]
            records.append(record)
    return records


def records_from_build_waveform(plugin, photons, windows, enable_noise, noise_offset):
    first_photon, stop_photon = photon_range_per_window(photons["pulse_id"], windows)
    buffer = np.zeros(int(np.ceil(windows["length"] / RECORD_LENGTH).sum()), dtype=plugin.dtype)
    n_records = build_waveform(
        windows,
        first_photon,
        stop_photon,
        photons["time"],
        photons["photon_gain"].astype(np.float64),
        buffer,
        DT,
        plugin.pmt_current_templates,
        plugin.current_2_adc,
        plugin.noise_per_channel,
        enable_noise,
        BASELINE,
        plugin.thresholds,
        int(plugin.trigger_window),
        noise_offset,
        ADC_MAXIMUM,
    )
    return buffer[:n_records], first_photon, stop_photon


@settings(deadline=None, max_examples=40)
@given(
    times=st.lists(st.integers(0, 50_000), min_size=1, max_size=30),
    channels=st.lists(st.integers(0, 3), min_size=30, max_size=30),
    gain_factors=st.lists(st.floats(0.01, 3.0), min_size=30, max_size=30),
    n_saturating_photons=st.sampled_from([0, 500, 1100, 3000]),
    enable_noise=st.booleans(),
    noise_offset=st.integers(0, 10_000),
)
def test_build_waveform_matches_the_reference(
    plugin_without_noise,
    times,
    channels,
    gain_factors,
    n_saturating_photons,
    enable_noise,
    noise_offset,
):
    """Includes optional bursts of photons at one time in one channel that saturate the
    digitizer (below -32768 ADC; 500 and 1100 photons wrapped to positive values before 0.0.3)."""
    times = np.append(np.array(times), np.full(n_saturating_photons, times[0])) + 1_000_000
    channels = np.append(
        channels[: len(times) - n_saturating_photons], np.zeros(n_saturating_photons)
    )
    gains = SPE_GAIN * np.append(
        gain_factors[: len(times) - n_saturating_photons], np.ones(n_saturating_photons)
    )
    photons, windows = merged_photons(times, channels.astype(np.int16), gains.astype(np.int32))
    photon_order = np.argsort(photons["pulse_id"], kind="stable")
    photons = photons[photon_order]

    records, first_photon, stop_photon = records_from_build_waveform(
        plugin_without_noise, photons, windows, enable_noise, noise_offset
    )

    expected = []
    for window, first, stop in zip(windows, first_photon, stop_photon):
        expected += reference_window_records(
            plugin_without_noise,
            window,
            photons["time"][first:stop],
            photons["photon_gain"][first:stop].astype(np.float64),
            enable_noise,
            noise_offset,
        )
    expected = np.array(expected, dtype=plugin_without_noise.dtype)
    assert records.tobytes() == expected.tobytes()


@pytest.mark.parametrize("n_photons", [500, 1100, 3000])
def test_saturated_pulses_stay_at_zero(plugin_without_noise, n_photons):
    """Pulses far below -32768 ADC must not wrap around to positive int16 values (before 0.0.3,
    500 and 1100 photons in one ns gave samples of 26 377 and 25 723 ADC)."""
    (chunk,) = run(
        plugin_without_noise,
        np.full(n_photons, 1_000_000),
        np.full(n_photons, 7),
        np.full(n_photons, SPE_GAIN),
    )
    data = np.concatenate([record["data"][: record["length"]] for record in chunk.data])
    assert np.any(data == 0)
    assert data.min() >= 0
    assert data.max() <= BASELINE


def test_samples_stay_below_the_digitizer_maximum(tmp_path):
    plugin = make_plugin(
        tmp_path, enable_noise=False, digitizer_reference_baseline=ADC_MAXIMUM + 1000
    )
    (chunk,) = run(plugin, [1_000_000], [7], [SPE_GAIN])
    data = np.concatenate([record["data"][: record["length"]] for record in chunk.data])
    assert data.max() == ADC_MAXIMUM
