"""PMT waveforms, digitization and zero-length encoding (port of fuse's ``PMTResponseAndDAQ``).

Differences to fuse:

- The photons of a pulse window are found with offsets into one array sorted by ``pulse_id``
  instead of one ``numba.typed.List`` entry per window built in a Python loop. The sort is stable
  and ``photon_summary`` is sorted by time, so the photons of a window stay sorted by time and
  ``add_current`` does not sort them again.
- Down-chunking: the gap before a window is measured from the latest end of *all* earlier windows
  (fuse: only the previous window, so a long window in another channel could still be open and the
  next chunk could start before the end of the previous one). The end of an intermediate chunk is
  the latest end of its windows (fuse: of its records, which fails if there are none).
- The noise is read at ``time // dt`` (fuse: ``time / 10``, i.e. assumes 10 ns samples).
- ``add_current`` cuts a template that would reach beyond the end of its window (only possible
  for windows clipped at the chunk end) instead of failing.
- ``find_intervals_below_threshold`` stops when its result buffer is full (fuse writes behind the
  end of the buffer; this cannot happen with the buffer size used here).
- Every sample is clipped to the digitizer range ``[0, 2**digitizer_bits - 1]`` before it is
  written (fuse writes the int64 value into the int16 field and clips negative values afterwards,
  so samples below -32768 ADC, i.e. very large S2s, wrap around to positive values).
- The waveform of a window is built in one pass over its samples (current, rounding, noise,
  baseline, clipping) with scratch arrays allocated once per chunk instead of several temporary
  arrays per window.
- The SPE pulse shape is one option (``pe_pulse_shape``, a file with fuse's keys ``pe_pulse_ts``
  and ``pe_pulse_ys``) instead of two.

Kept from fuse: the template placement puts the pulse ``samples_before_pulse_center`` samples
after the photon time; the waveform area in PE is the photon PE times
``external_amplification / sample_duration`` (= 1 for 10 / 10 ns).
"""

import numba
import numpy as np
import strax
from scipy.interpolate import interp1d

from ...common import stable_argsort
from ...plugin import EmberBaseDownChunkingPlugin

export, __all__ = strax.exporter()

RECORD_LENGTH = strax.DEFAULT_RECORD_LENGTH


@export
class PMTResponseAndDAQ(EmberBaseDownChunkingPlugin):
    """Plugin to simulate the PMT response and DAQ effects.

    First the single PMT waveform is simulated based on the photon timing and gain information.
    Next the waveform is converted to ADC counts, noise and a baseline are added. Then hitfinding
    is performed and the found intervals are split into multiple fragments of fixed length (if
    needed). Finally the data is saved as raw_records.
    """

    __version__ = "0.0.3"

    depends_on = ("photon_summary", "pulse_ids", "pulse_windows")

    provides = "raw_records"
    data_kind = "raw_records"

    dtype = strax.raw_record_dtype(samples_per_record=RECORD_LENGTH)

    save_when = strax.SaveWhen.TARGET

    dt = strax.URLConfig(
        default="detector://sample_duration",
        type=int,
        help="Width of one sample [ns]",
    )

    pmt_circuit_load_resistor = strax.URLConfig(
        default="detector://pmt_circuit_load_resistor",
        type=(int, float),
        help="Conversion constant from PMT current to voltage (fuse's 'PMT circuit load resistor')",
    )

    external_amplification = strax.URLConfig(
        default="detector://external_amplification",
        type=(int, float),
        help="External amplification factor",
    )

    digitizer_bits = strax.URLConfig(
        default="detector://digitizer_bits",
        type=(int, float),
        help="Number of bits of the digitizer boards",
    )

    digitizer_voltage_range = strax.URLConfig(
        default="detector://digitizer_voltage_range",
        type=(int, float),
        help="Voltage range of the digitizer boards [V]",
    )

    noise_data = strax.URLConfig(
        default="ember_file://noise_dummy.npz",
        cache=True,
        help="Noise data: dict with 'arr_0' of shape (n_samples, n_tpc_pmts) [ADC]",
    )

    enable_noise = strax.URLConfig(
        default="detector://enable_noise",
        type=bool,
        help="Option to enable or disable noise",
    )

    pe_pulse_shape = strax.URLConfig(
        default="ember_file://pmt_pulse_shape_dummy.json",
        cache=True,
        help="SPE pulse shape: dict with 'pe_pulse_ts' [ns] and 'pe_pulse_ys' (any normalization)",
    )

    pmt_pulse_time_rounding = strax.URLConfig(
        default="detector://pmt_pulse_time_rounding",
        type=(int, float),
        help="Time rounding of the PMT pulse [ns], must be 1",
    )

    samples_after_pulse_center = strax.URLConfig(
        default="detector://samples_after_pulse_center",
        type=(int, float),
        help="Number of samples after the pulse center",
    )

    samples_before_pulse_center = strax.URLConfig(
        default="detector://samples_before_pulse_center",
        type=(int, float),
        help="Number of samples before the pulse center",
    )

    digitizer_reference_baseline = strax.URLConfig(
        default="detector://digitizer_reference_baseline",
        type=(int, float),
        help="Digitizer reference baseline [ADC]",
    )

    zle_threshold = strax.URLConfig(
        default="detector://zle_threshold",
        type=(int, float),
        help="Threshold for the zero length encoding [ADC below the baseline]",
    )

    trigger_window = strax.URLConfig(
        default="detector://trigger_window",
        type=(int, float),
        help="Samples stored before and after a threshold crossing",
    )

    special_thresholds = strax.URLConfig(
        default="detector://special_thresholds",
        help="Special thresholds for certain PMTs: {channel: zle_threshold}",
    )

    n_tpc_pmts = strax.URLConfig(
        default="detector://n_tpc_pmts",
        type=int,
        help="Number of PMTs in the TPC",
    )

    raw_records_file_size_target = strax.URLConfig(
        type=(int, float),
        default=200,
        track=False,
        help="Target for the raw records file size [MB]",
    )

    min_records_gap_length_for_splitting = strax.URLConfig(
        type=(int, float),
        default=1e5,
        track=False,
        help="Chunk can not be split if gap between pulses is smaller than this value [ns]",
    )

    def setup(self):
        super().setup()

        self.current_2_adc = (
            self.pmt_circuit_load_resistor
            * self.external_amplification
            / (self.digitizer_voltage_range / 2**self.digitizer_bits)
        )

        self.pmt_current_templates = init_pmt_current_templates(
            self.pe_pulse_shape["pe_pulse_ts"],
            self.pe_pulse_shape["pe_pulse_ys"],
            self.dt,
            int(self.samples_before_pulse_center),
            int(self.samples_after_pulse_center),
            self.pmt_pulse_time_rounding,
        )

        self.thresholds = zle_thresholds(
            self.n_tpc_pmts,
            self.digitizer_reference_baseline,
            self.zle_threshold,
            self.special_thresholds,
        )

        noise = np.asarray(self.noise_data["arr_0"])
        if noise.ndim != 2 or noise.shape[1] != self.n_tpc_pmts:
            raise ValueError(
                f"Noise data must have shape (n_samples, {self.n_tpc_pmts}), got {noise.shape}."
            )
        self.noise_per_channel = np.ascontiguousarray(noise.T)

    def compute(self, propagated_photons, pulse_windows, start, end):
        if len(propagated_photons) == 0 or len(pulse_windows) == 0:
            self.log.debug("No photons or pulse windows found for chunk!")
            yield self.chunk(start=start, end=end, data=np.zeros(0, dtype=self.dtype))
            return

        split_index = find_split_index(
            pulse_windows,
            file_size_limit=self.raw_records_file_size_target,
            min_gap_length=self.min_records_gap_length_for_splitting,
        )
        pulse_window_chunks = np.split(pulse_windows, split_index)
        n_chunks = len(pulse_window_chunks)
        if n_chunks > 1:
            self.log.info(
                f"Chunk size exceeding file size target. Downchunking to {n_chunks} chunks"
            )

        photon_order = stable_argsort(propagated_photons["pulse_id"])
        photon_times = propagated_photons["time"][photon_order]
        photon_gains = propagated_photons["photon_gain"][photon_order].astype(np.float64)
        photon_pulse_ids = propagated_photons["pulse_id"][photon_order]

        chunk_start = start
        for chunk_index, windows in enumerate(pulse_window_chunks):
            first_photon, stop_photon = photon_range_per_window(photon_pulse_ids, windows)
            records = self.compute_chunk(
                windows, first_photon, stop_photon, photon_times, photon_gains
            )
            if chunk_index < n_chunks - 1:
                chunk_end = int(strax.endtime(windows).max())
            else:
                chunk_end = end
            yield self.chunk(start=chunk_start, end=chunk_end, data=records)
            chunk_start = chunk_end

    def compute_chunk(
        self,
        windows: np.ndarray,
        first_photon: np.ndarray,
        stop_photon: np.ndarray,
        photon_times: np.ndarray,
        photon_gains: np.ndarray,
    ) -> np.ndarray:
        """Return the raw_records of a group of pulse windows, sorted by time.

        :param windows: pulse windows.
        :param first_photon: index of the first photon of each window in the photon arrays.
        :param stop_photon: index after the last photon of each window.
        :param photon_times: photon times, sorted by ``pulse_id`` and time [ns].
        :param photon_gains: photon gains, same order.
        :return: the raw_records.
        """
        n_records_upper_limit = int(np.sum(np.ceil(windows["length"] / RECORD_LENGTH)))
        waveform_buffer = np.zeros(n_records_upper_limit, dtype=self.dtype)

        noise_offset = int(self.rng.integers(0, self.noise_per_channel.shape[1]))
        adc_maximum = 2 ** int(self.digitizer_bits) - 1

        buffer_level = build_waveform(
            windows,
            first_photon,
            stop_photon,
            photon_times,
            photon_gains,
            waveform_buffer,
            self.dt,
            self.pmt_current_templates,
            self.current_2_adc,
            self.noise_per_channel,
            self.enable_noise,
            self.digitizer_reference_baseline,
            self.thresholds,
            int(self.trigger_window),
            noise_offset,
            adc_maximum,
        )
        return strax.sort_by_time(waveform_buffer[:buffer_level])


def init_pmt_current_templates(
    pulse_times: np.ndarray,
    pulse_amplitudes: np.ndarray,
    sample_duration: int,
    samples_before: int,
    samples_after: int,
    pmt_pulse_time_rounding: float,
) -> np.ndarray:
    """Create the SPE current templates for every ns offset of a photon within a sample.

    For 10 ns samples and 1 ns rounding, ``templates[i]`` is used for photons arriving at
    ``10 m + i`` ns. Each template sums to ``1 / sample_duration`` (1 PE per sample duration).

    :param pulse_times: times of the SPE pulse shape [ns].
    :param pulse_amplitudes: amplitudes of the SPE pulse shape (any normalization).
    :param sample_duration: duration of one sample [ns].
    :param samples_before: template samples before the pulse center.
    :param samples_after: template samples after the pulse center.
    :param pmt_pulse_time_rounding: time resolution of the templates [ns], must be 1.
    :return: array ``(sample_duration, samples_before + samples_after)``.
    """
    if pmt_pulse_time_rounding != 1:
        raise ValueError(f"pmt_pulse_time_rounding must be 1, got {pmt_pulse_time_rounding}.")
    pulse_amplitudes = np.asarray(pulse_amplitudes, dtype=np.float64)
    pe_pulse_function = interp1d(
        pulse_times,
        np.cumsum(pulse_amplitudes) / np.sum(pulse_amplitudes),
        bounds_error=False,
        fill_value=(0, 1),
    )

    sample_edges = np.linspace(
        -samples_before * sample_duration,
        samples_after * sample_duration,
        1 + samples_before + samples_after,
    )

    templates = []
    for offset in np.arange(0, sample_duration, pmt_pulse_time_rounding):
        pmt_current = np.diff(pe_pulse_function(sample_edges - offset)) / sample_duration
        pmt_current *= (1 / sample_duration) / np.sum(pmt_current)
        templates.append(pmt_current)
    return np.array(templates)


def zle_thresholds(
    n_channels: int, baseline: float, zle_threshold: float, special_thresholds: dict
) -> np.ndarray:
    """Return the zero-length-encoding threshold per channel (as in fuse).

    A sample belongs to a pulse if it is below ``baseline - threshold - 1``.

    :param n_channels: number of channels.
    :param baseline: digitizer baseline [ADC].
    :param zle_threshold: default threshold below the baseline [ADC].
    :param special_thresholds: ``{channel: threshold}`` overrides; channels outside the range are
        ignored, as in fuse.
    :return: threshold per channel [ADC].
    """
    thresholds = np.full(n_channels, baseline - zle_threshold - 1, dtype=np.float64)
    for channel, threshold in special_thresholds.items():
        if int(channel) < n_channels:
            thresholds[int(channel)] = baseline - threshold - 1
    return thresholds


def photon_range_per_window(
    photon_pulse_ids: np.ndarray, windows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return for every window the index range of its photons.

    :param photon_pulse_ids: ``pulse_id`` of every photon, sorted.
    :param windows: pulse windows.
    :return: first index and index after the last photon of every window.
    """
    first_photon = np.searchsorted(photon_pulse_ids, windows["pulse_id"], side="left")
    stop_photon = np.searchsorted(photon_pulse_ids, windows["pulse_id"], side="right")
    if np.any(stop_photon == first_photon):
        raise ValueError("Every pulse window must contain at least one photon.")
    return first_photon, stop_photon


def find_split_index(
    pulses: np.ndarray, file_size_limit: float, min_gap_length: float
) -> np.ndarray:
    """Return the indices where the time-sorted pulse windows are split into chunks.

    A split is made after window ``i`` once the data size since the last split reached
    ``file_size_limit`` and the next window starts at least ``min_gap_length`` after the latest
    end of all windows up to ``i``.

    :param pulses: pulse windows sorted by time.
    :param file_size_limit: data size per chunk [MB], counted as 2 bytes per sample.
    :param min_gap_length: minimal gap for a split [ns].
    :return: split indices for ``np.split``.
    """
    latest_endtime = np.maximum.accumulate(strax.endtime(pulses))
    gaps = np.append(pulses["time"][1:] - latest_endtime[:-1], 0)
    return _find_split_index(pulses["length"], gaps, file_size_limit, min_gap_length)


@numba.njit(cache=True)
def _find_split_index(lengths, gaps, file_size_limit, min_gap_length):
    data_size_mb = 0.0
    split_index = []
    for i in range(len(lengths)):
        data_size_mb += lengths[i] * 2 / 1e6
        if data_size_mb < file_size_limit:
            continue
        if gaps[i] >= min_gap_length:
            data_size_mb = 0.0
            split_index.append(i + 1)
    return np.array(split_index, dtype=np.int64)


@numba.njit(cache=True)
def build_waveform(
    pulse_windows,
    first_photon,
    stop_photon,
    photon_times,
    photon_gains,
    waveform_buffer,
    dt,
    pmt_current_templates,
    current_2_adc,
    noise_per_channel,
    enable_noise,
    digitizer_reference_baseline,
    thresholds,
    trigger_window,
    noise_offset,
    adc_maximum,
):
    """Fill ``waveform_buffer`` with the raw_records of all pulse windows.

    The scratch arrays for the current, the digitized waveform and the ZLE intervals are allocated
    once with the length of the longest window and reused for every window.

    :param pulse_windows: pulse windows.
    :param first_photon: index of the first photon of each window in the photon arrays.
    :param stop_photon: index after the last photon of each window.
    :param photon_times: photon times, sorted by ``pulse_id`` and time [ns].
    :param photon_gains: photon gains, same order.
    :param waveform_buffer: output records, large enough for all windows.
    :param dt: sample duration [ns].
    :param pmt_current_templates: SPE current templates per ns offset within a sample.
    :param current_2_adc: conversion from current to ADC counts.
    :param noise_per_channel: noise samples, shape ``(n_channels, n_noise_samples)``.
    :param enable_noise: add the noise.
    :param digitizer_reference_baseline: baseline [ADC].
    :param thresholds: ZLE threshold per channel [ADC].
    :param trigger_window: samples stored before and after an interval below threshold.
    :param noise_offset: random offset into the noise samples.
    :param adc_maximum: largest ADC value of the digitizer (``2**bits - 1``).
    :return: number of records written.
    """
    max_window_length = 0
    for i in range(len(pulse_windows)):
        max_window_length = max(max_window_length, pulse_windows[i]["length"])
    current_scratch = np.zeros(max_window_length, dtype=np.float64)
    waveform_scratch = np.zeros(max_window_length, dtype=np.int64)
    intervals_scratch = np.zeros((max_window_length // 2, 2), dtype=np.int64)
    baseline = np.int64(digitizer_reference_baseline)
    n_noise_samples = noise_per_channel.shape[1]

    buffer_level = 0
    for i in range(len(pulse_windows)):
        pulse = pulse_windows[i]
        window_length = pulse["length"]
        channel = pulse["channel"]
        pulse_current = current_scratch[:window_length]
        pulse_current[:] = 0.0
        add_current(
            photon_times[first_photon[i] : stop_photon[i]],
            photon_gains[first_photon[i] : stop_photon[i]],
            pulse["time"] // dt,
            dt,
            pmt_current_templates,
            pulse_current,
        )

        noise_index = (pulse["time"] // dt + noise_offset + 1) % n_noise_samples
        for sample in range(window_length):
            adc_value = baseline - np.int64(np.rint(pulse_current[sample] * current_2_adc))
            if enable_noise:
                adc_value += noise_per_channel[channel, noise_index]
                noise_index += 1
                if noise_index == n_noise_samples:
                    noise_index = 0
            waveform_scratch[sample] = min(max(adc_value, 0), adc_maximum)

        buffer_level = convert_pulse_to_fragments(
            waveform_scratch[:window_length],
            waveform_buffer,
            buffer_level,
            thresholds[channel],
            trigger_window,
            channel,
            pulse["time"],
            dt,
            intervals_scratch[: window_length // 2],
        )
    return buffer_level


@numba.njit(cache=True)
def add_current(photon_times, photon_gains, pulse_left, dt, pmt_current_templates, pulse_current):
    """Add the SPE currents of time-sorted photons to the waveform of one pulse window.

    Photons with the same time (ns) are summed before their template is added.

    :param photon_times: photon times, sorted [ns].
    :param photon_gains: photon gains.
    :param pulse_left: first sample of the window (time // dt).
    :param dt: sample duration [ns].
    :param pmt_current_templates: templates per ns offset within a sample.
    :param pulse_current: waveform of the window, modified in place.
    """
    if len(photon_times) == 0:
        return

    gain_total = 0.0
    current_time = photon_times[0]
    for i in range(len(photon_times)):
        if photon_times[i] > current_time:
            _add_template(
                current_time, gain_total, pulse_left, dt, pmt_current_templates, pulse_current
            )
            gain_total = photon_gains[i]
            current_time = photon_times[i]
        else:
            gain_total += photon_gains[i]
    _add_template(current_time, gain_total, pulse_left, dt, pmt_current_templates, pulse_current)


@numba.njit(cache=True)
def _add_template(photon_time, gain, pulse_left, dt, pmt_current_templates, pulse_current):
    template_length = pmt_current_templates.shape[1]
    first_sample = photon_time // dt - pulse_left
    offset_in_sample = photon_time % dt
    stop_sample = min(first_sample + template_length, len(pulse_current))
    for sample in range(first_sample, stop_sample):
        pulse_current[sample] += (
            pmt_current_templates[offset_in_sample, sample - first_sample] * gain
        )


@numba.njit(cache=True)
def convert_pulse_to_fragments(
    single_waveform,
    waveform_buffer,
    buffer_level,
    threshold,
    trigger_window,
    pulse_channel,
    pulse_time,
    dt,
    intervals_scratch,
):
    """Zero-length encode one waveform and write its intervals as records into the buffer.

    Intervals below ``threshold`` are extended by ``trigger_window`` samples on both sides, put on
    even samples and split into records of ``RECORD_LENGTH`` samples.

    :param single_waveform: digitized waveform, values within the int16 range.
    :param waveform_buffer: output records.
    :param buffer_level: number of records already in ``waveform_buffer``.
    :param threshold: ZLE threshold [ADC].
    :param trigger_window: samples stored before and after an interval below threshold.
    :param pulse_channel: channel of the waveform.
    :param pulse_time: time of the first sample [ns].
    :param dt: sample duration [ns].
    :param intervals_scratch: ``(n, 2)`` integer array for the intervals; at most ``n`` intervals
        are found (``len(single_waveform) // 2`` as in fuse).
    :return: new fill level of ``waveform_buffer``.
    """
    n_intervals = find_intervals_below_threshold(
        single_waveform, threshold, 2 * trigger_window + 1, intervals_scratch
    )
    last_sample = len(single_waveform) - 1
    for interval_index in range(n_intervals):
        left = min(max(intervals_scratch[interval_index, 0] - trigger_window, 0), last_sample)
        right = min(max(intervals_scratch[interval_index, 1] + trigger_window, 0), last_sample)
        left = (left + 1) // 2 * 2
        right = right // 2 * 2
        pulse_length = right - left + 1
        records_needed = (pulse_length + RECORD_LENGTH - 1) // RECORD_LENGTH
        for record_i in range(records_needed):
            record = waveform_buffer[buffer_level + record_i]
            first_sample = left + record_i * RECORD_LENGTH
            n_samples = min(RECORD_LENGTH, right + 1 - first_sample)
            record["channel"] = pulse_channel
            record["time"] = dt * (pulse_time // dt + first_sample)
            record["dt"] = dt
            record["pulse_length"] = pulse_length
            record["record_i"] = record_i
            record["length"] = n_samples
            for sample in range(n_samples):
                record["data"][sample] = single_waveform[first_sample + sample]
        buffer_level += records_needed
    return buffer_level


@numba.njit(cache=True)
def find_intervals_below_threshold(waveform, threshold, holdoff, result_buffer):
    """Fill ``result_buffer`` with the inclusive bounds of the intervals where ``waveform`` is below
    ``threshold``.

    An interval ends once the waveform is back above the threshold for ``holdoff`` samples (or at
    the end of the waveform). Stops when the buffer is full.

    :param waveform: waveform [ADC].
    :param threshold: threshold [ADC].
    :param holdoff: samples after the last sample below threshold before an interval ends.
    :param result_buffer: ``(n, 2)`` integer array, filled in place.
    :return: number of intervals found.
    """
    result_buffer_size = len(result_buffer)
    if result_buffer_size == 0:
        return 0
    last_index = len(waveform) - 1

    in_interval = False
    n_intervals = 0
    interval_start = -1
    interval_end = -1

    for i in range(len(waveform)):
        if waveform[i] < threshold:
            if not in_interval:
                in_interval = True
                interval_start = i
            interval_end = i

        interval_is_closed = waveform[i] >= threshold and i >= interval_end + holdoff
        if in_interval and (i == last_index or interval_is_closed):
            in_interval = False
            result_buffer[n_intervals, 0] = interval_start
            result_buffer[n_intervals, 1] = interval_end
            n_intervals += 1
            if n_intervals == result_buffer_size:
                break
    return n_intervals
