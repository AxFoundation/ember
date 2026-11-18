"""Time windows in which the PMT pulses of photons can overlap (port of fuse's ``PulseWindow``).

Differences to fuse:

- ``pulse_id`` values are unique across chunks (fuse continues with ``max()`` instead of
  ``max() + 1``, so the last window of a chunk and the first window of the next share an id).
- Whether a channel already has an open window is stored in a separate flag. fuse uses
  ``time == 0`` as marker, which fails for a window that starts at 0 ns (e.g. the first event at
  t = 0 with ``fixed_event_spacing``): the window is then overwritten and its photons are lost.
- Every window is clipped to the chunk boundaries (fuse clips only the first window of a channel).
- ``pulse_ids`` is not sorted again: it has one row per photon in the order of ``photon_summary``,
  which is already sorted by time.
"""

import numba
import numpy as np
import strax

from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()

PULSE_WINDOW_DTYPE = strax.interval_dtype + [(("ID of the pulse window", "pulse_id"), np.int64)]


@export
class PulseWindow(EmberBasePlugin):
    """Plugin to compute time intervals (called ``pulse_windows``) in which the PMT response of
    photons can overlap.

    Additionally a ``pulse_id`` is computed for each propagated photon to identify the pulse
    window it belongs to.
    """

    __version__ = "0.0.1"

    depends_on = "photon_summary"

    provides = ("pulse_windows", "pulse_ids")
    data_kind = {"pulse_windows": "pulse_windows", "pulse_ids": "propagated_photons"}

    dtype = {
        "pulse_windows": PULSE_WINDOW_DTYPE,
        "pulse_ids": strax.time_fields
        + [(("Pulse id to map the photon to the pulse window", "pulse_id"), np.int64)],
    }

    save_when = strax.SaveWhen.TARGET

    dt = strax.URLConfig(
        default="detector://sample_duration",
        type=int,
        help="Width of one sample [ns]",
    )

    samples_after_pulse_center = strax.URLConfig(
        default="detector://samples_after_pulse_center",
        type=(int, float),
        help="Number of samples after the pulse center",
    )

    samples_to_store_after = strax.URLConfig(
        default="detector://samples_to_store_after",
        type=(int, float),
        help="Number of samples to store after the pulse center",
    )

    samples_before_pulse_center = strax.URLConfig(
        default="detector://samples_before_pulse_center",
        type=(int, float),
        help="Number of samples before the pulse center",
    )

    samples_to_store_before = strax.URLConfig(
        default="detector://samples_to_store_before",
        type=(int, float),
        help="Number of samples to store before the pulse center",
    )

    n_tpc_pmts = strax.URLConfig(
        default="detector://n_tpc_pmts",
        type=int,
        help="Number of PMTs in the TPC",
    )

    def setup(self):
        super().setup()

        self.pulse_ids_seen = 0

        # Twice the samples to store, to avoid overlapping records when triggering on noise
        self.pulse_left_extension = (
            np.int64(2 * self.samples_to_store_before) + self.samples_before_pulse_center
        )
        self.pulse_right_extension = (
            np.int64(2 * self.samples_to_store_after) + self.samples_after_pulse_center
        )

    def compute(self, propagated_photons, start, end):
        if len(propagated_photons) == 0:
            return {
                "pulse_windows": np.zeros(0, self.dtype["pulse_windows"]),
                "pulse_ids": np.zeros(0, self.dtype["pulse_ids"]),
            }

        single_photon_pulses = np.zeros(len(propagated_photons), dtype=strax.interval_dtype)
        single_photon_pulses["length"] = (
            self.samples_before_pulse_center + self.samples_after_pulse_center
        )
        single_photon_pulses["dt"] = self.dt
        single_photon_pulses["time"] = propagated_photons["time"]
        single_photon_pulses["channel"] = propagated_photons["channel"]

        photon_pulses, photon_pulse_ids = concat_overlapping_hits(
            single_photon_pulses,
            (self.pulse_left_extension, self.pulse_right_extension),
            self.n_tpc_pmts,
            start,
            end,
        )
        photon_pulses["pulse_id"] += self.pulse_ids_seen

        pulse_ids = np.zeros(len(photon_pulse_ids), self.dtype["pulse_ids"])
        pulse_ids["pulse_id"] = photon_pulse_ids + self.pulse_ids_seen
        pulse_ids["time"] = propagated_photons["time"]
        pulse_ids["endtime"] = propagated_photons["endtime"]

        self.pulse_ids_seen += photon_pulse_ids.max() + 1

        return {"pulse_windows": strax.sort_by_time(photon_pulses), "pulse_ids": pulse_ids}


def concat_overlapping_hits(
    hits: np.ndarray, extensions: tuple, n_channels: int, start: int, end: int
) -> tuple[np.ndarray, np.ndarray]:
    """Merge single-photon intervals of one channel that overlap after extending them.

    Modified from strax' ``concat_overlapping_hits`` (via fuse). Assumes the hits are sorted by
    time.

    In the numba generator a record view such as ``open_window`` is saved as a copy across a
    ``yield``, so the open window is updated before the result buffer is handed over.

    :param hits: single-photon intervals (``strax.interval_dtype``).
    :param extensions: ``(left, right)`` extension in samples.
    :param n_channels: number of channels; channels must lie in ``[0, n_channels)``.
    :param start: start of the chunk [ns]; windows are clipped to it.
    :param end: end of the chunk [ns]; windows are clipped to it.
    :return: the pulse windows (``PULSE_WINDOW_DTYPE``, ``pulse_id`` starting at 0, not sorted)
        and the ``pulse_id`` of every hit.
    """
    if len(hits) and (hits["channel"].min() < 0 or hits["channel"].max() >= n_channels):
        raise ValueError(f"Photon channels must lie in [0, {n_channels}).")
    open_windows = np.zeros(
        n_channels,
        dtype=strax.interval_dtype
        + [
            (("End time of the interval (ns since unix epoch)", "endtime"), np.int64),
            (("ID of the pulse window", "pulse_id"), np.int64),
        ],
    )
    channel_has_open_window = np.zeros(n_channels, dtype=np.bool_)
    photon_pulse_ids = np.zeros(len(hits), dtype=np.int64)
    if len(hits) == 0:
        return np.zeros(0, PULSE_WINDOW_DTYPE), photon_pulse_ids

    windows = _concat_overlapping_hits(
        hits,
        extensions,
        open_windows,
        channel_has_open_window,
        photon_pulse_ids,
        start,
        end,
    )
    return windows, photon_pulse_ids


@strax.utils.growing_result(PULSE_WINDOW_DTYPE, chunk_size=int(1e4))
@numba.njit(nogil=True, cache=True)
def _concat_overlapping_hits(
    hits,
    extensions,
    open_windows,
    channel_has_open_window,
    photon_pulse_ids,
    chunk_start,
    chunk_end,
    _result_buffer=None,
):
    buffer = _result_buffer
    result_offset = 0
    next_pulse_id = 0

    left_extension, right_extension = extensions
    dt = hits["dt"][0]
    assert np.all(hits["dt"] == dt), "All hits must have the same dt!"

    for i, hit in enumerate(hits):
        time_with_left_extension = max(hit["time"] - int(left_extension * dt), chunk_start)
        endtime_with_right_extension = min(
            hit["time"] + hit["length"] * dt + int(right_extension * dt), chunk_end
        )
        channel = hit["channel"]
        open_window = open_windows[channel]

        if channel_has_open_window[channel]:
            if open_window["endtime"] >= time_with_left_extension:
                open_window["endtime"] = endtime_with_right_extension
                photon_pulse_ids[i] = open_window["pulse_id"]
                continue

            result = buffer[result_offset]
            result["time"] = open_window["time"]
            result["length"] = (open_window["endtime"] - open_window["time"]) // dt
            result["channel"] = channel
            result["dt"] = dt
            result["pulse_id"] = open_window["pulse_id"]
            result_offset += 1

        channel_has_open_window[channel] = True
        open_window["time"] = time_with_left_extension
        open_window["endtime"] = endtime_with_right_extension
        open_window["channel"] = channel
        open_window["dt"] = dt
        open_window["pulse_id"] = next_pulse_id
        photon_pulse_ids[i] = next_pulse_id
        next_pulse_id += 1

        if result_offset == len(buffer):
            yield result_offset
            result_offset = 0

    for channel in range(len(open_windows)):
        if not channel_has_open_window[channel]:
            continue
        open_window = open_windows[channel]
        result = buffer[result_offset]
        result["time"] = open_window["time"]
        result["length"] = (open_window["endtime"] - open_window["time"]) // dt
        result["channel"] = channel
        result["dt"] = dt
        result["pulse_id"] = open_window["pulse_id"]
        result_offset += 1
        if result_offset == len(buffer):
            yield result_offset
            result_offset = 0
    yield result_offset
