"""PMT afterpulses as pseudo photons (port of fuse's ``PMTAfterPulses``).

Differences to fuse:

- Delay and size are drawn with the inverse cumulative distribution (``searchsorted``): the
  smallest bin whose cumulative probability reaches the random number. fuse takes the bin whose
  cumulative probability is nearest to the random number (``argmin |cdf - u|``), which shifts the
  delays by half a bin on average and is slower.
- The merged S1 + S2 photons are not sorted by time first (the afterpulses are drawn per photon
  and the output is sorted at the end).
- fuse's ``Uniform`` afterpulse elements are not supported; such entries raise an error.
- The afterpulse configuration is checked in ``setup()`` (one row per channel, non-decreasing,
  finite, probability at most 1).
- An error is raised if an afterpulse would end up behind the end of the chunk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import strax

from ...common import build_photon_propagation_output, pmt_gains
from ...dtypes import propagated_photons_fields
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()

CDF_ROUNDING_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AfterpulseElement:
    """Afterpulse properties of one ion species (one entry of fuse's afterpulse configuration).

    :param name: name of the ion species.
    :param delaytime_cdf: ``(n_channels, n_bins)`` cumulative delay probability per channel. The
        last value of a row is the afterpulse probability per photoelectron.
    :param delaytime_bin_size: delay of one bin [ns]; bin ``i`` stands for ``i * bin size``.
    :param amplitude_cdf: ``(n_channels, n_bins)`` cumulative size probability per channel,
        normalized to 1.
    :param amplitude_bin_size: size of one bin [PE]; bin ``i`` stands for ``i * bin size``.
    """

    name: str
    delaytime_cdf: np.ndarray
    delaytime_bin_size: float
    amplitude_cdf: np.ndarray
    amplitude_bin_size: float


@export
class PMTAfterPulses(EmberBasePlugin):
    """Plugin to simulate PMT afterpulses using a precomputed afterpulse cumulative distribution.

    In the simulation afterpulses are saved as "pseudo" photons (``photon_type = 0``). They are
    combined with the S1 and S2 photons to create the waveforms.
    """

    __version__ = "0.0.1"

    depends_on = ("propagated_s2_photons", "propagated_s1_photons")
    provides = "pmt_afterpulses"
    data_kind = "ap_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = propagated_photons_fields + strax.time_fields

    enable_pmt_afterpulses = strax.URLConfig(
        default="detector://enable_pmt_afterpulses",
        type=bool,
        help="Decide if you want to enable PMT afterpulsing",
    )

    pmt_ap_t_modifier = strax.URLConfig(
        default="detector://pmt_ap_t_modifier",
        type=(int, float),
        help="Time subtracted from every afterpulse delay [ns]",
    )

    pmt_ap_modifier = strax.URLConfig(
        default="detector://pmt_ap_modifier",
        type=(int, float),
        help="Factor scaling the afterpulse probability",
    )

    pmt_circuit_load_resistor = strax.URLConfig(
        default="detector://pmt_circuit_load_resistor",
        type=(int, float),
        help="Conversion constant from ADC counts to PMT gain (fuse's 'PMT circuit load resistor')",
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

    n_tpc_pmts = strax.URLConfig(
        default="detector://n_tpc_pmts",
        type=int,
        help="Number of PMTs in the TPC",
    )

    gain_model_mc = strax.URLConfig(
        default="ember_pmt_to_pe://",
        infer_type=False,
        help="ADC-to-PE factor per channel; channels with 0 are switched off",
    )

    photon_ap_cdfs = strax.URLConfig(
        default="ember_file://pmt_afterpulse_config_dummy.json.gz",
        cache=True,
        help="Afterpulse cumulative distribution functions (fuse's format)",
    )

    def setup(self):
        super().setup()

        self.gains = pmt_gains(
            self.gain_model_mc,
            digitizer_voltage_range=self.digitizer_voltage_range,
            digitizer_bits=self.digitizer_bits,
            pmt_circuit_load_resistor=self.pmt_circuit_load_resistor,
        )
        if len(self.gains) != self.n_tpc_pmts:
            raise ValueError(
                f"gain_model_mc has {len(self.gains)} channels, "
                f"but n_tpc_pmts is {self.n_tpc_pmts}."
            )
        if self.pmt_ap_modifier <= 0:
            raise ValueError(f"pmt_ap_modifier must be positive, got {self.pmt_ap_modifier}.")

        self.afterpulse_elements = read_afterpulse_elements(self.photon_ap_cdfs, self.n_tpc_pmts)
        for element in self.afterpulse_elements:
            largest_probability = element.delaytime_cdf[:, -1].max() * self.pmt_ap_modifier
            if largest_probability > 0.5:
                self.log.warning(
                    f"PMT afterpulse probability of {element.name} is {largest_probability}, "
                    "larger than 0.5?"
                )

    def compute(self, s1_photons, s2_photons, start, end):
        if not self.enable_pmt_afterpulses or (len(s1_photons) == 0 and len(s2_photons) == 0):
            return np.zeros(0, dtype=self.dtype)

        merged_photons = np.concatenate([s1_photons, s2_photons])
        afterpulse_times, afterpulse_channels, afterpulse_gains = draw_afterpulses(
            merged_photons["time"],
            merged_photons["channel"],
            merged_photons["dpe"],
            self.afterpulse_elements,
            self.gains,
            self.pmt_ap_modifier,
            self.pmt_ap_t_modifier,
            self.rng,
        )

        result = build_photon_propagation_output(
            dtype=self.dtype,
            photon_timings=afterpulse_times,
            photon_channels=afterpulse_channels,
            photon_gains=afterpulse_gains,
            photon_is_dpe=np.zeros(len(afterpulse_times), dtype=np.bool_),
            cluster_ids=np.full(len(afterpulse_times), -1),
            photon_type=0,
        )
        result = strax.sort_by_time(result)

        if len(result) and result["time"].max() >= end:
            raise ValueError(
                f"Afterpulses up to {result['time'].max() - end + 1} ns behind the chunk end. "
                "Increase the separation_scale of the input plugin."
            )
        return result


def read_afterpulse_elements(afterpulse_config: dict, n_channels: int) -> list[AfterpulseElement]:
    """Convert fuse's afterpulse configuration into checked :class:`AfterpulseElement` objects.

    A one-dimensional ``amplitude_cdf`` (same for all channels) is repeated for every channel.

    :param afterpulse_config: ``{name: {"delaytime_cdf", "delaytime_bin_size", "amplitude_cdf",
        "amplitude_bin_size"}}``.
    :param n_channels: number of channels.
    :return: one element per ion species.
    """
    elements = []
    for name, entry in afterpulse_config.items():
        if "Uniform" in name:
            raise ValueError(f"Uniform afterpulse elements ({name}) are not supported by ember.")
        delaytime_cdf = np.asarray(entry["delaytime_cdf"], dtype=np.float64)
        amplitude_cdf = np.asarray(entry["amplitude_cdf"], dtype=np.float64)
        if amplitude_cdf.ndim == 1:
            amplitude_cdf = np.tile(amplitude_cdf, (n_channels, 1))
        for cdf_name, cdf in (("delaytime_cdf", delaytime_cdf), ("amplitude_cdf", amplitude_cdf)):
            check_cumulative_distribution(cdf, n_channels, f"{name} {cdf_name}")
        elements.append(
            AfterpulseElement(
                name=name,
                delaytime_cdf=delaytime_cdf,
                delaytime_bin_size=float(entry["delaytime_bin_size"]),
                amplitude_cdf=amplitude_cdf,
                amplitude_bin_size=float(entry["amplitude_bin_size"]),
            )
        )
    return elements


def check_cumulative_distribution(cdf: np.ndarray, n_channels: int, label: str) -> None:
    """Raise if ``cdf`` is not a valid per-channel cumulative distribution.

    Values up to ``1 + CDF_ROUNDING_TOLERANCE`` are accepted (rounding of normalized sums).

    :param cdf: ``(n_channels, n_bins)`` array.
    :param n_channels: expected number of rows.
    :param label: name used in the error message.
    """
    if cdf.ndim != 2 or cdf.shape[0] != n_channels:
        raise ValueError(f"{label} must have shape ({n_channels}, n_bins), got {cdf.shape}.")
    if not np.all(np.isfinite(cdf)):
        raise ValueError(f"{label} contains non-finite values.")
    if (
        np.any(cdf < 0)
        or np.any(cdf > 1 + CDF_ROUNDING_TOLERANCE)
        or np.any(np.diff(cdf, axis=1) < 0)
    ):
        raise ValueError(f"{label} must be non-decreasing with values in [0, 1].")


def draw_from_cumulative_rows(
    cdf: np.ndarray, rows: np.ndarray, random_numbers: np.ndarray
) -> np.ndarray:
    """Draw bin indices with the inverse of the cumulative distribution of the given rows.

    For each entry, returns the smallest bin ``i`` with ``cdf[row, i] >= random_number``. All rows
    are shifted by ``2 * row`` and searched in one ``searchsorted`` call (values lie in [0, 1]).

    :param cdf: ``(n_rows, n_bins)`` non-decreasing rows with values in [0, 1].
    :param rows: row of each draw.
    :param random_numbers: numbers in (0, last value of the row].
    :return: bin index of each draw.
    """
    n_rows, n_bins = cdf.shape
    row_offsets = 2.0 * np.arange(n_rows)
    flat_cdf = (cdf + row_offsets[:, None]).ravel()
    flat_index = np.searchsorted(flat_cdf, random_numbers + row_offsets[rows], side="left")
    return np.minimum(flat_index - rows * n_bins, n_bins - 1)


def draw_afterpulses(
    photon_times: np.ndarray,
    photon_channels: np.ndarray,
    photon_is_dpe: np.ndarray,
    afterpulse_elements: list[AfterpulseElement],
    gains: np.ndarray,
    afterpulse_modifier: float,
    afterpulse_time_modifier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw the afterpulses of all photons.

    Per ion species and photon a number ``u`` in (0, 1] is drawn and divided by
    ``afterpulse_modifier`` (and by 2 for double photoelectron photons). The photon causes an
    afterpulse if ``u`` is at most the afterpulse probability of its channel; the same ``u`` then
    gives the delay. A second number gives the size, and the afterpulse gain is
    ``channel gain * size``.

    :param photon_times: photon times [ns].
    :param photon_channels: photon channels.
    :param photon_is_dpe: double photoelectron emission flags.
    :param afterpulse_elements: afterpulse properties per ion species.
    :param gains: gain per channel.
    :param afterpulse_modifier: factor scaling the afterpulse probability.
    :param afterpulse_time_modifier: time subtracted from every delay [ns].
    :param rng: random number generator.
    :return: afterpulse times [ns], channels and gains.
    """
    photon_channels = np.asarray(photon_channels, dtype=np.int64)
    afterpulse_times, afterpulse_channels, afterpulse_gains = [], [], []
    for element in afterpulse_elements:
        random_numbers = (1 - rng.random(len(photon_times))) / afterpulse_modifier
        random_numbers[photon_is_dpe] /= 2

        afterpulse_probability = element.delaytime_cdf[photon_channels, -1]
        selected = np.flatnonzero(random_numbers <= afterpulse_probability)
        if len(selected) == 0:
            continue
        selected_channels = photon_channels[selected]

        delay_bins = draw_from_cumulative_rows(
            element.delaytime_cdf, selected_channels, random_numbers[selected]
        )
        size_bins = draw_from_cumulative_rows(
            element.amplitude_cdf, selected_channels, 1 - rng.random(len(selected))
        )
        delays = delay_bins * element.delaytime_bin_size - afterpulse_time_modifier

        afterpulse_times.append(photon_times[selected] + delays)
        afterpulse_channels.append(selected_channels)
        afterpulse_gains.append(gains[selected_channels] * size_bins * element.amplitude_bin_size)

    if not afterpulse_times:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    return (
        np.concatenate(afterpulse_times),
        np.concatenate(afterpulse_channels),
        np.concatenate(afterpulse_gains),
    )
