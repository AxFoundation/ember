"""Channel, arrival time and gain of every detected S2 photon.

Ported from fuse's ``S2PhotonPropagationBase`` and ``S2PhotonPropagation`` (Henning's thesis,
section 4.6). Every extracted electron produces ``n_s2_photons`` detected photons
(``SecondaryScintillation``). For each photon:

* channel ``c ~ P(c | x, y)`` from the S2 pattern map at the electron position at the liquid
  surface (switched-off channels removed, the remaining channels renormalized),
* time = extraction time of the electron
  + luminescence delay (time of the emission on the way to the anode, inverse CDF ``u → t``)
  + gas excimer decay (singlet with probability ``singlet_fraction_gas``, otherwise triplet)
  + optical delay (inverse CDF ``u → t``, sub-maps ``top`` and ``bottom``)
  + PMT transit time,
* gain and double photoelectron emission as for S1.

Large outputs are split into several chunks at gaps of at least
``min_electron_gap_length_for_splitting`` between the electrons (downchunking, as fuse).

Deviations from fuse:

* Depends on ``s2_photons`` and ``extracted_electrons`` (fuse: the merged data types of the delayed
  electron simulation, which is not ported).
* The luminescence timing is a 1D inverse CDF ``u → t`` (``s2_luminescence_timing_map``) instead of
  fuse's Garfield table interpolated in the gas gap. The time starts at the extraction; fuse
  subtracts the mean of the drawn sample of each electron.
* No AFT smearing and no AFT rescaling of the pattern map: the AFT is part of the map.
* The output size for the downchunking is computed from the photon counts of the electrons, not
  from a mean gain; the electrons are not re-sorted by cluster id (they arrive sorted by time).
* The channel sampling interpolates the pattern map and draws the photons' channels in one numba
  kernel per electron (``RegularGridMap.draw_channels``), without storing the interpolated
  ``(n_electrons, n_channels)`` array. It never picks a channel with zero probability.
* An error is raised if a photon would end up behind the end of the chunk.
"""

import numpy as np
import strax

from ...common import (
    build_photon_propagation_output,
    init_spe_scaling_factor_distributions,
    photon_gain_calculation,
    pmt_gains,
    pmt_transit_time_spread,
)
from ...dtypes import propagated_photons_fields
from ...plugin import EmberBaseDownChunkingPlugin

export, __all__ = strax.exporter()

BYTES_PER_MEGABYTE = 1e6


@export
class S2PhotonPropagationBase(EmberBaseDownChunkingPlugin):
    """Base plugin to simulate the propagation of S2 photons in the detector.

    Photons are assigned to PMT channels with the S2 pattern map; the timing of the photons is
    calculated by the child plugin.
    """

    __version__ = "0.0.2"

    depends_on = ("s2_photons", "extracted_electrons")
    provides = "propagated_s2_photons"
    data_kind = "s2_photons"

    save_when = strax.SaveWhen.TARGET

    dtype = propagated_photons_fields + strax.time_fields

    p_double_pe_emission = strax.URLConfig(
        default="detector://p_double_pe_emission",
        type=(int, float),
        help="Probability of double photo-electron emission",
    )

    pmt_transit_time_spread = strax.URLConfig(
        default="detector://pmt_transit_time_spread",
        type=(int, float),
        help="Spread (FWHM) of the PMT transit times [ns]",
    )

    pmt_transit_time_mean = strax.URLConfig(
        default="detector://pmt_transit_time_mean",
        type=(int, float),
        help="Mean of the PMT transit times [ns]",
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

    n_top_pmts = strax.URLConfig(
        default="detector://n_top_pmts",
        type=int,
        help="Number of PMTs on top array",
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

    photon_area_distribution = strax.URLConfig(
        default="ember_table://spe_distributions_dummy.csv",
        cache=True,
        help="SPE area distribution per channel (columns 'charge', '0', '1', ...)",
    )

    s2_pattern_map = strax.URLConfig(
        default="ember_map://s2_pattern_map_dummy.pkl?method=linear",
        cache=True,
        help="Probability P(c | x, y) that an S2 photon of an electron extracted at (x, y) is "
        "detected in channel c (sum over the channels 1, AFT built in)",
    )

    singlet_fraction_gas = strax.URLConfig(
        default="detector://singlet_fraction_gas",
        type=(int, float),
        help="Fraction of singlet excimers in the gas",
    )

    singlet_lifetime_gas = strax.URLConfig(
        default="detector://singlet_lifetime_gas",
        type=(int, float),
        help="Lifetime of singlet excimers in the gas [ns]",
    )

    triplet_lifetime_gas = strax.URLConfig(
        default="detector://triplet_lifetime_gas",
        type=(int, float),
        help="Lifetime of triplet excimers in the gas [ns]",
    )

    propagated_s2_photons_file_size_target = strax.URLConfig(
        default=300,
        type=(int, float),
        track=False,
        help="Target for the propagated_s2_photons chunk size [MB]",
    )

    min_electron_gap_length_for_splitting = strax.URLConfig(
        default=2e6,
        type=(int, float),
        track=False,
        help="A chunk is only split at gaps between electrons of at least this length [ns]",
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
        self.pmt_mask = self.gains > 0
        if not np.any(self.pmt_mask):
            raise ValueError("All channels are switched off in gain_model_mc.")
        self.masked_s2_pattern_map = self.s2_pattern_map.with_masked_channels(self.pmt_mask)

        if not 0 <= self.singlet_fraction_gas <= 1:
            raise ValueError(
                f"singlet_fraction_gas must be in [0, 1], got {self.singlet_fraction_gas}"
            )

        self.spe_scaling_factor_distributions = init_spe_scaling_factor_distributions(
            self.photon_area_distribution, self.n_tpc_pmts
        )

    def compute(self, individual_electrons, start, end):
        electron_groups = np.split(
            individual_electrons,
            find_electron_split_index(
                electron_times=individual_electrons["time"],
                n_photons_per_electron=individual_electrons["n_s2_photons"],
                bytes_per_photon=np.dtype(self.dtype).itemsize,
                file_size_target=self.propagated_s2_photons_file_size_target,
                min_gap_length=self.min_electron_gap_length_for_splitting,
            ),
        )
        if len(electron_groups) > 1:
            self.log.info(
                f"Chunk size exceeding file size target. Downchunking to {len(electron_groups)} "
                "chunks"
            )

        chunk_start = start
        for group_index, electron_group in enumerate(electron_groups):
            result = self.propagate_photons(electron_group)
            if group_index < len(electron_groups) - 1:
                last_time = max(electron_group["time"].max(), strax.endtime(result).max(initial=0))
                chunk_end = int(last_time + 0.9 * self.min_electron_gap_length_for_splitting)
            else:
                chunk_end = end
                if len(result) and result["time"].max() >= end:
                    raise ValueError(
                        f"S2 photons up to {result['time'].max() - end + 1} ns behind the chunk "
                        "end. Increase the separation_scale of the input plugin."
                    )
            yield self.chunk(start=chunk_start, end=chunk_end, data=result)
            chunk_start = chunk_end

    def propagate_photons(self, electrons: np.ndarray) -> np.ndarray:
        """Return the propagated photons of a group of electrons.

        :param electrons: extracted electrons with ``x_interface``, ``y_interface``,
            ``cluster_id``, ``time`` and ``n_s2_photons``.
        :return: one row per photon, sorted by time.
        """
        n_photons = electrons["n_s2_photons"].astype(np.int64)
        if n_photons.sum() == 0:
            return np.zeros(0, dtype=self.dtype)
        positions = np.column_stack([electrons["x_interface"], electrons["y_interface"]])

        photon_channels = self.photon_channels(positions, n_photons)
        photon_timings = np.repeat(electrons["time"], n_photons).astype(np.int64)
        photon_timings += self.photon_timings(positions, n_photons, photon_channels).astype(
            np.int64
        )
        photon_timings = pmt_transit_time_spread(
            photon_timings=photon_timings,
            pmt_transit_time_mean=self.pmt_transit_time_mean,
            pmt_transit_time_spread=self.pmt_transit_time_spread,
            rng=self.rng,
        )

        photon_gains, photon_is_dpe = photon_gain_calculation(
            photon_channels=photon_channels,
            p_double_pe_emission=self.p_double_pe_emission,
            gains=self.gains,
            spe_scaling_factor_distributions=self.spe_scaling_factor_distributions,
            rng=self.rng,
        )

        result = build_photon_propagation_output(
            dtype=self.dtype,
            photon_timings=photon_timings,
            photon_channels=photon_channels,
            photon_gains=photon_gains,
            photon_is_dpe=photon_is_dpe,
            cluster_ids=np.repeat(electrons["cluster_id"], n_photons),
            photon_type=2,
        )
        return strax.sort_by_time(result)

    def photon_channels(self, positions: np.ndarray, n_photons: np.ndarray) -> np.ndarray:
        """Draw the channel of every photon from the S2 pattern map.

        :param positions: (x, y) of the electrons at the liquid surface [cm], shape ``(n, 2)``.
        :param n_photons: number of photons per electron.
        :return: channel of each photon, electron by electron, as int16.
        """
        return self.masked_s2_pattern_map.draw_channels(positions, n_photons, self.rng)

    def singlet_triplet_delays(self, number_of_photons: int) -> np.ndarray:
        """Draw the decay time of the excimer that emits each photon.

        :param number_of_photons: number of photons.
        :return: delays [ns]: exponential with the singlet lifetime with probability
            ``singlet_fraction_gas``, otherwise with the triplet lifetime.
        """
        is_singlet = self.rng.random(number_of_photons) < self.singlet_fraction_gas
        lifetimes = np.where(is_singlet, self.singlet_lifetime_gas, self.triplet_lifetime_gas)
        return self.rng.exponential(1.0, number_of_photons) * lifetimes

    def photon_timings(
        self, positions: np.ndarray, n_photons: np.ndarray, photon_channels: np.ndarray
    ) -> np.ndarray:
        """Return the delay of each photon after the extraction of its electron.

        Implemented by the child plugin.
        """
        raise NotImplementedError


@export
class S2PhotonPropagation(S2PhotonPropagationBase):
    """Simulate the propagation of S2 photons with luminescence and optical timing maps."""

    __version__ = "0.0.2"

    child_plugin = True

    s2_luminescence_timing_map = strax.URLConfig(
        default="ember_map://s2_luminescence_timing_dummy.json?method=linear",
        cache=True,
        help="Inverse CDF of the emission time after the extraction [ns] as a function of u. The "
        "default is a mesh anode (map_generators/dummy_s2_timing_maps.ipynb).",
    )

    s2_optical_propagation_spline = strax.URLConfig(
        default="ember_map://s2_optical_propagation_dummy.json?method=linear",
        cache=True,
        help="Inverse CDF of the optical propagation delay [ns] as a function of u, with the "
        "maps 'top' and 'bottom'. The default comes from ember's optical toy Monte Carlo "
        "(map_generators/dummy_s2_timing_maps.ipynb).",
    )

    def photon_timings(
        self, positions: np.ndarray, n_photons: np.ndarray, photon_channels: np.ndarray
    ) -> np.ndarray:
        """Return the delay of each photon after the extraction of its electron.

        :param positions: (x, y) of the electrons at the liquid surface [cm] (not used by the 1D
            maps of this plugin).
        :param n_photons: number of photons per electron.
        :param photon_channels: channel of each photon.
        :return: luminescence + excimer decay + optical delay [ns], float64.
        """
        number_of_photons = len(photon_channels)
        return (
            self.luminescence_timings(number_of_photons)
            + self.singlet_triplet_delays(number_of_photons)
            + self.optical_propagation(photon_channels)
        )

    def luminescence_timings(self, number_of_photons: int) -> np.ndarray:
        """Draw the emission time of each photon after the extraction of its electron.

        :param number_of_photons: number of photons.
        :return: emission times [ns].
        """
        return self.s2_luminescence_timing_map(self.rng.random((number_of_photons, 1)))

    def optical_propagation(self, channels: np.ndarray) -> np.ndarray:
        """Draw the optical propagation delay of each photon from the timing map.

        :param channels: channel of each photon.
        :return: delays [ns].
        """
        propagation_time = np.zeros(len(channels), dtype=np.float64)
        uniform_numbers = self.rng.random((len(channels), 1))
        is_top = channels < self.n_top_pmts
        propagation_time[is_top] = self.s2_optical_propagation_spline(
            uniform_numbers[is_top], map_name="top"
        )
        propagation_time[~is_top] = self.s2_optical_propagation_spline(
            uniform_numbers[~is_top], map_name="bottom"
        )
        return propagation_time


def find_electron_split_index(
    electron_times: np.ndarray,
    n_photons_per_electron: np.ndarray,
    bytes_per_photon: int,
    file_size_target: float,
    min_gap_length: float,
) -> np.ndarray:
    """Find where to split the electrons so that each part produces about the target output size.

    As fuse: walking through the time-sorted electrons, a split is made after an electron once the
    photons since the last split exceed ``file_size_target`` and the gap to the next electron is at
    least ``min_gap_length``.

    :param electron_times: times of the electrons [ns], sorted.
    :param n_photons_per_electron: number of photons of each electron.
    :param bytes_per_photon: size of one output row [bytes].
    :param file_size_target: target size of one chunk [MB].
    :param min_gap_length: minimal gap between two electrons for a split [ns].
    :return: indices for ``np.split`` (index of the first electron of each new part).
    """
    if len(electron_times) < 2:
        return np.zeros(0, dtype=np.int64)
    cumulative_size = (
        np.cumsum(n_photons_per_electron, dtype=np.float64) * bytes_per_photon / BYTES_PER_MEGABYTE
    )
    gap_to_next_electron = np.diff(electron_times)
    split_index = []
    size_at_last_split = 0.0
    for candidate in np.flatnonzero(gap_to_next_electron >= min_gap_length):
        if cumulative_size[candidate] - size_at_last_split >= file_size_target:
            split_index.append(candidate + 1)
            size_at_last_split = cumulative_size[candidate]
    return np.array(split_index, dtype=np.int64)
