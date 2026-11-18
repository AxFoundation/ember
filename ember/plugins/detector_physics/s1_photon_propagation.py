"""Channel, arrival time and gain of every detected S1 photon.

Ported from fuse's ``S1PhotonPropagationBase`` and ``S1PhotonPropagation`` (Henning's thesis,
section 4.5). Differences to fuse:

- nestpy >= 2.1 API (``nestpy.interactions``); the NEST calculator uses
  ``nestpy.detectors.VDetector`` (the photon times do not depend on the detector).
- fuse's override of the NR scintillation timing (``override_s1_nr_scint_time`` and the
  ``s1_nr_scint_time_*_override`` options) is not ported: the times always come from NEST.
- ``p_double_pe_emission`` is spelled correctly (fuse: ``p_double_pe_emision``).
- Switched-off channels are removed from the pattern map in ``setup()`` instead of in the URL.
- An error is raised if a photon would end up behind the end of the chunk.
"""

import nestpy
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
from ...plugin import EmberBasePlugin
from ..micro_physics.yields import nest_rng, suppress_c_stderr

export, __all__ = strax.exporter()


@export
class S1PhotonPropagationBase(EmberBasePlugin):
    """Base plugin to simulate the propagation of S1 photons in the detector.

    Photons are randomly assigned to PMT channels based on their starting position, and the timing
    of the photons is calculated by the child plugin.
    """

    __version__ = "0.0.1"

    depends_on = ("microphysics_summary", "s1_photon_hits")
    provides = "propagated_s1_photons"
    data_kind = "s1_photons"

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

    s1_pattern_map = strax.URLConfig(
        default="ember_map://s1_pattern_map_dummy.pkl?method=linear",
        cache=True,
        help="S1 light collection efficiency per channel [PE per emitted photon] as a function of "
        "(x, y, z)",
    )

    def setup(self):
        super().setup()

        if self.seed is not None:
            self.short_seed = int(repr(self.seed)[-8:])
        else:
            self.short_seed = int(self.rng.integers(1, 10**8))
        self.log.debug(f"Generating nest random numbers starting with seed {self.short_seed}")

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
        self.masked_s1_pattern_map = self.s1_pattern_map.with_masked_channels(self.pmt_mask)

        self.spe_scaling_factor_distributions = init_spe_scaling_factor_distributions(
            self.photon_area_distribution, self.n_tpc_pmts
        )

    def compute(self, interactions_in_roi, start, end):
        instruction = interactions_in_roi[interactions_in_roi["n_s1_photon_hits"] > 0]
        if len(instruction) == 0:
            return np.zeros(0, self.dtype)

        nest_rng.set_seed(self.short_seed)
        nest_rng.lock_seed()
        self.short_seed += 1
        try:
            result = self.propagate_photons(instruction)
        finally:
            nest_rng.unlock_seed()

        if len(result) and result["time"].max() >= end:
            raise ValueError(
                f"S1 photons up to {result['time'].max() - end + 1} ns behind the chunk end. "
                "Increase the separation_scale of the input plugin."
            )
        return result

    def propagate_photons(self, instruction: np.ndarray) -> np.ndarray:
        """Return the propagated photons of all interactions with photon hits.

        Warning: Computes properties per photon intermediate results are not
        sorted by time!

        :param instruction: interactions with ``n_s1_photon_hits > 0``.
        :return: one row per photon, sorted by time.
        """
        positions = np.column_stack([instruction["x"], instruction["y"], instruction["z"]])
        cluster_ids = np.repeat(instruction["cluster_id"], instruction["n_s1_photon_hits"])

        photon_channels = self.photon_channels(
            positions=positions,
            n_photon_hits=instruction["n_s1_photon_hits"],
        )

        photon_timings = self.photon_timings(
            t=instruction["time"],
            n_photon_hits=instruction["n_s1_photon_hits"],
            recoil_type=instruction["nestid"],
            channels=photon_channels,
            positions=positions,
            e_dep=instruction["ed"],
            n_photons_emitted=instruction["photons"].astype(np.int64),
            n_excitons=instruction["excitons"].astype(np.int64),
            local_field=instruction["e_field"],
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
            cluster_ids=cluster_ids,
            photon_type=1,
        )
        return strax.sort_by_time(result)

    def photon_channels(self, positions: np.ndarray, n_photon_hits: np.ndarray) -> np.ndarray:
        """Draw the channel of every detected photon from the S1 pattern map.

        :param positions: (x, y, z) of the interactions, shape ``(n, 3)``.
        :param n_photon_hits: number of detected photons per interaction.
        :return: channel of each photon, interaction by interaction.
        """
        channels = np.arange(self.n_tpc_pmts)
        p_per_channel = self.masked_s1_pattern_map(positions)

        photon_channels = []
        for probabilities, n_hits in zip(p_per_channel, n_photon_hits):
            photon_channels.append(
                self.rng.choice(
                    channels, size=n_hits, p=probabilities / np.sum(probabilities), replace=True
                )
            )
        return np.concatenate(photon_channels)

    def photon_timings(self, **kwargs) -> np.ndarray:
        """Return the arrival time of each photon. Implemented by the child plugin."""
        raise NotImplementedError


@export
class S1PhotonPropagation(S1PhotonPropagationBase):
    """Simulate the propagation of S1 photons with an optical timing map and NEST timing."""

    __version__ = "0.0.2"

    child_plugin = True

    maximum_recombination_time = strax.URLConfig(
        default=1000,
        type=(int, float),
        help="Maximum scintillation delay [ns]; longer delays are drawn again",
    )

    s1_optical_propagation_spline = strax.URLConfig(
        default="ember_map://s1_optical_propagation_dummy.json?method=linear",
        cache=True,
        help="Inverse CDF of the optical propagation delay [ns] as a function of (z, u), with the "
        "maps 'top' and 'bottom'. The default comes from ember's optical toy Monte Carlo "
        "(map_generators/dummy_s1_timing_maps.ipynb).",
    )

    nest_time_sampling_max_loops = strax.URLConfig(
        default=10,
        track=False,
        help="Maximum number of loops in the scintillation time calculation.\n"
        "This is used to ensure that the number of drawn photon times after truncation is greater "
        "or equal to the number of photon hits.",
    )

    def setup(self):
        super().setup()
        with suppress_c_stderr():
            self.nestpy_calc = nestpy.NESTcalc(nestpy.detectors.VDetector())

    def photon_timings(
        self,
        t: np.ndarray,
        n_photon_hits: np.ndarray,
        recoil_type: np.ndarray,
        channels: np.ndarray,
        positions: np.ndarray,
        e_dep: np.ndarray,
        n_photons_emitted: np.ndarray,
        n_excitons: np.ndarray,
        local_field: np.ndarray,
    ) -> np.ndarray:
        """Return the arrival time of each photon at the photocathode.

        :param t: interaction times [ns].
        :param n_photon_hits: number of detected photons per interaction.
        :param recoil_type: NEST interaction type per interaction.
        :param channels: channel of each photon.
        :param positions: (x, y, z) of the interactions.
        :param e_dep: deposited energy per interaction [keV].
        :param n_photons_emitted: number of emitted photons per interaction.
        :param n_excitons: number of excitons per interaction.
        :param local_field: drift field per interaction [V/cm].
        :return: photon times [ns], int64, interaction by interaction.
        """
        photon_timings = np.repeat(t, n_photon_hits).astype(np.int64)
        z_positions = np.repeat(positions[:, 2], n_photon_hits)

        photon_timings += self.optical_propagation(channels, z_positions).astype(np.int64)
        photon_timings += self.nest_scintillation_timing(
            n_photon_hits=n_photon_hits,
            n_photons_emitted=n_photons_emitted,
            recoil_type=recoil_type,
            n_excitons=n_excitons,
            local_field=local_field,
            e_dep=e_dep,
        )
        return photon_timings

    def nest_scintillation_timing(
        self,
        n_photon_hits: np.ndarray,
        n_photons_emitted: np.ndarray,
        recoil_type: np.ndarray,
        n_excitons: np.ndarray,
        local_field: np.ndarray,
        e_dep: np.ndarray,
    ) -> np.ndarray:
        """Draw the scintillation delay of each photon with NEST.

        Delays above ``maximum_recombination_time`` are rejected and drawn again.

        :return: scintillation delays [ns], int64, interaction by interaction.
        """
        if np.any(n_photon_hits > n_photons_emitted):
            raise ValueError(
                "Number of photon hits must be less than or equal to number of photons"
            )

        exciton_to_photon_ratio = np.divide(
            n_excitons.astype(np.float32),
            n_photons_emitted.astype(np.float32),
            out=np.zeros(len(n_photons_emitted), dtype=np.float32),
            where=n_photons_emitted != 0,
        )

        scintillation_times = []
        for index, counts in enumerate(n_photon_hits):
            sampled_times = np.array([])
            n_times_to_sample = counts
            loop_count = 0
            while n_times_to_sample > 0:
                times = np.array(
                    self.nestpy_calc.GetPhotonTimes(
                        nestpy.interactions(int(recoil_type[index])),
                        int(n_times_to_sample),
                        int(round(exciton_to_photon_ratio[index] * n_times_to_sample)),
                        float(local_field[index]),
                        float(e_dep[index]),
                    )
                )
                times = times[times < self.maximum_recombination_time]
                sampled_times = np.concatenate([sampled_times, times])
                n_times_to_sample = counts - len(sampled_times)

                if loop_count > self.nest_time_sampling_max_loops:
                    raise ValueError(
                        "Maximum number of loops reached in scintillation time calculation without"
                        " reaching the number of required photon times. This is likely due to a too"
                        " low maximum recombination time."
                    )
                loop_count += 1
            scintillation_times.append(sampled_times)

        return np.concatenate(scintillation_times).astype(np.int64)

    def optical_propagation(self, channels: np.ndarray, z_positions: np.ndarray) -> np.ndarray:
        """Draw the optical propagation delay of each photon from the timing map.

        :param channels: channel of each photon.
        :param z_positions: z of the interaction of each photon [cm].
        :return: delays [ns].
        """
        if len(z_positions) != len(channels):
            raise ValueError("Give each photon a z position")

        propagation_time = np.zeros(len(channels), dtype=np.float64)
        z_and_u = np.column_stack([z_positions, self.rng.random(len(channels))])

        is_top = channels < self.n_top_pmts
        propagation_time[is_top] = self.s1_optical_propagation_spline(
            z_and_u[is_top], map_name="top"
        )
        propagation_time[~is_top] = self.s1_optical_propagation_spline(
            z_and_u[~is_top], map_name="bottom"
        )
        return propagation_time
