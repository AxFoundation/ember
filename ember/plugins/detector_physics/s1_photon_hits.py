"""Number of detected S1 photons per interaction (ported from fuse's ``S1PhotonHits``).

The number of detected photons is drawn from a binomial distribution,
``n_hits ~ Binomial(photons, p)`` with
``p = LCE(x, y, z) * s1_detection_efficiency / (1 + p_double_pe_emission)``, where ``LCE`` is the
S1 pattern map summed over all active channels (Henning's thesis, eq. 4.11-4.13).
"""

import numpy as np
import strax

from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class S1PhotonHits(EmberBasePlugin):
    """Simulate the number of detected S1 photons with the S1 light collection efficiency map."""

    __version__ = "0.0.2"

    depends_on = "microphysics_summary"
    provides = "s1_photon_hits"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.ALWAYS

    dtype = [
        (("Number detected S1 photons", "n_s1_photon_hits"), np.int32),
    ] + strax.time_fields

    gain_model_mc = strax.URLConfig(
        default="ember_pmt_to_pe://",
        infer_type=False,
        help="ADC-to-PE factor per channel; channels with 0 are switched off",
    )

    s1_pattern_map = strax.URLConfig(
        default="ember_map://s1_pattern_map_dummy.pkl?method=linear",
        cache=True,
        help="S1 light collection efficiency per channel [PE per emitted photon] as a function of "
        "(x, y, z)",
    )

    p_double_pe_emission = strax.URLConfig(
        default="detector://p_double_pe_emission",
        type=(int, float),
        help="Probability of double photo-electron emission",
    )

    s1_detection_efficiency = strax.URLConfig(
        default="detector://s1_detection_efficiency",
        type=(int, float),
        help="S1 detection efficiency, a global scaling factor of the light collection efficiency",
    )

    def setup(self):
        super().setup()

        self.pmt_mask = np.asarray(self.gain_model_mc) > 0
        self.s1_lce_correction_map = self.s1_pattern_map.with_masked_channels(
            self.pmt_mask
        ).summed_over_channels()

    def compute(self, interactions_in_roi):
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        mask = interactions_in_roi["photons"] > 0
        if not np.any(mask):
            return result

        positions = np.column_stack(
            [
                interactions_in_roi["x"][mask],
                interactions_in_roi["y"][mask],
                interactions_in_roi["z"][mask],
            ]
        )
        result["n_s1_photon_hits"][mask] = self.get_n_photons(
            n_photons=interactions_in_roi["photons"][mask].astype(np.int64),
            positions=positions,
        )
        return result

    def get_n_photons(self, n_photons: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Draw the number of detected photons.

        :param n_photons: number of emitted S1 photons per interaction.
        :param positions: (x, y, z) of the interactions, shape ``(n, 3)``.
        :return: number of detected photons per interaction.
        """
        light_yield = self.s1_lce_correction_map(positions)
        light_yield = light_yield / (1 + self.p_double_pe_emission)
        light_yield = light_yield * self.s1_detection_efficiency
        return self.rng.binomial(n=n_photons, p=light_yield)
