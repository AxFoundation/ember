"""Electroluminescence photons of the extracted electrons (ported from fuse's
``SecondaryScintillation``).

Every extracted electron produces ``N ~ Poisson(μ)`` detected S2 photons with (Henning's thesis,
eq. 4.9, 4.10)

    μ(x, y) = c_xy(x, y) · g_SE / (1 + p_DPE),    g_SE = g2 / ε_extraction,

where ``c_xy`` is the relative S2 correction map (mean 1), ``g_SE`` the single electron gain in PE
per extracted electron and ``p_DPE`` the probability of a double photoelectron emission. Only the
photons that reach a PMT are simulated; photons that are lost are part of ``g_SE``.

Two outputs:

* ``s2_photons``: the number of photons of each extracted electron, row by row aligned with
  ``extracted_electrons`` (data kind ``individual_electrons``),
* ``s2_photons_sum``: the sum per interaction (data kind ``interactions_in_roi``).

Deviations from fuse:

* The single electron gain is not an option of its own: it is ``g2 / electron_extraction_yield``,
  so ``g2`` stays the single source of truth. ``electron_extraction_yield`` has the same name and
  default as in ``ElectronExtraction``, so changing it keeps ``g2`` fixed.
* Only the S2 correction map path is ported (no single electron gain map).
* The sum per interaction is a vectorized group-by sum (:func:`sum_per_cluster`) instead of a sort,
  a split and two Python loops.
* No dependency on ``drifted_electrons`` (not used) and no PMT gains (fuse computes them but does
  not use them here).
"""

import numpy as np
import strax
from immutabledict import immutabledict

from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class SecondaryScintillation(EmberBasePlugin):
    """Simulate the number of detected electroluminescence photons per extracted electron."""

    __version__ = "0.0.1"

    result_name_photons = "s2_photons"
    result_name_photons_sum = "s2_photons_sum"

    depends_on = ("microphysics_summary", "extracted_electrons")
    provides = (result_name_photons, result_name_photons_sum)
    data_kind = dict(
        {
            result_name_photons: "individual_electrons",
            result_name_photons_sum: "interactions_in_roi",
        }
    )

    dtype = dict(
        {
            result_name_photons: [
                (
                    ("Number of photons produced by the extracted electron", "n_s2_photons"),
                    np.int32,
                ),
            ]
            + strax.time_fields,
            result_name_photons_sum: [
                (
                    (
                        "Sum of all photons produced by electrons originating from the same "
                        "cluster",
                        "sum_s2_photons",
                    ),
                    np.int32,
                ),
            ]
            + strax.time_fields,
        }
    )

    save_when = immutabledict(
        {result_name_photons: strax.SaveWhen.TARGET, result_name_photons_sum: strax.SaveWhen.ALWAYS}
    )

    g2 = strax.URLConfig(
        default="detector://g2",
        type=(int, float),
        help="S2 gain [PE per electron at the liquid surface]",
    )

    electron_extraction_yield = strax.URLConfig(
        default="detector://electron_extraction_efficiency",
        type=(int, float),
        help="Probability that an electron at the liquid surface is extracted into the gas",
    )

    s2_correction_map = strax.URLConfig(
        default="ember_map://s2_correction_map_dummy.json?method=linear",
        cache=True,
        help="Relative S2 light yield c_xy(x, y) (mean 1 over the extraction area)",
    )

    p_double_pe_emission = strax.URLConfig(
        default="detector://p_double_pe_emission",
        type=(int, float),
        help="Probability of double photo-electron emission",
    )

    def setup(self):
        super().setup()
        if not 0 < self.electron_extraction_yield <= 1:
            raise ValueError(
                "electron_extraction_yield must be in (0, 1] to compute the single electron gain, "
                f"got {self.electron_extraction_yield}"
            )
        self.single_electron_gain = self.g2 / self.electron_extraction_yield

    def compute(self, individual_electrons, interactions_in_roi):
        result_photons = np.zeros(
            len(individual_electrons), dtype=self.dtype[self.result_name_photons]
        )
        result_photons["time"] = individual_electrons["time"]
        result_photons["endtime"] = individual_electrons["endtime"]

        result_photons_sum = np.zeros(
            len(interactions_in_roi), dtype=self.dtype[self.result_name_photons_sum]
        )
        result_photons_sum["time"] = interactions_in_roi["time"]
        result_photons_sum["endtime"] = interactions_in_roi["endtime"]

        if len(individual_electrons):
            positions = np.column_stack(
                [individual_electrons["x_interface"], individual_electrons["y_interface"]]
            )
            result_photons["n_s2_photons"] = self.rng.poisson(self.get_s2_light_yield(positions))
            result_photons_sum["sum_s2_photons"] = sum_per_cluster(
                interactions_in_roi["cluster_id"],
                individual_electrons["cluster_id"],
                result_photons["n_s2_photons"],
            )

        return {
            self.result_name_photons: result_photons,
            self.result_name_photons_sum: result_photons_sum,
        }

    def get_s2_light_yield(self, positions: np.ndarray) -> np.ndarray:
        """Mean number of detected photons per extracted electron.

        :param positions: (x, y) of the electrons at the liquid surface [cm], shape ``(n, 2)``.
        :return: mean number of photons per electron, shape ``(n,)``. NaN values of the map
            become 0.
        """
        light_yield = self.s2_correction_map(positions) * self.single_electron_gain
        light_yield = light_yield / (1 + self.p_double_pe_emission)
        return np.nan_to_num(light_yield, nan=0.0)


def sum_per_cluster(
    interaction_cluster_ids: np.ndarray, electron_cluster_ids: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Sum the values of all electrons of each interaction (a group-by sum).

    Works for electrons in any order. Interactions without electrons get 0.

    :param interaction_cluster_ids: cluster id of each interaction (unique).
    :param electron_cluster_ids: cluster id of each electron.
    :param values: value of each electron, e.g. its number of photons (integers).
    :return: sum per interaction, in the order of ``interaction_cluster_ids``, as int64.
    """
    result = np.zeros(len(interaction_cluster_ids), dtype=np.int64)
    if len(electron_cluster_ids) == 0 or len(interaction_cluster_ids) == 0:
        return result
    unique_ids, inverse = np.unique(electron_cluster_ids, return_inverse=True)
    sums = np.bincount(inverse, weights=values).astype(np.int64)
    index = np.clip(np.searchsorted(unique_ids, interaction_cluster_ids), 0, len(unique_ids) - 1)
    has_electrons = unique_ids[index] == interaction_cluster_ids
    result[has_electrons] = sums[index[has_electrons]]
    return result
