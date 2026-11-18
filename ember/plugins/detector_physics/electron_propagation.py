"""Individual electrons at the liquid surface (ported from fuse's ``ElectronPropagation``).

The electron cloud of every interaction is split into its electrons (Henning's thesis, eq. 4.5 and
4.6). For each electron:

* arrival time at the liquid surface ``t ~ N(t_drift, σ_t)`` with mean and spread from
  ``drifted_electrons``,
* position at the liquid surface ``(x_obs + Δx, y_obs + Δy)`` with independent
  ``Δx, Δy ~ N(0, √(2 D_T t))``, where ``t`` is the electron's own drift time and ``D_T`` the
  transverse diffusion constant of the drift map at the interaction position.

Deviations from fuse:

* ``D_T`` always comes from the drift map ``field_dependencies_map`` (sub-map
  ``diffusion_transverse_map``).
* Diffusion is isotropic: ``Δx`` and ``Δy`` are drawn directly. fuse draws a radial and an azimuthal
  shift (for anisotropic diffusion maps near the wall) and rotates them to x, y.
* No perpendicular-wire effects (fuse's ``ElectronPropagationPerpWires``).
* The output is sorted by time (fuse keeps the cluster order). Electrons of interactions at
  different depths arrive in a different order than the interactions happened; strax splits chunks
  by time and the downchunking in ``S2PhotonPropagation`` needs time-ordered electrons.
* Negative drift times (possible only for a spread larger than the mean) are set to 0.
* An error is raised if an electron would end up behind the end of the chunk.
"""

import numpy as np
import strax

from ...dtypes import individual_electron_fields
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronPropagation(EmberBasePlugin):
    """Split the electron clouds into individual electrons at the liquid surface."""

    __version__ = "0.0.1"

    depends_on = ("drifted_electrons", "microphysics_summary")
    provides = "electrons_at_interface"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET

    dtype = individual_electron_fields + strax.time_fields

    field_dependencies_map = strax.URLConfig(
        default="ember_map://field_dependencies_dummy.json?method=linear",
        cache=True,
        help="(r, z) map with the transverse diffusion constant (diffusion_transverse_map) "
        "[cm²/ns]",
    )

    def compute(self, interactions_in_roi, start, end):
        clusters = interactions_in_roi[interactions_in_roi["n_electron_interface"] > 0]
        if len(clusters) == 0:
            return np.zeros(0, dtype=self.dtype)

        n_electrons = clusters["n_electron_interface"].astype(np.int64)
        drift_times = self.draw_drift_times(
            n_electrons, clusters["drift_time_mean"], clusters["drift_time_spread"]
        )

        positions_rz = np.column_stack(
            [np.hypot(clusters["x_obs"], clusters["y_obs"]), clusters["z_obs"]]
        )
        diffusion_transverse = self.field_dependencies_map(
            positions_rz, map_name="diffusion_transverse_map"
        )
        transverse_spread = np.sqrt(2 * np.repeat(diffusion_transverse, n_electrons) * drift_times)

        result = np.zeros(len(drift_times), dtype=self.dtype)
        result["time"] = np.repeat(clusters["time"], n_electrons) + drift_times
        result["endtime"] = result["time"]
        result["x_interface"] = np.repeat(clusters["x_obs"], n_electrons) + self.rng.normal(
            0.0, transverse_spread
        )
        result["y_interface"] = np.repeat(clusters["y_obs"], n_electrons) + self.rng.normal(
            0.0, transverse_spread
        )
        result["cluster_id"] = np.repeat(clusters["cluster_id"], n_electrons)
        result = strax.sort_by_time(result)

        if result["time"].max() >= end:
            raise ValueError(
                f"Electrons up to {result['time'].max() - end + 1} ns behind the chunk end. "
                "Increase the separation_scale of the input plugin."
            )
        return result

    def draw_drift_times(
        self, n_electrons: np.ndarray, drift_time_mean: np.ndarray, drift_time_spread: np.ndarray
    ) -> np.ndarray:
        """Draw the drift time of every electron.

        :param n_electrons: number of electrons per cluster.
        :param drift_time_mean: mean drift time per cluster [ns].
        :param drift_time_spread: drift time spread per cluster [ns].
        :return: drift time per electron [ns], clusters one after the other, as int64.
        """
        drift_times = self.rng.normal(
            np.repeat(drift_time_mean, n_electrons).astype(np.float64),
            np.repeat(drift_time_spread, n_electrons).astype(np.float64),
        )
        return np.clip(drift_times, 0, None).astype(np.int64)
