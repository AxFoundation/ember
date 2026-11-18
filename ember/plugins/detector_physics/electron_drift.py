"""Drift of the electrons to the liquid surface (ported from fuse's ``ElectronDrift``).

For every interaction with electrons (Henning's thesis, eq. 4.1-4.4):

1. Charge-insensitive volume: ``N' ~ Binomial(N_e, p_surv(r, z))`` with the survival map.
2. Mean drift time ``t = t_below + t_gate`` with ``t_below = (z_gate - z) / v̄(r, z)``, where
   ``v̄`` is the average drift velocity from the interaction up to the gate (drift map), and
   ``t_gate`` the drift time from the gate to the liquid surface.
3. Drift time spread from longitudinal diffusion, added in quadrature for both regions:
   ``σ_t² = 2 D_L t_below / v̄² + 2 D_L,gate t_gate / v_gate²``.
4. Electron lifetime: ``N_interface ~ Binomial(N', exp(-t / τ_e))``.

Deviations from fuse:

* No field distortion: the observed position is the true position (``x_obs = x``, ...).
* Drift velocity and longitudinal diffusion below the gate always come from the (r, z) drift map
  ``field_dependencies_map`` (NEST, built in ``map_generators/dummy_field_maps.ipynb``), not from
  scalar options. Above the gate a separate diffusion constant is used.
* The gate position is a config option (fuse assumes ``z_gate = 0``).
* The survival map is a separate file (fuse: a sub-map of ``field_dependencies_map``).
"""

import numpy as np
import strax

from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronDrift(EmberBasePlugin):
    """Simulate the drift of the electrons from the interaction site to the liquid surface."""

    __version__ = "0.0.1"

    depends_on = "microphysics_summary"
    provides = "drifted_electrons"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.TARGET

    dtype = [
        (
            ("Number of electrons reaching the liquid gas interface", "n_electron_interface"),
            np.int32,
        ),
        (("Mean drift time of the electrons in the cluster [ns]", "drift_time_mean"), np.int32),
        (
            ("Spread of the drift time of the electrons in the cluster [ns]", "drift_time_spread"),
            np.int32,
        ),
        (("Observed x position of the cluster at liquid-gas interface [cm]", "x_obs"), np.float32),
        (("Observed y position of the cluster at liquid-gas interface [cm]", "y_obs"), np.float32),
        (
            ("Observed z position of the cluster after field distortion correction [cm]", "z_obs"),
            np.float32,
        ),
    ] + strax.time_fields

    field_dependencies_map = strax.URLConfig(
        default="ember_map://field_dependencies_dummy.json?method=linear",
        cache=True,
        help="(r, z) map with the average drift velocity to the gate (drift_speed_map) [cm/ns] and "
        "the longitudinal and transverse diffusion constants [cm²/ns]",
    )

    survival_probability_map = strax.URLConfig(
        default="ember_map://electron_survival_below_cathode.json?method=nearest",
        cache=True,
        help="(r, z) map of the probability that an electron reaches the liquid surface",
    )

    electron_lifetime_liquid = strax.URLConfig(
        default="detector://electron_lifetime_liquid",
        type=(int, float),
        help="Electron lifetime in liquid xenon [ns]; 0 or less disables the lifetime",
    )

    drift_time_gate = strax.URLConfig(
        default="detector://drift_time_gate",
        type=(int, float),
        help="Drift time from the gate to the liquid surface [ns]",
    )

    diffusion_constant_longitudinal_gate = strax.URLConfig(
        default="detector://diffusion_constant_longitudinal_gate",
        type=(int, float),
        help="Longitudinal diffusion constant between gate and liquid surface [cm²/ns]",
    )

    z_gate = strax.URLConfig(
        default="detector://z_gate",
        type=(int, float),
        help="z position of the gate [cm]",
    )

    z_liquid_level = strax.URLConfig(
        default="detector://z_liquid_level",
        type=(int, float),
        help="z position of the liquid surface [cm]",
    )

    def setup(self):
        super().setup()
        if not self.z_liquid_level > self.z_gate:
            raise ValueError("The liquid surface must lie above the gate.")
        if self.drift_time_gate <= 0:
            raise ValueError("drift_time_gate must be positive.")
        self.drift_velocity_gate = (self.z_liquid_level - self.z_gate) / self.drift_time_gate

    def compute(self, interactions_in_roi):
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]

        mask = interactions_in_roi["electrons"] > 0
        if not np.any(mask):
            return result

        x = interactions_in_roi["x"][mask]
        y = interactions_in_roi["y"][mask]
        z = interactions_in_roi["z"][mask]
        n_electron = interactions_in_roi["electrons"][mask].astype(np.int64)
        positions_rz = np.column_stack([np.hypot(x, y), z])

        survival_probability = np.clip(self.survival_probability_map(positions_rz), 0, 1)
        n_electron = self.rng.binomial(n=n_electron, p=survival_probability)

        drift_time_mean, drift_time_spread = self.get_s2_drift_time_params(positions_rz)

        if self.electron_lifetime_liquid > 0:
            survival_after_lifetime = np.exp(-drift_time_mean / self.electron_lifetime_liquid)
            n_electron = self.rng.binomial(n=n_electron, p=survival_after_lifetime)
        else:
            self.log.debug("No electron lifetime applied")

        result["n_electron_interface"][mask] = n_electron
        result["drift_time_mean"][mask] = drift_time_mean
        result["drift_time_spread"][mask] = drift_time_spread
        result["x_obs"][mask] = x
        result["y_obs"][mask] = y
        result["z_obs"][mask] = z
        return result

    def get_s2_drift_time_params(self, positions_rz: np.ndarray) -> tuple:
        """Mean and spread of the drift time from the interaction to the liquid surface.

        :param positions_rz: (r, z) of the interactions [cm], shape ``(n, 2)``.
        :return: mean drift time [ns] and drift time spread [ns], arrays of length ``n``.
        """
        drift_velocity_below_gate = self.field_dependencies_map(
            positions_rz, map_name="drift_speed_map"
        )
        diffusion_below_gate = self.field_dependencies_map(
            positions_rz, map_name="diffusion_longitudinal_map"
        )
        drift_length_below_gate = np.clip(self.z_gate - positions_rz[:, 1], 0, None)
        drift_time_below_gate = drift_length_below_gate / drift_velocity_below_gate
        drift_time_mean = drift_time_below_gate + self.drift_time_gate

        spread_squared_below_gate = (
            2 * diffusion_below_gate * drift_time_below_gate / drift_velocity_below_gate**2
        )
        spread_squared_above_gate = (
            2
            * self.diffusion_constant_longitudinal_gate
            * self.drift_time_gate
            / self.drift_velocity_gate**2
        )
        drift_time_spread = np.sqrt(spread_squared_below_gate + spread_squared_above_gate)
        return drift_time_mean, drift_time_spread
