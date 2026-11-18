"""Assign each interaction to a detector volume (ported from fuse's ``VolumeProperties``).

Volumes (z intervals are lower-inclusive and upper-exclusive, r < tpc_radius):

* ``tpc``: ``z_cathode <= z < z_gate``, liquid xenon density.
* ``below_cathode``: ``z_bottom_pmts <= z < z_cathode``, liquid xenon density.
* ``gas_phase``: ``z_liquid_level <= z < z_top_pmts``, gaseous xenon density.
* everything else is ``undefined`` (id 0).

Unlike fuse there is no ``create_S2`` flag: electrons from below the cathode are removed later
by the electron survival map during the electron drift.
"""

import numba
import numpy as np
import strax

from ...common import VOLUMES_IDS
from ...dtypes import volume_properties_fields
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class VolumeProperties(EmberBasePlugin):
    """Assign volume id and xenon density to every interaction."""

    __version__ = "0.0.1"

    depends_on = "clustered_interactions"
    provides = "volume_properties"

    dtype = volume_properties_fields + strax.time_fields

    tpc_radius = strax.URLConfig(
        default="detector://tpc_radius",
        type=(int, float),
        help="Radius of the sensitive volume [cm]",
    )

    z_gate = strax.URLConfig(
        default="detector://z_gate",
        type=(int, float),
        help="z position of the gate electrode [cm]",
    )

    z_cathode = strax.URLConfig(
        default="detector://z_cathode",
        type=(int, float),
        help="z position of the cathode electrode [cm]",
    )

    z_bottom_pmts = strax.URLConfig(
        default="detector://z_bottom_pmts",
        type=(int, float),
        help="z position of the bottom PMT array [cm]",
    )

    z_liquid_level = strax.URLConfig(
        default="detector://z_liquid_level",
        type=(int, float),
        help="z position of the liquid-gas interface [cm]",
    )

    z_top_pmts = strax.URLConfig(
        default="detector://z_top_pmts",
        type=(int, float),
        help="z position of the top PMT array [cm]",
    )

    xenon_density_liquid = strax.URLConfig(
        default="detector://xenon_density_liquid",
        type=(int, float),
        help="Density of liquid xenon [g/cm3]",
    )

    xenon_density_gas = strax.URLConfig(
        default="detector://xenon_density_gas",
        type=(int, float),
        help="Density of gaseous xenon [g/cm3]",
    )

    def compute(self, clustered_interactions):
        result = np.zeros(len(clustered_interactions), dtype=self.dtype)
        if len(clustered_interactions) == 0:
            return result

        result["time"] = clustered_interactions["time"]
        result["endtime"] = clustered_interactions["endtime"]

        volumes = [
            ("tpc", self.z_cathode, self.z_gate, self.xenon_density_liquid),
            ("below_cathode", self.z_bottom_pmts, self.z_cathode, self.xenon_density_liquid),
            ("gas_phase", self.z_liquid_level, self.z_top_pmts, self.xenon_density_gas),
        ]
        for volume_name, min_z, max_z, density in volumes:
            mask = in_cylinder(
                clustered_interactions["x"],
                clustered_interactions["y"],
                clustered_interactions["z"],
                min_z,
                max_z,
                self.tpc_radius,
            )
            result["vol_id"][mask] = VOLUMES_IDS[volume_name]
            result["xe_density"][mask] = density

        return result


@numba.njit
def in_cylinder(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, min_z: float, max_z: float, max_r: float
) -> np.ndarray:
    """Check which positions lie inside a cylinder around the z axis.

    :param x: x positions [cm].
    :param y: y positions [cm].
    :param z: z positions [cm].
    :param min_z: inclusive lower z boundary.
    :param max_z: exclusive upper z boundary.
    :param max_r: exclusive radial boundary.
    :return: boolean mask.
    """
    r = np.sqrt(x**2 + y**2)
    return (r < max_r) & (z < max_z) & (z >= min_z)
