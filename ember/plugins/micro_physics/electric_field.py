"""Electric drift field at the interaction position (ported from fuse's ``ElectricField``)."""

import numpy as np
import strax

from ...dtypes import electric_fields
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class ElectricField(EmberBasePlugin):
    """Evaluate the drift field map at the (r, z) position of each interaction."""

    __version__ = "0.0.1"

    depends_on = "interactions_in_roi"
    provides = "electric_field_values"
    data_kind = "interactions_in_roi"

    save_when = strax.SaveWhen.TARGET

    dtype = electric_fields + strax.time_fields

    efield_map = strax.URLConfig(
        default="ember_map://drift_field_constant.json?method=linear",
        cache=True,
        help="Map of the drift field [V/cm] as a function of (r, z)",
    )

    def compute(self, interactions_in_roi):
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        if len(interactions_in_roi) == 0:
            return result

        r = np.sqrt(interactions_in_roi["x"] ** 2 + interactions_in_roi["y"] ** 2)
        positions = np.stack((r, interactions_in_roi["z"]), axis=1)
        e_field = self.efield_map(positions)

        number_of_negative_values = np.sum(e_field < 0)
        if number_of_negative_values > 0:
            self.log.warning(
                f"Found {number_of_negative_values} negative electric field values. Clipping to 0."
            )
        number_of_nan_values = np.sum(np.isnan(e_field))
        if number_of_nan_values > 0:
            self.log.warning(
                f"Found {number_of_nan_values} NaN electric field values. Clipping to 0."
            )
        result["e_field"] = np.clip(np.nan_to_num(e_field), 0, None)
        return result
