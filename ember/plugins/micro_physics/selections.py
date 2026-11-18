"""Volume selection and merging of selections (ported from fuse's ``cuts_and_selections``)."""

import numpy as np
import strax

from ...common import VOLUMES_IDS
from ...dtypes import clustered_interactions_fields, volume_properties_fields
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class VolumeSelection(strax.CutPlugin):
    """Select interactions inside the accepted detector volumes."""

    __version__ = "0.0.1"

    depends_on = "volume_properties"
    provides = "volume_selection"
    cut_name = "volume_selection"
    cut_description = "Interaction is inside an accepted detector volume"

    accept_volumes = ["tpc", "below_cathode"]

    def cut_by(self, clustered_interactions):
        mask = np.zeros(len(clustered_interactions), dtype=bool)
        for volume_name in self.accept_volumes:
            mask |= clustered_interactions["vol_id"] == VOLUMES_IDS[volume_name]
        return mask


@export
class SelectionMerger(EmberBasePlugin):
    """Keep only interactions passing the selection logic and attach the volume properties.

    The selection logic is a string expression over boolean fields of the dependencies, using
    ``&``, ``|``, ``~`` and parentheses, e.g. ``"volume_selection & energy_range_cut"``.
    """

    __version__ = "0.0.1"
    save_when = strax.SaveWhen.TARGET
    provides = "interactions_in_roi"
    data_kind = "interactions_in_roi"

    selection_logic = "volume_selection"

    dtype = clustered_interactions_fields + volume_properties_fields + strax.time_fields

    @staticmethod
    def evaluate_selection_logic(interactions: np.ndarray, expression: str) -> np.ndarray:
        """Evaluate a boolean expression over the fields of ``interactions``.

        :param interactions: structured array with the selection fields.
        :param expression: e.g. ``"volume_selection & ~some_cut"``.
        :return: boolean mask.
        """
        fields = {name: interactions[name].astype(bool) for name in interactions.dtype.names}
        return np.asarray(eval(expression, {"__builtins__": None}, fields), dtype=np.bool_)

    def compute(self, clustered_interactions):
        if len(clustered_interactions) == 0:
            return np.zeros(0, dtype=self.dtype)

        mask = self.evaluate_selection_logic(clustered_interactions, self.selection_logic)
        selected = clustered_interactions[mask]
        result = np.zeros(len(selected), dtype=self.dtype)
        strax.copy_to_buffer(selected, result, "_copy_selected_interactions")
        return result


@export
class DefaultSimulation(SelectionMerger):
    """Default selection: interactions in the TPC and below the cathode."""

    depends_on = ("clustered_interactions", "volume_properties", "volume_selection")
    __version__ = "0.0.1"
    selection_logic = "volume_selection"
