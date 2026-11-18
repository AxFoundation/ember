"""Summary of the microphysics simulation (ported from fuse's ``MicroPhysicsSummary``)."""

import strax

export, __all__ = strax.exporter()


@export
class MicroPhysicsSummary(strax.MergeOnlyPlugin):
    """Merge interactions, quanta and electric field into ``microphysics_summary``.

    This data type is the interface to the detector-physics simulation. It contains all fields
    of fuse's detector-physics CSV input (x, y, z, ed, nestid, cluster_id, eventid, t, photons,
    electrons, excitons, e_field) plus A, Z, xe_density, vol_id and the input-quanta fields.
    """

    depends_on = ("interactions_in_roi", "quanta", "electric_field_values")
    rechunk_on_save = False
    save_when = strax.SaveWhen.ALWAYS
    provides = "microphysics_summary"
    __version__ = "0.0.1"
