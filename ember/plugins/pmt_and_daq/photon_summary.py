"""Merge the S1, S2 and afterpulse photons into one data type (port of fuse's ``PhotonSummary``)."""

import strax

from ...plugin import EmberVerticalMergerPlugin

export, __all__ = strax.exporter()


@export
class PhotonSummary(EmberVerticalMergerPlugin):
    """Plugin that concatenates propagated photons for S1, S2 and PMT afterpulses."""

    __version__ = "0.0.1"

    depends_on = ("propagated_s2_photons", "propagated_s1_photons", "pmt_afterpulses")
    provides = "photon_summary"
    data_kind = "propagated_photons"
