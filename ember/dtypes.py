"""Field definitions of the ember data types (ported from ``fuse.dtypes``)."""

import numpy as np

cluster_positions_fields = [
    (("x position of the cluster [cm]", "x"), np.float32),
    (("y position of the cluster [cm]", "y"), np.float32),
    (("z position of the cluster [cm]", "z"), np.float32),
]

cluster_id_fields = [
    (("Energy of the cluster [keV]", "ed"), np.float32),
    (("NEST interaction type", "nestid"), np.int8),
    (("ID of the cluster", "cluster_id"), np.int32),
]

csv_cluster_misc_fields = [
    (("Event ID of the input file", "eventid"), np.int32),
    (("Time of the interaction relative to the event start [ns]", "t"), np.int64),
]

nuclear_fields = [
    (("Mass number of the interacting particle (ions only)", "A"), np.int16),
    (("Charge number of the interacting particle (ions only)", "Z"), np.int16),
]

input_quanta_fields = [
    (("Number of photons given in the input file, -1 if not given", "input_photons"), np.int32),
    (
        ("Number of electrons given in the input file, -1 if not given", "input_electrons"),
        np.int32,
    ),
    (("True if ed was estimated from the given quanta", "ed_is_estimated"), np.bool_),
]

volume_properties_fields = [
    (("Xenon density at the cluster position [g/cm3]", "xe_density"), np.float32),
    (("ID of the volume in which the cluster occured", "vol_id"), np.int8),
]

quanta_fields = [
    (("Number of photons at interaction position", "photons"), np.int32),
    (("Number of electrons at interaction position", "electrons"), np.int32),
    (("Number of excitons at interaction position", "excitons"), np.int32),
]

electric_fields = [
    (("Electric field value at the cluster position [V/cm]", "e_field"), np.float32),
]

clustered_interactions_fields = (
    cluster_positions_fields
    + cluster_id_fields
    + csv_cluster_misc_fields
    + nuclear_fields
    + input_quanta_fields
)

propagated_photons_fields = [
    (("PMT channel of the photon", "channel"), np.int16),
    (("Photon creates a double photo-electron emission", "dpe"), np.bool_),
    (("Sampled PMT gain for the photon", "photon_gain"), np.int32),
    (("ID of the cluster creating the photon", "cluster_id"), np.int32),
    (("Type of the photon. S1 (1), S2 (2) or PMT AP (0)", "photon_type"), np.int8),
]

individual_electron_fields = [
    (("x position of the electron at the interface [cm]", "x_interface"), np.float32),
    (("y position of the electron at the interface [cm]", "y_interface"), np.float32),
    (("ID of the cluster creating the electron", "cluster_id"), np.int32),
]
