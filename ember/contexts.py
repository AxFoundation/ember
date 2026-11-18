"""Context builders for ember.

The default context reads interaction instructions from a CSV file, simulates the microphysics up
to ``microphysics_summary``, the S1 photons up to ``propagated_s1_photons``, the S2 photons up to
``propagated_s2_photons`` and the PMT response and DAQ up to ``raw_records``. No database, no
network: all detector parameters come from ``ember/config/detector.yaml`` (``detector://``, see
:mod:`ember.url_protocols`) and all maps, tables and files from ``ember/dummy_maps``
(``ember_map://``, ``ember_table://``, ``ember_file://``).

The context stops at ``raw_records``. To process them with the optional package abstrax, register
abstrax's plugins in this context (``context.register(abstrax.contexts.ABSTRAX_PLUGINS)``) and set
``gain_model`` to ``ember_pmt_to_pe://`` (see tutorial 4).
"""

from __future__ import annotations

from pathlib import Path

import strax

from ember.plugins import detector_physics, micro_physics, pmt_and_daq

MICROPHYSICS_PLUGINS = [
    micro_physics.CsvInput,
    micro_physics.VolumeProperties,
    micro_physics.VolumeSelection,
    micro_physics.DefaultSimulation,
    micro_physics.ElectricField,
    micro_physics.NestYields,
    micro_physics.MicroPhysicsSummary,
]

S1_PLUGINS = [
    detector_physics.S1PhotonHits,
    detector_physics.S1PhotonPropagation,
]

S2_PLUGINS = [
    detector_physics.ElectronDrift,
    detector_physics.ElectronPropagation,
    detector_physics.ElectronExtraction,
    detector_physics.SecondaryScintillation,
    detector_physics.S2PhotonPropagation,
]

DAQ_PLUGINS = [
    pmt_and_daq.PMTAfterPulses,
    pmt_and_daq.PhotonSummary,
    pmt_and_daq.PulseWindow,
    pmt_and_daq.PMTResponseAndDAQ,
]


def ember_context(output_folder: str | Path = "./strax_data", **context_kwargs) -> strax.Context:
    """Build a strax context with all ember plugins registered.

    Set the CSV file with ``st.set_config({"input_file": "my_instructions.csv"})``.

    :param output_folder: directory for the simulated data. Created if it does not exist.
    :param context_kwargs: forwarded to ``strax.Context``.
    :return: the context.
    """
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    context = strax.Context(
        storage=[strax.DataDirectory(str(output_folder), readonly=False)],
        register=MICROPHYSICS_PLUGINS + S1_PLUGINS + S2_PLUGINS + DAQ_PLUGINS,
        **context_kwargs,
    )
    return context
