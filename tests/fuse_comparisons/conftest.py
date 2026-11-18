"""Fixtures for the temporary fuse comparison (run with ``pytest tests/fuse_comparisons``).

Skipped if fuse or straxen are not installed or fuse's public config file is not found.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

pytest.importorskip("fuse")
pytest.importorskip("straxen")

from fuse_adapters import (  # noqa: E402
    FUSE_PUBLIC_CONFIG,
    add_fuse_s1_plugins,
    add_fuse_s2_plugins,
    ember_drift_constants,
    beta_instructions,
    ember_to_fuse_instructions,
    fuse_daq_context,
    fuse_microphysics_context,
)

import ember  # noqa: E402

NUMBER_OF_EVENTS = 1000
ENERGY = 40.0
RUN_ID = "000000"


@pytest.fixture(scope="session")
def simulations(tmp_path_factory) -> dict:
    """Simulate the same 40 keV betas with ember and with fuse (microphysics_summary)."""
    if not FUSE_PUBLIC_CONFIG.exists():
        pytest.skip(f"fuse public config not found at {FUSE_PUBLIC_CONFIG}")

    folder = tmp_path_factory.mktemp("fuse_comparison")
    instructions = beta_instructions(NUMBER_OF_EVENTS, ENERGY, seed=2026)
    ember_csv = folder / "beta_40keV_ember.csv"
    instructions.to_csv(ember_csv, index=False)
    ember_to_fuse_instructions(instructions).to_csv(folder / "beta_40keV_fuse.csv", index=False)

    ember_context = ember.ember_context(output_folder=folder / "ember_data")
    ember_context.set_config({"input_file": str(ember_csv), "n_interactions_per_chunk": 250})
    fuse_context = fuse_microphysics_context(
        str(folder / "fuse_data"), str(folder), "beta_40keV_fuse.csv"
    )
    return {
        "instructions": instructions,
        "ember": ember_context.get_array(RUN_ID, "microphysics_summary"),
        "fuse": fuse_context.get_array(RUN_ID, "microphysics_summary"),
        "ember_context": ember_context,
        "fuse_context": fuse_context,
    }


@pytest.fixture(scope="session")
def s1_simulations(simulations) -> dict:
    """S1 hits and photons of the same betas, simulated with ember and with fuse's S1 plugins."""
    fuse_context = simulations["fuse_context"]
    add_fuse_s1_plugins(fuse_context)
    result = {}
    for name in ["ember", "fuse"]:
        context = simulations[f"{name}_context"]
        result[name] = {
            "interactions": context.get_array(RUN_ID, ["microphysics_summary", "s1_photon_hits"]),
            "photons": context.get_array(RUN_ID, "propagated_s1_photons"),
        }
    return result


S2_DATA_TYPES = {
    "interactions": ["microphysics_summary", "drifted_electrons", "s2_photons_sum"],
    "electrons_at_interface": ["electrons_at_interface"],
    "extracted": ["extracted_electrons", "s2_photons"],
    "photons": ["propagated_s2_photons"],
}


@pytest.fixture(scope="session")
def s2_simulations(simulations, s1_simulations, tmp_path_factory) -> dict:
    """S2 electrons and photons of the same betas, simulated with ember and with fuse.

    fuse uses one longitudinal diffusion constant below and above the gate, so ember's
    ``diffusion_constant_longitudinal_gate`` is set to the drift-map value for the comparison.
    """
    add_fuse_s2_plugins(simulations["fuse_context"], tmp_path_factory.mktemp("fuse_s2_maps"))
    simulations["ember_context"].set_config(
        {"diffusion_constant_longitudinal_gate": ember_drift_constants()["diffusion_longitudinal"]}
    )
    result = {}
    for name in ["ember", "fuse"]:
        context = simulations[f"{name}_context"]
        result[name] = {
            key: context.get_array(RUN_ID, data_types) for key, data_types in S2_DATA_TYPES.items()
        }
    return result


NUMBER_OF_DAQ_EVENTS = 200


@pytest.fixture(scope="module")
def daq_simulations(tmp_path_factory) -> dict:
    """raw_records of the same ember photons, made by ember and by fuse's PMT and DAQ plugins.

    Module scope: the arrays are released after the DAQ comparison (memory of the full run).

    ``exact``: fuse's PulseWindow and PMTResponseAndDAQ on ember's photon_summary, noise off.
    ``statistical``: fuse's PMTAfterPulses, PhotonSummary, PulseWindow and PMTResponseAndDAQ on
    ember's S1 and S2 photons, noise and afterpulses on (different random numbers than ember).
    """
    folder = tmp_path_factory.mktemp("daq_comparison")
    instructions = beta_instructions(NUMBER_OF_DAQ_EVENTS, ENERGY, seed=2027)
    ember_csv = folder / "beta_40keV_daq.csv"
    instructions.to_csv(ember_csv, index=False)
    ember_context = ember.ember_context(output_folder=folder / "ember_data")
    ember_context.set_config({"input_file": str(ember_csv)})

    without_noise = {"enable_noise": False}
    fuse_exact = fuse_daq_context(
        str(folder / "fuse_exact"),
        ember_context,
        RUN_ID,
        {"photon_summary": "propagated_photons"},
    )
    fuse_exact.set_config(without_noise)
    fuse_statistical = fuse_daq_context(
        str(folder / "fuse_statistical"),
        ember_context,
        RUN_ID,
        {"propagated_s1_photons": "s1_photons", "propagated_s2_photons": "s2_photons"},
    )
    return {
        "exact": {
            "ember": {
                "raw_records": ember_context.get_array(RUN_ID, "raw_records", config=without_noise),
                "pulse_windows": ember_context.get_array(RUN_ID, "pulse_windows"),
            },
            "fuse": {
                "raw_records": fuse_exact.get_array(RUN_ID, "raw_records"),
                "pulse_windows": fuse_exact.get_array(RUN_ID, "pulse_windows"),
            },
        },
        "statistical": {
            name: {
                "raw_records": context.get_array(RUN_ID, "raw_records"),
                "pmt_afterpulses": context.get_array(RUN_ID, "pmt_afterpulses"),
            }
            for name, context in [("ember", ember_context), ("fuse", fuse_statistical)]
        },
        "photons": ember_context.get_array(RUN_ID, "photon_summary"),
        "spe_gain": ember_context.get_single_plugin(RUN_ID, "pmt_afterpulses").gains[0],
    }
