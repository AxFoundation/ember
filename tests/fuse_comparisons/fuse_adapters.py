"""Everything fuse needs for the comparison with ember.

TEMPORARY: this folder is deleted once ember is finished. Nothing in the ember package may
import from here.

* :class:`FuseCompatibleNestYields`: ember's ``NestYields`` adapted to fuse's
  ``interactions_in_roi``. The installed fuse (1.6.3) cannot run its own ``NestYields`` with
  nestpy 2.1.3 (renamed API), so ember's plugin replaces it in the fuse context.
* :func:`ember_to_fuse_instructions`: converts an ember CSV table into fuse's Geant4-like
  microphysics CSV format.
* :func:`fuse_microphysics_context`: fuse's microphysics context with the public XENONnT config,
  a constant field of 23 V/cm and ember's ``NestYields``.
* :class:`FuseCompatibleS1PhotonPropagation`: fuse's ``S1PhotonPropagation`` with the nestpy 2.1.3
  API. Its scintillation timing is ember's (same NEST call, without fuse's NR override).
* :func:`add_fuse_s1_plugins`: registers fuse's S1 plugins and points them to ember's dummy maps,
  SPE table and gains.
* :func:`add_fuse_s2_plugins`: registers fuse's five S2 plugins (without delayed electrons) with
  ember's maps (luminescence map converted by :func:`write_fuse_luminescence_files`) and
  ember's constant-field drift values.
* :func:`fuse_daq_context`: fuse's PMT and DAQ plugins in their own context, fed with ember's
  stored photons by replay source plugins (:func:`make_replay_plugin`), with ember's detector
  values, pulse shape, noise and afterpulse files.
"""

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

import nestpy
import strax

from ember.plugins.detector_physics import S1PhotonPropagation as EmberS1PhotonPropagation
from ember.plugins.micro_physics import NestYields
from ember.maps import save_map_data
from ember.url_protocols import DUMMY_MAP_DIRECTORY

DEFAULT_FUSE_PUBLIC_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "fuse"
    / "examples"
    / "files"
    / "XENONnT_public_config.json"
)
FUSE_PUBLIC_CONFIG = Path(os.environ.get("EMBER_FUSE_PUBLIC_CONFIG", DEFAULT_FUSE_PUBLIC_CONFIG))

# fuse classifies the cluster type from the Geant4 particle name (fuse.merge_cluster.classify).
GEANT4_PARTICLE_FOR_NESTID = {7: "gamma", 8: "e-"}


class FuseCompatibleNestYields(NestYields):
    """ember's NestYields running on fuse's ``interactions_in_roi``.

    fuse's data has no input-quanta fields (NEST computes everything) and uses ``create_S2`` to
    remove electrons outside the TPC.
    """

    def compute(self, interactions_in_roi):
        extended_dtype = interactions_in_roi.dtype.descr + [
            ("input_photons", np.int32),
            ("input_electrons", np.int32),
        ]
        extended = np.zeros(len(interactions_in_roi), dtype=extended_dtype)
        for name in interactions_in_roi.dtype.names:
            extended[name] = interactions_in_roi[name]
        extended["input_photons"] = -1
        extended["input_electrons"] = -1

        result = super().compute(extended)
        result["electrons"][~interactions_in_roi["create_S2"]] = 0
        return result


def ember_to_fuse_instructions(ember_instructions: pd.DataFrame) -> pd.DataFrame:
    """Convert ember instructions (cm, nestid) to fuse's microphysics CSV (mm, particle names).

    Only gammas (nestid 7) and betas (nestid 8) with given energies are supported.

    :param ember_instructions: ember CSV table.
    :return: table in fuse's ``ChunkInput`` CSV format.
    """
    unsupported = ~ember_instructions["nestid"].isin(list(GEANT4_PARTICLE_FOR_NESTID))
    if unsupported.any() or (ember_instructions["ed"] < 0).any():
        raise ValueError("Only gammas and betas with given energies can be converted.")

    fuse_instructions = pd.DataFrame()
    for axis in ["x", "y", "z"]:
        fuse_instructions[f"{axis}p"] = ember_instructions[axis] * 10
        fuse_instructions[f"{axis}p_pri"] = ember_instructions[axis] * 10
    fuse_instructions["ed"] = ember_instructions["ed"]
    fuse_instructions["time"] = ember_instructions["t"]
    fuse_instructions["eventid"] = ember_instructions["eventid"]
    fuse_instructions["type"] = ember_instructions["nestid"].map(GEANT4_PARTICLE_FOR_NESTID)
    fuse_instructions["trackid"] = 1
    fuse_instructions["parentid"] = 0
    fuse_instructions["creaproc"] = "None"
    fuse_instructions["parenttype"] = "None"
    fuse_instructions["edproc"] = "None"
    return fuse_instructions


def fuse_microphysics_context(output_folder: str, csv_folder: str, csv_name: str):
    """fuse microphysics context matching ember's default settings.

    :param output_folder: storage folder of the fuse data.
    :param csv_folder: folder of the fuse CSV file.
    :param csv_name: name of the fuse CSV file.
    :return: the fuse (strax) context.
    """
    import fuse

    context = fuse.context.microphysics_context(
        output_folder=output_folder, simulation_config_file=str(FUSE_PUBLIC_CONFIG)
    )
    context.register(FuseCompatibleNestYields)
    context.set_config(
        {
            "path": csv_folder,
            "file_name": csv_name,
            "n_interactions_per_chunk": 250,
            "efield_map": "constant_dummy_map://23",
            "nest_width_parameters": {},
            "nest_er_yields_parameters": -1,
            "nest_nr_yields_parameters": -1,
        }
    )
    return context


def beta_instructions(number_of_events: int, energy: float, seed: int) -> pd.DataFrame:
    """Mono-energetic betas uniformly distributed in the TPC (one interaction per event)."""
    import strax

    tpc_radius = strax.URLConfig.evaluate_dry("detector://tpc_radius")
    z_cathode = strax.URLConfig.evaluate_dry("detector://z_cathode")
    z_gate = strax.URLConfig.evaluate_dry("detector://z_gate")
    rng = np.random.default_rng(seed)
    radius = tpc_radius * np.sqrt(rng.uniform(0, 1, number_of_events))
    angle = rng.uniform(-np.pi, np.pi, number_of_events)
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": radius * np.cos(angle),
            "y": radius * np.sin(angle),
            "z": rng.uniform(z_cathode, z_gate, number_of_events),
            "ed": np.full(number_of_events, energy),
            "nestid": np.full(number_of_events, 8),
            "A": np.zeros(number_of_events, dtype=int),
            "Z": np.zeros(number_of_events, dtype=int),
            "photons": np.full(number_of_events, -1),
            "electrons": np.full(number_of_events, -1),
        }
    )


def _fuse_s1_propagation_class():
    from fuse.plugins.detector_physics import S1PhotonPropagation, S1PhotonPropagationBase

    class FuseCompatibleS1PhotonPropagation(S1PhotonPropagation):
        """fuse's S1PhotonPropagation with the nestpy 2.1.3 API.

        fuse 1.6.3 calls ``nestpy.DetectorExample_XENON10`` and ``nestpy.INTERACTION_TYPE``, which
        do not exist in nestpy 2.1.3, and ``np.in1d``, which the installed numpy no longer has.
        Channel assignment (fuse's code with ``np.isin``), optical propagation, transit time and
        gains are fuse's code; only the NEST timing call is ember's.
        """

        def setup(self):
            S1PhotonPropagationBase.setup(self)
            self.nestpy_calc = nestpy.NESTcalc(nestpy.detectors.VDetector())

        def photon_channels(self, positions, n_photon_hits):
            channels = np.arange(self.n_tpc_pmts)
            p_per_channel = self.s1_pattern_map(positions)
            p_per_channel[:, np.isin(channels, self.turned_off_pmts)] = 0

            photon_channels = []
            for probabilities, n_hits in zip(p_per_channel, n_photon_hits):
                photon_channels.append(
                    self.rng.choice(
                        channels, size=n_hits, p=probabilities / np.sum(probabilities), replace=True
                    )
                )
            return np.concatenate(photon_channels)

        nest_scintillation_timing = EmberS1PhotonPropagation.nest_scintillation_timing

    return FuseCompatibleS1PhotonPropagation


def add_fuse_s1_plugins(context) -> None:
    """Register fuse's S1 plugins with ember's dummy maps, SPE table, gains and detector values.

    fuse's pattern map is interpolated with ``RegularGridInterpolator`` (ember interpolates
    linearly on the same grid) instead of fuse's default ``WeightedNearestNeighbors``.

    :param context: fuse context providing ``microphysics_summary``.
    """
    from fuse.plugins.detector_physics import S1PhotonHits

    def detector_value(key):
        return strax.URLConfig.evaluate_dry(f"detector://{key}")

    context.register([S1PhotonHits, _fuse_s1_propagation_class()])
    pattern_map_file = DUMMY_MAP_DIRECTORY / "s1_pattern_map_dummy.pkl"
    timing_map_file = DUMMY_MAP_DIRECTORY / "s1_optical_propagation_dummy.json"
    spe_file = DUMMY_MAP_DIRECTORY / "spe_distributions_dummy.csv"
    context.set_config(
        {
            "s1_pattern_map": (
                f"pattern_map://resource://{pattern_map_file}?&fmt=pkl"
                "&pmt_mask=plugin.pmt_mask&method=RegularGridInterpolator"
            ),
            "s1_optical_propagation_spline": (
                f"itp_map://resource://{timing_map_file}?&fmt=json&method=RegularGridInterpolator"
            ),
            "photon_area_distribution": f"simple_load://resource://{spe_file}?&fmt=csv",
            "gain_model_mc": strax.URLConfig.evaluate_dry("ember_pmt_to_pe://").tolist(),
            "p_double_pe_emision": detector_value("p_double_pe_emission"),
            "s1_detection_efficiency": detector_value("s1_detection_efficiency"),
            "pmt_transit_time_mean": detector_value("pmt_transit_time_mean"),
            "pmt_transit_time_spread": detector_value("pmt_transit_time_spread"),
            "pmt_circuit_load_resistor": detector_value("pmt_circuit_load_resistor"),
            "digitizer_bits": detector_value("digitizer_bits"),
            "digitizer_voltage_range": detector_value("digitizer_voltage_range"),
            "n_top_pmts": detector_value("n_top_pmts"),
            "n_tpc_pmts": detector_value("n_tpc_pmts"),
            "maximum_recombination_time": 1000,
            "override_s1_nr_scint_time": False,
            "s1_nr_scint_time_ed_override": 10.0,
            "s1_nr_scint_time_nesttype_override": 8,
            "s1_nr_scint_time_excitonfrac_override": 0.15,
        }
    )


def _fuse_s2_classes() -> list:
    from fuse.plugins.detector_physics import (
        ElectronDrift,
        ElectronExtraction,
        ElectronPropagation,
        S2PhotonPropagation,
        SecondaryScintillation,
    )

    class FuseCompatibleSecondaryScintillation(SecondaryScintillation):
        """fuse's SecondaryScintillation; ``np.in1d`` (removed in numpy 2.4) is mapped to
        ``np.isin`` while ``compute`` runs."""

        def compute(self, individual_electrons, interactions_in_roi):
            with mock.patch.object(np, "in1d", np.isin, create=True):
                return super().compute(individual_electrons, interactions_in_roi)

    class FuseS2PhotonPropagationWithoutDelayedElectrons(S2PhotonPropagation):
        """fuse's S2PhotonPropagation (Garfield luminescence) on the regular electrons only.

        fuse depends on the merged data types of the delayed electron simulation, which is not
        part of the comparison; without delayed electrons they equal ``s2_photons`` and
        ``extracted_electrons``.
        """

        depends_on = ("s2_photons", "extracted_electrons")

    return [
        ElectronDrift,
        ElectronPropagation,
        ElectronExtraction,
        FuseCompatibleSecondaryScintillation,
        FuseS2PhotonPropagationWithoutDelayedElectrons,
    ]


def ember_drift_constants() -> dict:
    """Drift velocity [cm/ns] and diffusion constants [cm²/ns] of ember's drift map in the TPC."""
    drift_map = strax.URLConfig.evaluate_dry(
        "ember_map://field_dependencies_dummy.json?method=linear"
    )
    position = np.array([[0.0, -70.0]])
    return {
        "drift_velocity": float(drift_map(position, map_name="drift_speed_map")[0]),
        "diffusion_longitudinal": float(
            drift_map(position, map_name="diffusion_longitudinal_map")[0]
        ),
        "diffusion_transverse": float(drift_map(position, map_name="diffusion_transverse_map")[0]),
    }


def write_fuse_luminescence_files(folder: Path) -> tuple:
    """Write ember's luminescence timing map in fuse's Garfield format.

    fuse interpolates the inverse CDF between gas-gap rows and draws from its points
    ``0 ... n - 2`` only, so ember's inverse CDF is written twice (two identical gas-gap rows) with
    its last point repeated. The gas-gap map is constant (ember's gas gap).

    :param folder: output folder.
    :return: paths of the timing table (``.npy``) and the gas-gap map (``.json``).
    """
    luminescence_map = strax.URLConfig.evaluate_dry("ember_map://s2_luminescence_timing_dummy.json")
    inverse_cdf = np.append(luminescence_map.values, luminescence_map.values[-1])
    gas_gap = strax.URLConfig.evaluate_dry("detector://z_anode") - strax.URLConfig.evaluate_dry(
        "detector://z_liquid_level"
    )

    table = np.zeros(
        2, dtype=[("gas_gap", np.float32), ("timing_inv_cdf", np.float32, len(inverse_cdf))]
    )
    table["gas_gap"] = [0.2, 0.4]
    table["timing_inv_cdf"] = inverse_cdf
    table_path = Path(folder) / "fuse_luminescence_timing.npy"
    np.save(table_path, table)

    gas_gap_path = Path(folder) / "fuse_gas_gap_map.json"
    save_map_data(
        {
            "coordinate_system": [["x", [-100.0, 100.0, 2]], ["y", [-100.0, 100.0, 2]]],
            "map": np.full((2, 2), gas_gap),
        },
        gas_gap_path,
    )
    return table_path, gas_gap_path


def add_fuse_s2_plugins(context, folder: Path) -> None:
    """Register fuse's five S2 plugins with ember's maps and constant-field detector values.

    Field distortion and all fuse maps of the drift are off; drift velocity and diffusion
    constants are ember's drift-map values at 23 V/cm. Options that are not set here come from
    fuse's public config.

    :param context: fuse context providing ``microphysics_summary`` (S1 plugins already added).
    :param folder: folder for the luminescence files in fuse's format.
    """
    import straxen  # noqa: F401  (registers resource:// and itp_map://)

    def detector_value(key):
        return strax.URLConfig.evaluate_dry(f"detector://{key}")

    plugins = _fuse_s2_classes()
    context.register(plugins)
    for plugin in plugins:
        for option_key, option in plugin.takes_config.items():
            if option_key in context.config:
                continue
            if isinstance(option.default, str) and "SIMULATION_CONFIG_FILE.json" in option.default:
                context.config[option_key] = option.default.replace(
                    "SIMULATION_CONFIG_FILE.json", str(FUSE_PUBLIC_CONFIG)
                )

    constants = ember_drift_constants()
    luminescence_table, gas_gap_map = write_fuse_luminescence_files(folder)
    field_map_file = DUMMY_MAP_DIRECTORY / "field_dependencies_dummy.json"
    unused_field_map = (
        f"itp_map://resource://{field_map_file}?&fmt=json&method=RegularGridInterpolator"
    )
    pattern_map_file = DUMMY_MAP_DIRECTORY / "s2_pattern_map_dummy.pkl"
    correction_map_file = DUMMY_MAP_DIRECTORY / "s2_correction_map_dummy.json"
    optical_map_file = DUMMY_MAP_DIRECTORY / "s2_optical_propagation_dummy.json"
    single_electron_gain = detector_value("g2") / detector_value("electron_extraction_efficiency")
    context.set_config(
        {
            "drift_velocity_liquid": constants["drift_velocity"],
            "diffusion_constant_longitudinal": constants["diffusion_longitudinal"],
            "diffusion_constant_transverse": constants["diffusion_transverse"],
            "drift_time_gate": detector_value("drift_time_gate"),
            "electron_lifetime_liquid": detector_value("electron_lifetime_liquid"),
            "enable_survival_probability_map": False,
            "enable_drift_velocity_map": False,
            "norm_drift_velocity": False,
            "enable_diffusion_longitudinal_map": False,
            "enable_diffusion_transverse_map": False,
            "field_distortion_model": "none",
            "fdc_map_fuse": unused_field_map,
            "field_dependencies_map_tmp": unused_field_map,
            "diffusion_longitudinal_map_tmp": unused_field_map,
            "tpc_length": -detector_value("z_cathode"),
            "tpc_radius": detector_value("tpc_radius"),
            "gate_to_anode_distance": detector_value("z_anode") - detector_value("z_gate"),
            "elr_gas_gap_length": detector_value("z_anode") - detector_value("z_liquid_level"),
            "electron_extraction_yield": detector_value("electron_extraction_efficiency"),
            "electron_trapping_time": detector_value("electron_trapping_time"),
            "ext_eff_from_map": False,
            "se_gain_from_map": False,
            "g2_mean": detector_value("g2"),
            "s2_secondary_sc_gain_mc": single_electron_gain,
            "s2_correction_map": (
                f"itp_map://resource://{correction_map_file}?&fmt=json"
                "&method=RegularGridInterpolator"
            ),
            "se_gain_map": "constant_dummy_map://1",
            "s2_pattern_map": (
                f"pattern_map://resource://{pattern_map_file}?&fmt=pkl"
                "&pmt_mask=plugin.pmt_mask&method=RegularGridInterpolator"
            ),
            "s2_mean_area_fraction_top": detector_value("s2_mean_area_fraction_top"),
            "s2_aft_sigma": 0.0,
            "s2_aft_skewness": 0.0,
            "phase_s2": "gas",
            "singlet_fraction_gas": detector_value("singlet_fraction_gas"),
            "singlet_lifetime_gas": detector_value("singlet_lifetime_gas"),
            "triplet_lifetime_gas": detector_value("triplet_lifetime_gas"),
            "s2_luminescence_map": f"simple_load://resource://{luminescence_table}?&fmt=npy",
            "garfield_gas_gap_map": (
                f"itp_map://resource://{gas_gap_map}?&fmt=json&method=RegularGridInterpolator"
            ),
            "s2_optical_propagation_spline": (
                f"itp_map://resource://{optical_map_file}?&fmt=json&method=RegularGridInterpolator"
            ),
        }
    )


def make_replay_plugin(data_type: str, data_kind: str, chunks: list) -> type:
    """Return a source plugin that hands over stored ember chunks.

    :param data_type: ember data type, e.g. ``propagated_s1_photons``.
    :param data_kind: data kind of the replayed data.
    :param chunks: the ``strax.Chunk`` objects of the data type, loaded from ember's storage.
    :return: the plugin class.
    """

    class EmberReplay(strax.Plugin):
        __version__ = "0.0.0"
        depends_on = tuple()
        provides = data_type
        rechunk_on_save = False
        save_when = strax.SaveWhen.NEVER
        dtype = chunks[0].data.dtype

        def compute(self, chunk_i):
            self.source_done = chunk_i == len(chunks) - 1
            return self.chunk(
                start=chunks[chunk_i].start, end=chunks[chunk_i].end, data=chunks[chunk_i].data
            )

        def source_finished(self):
            return self.source_done

        def is_ready(self, chunk_i):
            if "ready" not in self.__dict__:
                self.ready = False
            self.ready ^= True
            return self.ready

    EmberReplay.data_kind = data_kind
    EmberReplay.__name__ = f"EmberReplay_{data_type}"
    return EmberReplay


def fuse_daq_config() -> dict:
    """Options of fuse's PMT and DAQ plugins set to ember's values, files and gains."""

    def detector_value(key):
        return strax.URLConfig.evaluate_dry(f"detector://{key}")

    pulse_shape = strax.URLConfig.evaluate_dry("ember_file://pmt_pulse_shape_dummy.json")
    config = {
        key: detector_value(key)
        for key in [
            "pmt_circuit_load_resistor",
            "external_amplification",
            "digitizer_bits",
            "digitizer_voltage_range",
            "enable_noise",
            "pmt_pulse_time_rounding",
            "samples_after_pulse_center",
            "samples_before_pulse_center",
            "samples_to_store_after",
            "samples_to_store_before",
            "digitizer_reference_baseline",
            "zle_threshold",
            "trigger_window",
            "special_thresholds",
            "n_tpc_pmts",
            "enable_pmt_afterpulses",
            "pmt_ap_modifier",
            "pmt_ap_t_modifier",
        ]
    }
    config.update(
        {
            "dt": detector_value("sample_duration"),
            "pe_pulse_ts": list(pulse_shape["pe_pulse_ts"]),
            "pe_pulse_ys": list(pulse_shape["pe_pulse_ys"]),
            "noise_data": "ember_file://noise_dummy.npz",
            "photon_ap_cdfs": "ember_file://pmt_afterpulse_config_dummy.json.gz",
            "gain_model_mc": "ember_pmt_to_pe://",
        }
    )
    return config


def fuse_daq_context(output_folder: str, ember_context, run_id: str, replayed: dict):
    """Context with fuse's PMT and DAQ plugins fed with ember's stored data.

    The ember data is loaded before fuse (and with it straxen, which patches strax' context) is
    imported.

    :param output_folder: storage folder of the fuse data.
    :param ember_context: ember context in which the replayed data is stored.
    :param run_id: run id of the ember data; the fuse data gets the same run id.
    :param replayed: ``{data_type: data_kind}`` of the ember data handed to fuse, e.g.
        ``{"photon_summary": "propagated_photons"}`` (fuse computes only the DAQ) or the S1 and S2
        photons (fuse also computes the afterpulses and the photon summary).
    :return: the context.
    """
    chunks = {name: list(ember_context.get_iter(run_id, name)) for name in replayed}
    from fuse.plugins.pmt_and_daq import (
        PhotonSummary,
        PMTAfterPulses,
        PMTResponseAndDAQ,
        PulseWindow,
    )

    plugins = [PulseWindow, PMTResponseAndDAQ]
    if "photon_summary" not in replayed:
        plugins = [PMTAfterPulses, PhotonSummary] + plugins
    context = strax.Context(
        storage=[strax.DataDirectory(output_folder, readonly=False)],
        register=[make_replay_plugin(name, kind, chunks[name]) for name, kind in replayed.items()]
        + plugins,
    )
    context.set_config(fuse_daq_config())
    return context
