"""End-to-end tests of the ember context (microphysics, S1 and S2 photons, raw_records)."""

import numpy as np
import pytest
import strax
from instruction_helpers import make_instructions, write_csv

import ember

FUSE_DETECTOR_PHYSICS_FIELDS = [
    "x",
    "y",
    "z",
    "ed",
    "nestid",
    "cluster_id",
    "eventid",
    "t",
    "photons",
    "electrons",
    "excitons",
    "e_field",
    "time",
    "endtime",
]


def test_context_registers_all_plugins(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    for data_type in [
        "clustered_interactions",
        "volume_properties",
        "volume_selection",
        "interactions_in_roi",
        "electric_field_values",
        "quanta",
        "microphysics_summary",
        "s1_photon_hits",
        "propagated_s1_photons",
        "drifted_electrons",
        "electrons_at_interface",
        "extracted_electrons",
        "s2_photons",
        "s2_photons_sum",
        "propagated_s2_photons",
        "pmt_afterpulses",
        "photon_summary",
        "pulse_windows",
        "pulse_ids",
        "raw_records",
    ]:
        assert data_type in context._plugin_class_registry


def test_microphysics_summary_end_to_end(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    context.set_config({"input_file": write_csv(make_instructions(60), tmp_path)})
    context.make("000000", "microphysics_summary")
    assert context.is_stored("000000", "microphysics_summary")

    summary = context.get_array("000000", "microphysics_summary")
    assert len(summary) == 60
    assert set(FUSE_DETECTOR_PHYSICS_FIELDS) <= set(summary.dtype.names)
    assert np.all(np.diff(summary["time"]) >= 0)
    assert np.all(summary["photons"] > 0)
    assert np.all(summary["electrons"] > 0)
    assert np.all(summary["excitons"] <= summary["photons"])


def test_stored_data_is_reloaded(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    context.set_config({"input_file": write_csv(make_instructions(20), tmp_path)})
    first = context.get_array("000000", "microphysics_summary")
    reloaded = ember.ember_context(output_folder=tmp_path / "data").get_array(
        "000000", "microphysics_summary", config=dict(input_file="not_needed_when_stored.csv")
    )
    np.testing.assert_array_equal(first, reloaded)


def test_s1_photons_end_to_end_and_reload(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    context.set_config({"input_file": write_csv(make_instructions(30), tmp_path)})
    context.make("000000", "propagated_s1_photons")
    assert context.is_stored("000000", "s1_photon_hits")
    assert context.is_stored("000000", "propagated_s1_photons")

    photons = context.get_array("000000", "propagated_s1_photons")
    reloaded = ember.ember_context(output_folder=tmp_path / "data").get_array(
        "000000", "propagated_s1_photons", config=dict(input_file="not_needed_when_stored.csv")
    )
    np.testing.assert_array_equal(photons, reloaded)


def test_s2_photons_end_to_end_and_reload(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    context.set_config({"input_file": write_csv(make_instructions(30), tmp_path)})
    context.make("000000", "propagated_s2_photons")
    assert context.is_stored("000000", "s2_photons_sum")
    assert context.is_stored("000000", "propagated_s2_photons")

    photons = context.get_array("000000", "propagated_s2_photons")
    photons_per_interaction = context.get_array("000000", "s2_photons_sum")
    assert len(photons) == photons_per_interaction["sum_s2_photons"].sum() > 0
    assert np.all(np.diff(photons["time"]) >= 0)
    assert np.all(photons["photon_type"] == 2)

    reloaded = ember.ember_context(output_folder=tmp_path / "data").get_array(
        "000000", "propagated_s2_photons", config=dict(input_file="not_needed_when_stored.csv")
    )
    np.testing.assert_array_equal(photons, reloaded)


def test_raw_records_end_to_end_and_reload(tmp_path):
    context = ember.ember_context(output_folder=tmp_path / "data")
    context.set_config({"input_file": write_csv(make_instructions(30), tmp_path)})
    context.make("000000", "raw_records")

    photons = context.get_array("000000", "photon_summary")
    n_s1_s2_afterpulse_photons = sum(
        len(context.get_array("000000", data_type))
        for data_type in ["propagated_s1_photons", "propagated_s2_photons", "pmt_afterpulses"]
    )
    assert len(photons) == n_s1_s2_afterpulse_photons
    assert set(np.unique(photons["photon_type"])) == {0, 1, 2}

    windows = context.get_array("000000", "pulse_windows")
    assert len(np.unique(windows["pulse_id"])) == len(windows) > 10_000

    raw_records = context.get_array("000000", "raw_records")
    assert np.all(np.diff(raw_records["time"]) >= 0)
    first_records = raw_records[raw_records["record_i"] == 0]
    for channel in np.unique(first_records["channel"]):
        pulses = first_records[first_records["channel"] == channel]
        pulse_ends = pulses["time"] + pulses["pulse_length"] * pulses["dt"]
        assert np.all(pulses["time"][1:] >= pulse_ends[:-1])

    baseline = strax.URLConfig.evaluate_dry("detector://digitizer_reference_baseline")
    pmt_to_pe = strax.URLConfig.evaluate_dry("detector://pmt_to_pe")
    spe_gain = context.get_single_plugin("000000", "pmt_afterpulses").gains[0]
    in_record = np.arange(raw_records["data"].shape[1])[None, :] < raw_records["length"][:, None]
    raw_records_pe = (
        np.sum((baseline - raw_records["data"].astype(np.int64)) * in_record) * pmt_to_pe
    )
    photon_pe = photons["photon_gain"].sum() / spe_gain
    assert raw_records_pe == pytest.approx(photon_pe, rel=0.005)

    reloaded = ember.ember_context(output_folder=tmp_path / "data").get_array(
        "000000", "raw_records", config=dict(input_file="not_needed_when_stored.csv")
    )
    np.testing.assert_array_equal(raw_records, reloaded)
