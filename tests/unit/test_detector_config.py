"""Consistency of ember's detector file, also with abstrax's file (needs abstrax)."""

import numpy as np
import pytest
import strax
from optional_packages import abstrax_is_installed, requires_abstrax

import ember
from ember.url_protocols import DETECTOR_CONFIG_FILE as EMBER_DETECTOR_CONFIG_FILE
from ember.url_protocols import load_detector_config, lookup_detector_value

if abstrax_is_installed:
    from abstrax.url_protocols import DETECTOR_CONFIG_FILE as ABSTRAX_DETECTOR_CONFIG_FILE

    ABSTRAX_KEYS = sorted(load_detector_config(ABSTRAX_DETECTOR_CONFIG_FILE))
else:
    ABSTRAX_DETECTOR_CONFIG_FILE = None
    ABSTRAX_KEYS = []


def ember_value(key: str):
    return lookup_detector_value(EMBER_DETECTOR_CONFIG_FILE, key)


@requires_abstrax
@pytest.mark.parametrize("key", ["n_tpc_pmts", "tpc_length", "pmt_to_pe"])
def test_detector_protocol_reads_the_ember_file_with_abstrax_imported(key):
    import abstrax  # noqa: F401

    assert strax.URLConfig.evaluate_dry(f"detector://{key}") == ember_value(key)


@requires_abstrax
@pytest.mark.parametrize("key", ABSTRAX_KEYS)
def test_every_abstrax_key_has_the_same_value_in_ember(key):
    abstrax_value = lookup_detector_value(ABSTRAX_DETECTOR_CONFIG_FILE, key)
    if isinstance(abstrax_value, float):
        assert ember_value(key) == pytest.approx(abstrax_value, rel=1e-12)
    else:
        assert ember_value(key) == abstrax_value


@requires_abstrax
def test_shared_keys_are_marked_in_the_ember_file():
    ember_lines = EMBER_DETECTOR_CONFIG_FILE.read_text().splitlines()
    for key in ABSTRAX_KEYS:
        (line,) = [line for line in ember_lines if line.startswith(f"{key}:")]
        assert line.endswith("[shared]"), line


def test_tpc_length_is_the_gate_to_cathode_distance():
    assert ember_value("tpc_length") == pytest.approx(
        ember_value("z_gate") - ember_value("z_cathode"), rel=1e-12
    )


def test_drift_velocity_agrees_with_the_drift_map():
    drift_map = strax.URLConfig.evaluate_dry(
        "ember_map://field_dependencies_dummy.json?method=linear"
    )
    positions = np.array([[0.0, -10.0], [30.0, -70.0], [60.0, -140.0]])
    map_velocity = drift_map(positions, map_name="drift_speed_map")
    np.testing.assert_allclose(map_velocity, ember_value("drift_velocity_liquid"), rtol=2e-3)


@requires_abstrax
def test_abstrax_plugins_can_be_registered_in_the_ember_context(tmp_path):
    from abstrax.contexts import ABSTRAX_PLUGINS

    context = ember.ember_context(output_folder=tmp_path / "data")
    context.register(ABSTRAX_PLUGINS)
    assert "event_info" in context._plugin_class_registry
    assert context.get_single_plugin("0", "records").n_tpc_pmts == ember_value("n_tpc_pmts")
