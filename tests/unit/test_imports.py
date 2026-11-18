"""Import smoke tests."""

import importlib

import pytest

MODULES = [
    "ember",
    "ember.url_protocols",
    "ember.maps",
    "ember.contexts",
    "ember.plugins.micro_physics",
    "ember.plugins.detector_physics",
    "ember.plugins.detector_physics.electron_drift",
    "ember.plugins.detector_physics.electron_propagation",
    "ember.plugins.detector_physics.electron_extraction",
    "ember.plugins.detector_physics.secondary_scintillation",
    "ember.plugins.detector_physics.s2_photon_propagation",
    "ember.plugins.pmt_and_daq",
    "ember.plugins.pmt_and_daq.pmt_afterpulses",
    "ember.plugins.pmt_and_daq.photon_summary",
    "ember.plugins.pmt_and_daq.photon_pulses",
    "ember.plugins.pmt_and_daq.pmt_response_and_daq",
    "ember.pmt_pulse_model",
    "ember.electron_transport",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name):
    importlib.import_module(module_name)


def test_ember_does_not_import_fuse_or_straxen():
    import subprocess
    import sys

    code = "import sys, ember; print('fuse' in sys.modules, 'straxen' in sys.modules)"
    output = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.split()
    assert output[-2:] == ["False", "False"]
