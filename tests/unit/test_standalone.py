"""ember runs without abstrax, and detector:// reads ember's file whatever the import order."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import strax
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st_hyp
from optional_packages import requires_abstrax

from ember import url_protocols
from ember.url_protocols import DETECTOR_CONFIG_FILE, lookup_detector_value

BLOCK_ABSTRAX = "import sys; sys.modules['abstrax'] = None"
PRINT_N_TPC_PMTS = "import strax; print(strax.URLConfig.evaluate_dry('detector://n_tpc_pmts'))"


def run_python(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return the last line of its output.

    :param code: Python statements separated by newlines.
    :return: last line printed to stdout.
    """
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def ember_n_tpc_pmts() -> str:
    return str(lookup_detector_value(DETECTOR_CONFIG_FILE, "n_tpc_pmts"))


def test_ember_runs_without_abstrax():
    code = "\n".join(
        [
            BLOCK_ABSTRAX,
            "import ember",
            "ember.ember_context(output_folder=__import__('tempfile').mkdtemp())",
            "loaded = {name.split('.')[0] for name, module in sys.modules.items() if module}",
            "assert 'abstrax' not in loaded",
            PRINT_N_TPC_PMTS,
        ]
    )
    assert run_python(code) == ember_n_tpc_pmts()


def test_ember_registers_its_own_detector_protocol_without_abstrax():
    code = "\n".join(
        [
            BLOCK_ABSTRAX,
            "import ember, strax",
            "from ember.url_protocols import detector_protocol",
            "print(strax.URLConfig._LOOKUP['detector'] is detector_protocol)",
        ]
    )
    assert run_python(code) == "True"


@requires_abstrax
@pytest.mark.parametrize(
    "imports", ["import abstrax; import ember", "import ember; import abstrax"]
)
def test_detector_protocol_reads_the_ember_file_in_any_import_order(imports):
    assert run_python(f"{imports}\n{PRINT_N_TPC_PMTS}") == ember_n_tpc_pmts()


@requires_abstrax
def test_abstrax_imported_first_reads_the_ember_file():
    code = "\n".join(
        [
            "import abstrax.url_protocols as abstrax_protocols",
            "import ember",
            "print(abstrax_protocols.active_detector_config_file())",
        ]
    )
    assert run_python(code) == str(DETECTOR_CONFIG_FILE)


def test_activation_twice_keeps_the_ember_protocol():
    url_protocols.activate_detector_protocol()
    assert strax.URLConfig.evaluate_dry("detector://n_tpc_pmts") == int(ember_n_tpc_pmts())


def test_a_foreign_detector_protocol_raises(monkeypatch):
    def foreign_detector_protocol(key: str, **_):
        return key

    monkeypatch.setitem(strax.URLConfig._LOOKUP, "detector", foreign_detector_protocol)
    with pytest.raises(RuntimeError, match="already registered"):
        url_protocols.activate_detector_protocol()


def test_lookup_of_a_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        lookup_detector_value(tmp_path / "missing.yaml", "n_tpc_pmts")


def test_lookup_of_an_unknown_key_raises():
    with pytest.raises(KeyError, match="no entry"):
        lookup_detector_value(DETECTOR_CONFIG_FILE, "this_key_does_not_exist")


config_keys = st_hyp.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20)
config_values = st_hyp.one_of(
    st_hyp.integers(-(10**9), 10**9),
    st_hyp.floats(allow_nan=False, allow_infinity=False),
    st_hyp.booleans(),
    st_hyp.lists(st_hyp.integers(0, 2000), max_size=5),
)


@settings(max_examples=50, deadline=None)
@given(config=st_hyp.dictionaries(config_keys, config_values, min_size=1, max_size=10))
def test_lookup_returns_every_value_of_the_file(config):
    path = Path(tempfile.mkdtemp()) / "detector.yaml"
    path.write_text(yaml.safe_dump(config))
    for key, value in config.items():
        assert lookup_detector_value(path, key) == value
