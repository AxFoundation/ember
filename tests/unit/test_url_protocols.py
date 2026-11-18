"""Tests for ember's URLConfig protocols."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from strax import URLConfig

import ember  # noqa: F401  (registers the protocols)
from ember.url_protocols import DETECTOR_CONFIG_FILE, pmt_to_pe_per_channel, resolve_map_path


def test_detector_protocol_known_key():
    assert URLConfig.evaluate_dry("detector://n_tpc_pmts") == 494


def test_detector_protocol_unknown_key_raises():
    with pytest.raises(KeyError):
        URLConfig.evaluate_dry("detector://this_key_does_not_exist")


def test_detector_protocol_reads_the_ember_file():
    assert DETECTOR_CONFIG_FILE.parts[-3:] == ("ember", "config", "detector.yaml")
    assert URLConfig.evaluate_dry("detector://z_cathode") == pytest.approx(-148.6515)


def test_channel_split_is_consistent():
    n_tpc_pmts = URLConfig.evaluate_dry("detector://n_tpc_pmts")
    n_top_pmts = URLConfig.evaluate_dry("detector://n_top_pmts")
    n_bottom_pmts = URLConfig.evaluate_dry("detector://n_bottom_pmts")
    assert n_top_pmts + n_bottom_pmts == n_tpc_pmts
    assert URLConfig.evaluate_dry("detector://top_pmt_indices") == [0, n_top_pmts]
    assert URLConfig.evaluate_dry("detector://bottom_pmt_indices") == [
        n_top_pmts,
        n_tpc_pmts,
    ]


def test_geometry_is_ordered_from_bottom_to_top():
    names = ["z_bottom_pmts", "z_cathode", "z_gate", "z_liquid_level", "z_anode", "z_top_pmts"]
    z_values = [URLConfig.evaluate_dry(f"detector://{name}") for name in names]
    assert z_values == sorted(z_values)
    assert len(set(z_values)) == len(z_values)


@pytest.mark.parametrize(
    "name",
    [
        "extraction_field_liquid",
        "drift_time_gate",
        "diffusion_constant_longitudinal_gate",
        "electron_trapping_time",
        "lxe_dielectric_constant",
        "singlet_lifetime_gas",
        "triplet_lifetime_gas",
        "pmt_diameter",
    ],
)
def test_s2_parameter_is_positive(name):
    assert URLConfig.evaluate_dry(f"detector://{name}") > 0


@pytest.mark.parametrize(
    "name", ["singlet_fraction_gas", "s2_mean_area_fraction_top", "electron_extraction_efficiency"]
)
def test_s2_fraction_is_a_probability(name):
    assert 0 < URLConfig.evaluate_dry(f"detector://{name}") <= 1


def test_pmt_diameter_is_larger_than_photocathode():
    pmt_diameter = URLConfig.evaluate_dry("detector://pmt_diameter")
    photocathode_diameter = URLConfig.evaluate_dry("detector://pmt_photocathode_diameter")
    assert pmt_diameter > photocathode_diameter


def test_resolve_map_path_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        resolve_map_path("this_map_does_not_exist.json")


def test_resolve_map_path_accepts_absolute_path(tmp_path):
    path = tmp_path / "map.json"
    path.write_text("{}")
    assert resolve_map_path(str(path)) == path


def test_pmt_to_pe_protocol_gives_one_value_per_channel():
    to_pe = URLConfig.evaluate_dry("ember_pmt_to_pe://")
    n_tpc_pmts = URLConfig.evaluate_dry("detector://n_tpc_pmts")
    assert to_pe.shape == (n_tpc_pmts,)
    np.testing.assert_allclose(to_pe, URLConfig.evaluate_dry("detector://pmt_to_pe"))


def test_disabled_channels_get_zero():
    to_pe = pmt_to_pe_per_channel(n_tpc_pmts=10, pmt_to_pe=0.005, disabled_channels=[2, 7])
    assert np.flatnonzero(to_pe == 0).tolist() == [2, 7]
    assert np.all(to_pe[[0, 1, 3, 4, 5, 6, 8, 9]] == 0.005)


def test_disabled_channel_out_of_range_raises():
    with pytest.raises(ValueError, match="Disabled channels"):
        pmt_to_pe_per_channel(n_tpc_pmts=10, pmt_to_pe=0.005, disabled_channels=[10])


@settings(deadline=None, max_examples=50)
@given(
    n_tpc_pmts=st.integers(1, 600),
    data=st.data(),
)
def test_pmt_to_pe_has_zeros_exactly_at_disabled_channels(n_tpc_pmts, data):
    disabled = data.draw(st.sets(st.integers(0, n_tpc_pmts - 1)))
    to_pe = pmt_to_pe_per_channel(n_tpc_pmts, 0.01, sorted(disabled))
    assert set(np.flatnonzero(to_pe == 0)) == disabled


def test_table_protocol_round_trip(tmp_path):
    table = pd.DataFrame({"charge": [0.0, 1.0], "0": [0.2, 0.8], "1": [0.5, 0.5]})
    path = tmp_path / "table.csv"
    table.to_csv(path, index=False)
    pd.testing.assert_frame_equal(URLConfig.evaluate_dry(f"ember_table://{path}"), table)


def test_numbers_in_the_detector_file_are_not_read_as_strings():
    """PyYAML reads e.g. 2.0e7 (exponent without sign) as a string; write 2.0e+7 or 20000000.0."""
    import yaml

    with open(DETECTOR_CONFIG_FILE) as config_file:
        config = yaml.safe_load(config_file)
    for key, value in config.items():
        if isinstance(value, str):
            with pytest.raises(ValueError):
                float(value)


def detector_value(name: str):
    """Return the value of ``name`` in ember's detector.yaml.

    :param name: key in detector.yaml.
    :return: the value.
    """
    return URLConfig.evaluate_dry(f"detector://{name}")


@pytest.mark.parametrize(
    "name",
    [
        "sample_duration",
        "samples_before_pulse_center",
        "samples_after_pulse_center",
        "samples_to_store_before",
        "samples_to_store_after",
        "external_amplification",
        "zle_threshold",
        "trigger_window",
        "noise_sigma",
        "pmt_pulse_smearing_sigma",
        "pmt_ap_modifier",
    ],
)
def test_daq_parameter_is_positive(name):
    assert detector_value(name) > 0


@pytest.mark.parametrize(
    "name",
    [
        "sample_duration",
        "samples_before_pulse_center",
        "samples_after_pulse_center",
        "samples_to_store_before",
        "samples_to_store_after",
        "digitizer_bits",
        "digitizer_reference_baseline",
        "trigger_window",
    ],
)
def test_daq_sample_counts_are_integers(name):
    assert isinstance(detector_value(name), int)


def test_pmt_pulse_time_rounding_is_one_ns():
    assert detector_value("pmt_pulse_time_rounding") == 1


def test_zle_threshold_lies_between_zero_and_baseline():
    baseline = detector_value("digitizer_reference_baseline")
    assert 0 < baseline - detector_value("zle_threshold") < baseline


def test_baseline_fits_into_the_digitizer_range():
    assert (
        0 < detector_value("digitizer_reference_baseline") < 2 ** detector_value("digitizer_bits")
    )


def test_special_thresholds_refer_to_existing_channels():
    n_tpc_pmts = detector_value("n_tpc_pmts")
    baseline = detector_value("digitizer_reference_baseline")
    for channel, threshold in detector_value("special_thresholds").items():
        assert 0 <= int(channel) < n_tpc_pmts
        assert 0 < threshold < baseline


@pytest.mark.parametrize("name", ["enable_noise", "enable_pmt_afterpulses"])
def test_daq_switches_are_booleans(name):
    assert isinstance(detector_value(name), bool)


def test_file_protocol_reads_json_and_json_gz(tmp_path):
    import gzip
    import json

    content = {"pe_pulse_ts": [0, 1], "nested": {"a": [1.5]}}
    json_path = tmp_path / "content.json"
    json_path.write_text(json.dumps(content))
    gz_path = tmp_path / "content.json.gz"
    with gzip.open(gz_path, "wt") as gz_file:
        json.dump(content, gz_file)
    assert URLConfig.evaluate_dry(f"ember_file://{json_path}") == content
    assert URLConfig.evaluate_dry(f"ember_file://{gz_path}") == content


def test_file_protocol_reads_npz_as_dict_of_arrays(tmp_path):
    path = tmp_path / "noise.npz"
    noise = np.arange(12.0).reshape(4, 3)
    np.savez(path, arr_0=noise)
    loaded = URLConfig.evaluate_dry(f"ember_file://{path}")
    assert list(loaded) == ["arr_0"]
    np.testing.assert_array_equal(loaded["arr_0"], noise)


def test_file_protocol_rejects_other_file_types(tmp_path):
    path = tmp_path / "table.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError):
        URLConfig.evaluate_dry(f"ember_file://{path}")
