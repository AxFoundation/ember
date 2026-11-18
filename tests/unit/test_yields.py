"""Tests for the NestYields plugin."""

from types import SimpleNamespace

import nestpy
import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given
from hypothesis import strategies as st
from instruction_helpers import COLUMNS, make_instructions, write_csv

from ember.plugins.micro_physics import (
    CsvInput,
    DefaultSimulation,
    ElectricField,
    NestYields,
    VolumeProperties,
    VolumeSelection,
)
from ember.plugins.micro_physics.yields import mean_excitons, scaled_excitons

PLUGINS = [
    CsvInput,
    VolumeProperties,
    VolumeSelection,
    DefaultSimulation,
    ElectricField,
    NestYields,
]
DRIFT_FIELD = strax.URLConfig.evaluate_dry("detector://drift_field")
DENSITY = strax.URLConfig.evaluate_dry("detector://xenon_density_liquid")


def quanta_for(tmp_path, data_frame: pd.DataFrame, run_id: str = "000000", **config):
    context = strax.Context(
        storage=[],
        register=PLUGINS,
        config={"input_file": write_csv(data_frame, tmp_path), **config},
    )
    result = context.get_array(run_id, ["interactions_in_roi", "quanta"])
    return result[np.argsort(result["eventid"])]


def rows(*changes: dict) -> pd.DataFrame:
    data_frame = make_instructions(len(changes))
    for index, change in enumerate(changes):
        for column, value in change.items():
            data_frame.loc[index, column] = value
    return data_frame[COLUMNS]


def nest_mean_yields(energy: float, nestid: int = 8):
    calculator = nestpy.NESTcalc(nestpy.detectors.VDetector())
    return calculator.GetYields(
        interaction=nestpy.interactions(nestid),
        energy=energy,
        density=DENSITY,
        drift_field=DRIFT_FIELD,
        A=0,
        Z=0,
    )


def test_mean_yields_match_nestpy(tmp_path):
    result = quanta_for(tmp_path, make_instructions(5), return_yields_only=True)
    expected = nest_mean_yields(40.0)
    np.testing.assert_array_equal(result["photons"], int(expected.PhotonYield))
    np.testing.assert_array_equal(result["electrons"], int(expected.ElectronYield))


def test_sampled_quanta_scatter_around_mean_yields(tmp_path):
    result = quanta_for(tmp_path, make_instructions(400))
    expected = nest_mean_yields(40.0)
    for field, mean in [("photons", expected.PhotonYield), ("electrons", expected.ElectronYield)]:
        uncertainty_of_mean = np.std(result[field]) / np.sqrt(len(result))
        assert abs(np.mean(result[field]) - mean) < 5 * uncertainty_of_mean
    assert np.mean(result["excitons"]) == pytest.approx(mean_excitons(expected), rel=0.05)


def test_quanta_are_reproducible_and_depend_on_run_id(tmp_path):
    data_frame = make_instructions(20)
    first = quanta_for(tmp_path, data_frame)
    second = quanta_for(tmp_path, data_frame)
    other_run = quanta_for(tmp_path, data_frame, run_id="000001")
    np.testing.assert_array_equal(first["photons"], second["photons"])
    assert not np.array_equal(first["photons"], other_run["photons"])


def test_rules_for_given_quanta(tmp_path):
    data_frame = rows(
        {"photons": -1, "electrons": -1},
        {"photons": 1000, "electrons": -1},
        {"photons": -1, "electrons": 300},
        {"photons": 1000, "electrons": 300},
        {"ed": -1.0, "photons": 1000, "electrons": -1},
        {"ed": -1.0, "photons": -1, "electrons": 300},
    )
    result = quanta_for(tmp_path, data_frame)
    np.testing.assert_array_equal(result["photons"][[1, 3, 4]], 1000)
    np.testing.assert_array_equal(result["electrons"][[2, 3, 5]], 300)
    assert result["electrons"][4] == 0
    assert result["photons"][5] == 0
    assert result["excitons"][5] == 0
    assert result["photons"][0] > 1000 and result["electrons"][0] > 0

    ratio = mean_excitons(nest_mean_yields(40.0)) / nest_mean_yields(40.0).PhotonYield
    assert result["excitons"][1] == round(ratio * 1000)
    estimated = nest_mean_yields(13.7e-3 * 1000)
    assert result["excitons"][4] == round(mean_excitons(estimated) / estimated.PhotonYield * 1000)
    assert result["ed"][4] == pytest.approx(13.7)
    assert result["ed_is_estimated"][4]


def test_excitons_never_exceed_photons(tmp_path):
    result = quanta_for(tmp_path, make_instructions(100))
    assert np.all(result["excitons"] <= result["photons"])


def test_zero_energy_gives_no_quanta(tmp_path):
    result = quanta_for(tmp_path, rows({"ed": 0.0}, {"ed": 0.0, "photons": 5}))
    np.testing.assert_array_equal(result["photons"], [0, 5])
    np.testing.assert_array_equal(result["electrons"], [0, 0])
    np.testing.assert_array_equal(result["excitons"], [0, 0])


@pytest.mark.parametrize(
    "change",
    [
        {"nestid": 7},
        {"nestid": 0, "ed": 10.0},
        {"nestid": 6, "A": 4, "Z": 2, "ed": 5000.0},
        {"nestid": 11, "ed": 32.1},
    ],
)
def test_other_interaction_types_produce_quanta(tmp_path, change):
    result = quanta_for(tmp_path, rows(change))
    assert result["photons"][0] > 0
    assert result["electrons"][0] > 0


def test_kr83m_9_kev_line_has_no_electrons_in_nest_at_low_field(tmp_path):
    result = quanta_for(tmp_path, rows({"nestid": 11, "ed": 9.4}))
    assert result["photons"][0] > 0
    assert result["electrons"][0] == 0


def test_unknown_width_parameter_raises(tmp_path):
    with pytest.raises(Exception, match="Unknown NEST width parameter"):
        quanta_for(tmp_path, make_instructions(2), nest_width_parameters={"fano": 1.0})


def test_wrong_number_of_yield_parameters_raises(tmp_path):
    with pytest.raises(Exception, match="Expected 10 yield parameters"):
        quanta_for(tmp_path, make_instructions(2), nest_er_yields_parameters=[-1.0, -1.0])


def test_width_parameters_change_the_spread(tmp_path):
    data_frame = make_instructions(200)
    default = quanta_for(tmp_path, data_frame)
    wide = quanta_for(tmp_path, data_frame, nest_width_parameters={"fano_ER": 20.0})
    assert np.std(wide["photons"] + wide["electrons"]) > np.std(
        default["photons"] + default["electrons"]
    )


@given(
    photon_yield=st.floats(0, 1e6),
    electron_yield=st.floats(0, 1e6),
    exciton_ratio=st.floats(0, 10),
    photons=st.integers(0, 10**6),
)
def test_scaled_excitons_properties(photon_yield, electron_yield, exciton_ratio, photons):
    yields_result = SimpleNamespace(
        PhotonYield=photon_yield, ElectronYield=electron_yield, ExcitonRatio=exciton_ratio
    )
    excitons = scaled_excitons(yields_result, photons)
    assert 0 <= excitons <= photons
    if photon_yield == 0:
        assert excitons == 0
