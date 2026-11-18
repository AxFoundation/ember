"""Compare ember and fuse microphysics_summary for the same 40 keV beta instructions.

Both chains use the same NEST implementation (ember's NestYields is registered in fuse), so
these tests check the input handling, chunking, volume handling, field and the wiring of the
NEST yields. Random seeds differ between the two chains, so quanta are compared statistically.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp

MINIMAL_P_VALUE = 1e-3
MAXIMAL_PULL = 5


def sorted_by_event(data: np.ndarray) -> np.ndarray:
    return data[np.argsort(data["eventid"], kind="mergesort")]


def test_same_number_of_interactions(simulations):
    assert len(simulations["ember"]) == len(simulations["instructions"])
    assert len(simulations["fuse"]) == len(simulations["instructions"])


def test_same_positions_and_energies(simulations):
    ember_data = sorted_by_event(simulations["ember"])
    fuse_data = sorted_by_event(simulations["fuse"])
    np.testing.assert_array_equal(ember_data["eventid"], fuse_data["eventid"])
    for field in ["x", "y", "z"]:
        np.testing.assert_allclose(ember_data[field], fuse_data[field], atol=1e-3)
    np.testing.assert_allclose(ember_data["ed"], fuse_data["ed"])


@pytest.mark.parametrize("field", ["nestid", "e_field", "xe_density", "vol_id"])
def test_same_interaction_properties(simulations, field):
    ember_values = np.unique(simulations["ember"][field])
    fuse_values = np.unique(simulations["fuse"][field])
    np.testing.assert_allclose(ember_values, fuse_values)
    assert len(ember_values) == 1


@pytest.mark.parametrize("field", ["photons", "electrons", "excitons"])
def test_quanta_agree_statistically(simulations, field):
    ember_values = simulations["ember"][field].astype(float)
    fuse_values = simulations["fuse"][field].astype(float)

    ratio_of_means = np.mean(ember_values) / np.mean(fuse_values)
    assert 0.1 < ratio_of_means < 10, "deviation of more than an order of magnitude"

    uncertainty = np.sqrt(
        np.var(ember_values) / len(ember_values) + np.var(fuse_values) / len(fuse_values)
    )
    pull = abs(np.mean(ember_values) - np.mean(fuse_values)) / uncertainty
    assert pull < MAXIMAL_PULL

    assert ks_2samp(ember_values, fuse_values).pvalue > MINIMAL_P_VALUE


def test_event_times_agree_statistically(simulations):
    maximal_time = len(simulations["instructions"]) * 1e9
    for name in ["ember", "fuse"]:
        times = simulations[name]["time"]
        assert np.all((times >= 0) & (times <= maximal_time))
    pvalue = ks_2samp(simulations["ember"]["time"], simulations["fuse"]["time"]).pvalue
    assert pvalue > MINIMAL_P_VALUE
