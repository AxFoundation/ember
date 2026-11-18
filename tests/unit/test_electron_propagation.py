"""Tests of the ElectronPropagation plugin."""

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import COLUMNS, make_instructions, write_csv

import ember
from ember.electron_transport import transverse_diffusion_constant_liquid

DIFFUSION_TRANSVERSE = float(
    transverse_diffusion_constant_liquid(strax.URLConfig.evaluate_dry("detector://drift_field"))
)


def electron_instructions(rows: list) -> pd.DataFrame:
    """One row per interaction: (eventid, t, x, y, z, electrons), no photons."""
    eventid, t, x, y, z, electrons = (np.array(column) for column in zip(*rows))
    number = len(rows)
    return pd.DataFrame(
        {
            "eventid": eventid,
            "t": t,
            "x": x,
            "y": y,
            "z": z,
            "ed": np.full(number, -1.0),
            "nestid": np.full(number, 8),
            "A": np.zeros(number, dtype=int),
            "Z": np.zeros(number, dtype=int),
            "photons": np.zeros(number, dtype=int),
            "electrons": electrons,
        }
    )[COLUMNS]


def make_context(folder, instructions: pd.DataFrame, **config) -> strax.Context:
    context = ember.ember_context(output_folder=folder / "data")
    context.set_config(
        {"input_file": write_csv(instructions, folder), "electron_lifetime_liquid": 0, **config}
    )
    return context


@pytest.fixture(scope="module")
def beta_electrons(tmp_path_factory):
    """Clusters and electrons at the interface of 100 betas of 40 keV."""
    folder = tmp_path_factory.mktemp("beta")
    context = make_context(folder, make_instructions(100))
    clusters = context.get_array("0", ["microphysics_summary", "drifted_electrons"])
    electrons = context.get_array("0", "electrons_at_interface")
    return clusters, electrons


@pytest.fixture(scope="module")
def deep_cluster_electrons(tmp_path_factory):
    """Many electrons of one interaction deep in the TPC."""
    folder = tmp_path_factory.mktemp("deep")
    instructions = electron_instructions([(0, 0, 20.0, -10.0, -120.0, 20000)])
    context = make_context(folder, instructions)
    cluster = context.get_array("0", ["microphysics_summary", "drifted_electrons"])
    electrons = context.get_array("0", "electrons_at_interface")
    return cluster[0], electrons


@pytest.fixture(scope="module")
def propagation_plugin(tmp_path_factory):
    folder = tmp_path_factory.mktemp("plugin")
    return make_context(folder, make_instructions(5)).get_single_plugin(
        "0", "electrons_at_interface"
    )


def test_one_row_per_electron_at_the_interface(beta_electrons):
    clusters, electrons = beta_electrons
    assert len(electrons) == clusters["n_electron_interface"].sum()
    cluster_ids, counts = np.unique(electrons["cluster_id"], return_counts=True)
    expected = dict(zip(clusters["cluster_id"], clusters["n_electron_interface"]))
    assert all(expected[cluster_id] == count for cluster_id, count in zip(cluster_ids, counts))


def test_electrons_are_sorted_by_time(beta_electrons):
    _, electrons = beta_electrons
    assert np.all(np.diff(electrons["time"]) >= 0)
    np.testing.assert_array_equal(electrons["time"], electrons["endtime"])


def test_drift_time_distribution_follows_the_cluster(deep_cluster_electrons):
    cluster, electrons = deep_cluster_electrons
    drift_times = electrons["time"] - cluster["time"]
    number = len(drift_times)
    assert number == cluster["n_electron_interface"]
    assert abs(drift_times.mean() - cluster["drift_time_mean"]) < 5 * cluster[
        "drift_time_spread"
    ] / np.sqrt(number)
    assert drift_times.std() == pytest.approx(cluster["drift_time_spread"], rel=0.03)


def test_transverse_spread_follows_the_diffusion(deep_cluster_electrons):
    cluster, electrons = deep_cluster_electrons
    expected_spread = np.sqrt(2 * DIFFUSION_TRANSVERSE * cluster["drift_time_mean"])
    for coordinate, centre in [
        ("x_interface", cluster["x_obs"]),
        ("y_interface", cluster["y_obs"]),
    ]:
        assert electrons[coordinate].mean() == pytest.approx(centre, abs=5 * expected_spread / 100)
        assert electrons[coordinate].std() == pytest.approx(expected_spread, rel=0.03)


def test_shallow_electrons_come_first_although_their_interaction_was_later(tmp_path):
    deep = (0, 0, 0.0, 0.0, -145.0, 50)
    shallow = (0, 500_000, 0.0, 0.0, -1.0, 50)
    context = make_context(tmp_path, electron_instructions([deep, shallow]))
    clusters = context.get_array("0", "microphysics_summary")
    electrons = context.get_array("0", "electrons_at_interface")
    shallow_cluster_id = clusters["cluster_id"][np.argmax(clusters["z"])]
    assert np.all(electrons["cluster_id"][:50] == shallow_cluster_id)
    assert np.all(np.diff(electrons["time"]) >= 0)


def test_clusters_without_electrons_give_no_electrons(tmp_path):
    instructions = electron_instructions([(0, 0, 0.0, 0.0, -50.0, 0), (1, 0, 0.0, 0.0, -50.0, 0)])
    electrons = make_context(tmp_path, instructions).get_array("0", "electrons_at_interface")
    assert len(electrons) == 0


def test_results_are_reproducible(tmp_path):
    instructions = make_instructions(10)
    first = make_context(tmp_path / "a", instructions).get_array("0", "electrons_at_interface")
    second = make_context(tmp_path / "b", instructions).get_array("0", "electrons_at_interface")
    np.testing.assert_array_equal(first, second)


@settings(deadline=None, max_examples=50)
@given(
    clusters=st.lists(
        st.tuples(st.integers(0, 200), st.floats(0.0, 3e6), st.floats(0.0, 1e4)),
        min_size=1,
        max_size=20,
    )
)
def test_drift_times_are_non_negative_with_one_per_electron(propagation_plugin, clusters):
    n_electrons, mean, spread = (np.array(column) for column in zip(*clusters))
    drift_times = propagation_plugin.draw_drift_times(n_electrons, mean, spread)
    assert len(drift_times) == n_electrons.sum()
    assert drift_times.dtype == np.int64
    assert np.all(drift_times >= 0)


def test_electrons_behind_the_chunk_end_raise(propagation_plugin):
    input_dtype = strax.merged_dtype(
        [
            propagation_plugin.deps["drifted_electrons"].dtype_for("drifted_electrons"),
            propagation_plugin.deps["microphysics_summary"].dtype_for("microphysics_summary"),
        ]
    )
    data = np.zeros(1, dtype=input_dtype)
    data["n_electron_interface"] = 10
    data["drift_time_mean"] = 2_000_000
    data["time"] = 1000
    with pytest.raises(ValueError, match="behind the chunk end"):
        propagation_plugin.compute(data, start=0, end=1_000_000)
