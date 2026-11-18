"""Tests of the S1 photon simulation: S1PhotonHits and S1PhotonPropagation."""

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from instruction_helpers import COLUMNS, make_instructions, write_csv

import ember
from ember.maps import save_map_data

N_TPC_PMTS = strax.URLConfig.evaluate_dry("detector://n_tpc_pmts")
N_TOP_PMTS = strax.URLConfig.evaluate_dry("detector://n_top_pmts")
P_DOUBLE_PE = strax.URLConfig.evaluate_dry("detector://p_double_pe_emission")


def fixed_photon_instructions(
    number_of_events: int, photons: int, position: tuple, nestid: int = 8, energy: float = -1
) -> pd.DataFrame:
    """Events with a given number of photons at one position (no electrons)."""
    x_value, y_value, z_value = position
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": np.full(number_of_events, x_value),
            "y": np.full(number_of_events, y_value),
            "z": np.full(number_of_events, z_value),
            "ed": np.full(number_of_events, energy),
            "nestid": np.full(number_of_events, nestid),
            "A": np.zeros(number_of_events, dtype=int),
            "Z": np.zeros(number_of_events, dtype=int),
            "photons": np.full(number_of_events, photons),
            "electrons": np.zeros(number_of_events, dtype=int),
        }
    )[COLUMNS]


def make_context(folder, instructions: pd.DataFrame, **config) -> strax.Context:
    context = ember.ember_context(output_folder=folder / "data")
    context.set_config({"input_file": write_csv(instructions, folder), **config})
    return context


def total_lce(position: tuple) -> float:
    pattern_map = strax.URLConfig.evaluate_dry("ember_map://s1_pattern_map_dummy.pkl")
    return float(pattern_map.summed_over_channels()(np.array([position]))[0])


@pytest.fixture(scope="module")
def beta_photons(tmp_path_factory):
    """Interactions with hits and the propagated photons of 100 betas of 40 keV."""
    folder = tmp_path_factory.mktemp("beta")
    context = make_context(folder, make_instructions(100))
    interactions = context.get_array("0", ["microphysics_summary", "s1_photon_hits"])
    photons = context.get_array("0", "propagated_s1_photons")
    return interactions, photons


@pytest.fixture(scope="module")
def propagation_plugin(tmp_path_factory):
    """A set-up S1PhotonPropagation plugin."""
    folder = tmp_path_factory.mktemp("plugin")
    context = make_context(folder, make_instructions(5))
    return context.get_single_plugin("0", "propagated_s1_photons")


# S1PhotonHits


def test_zero_photons_give_zero_hits(tmp_path):
    instructions = fixed_photon_instructions(10, photons=0, position=(0.0, 0.0, -50.0))
    instructions["electrons"] = 100
    hits = make_context(tmp_path, instructions).get_array("0", "s1_photon_hits")
    assert len(hits) == 10
    assert np.all(hits["n_s1_photon_hits"] == 0)


def test_zero_detection_efficiency_gives_zero_hits(tmp_path):
    instructions = fixed_photon_instructions(10, photons=1000, position=(0.0, 0.0, -50.0))
    context = make_context(tmp_path, instructions, s1_detection_efficiency=0.0)
    assert np.all(context.get_array("0", "s1_photon_hits")["n_s1_photon_hits"] == 0)


@pytest.mark.parametrize("position", [(0.0, 0.0, -10.0), (30.0, -20.0, -75.0), (5.0, 50.0, -140.0)])
def test_mean_hits_follow_the_lce_map(tmp_path, position):
    number_of_events, photons = 200, 2000
    instructions = fixed_photon_instructions(number_of_events, photons, position)
    hits = make_context(tmp_path, instructions).get_array("0", "s1_photon_hits")
    probability = total_lce(position) / (1 + P_DOUBLE_PE)
    expected = photons * probability
    sigma_of_mean = np.sqrt(photons * probability * (1 - probability) / number_of_events)
    assert abs(hits["n_s1_photon_hits"].mean() - expected) < 5 * sigma_of_mean


def test_disabled_channels_lower_the_light_collection(tmp_path):
    position = (0.0, 0.0, -50.0)
    disabled_channels = np.arange(0, N_TPC_PMTS, 2)
    to_pe = np.full(N_TPC_PMTS, 0.005)
    to_pe[disabled_channels] = 0.0
    instructions = fixed_photon_instructions(200, 2000, position)
    context = make_context(tmp_path, instructions, gain_model_mc=to_pe.tolist())
    hits = context.get_array("0", "s1_photon_hits")

    pattern_map = strax.URLConfig.evaluate_dry("ember_map://s1_pattern_map_dummy.pkl")
    per_channel = pattern_map(np.array([position]))[0]
    active_fraction = 1 - per_channel[disabled_channels].sum() / per_channel.sum()
    expected = 2000 * total_lce(position) * active_fraction / (1 + P_DOUBLE_PE)
    assert hits["n_s1_photon_hits"].mean() == pytest.approx(expected, rel=0.03)

    photons = context.get_array("0", "propagated_s1_photons")
    assert not np.any(np.isin(photons["channel"], disabled_channels))


def test_pmt_mask_follows_the_gain_model(tmp_path):
    to_pe = np.full(N_TPC_PMTS, 0.005)
    to_pe[[0, 7, N_TPC_PMTS - 1]] = 0.0
    context = make_context(tmp_path, make_instructions(5), gain_model_mc=to_pe.tolist())
    plugin = context.get_single_plugin("0", "s1_photon_hits")
    np.testing.assert_array_equal(plugin.pmt_mask, to_pe > 0)


def test_digitizer_constants_do_not_change_the_hits_lineage(tmp_path):
    context = make_context(tmp_path, make_instructions(5))
    default_hits_key = context.key_for("0", "s1_photon_hits").lineage_hash
    default_propagation_key = context.key_for("0", "propagated_s1_photons").lineage_hash
    context.set_config({"digitizer_bits": 12, "pmt_circuit_load_resistor": 1.0})
    assert context.key_for("0", "s1_photon_hits").lineage_hash == default_hits_key
    assert context.key_for("0", "propagated_s1_photons").lineage_hash != default_propagation_key


@settings(deadline=None, max_examples=30)
@given(
    photons=st.lists(st.integers(0, 10**5), min_size=1, max_size=20),
    data=st.data(),
)
def test_hits_never_exceed_photons(tmp_path_factory, photons, data):
    plugin = _hits_plugin(tmp_path_factory)
    positions = np.array(
        [
            [
                data.draw(st.floats(-80, 80)),
                data.draw(st.floats(-80, 80)),
                data.draw(st.floats(-160, 10)),
            ]
            for _ in photons
        ]
    )
    n_photons = np.array(photons, dtype=np.int64)
    hits = plugin.get_n_photons(n_photons, positions)
    assert np.all(hits >= 0)
    assert np.all(hits <= n_photons)


_HITS_PLUGIN_CACHE = {}


def _hits_plugin(tmp_path_factory):
    if "plugin" not in _HITS_PLUGIN_CACHE:
        folder = tmp_path_factory.mktemp("hits_plugin")
        context = make_context(folder, make_instructions(5))
        _HITS_PLUGIN_CACHE["plugin"] = context.get_single_plugin("0", "s1_photon_hits")
    return _HITS_PLUGIN_CACHE["plugin"]


# S1PhotonPropagation


def test_one_row_per_hit_and_photon_type(beta_photons):
    interactions, photons = beta_photons
    assert len(photons) == interactions["n_s1_photon_hits"].sum()
    assert np.all(photons["photon_type"] == 1)
    assert np.all(np.diff(photons["time"]) >= 0)
    np.testing.assert_array_equal(photons["time"], photons["endtime"])


def test_hits_per_cluster_match(beta_photons):
    interactions, photons = beta_photons
    cluster_ids, counts = np.unique(photons["cluster_id"], return_counts=True)
    expected = dict(zip(interactions["cluster_id"], interactions["n_s1_photon_hits"]))
    assert all(expected[cluster] == count for cluster, count in zip(cluster_ids, counts))


def test_photons_arrive_after_the_interaction(beta_photons):
    interactions, photons = beta_photons
    interaction_time = dict(zip(interactions["cluster_id"], interactions["time"]))
    delay = photons["time"] - np.array([interaction_time[c] for c in photons["cluster_id"]])
    assert np.all(delay > 0)
    assert np.all(delay < 1000 + 46 + 10 * 9.0)


def test_area_fraction_top_follows_the_map(tmp_path):
    position = (10.0, 10.0, -20.0)
    instructions = fixed_photon_instructions(50, 2000, position)
    photons = make_context(tmp_path, instructions).get_array("0", "propagated_s1_photons")
    pattern_map = strax.URLConfig.evaluate_dry("ember_map://s1_pattern_map_dummy.pkl")
    per_channel = pattern_map(np.array([position]))[0]
    expected_aft = per_channel[:N_TOP_PMTS].sum() / per_channel.sum()
    measured_aft = np.mean(photons["channel"] < N_TOP_PMTS)
    sigma = np.sqrt(expected_aft * (1 - expected_aft) / len(photons))
    assert abs(measured_aft - expected_aft) < 5 * sigma


def test_dpe_fraction_and_gain(beta_photons):
    _, photons = beta_photons
    assert photons["dpe"].mean() == pytest.approx(P_DOUBLE_PE, abs=0.01)
    single_pe_gain = 2.25 / 2**14 / 8.010882825e-09 / 0.005
    median_ratio = np.median(photons["photon_gain"][~photons["dpe"]]) / single_pe_gain
    assert median_ratio == pytest.approx(1.0, abs=0.02)


def test_output_is_reproducible(tmp_path):
    instructions = make_instructions(20)
    first = make_context(tmp_path / "a", instructions).get_array("0", "propagated_s1_photons")
    second = make_context(tmp_path / "b", instructions).get_array("0", "propagated_s1_photons")
    np.testing.assert_array_equal(first, second)


def test_scintillation_delays_stay_below_maximum(propagation_plugin):
    n_hits = np.array([500, 500])
    delays = propagation_plugin.nest_scintillation_timing(
        n_photon_hits=n_hits,
        n_photons_emitted=np.array([3000, 3000]),
        recoil_type=np.array([8, 0]),
        n_excitons=np.array([500, 2000]),
        local_field=np.array([23.0, 23.0]),
        e_dep=np.array([40.0, 20.0]),
    )
    assert len(delays) == n_hits.sum()
    assert np.all(delays >= 0)
    assert np.all(delays < propagation_plugin.maximum_recombination_time)


def test_nuclear_recoils_have_faster_scintillation(propagation_plugin):
    def median_delay(nestid, n_excitons, energy):
        return np.median(
            propagation_plugin.nest_scintillation_timing(
                n_photon_hits=np.array([20000]),
                n_photons_emitted=np.array([20000]),
                recoil_type=np.array([nestid]),
                n_excitons=np.array([n_excitons]),
                local_field=np.array([23.0]),
                e_dep=np.array([energy]),
            )
        )

    assert median_delay(0, 15000, 20.0) < 0.6 * median_delay(8, 3000, 40.0)


def test_more_hits_than_photons_raises(propagation_plugin):
    with pytest.raises(ValueError, match="less than or equal"):
        propagation_plugin.nest_scintillation_timing(
            n_photon_hits=np.array([10]),
            n_photons_emitted=np.array([5]),
            recoil_type=np.array([8]),
            n_excitons=np.array([1]),
            local_field=np.array([23.0]),
            e_dep=np.array([1.0]),
        )


def test_optical_timing_map_is_used_per_array(tmp_path):
    map_path = tmp_path / "timing.json"
    save_map_data(
        {
            "coordinate_system": [["z", [-160.0, 10.0, 2]], ["u", [0.0, 1.0, 2]]],
            "top": [[100.0, 100.0], [100.0, 100.0]],
            "bottom": [[0.0, 400.0], [0.0, 400.0]],
        },
        map_path,
    )
    folder = tmp_path / "run"
    folder.mkdir()
    context = make_context(
        folder,
        make_instructions(5),
        s1_optical_propagation_spline=f"ember_map://{map_path}?method=linear",
    )
    plugin = context.get_single_plugin("0", "propagated_s1_photons")
    channels = np.array([0, N_TOP_PMTS - 1] * 1000 + [N_TOP_PMTS, N_TPC_PMTS - 1] * 1000)
    delays = plugin.optical_propagation(channels, np.full(len(channels), -50.0))
    is_top = channels < N_TOP_PMTS
    np.testing.assert_allclose(delays[is_top], 100.0)
    assert delays[~is_top].mean() == pytest.approx(200.0, rel=0.05)
    assert delays[~is_top].min() >= 0 and delays[~is_top].max() <= 400


def test_default_optical_delay_is_at_least_the_direct_flight_time(propagation_plugin):
    channels = np.array([0, N_TOP_PMTS, N_TPC_PMTS - 1])
    refractive_index = strax.URLConfig.evaluate_dry("detector://lxe_refractive_index")
    light_speed_in_liquid_xenon = 29.98 / refractive_index
    delays = propagation_plugin.optical_propagation(channels, np.full(3, -70.0))
    assert np.all(delays >= 70.0 / light_speed_in_liquid_xenon * 0.99)


def test_photons_behind_the_chunk_end_raise(propagation_plugin, beta_photons):
    interactions, _ = beta_photons
    first_interaction = interactions[interactions["n_s1_photon_hits"] > 0][:1]
    with pytest.raises(ValueError, match="behind the chunk end"):
        propagation_plugin.compute(
            first_interaction,
            start=int(first_interaction["time"][0]) - 10,
            end=int(first_interaction["time"][0]) + 1,
        )


def test_gain_model_with_wrong_length_raises(tmp_path):
    context = make_context(tmp_path, make_instructions(5), gain_model_mc=[0.005] * 10)
    with pytest.raises(ValueError, match="channels"):
        context.get_array("0", "propagated_s1_photons")
