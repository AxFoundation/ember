"""Compare ember's and fuse's S2 simulation for the same 40 keV betas.

fuse's ElectronDrift, ElectronPropagation, ElectronExtraction, SecondaryScintillation and
S2PhotonPropagation (Garfield child, without delayed electrons) run on fuse's microphysics_summary
with ember's maps and ember's constant-field drift values. Random seeds and the microphysics
differ between the two chains, so all quantities are compared statistically. KS tests use all
values when they are independent; the photon delays are dominated by the drift time of their
electron, so they are subsampled and their pull uses the number of electrons.
"""

import numpy as np
import pytest
import strax
from scipy.stats import chi2, ks_2samp

MINIMAL_P_VALUE = 1e-3
MAXIMAL_PULL = 5
RUN_ID = "000000"
N_TOP_PMTS = strax.URLConfig.evaluate_dry("detector://n_top_pmts")


def pull(first: np.ndarray, second: np.ndarray) -> float:
    uncertainty = np.sqrt(np.var(first) / len(first) + np.var(second) / len(second))
    return abs(np.mean(first) - np.mean(second)) / uncertainty


def subsampled_ks_p_value(first: np.ndarray, second: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    first_subsample = rng.choice(first, 20000, replace=False)
    second_subsample = rng.choice(second, 20000, replace=False)
    return ks_2samp(first_subsample, second_subsample).pvalue


def interactions_with_electrons(simulation: dict) -> np.ndarray:
    interactions = simulation["interactions"]
    return interactions[interactions["electrons"] > 0]


def interaction_index(interactions: np.ndarray, cluster_ids: np.ndarray) -> np.ndarray:
    sort_index = np.argsort(interactions["cluster_id"])
    return sort_index[np.searchsorted(interactions["cluster_id"][sort_index], cluster_ids)]


def arrival_time_residuals(simulation: dict) -> np.ndarray:
    interactions, electrons = simulation["interactions"], simulation["electrons_at_interface"]
    index = interaction_index(interactions, electrons["cluster_id"])
    drift_times = electrons["time"] - interactions["time"][index]
    return (drift_times - interactions["drift_time_mean"][index]).astype(float)


def photon_delays_after_mean_drift(simulation: dict) -> np.ndarray:
    interactions, photons = simulation["interactions"], simulation["photons"]
    index = interaction_index(interactions, photons["cluster_id"])
    delays = photons["time"] - interactions["time"][index] - interactions["drift_time_mean"][index]
    return delays.astype(float)


def luminescence_mean() -> float:
    luminescence_map = strax.URLConfig.evaluate_dry("ember_map://s2_luminescence_timing_dummy.json")
    return float(np.mean(luminescence_map.values))


@pytest.fixture(scope="module")
def plugins(simulations) -> dict:
    return {
        name: simulations[f"{name}_context"].get_single_plugin(RUN_ID, "propagated_s2_photons")
        for name in ["ember", "fuse"]
    }


def test_drift_time_parameters_are_identical(s2_simulations):
    ember_interactions = interactions_with_electrons(s2_simulations["ember"])
    fuse_interactions = interactions_with_electrons(s2_simulations["fuse"])
    for field in ["drift_time_mean", "drift_time_spread"]:
        np.testing.assert_allclose(ember_interactions[field], fuse_interactions[field], atol=1)


def test_fraction_of_electrons_reaching_the_surface_agrees(s2_simulations):
    fractions = {}
    for name, simulation in s2_simulations.items():
        interactions = interactions_with_electrons(simulation)
        fractions[name] = interactions["n_electron_interface"] / interactions["electrons"]
    assert pull(fractions["ember"], fractions["fuse"]) < MAXIMAL_PULL
    assert ks_2samp(fractions["ember"], fractions["fuse"]).pvalue > MINIMAL_P_VALUE


def test_arrival_time_spread_agrees(s2_simulations):
    ember_residuals = arrival_time_residuals(s2_simulations["ember"])
    fuse_residuals = arrival_time_residuals(s2_simulations["fuse"])
    assert pull(ember_residuals, fuse_residuals) < MAXIMAL_PULL
    assert np.std(ember_residuals) / np.std(fuse_residuals) == pytest.approx(1, abs=0.01)
    assert ks_2samp(ember_residuals, fuse_residuals).pvalue > MINIMAL_P_VALUE


@pytest.mark.parametrize("axis", ["x", "y"])
def test_transverse_diffusion_agrees(s2_simulations, axis):
    shifts = {}
    for name, simulation in s2_simulations.items():
        interactions, electrons = simulation["interactions"], simulation["electrons_at_interface"]
        index = interaction_index(interactions, electrons["cluster_id"])
        shifts[name] = (electrons[f"{axis}_interface"] - interactions[f"{axis}_obs"][index]).astype(
            float
        )
    assert pull(shifts["ember"], shifts["fuse"]) < MAXIMAL_PULL
    assert np.std(shifts["ember"]) / np.std(shifts["fuse"]) == pytest.approx(1, abs=0.01)
    assert ks_2samp(shifts["ember"], shifts["fuse"]).pvalue > MINIMAL_P_VALUE


def test_extraction_fraction_agrees(s2_simulations):
    fractions, uncertainties = {}, {}
    for name, simulation in s2_simulations.items():
        n_at_surface = len(simulation["electrons_at_interface"])
        fractions[name] = len(simulation["extracted"]) / n_at_surface
        uncertainties[name] = np.sqrt(fractions[name] * (1 - fractions[name]) / n_at_surface)
    difference = abs(fractions["ember"] - fractions["fuse"])
    assert difference < MAXIMAL_PULL * np.hypot(uncertainties["ember"], uncertainties["fuse"])


def test_photons_per_electron_agree(s2_simulations):
    ember_photons = s2_simulations["ember"]["extracted"]["n_s2_photons"].astype(float)
    fuse_photons = s2_simulations["fuse"]["extracted"]["n_s2_photons"].astype(float)
    assert pull(ember_photons, fuse_photons) < MAXIMAL_PULL
    assert np.var(ember_photons) / np.var(fuse_photons) == pytest.approx(1, abs=0.01)


def test_photons_per_interaction_agree(s2_simulations):
    ember_sums = interactions_with_electrons(s2_simulations["ember"])["sum_s2_photons"]
    fuse_sums = interactions_with_electrons(s2_simulations["fuse"])["sum_s2_photons"]
    assert 0.1 < ember_sums.mean() / fuse_sums.mean() < 10
    assert pull(ember_sums.astype(float), fuse_sums.astype(float)) < MAXIMAL_PULL
    assert ks_2samp(ember_sums, fuse_sums).pvalue > MINIMAL_P_VALUE


def test_one_row_per_photon_in_both(s2_simulations):
    for simulation in s2_simulations.values():
        assert len(simulation["photons"]) == simulation["extracted"]["n_s2_photons"].sum()
        assert np.all(simulation["photons"]["photon_type"] == 2)


def test_area_fraction_top_agrees(s2_simulations):
    is_top = {
        name: (simulation["photons"]["channel"] < N_TOP_PMTS).astype(float)
        for name, simulation in s2_simulations.items()
    }
    assert pull(is_top["ember"], is_top["fuse"]) < MAXIMAL_PULL


@pytest.mark.parametrize("name", ["ember", "fuse"])
def test_channel_occupancy_follows_the_pattern_map(s2_simulations, plugins, name):
    """Observed photons per channel vs Σ_e n_e P(c | x_e), with each chain's own electrons."""
    electrons = s2_simulations[name]["extracted"]
    pattern_map = plugins["ember"].masked_s2_pattern_map
    positions = np.column_stack([electrons["x_interface"], electrons["y_interface"]])
    expected = np.zeros(pattern_map.values.shape[-1])
    for block_start in range(0, len(positions), 50_000):
        block = slice(block_start, block_start + 50_000)
        probabilities = pattern_map(positions[block])
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected += electrons["n_s2_photons"][block] @ probabilities
    observed = np.bincount(s2_simulations[name]["photons"]["channel"], minlength=len(expected))
    has_light = expected > 0
    assert np.all(observed[~has_light] == 0)
    chi_square = np.sum((observed - expected)[has_light] ** 2 / expected[has_light])
    assert chi2.sf(chi_square, has_light.sum() - 1) > MINIMAL_P_VALUE


def test_photon_timing_model_agrees(plugins):
    """Luminescence + excimer decay + optical delay with many photons per electron.

    fuse subtracts the sample mean of the luminescence times of every electron; with 10⁴ photons
    per electron this is the mean of the map, which is added back here.
    """
    n_photons = np.full(50, 10_000)
    positions = np.zeros((len(n_photons), 2))
    channels = np.tile(np.array([0, N_TOP_PMTS]), n_photons.sum() // 2).astype(np.int16)
    timings = {
        name: plugin.photon_timings(positions, n_photons, channels).astype(float)
        for name, plugin in plugins.items()
    }
    timings["fuse"] += luminescence_mean()
    for is_top in [True, False]:
        selection = (channels < N_TOP_PMTS) == is_top
        ember_times, fuse_times = timings["ember"][selection], timings["fuse"][selection]
        assert np.mean(ember_times) == pytest.approx(np.mean(fuse_times), abs=2)
        assert np.std(ember_times) / np.std(fuse_times) == pytest.approx(1, abs=0.01)
        assert ks_2samp(ember_times, fuse_times).pvalue > MINIMAL_P_VALUE


def test_photon_delays_after_the_drift_agree(s2_simulations):
    ember_delays = photon_delays_after_mean_drift(s2_simulations["ember"])
    fuse_delays = photon_delays_after_mean_drift(s2_simulations["fuse"]) + luminescence_mean()
    uncertainty = np.sqrt(
        np.var(ember_delays) / len(s2_simulations["ember"]["extracted"])
        + np.var(fuse_delays) / len(s2_simulations["fuse"]["extracted"])
    )
    assert abs(np.mean(ember_delays) - np.mean(fuse_delays)) < MAXIMAL_PULL * uncertainty
    assert subsampled_ks_p_value(ember_delays, fuse_delays, seed=3) > MINIMAL_P_VALUE


@pytest.mark.parametrize("field", ["photon_gain", "dpe"])
def test_gains_agree(s2_simulations, field):
    ember_values = s2_simulations["ember"]["photons"][field].astype(float)
    fuse_values = s2_simulations["fuse"]["photons"][field].astype(float)
    assert pull(ember_values, fuse_values) < MAXIMAL_PULL
    if field == "photon_gain":
        assert ks_2samp(ember_values, fuse_values).pvalue > MINIMAL_P_VALUE
