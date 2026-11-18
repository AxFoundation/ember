"""Compare ember's and fuse's S1 photon simulation for the same 40 keV betas.

fuse's S1PhotonHits and S1PhotonPropagation run on fuse's microphysics_summary with ember's dummy
maps, SPE table and gains. Random seeds and the microphysics differ between the two chains, so
all quantities are compared statistically.
"""

import numpy as np
import pytest
from scipy.stats import ks_2samp

MINIMAL_P_VALUE = 1e-3
MAXIMAL_PULL = 5
N_TOP_PMTS = 253


def pull(first: np.ndarray, second: np.ndarray) -> float:
    uncertainty = np.sqrt(np.var(first) / len(first) + np.var(second) / len(second))
    return abs(np.mean(first) - np.mean(second)) / uncertainty


def photon_delays(simulation: dict) -> np.ndarray:
    interactions, photons = simulation["interactions"], simulation["photons"]
    interaction_time = dict(zip(interactions["cluster_id"], interactions["time"]))
    return photons["time"] - np.array([interaction_time[c] for c in photons["cluster_id"]])


def test_hits_per_interaction_agree(s1_simulations):
    ember_hits = s1_simulations["ember"]["interactions"]["n_s1_photon_hits"].astype(float)
    fuse_hits = s1_simulations["fuse"]["interactions"]["n_s1_photon_hits"].astype(float)
    assert 0.1 < ember_hits.mean() / fuse_hits.mean() < 10
    assert pull(ember_hits, fuse_hits) < MAXIMAL_PULL
    assert ks_2samp(ember_hits, fuse_hits).pvalue > MINIMAL_P_VALUE


def test_detected_fraction_agrees(s1_simulations):
    fractions = {}
    for name, simulation in s1_simulations.items():
        interactions = simulation["interactions"]
        fractions[name] = interactions["n_s1_photon_hits"] / interactions["photons"]
    assert pull(fractions["ember"], fractions["fuse"]) < MAXIMAL_PULL
    assert ks_2samp(fractions["ember"], fractions["fuse"]).pvalue > MINIMAL_P_VALUE


def test_one_photon_per_hit_in_both(s1_simulations):
    for simulation in s1_simulations.values():
        assert len(simulation["photons"]) == simulation["interactions"]["n_s1_photon_hits"].sum()


def test_area_fraction_top_agrees(s1_simulations):
    is_top = {
        name: (simulation["photons"]["channel"] < N_TOP_PMTS).astype(float)
        for name, simulation in s1_simulations.items()
    }
    assert pull(is_top["ember"], is_top["fuse"]) < MAXIMAL_PULL


def test_photon_delays_agree(s1_simulations):
    ember_delays = photon_delays(s1_simulations["ember"])
    fuse_delays = photon_delays(s1_simulations["fuse"])
    assert pull(ember_delays.astype(float), fuse_delays.astype(float)) < MAXIMAL_PULL
    rng = np.random.default_rng(0)
    assert (
        ks_2samp(rng.choice(ember_delays, 20000), rng.choice(fuse_delays, 20000)).pvalue
        > MINIMAL_P_VALUE
    )


@pytest.mark.parametrize("field", ["photon_gain", "dpe"])
def test_gains_agree(s1_simulations, field):
    ember_values = s1_simulations["ember"]["photons"][field].astype(float)
    fuse_values = s1_simulations["fuse"]["photons"][field].astype(float)
    assert pull(ember_values, fuse_values) < MAXIMAL_PULL
    rng = np.random.default_rng(1)
    assert (
        ks_2samp(rng.choice(ember_values, 20000), rng.choice(fuse_values, 20000)).pvalue
        > MINIMAL_P_VALUE
    )


def test_photon_type_is_s1(s1_simulations):
    for simulation in s1_simulations.values():
        assert np.all(simulation["photons"]["photon_type"] == 1)
