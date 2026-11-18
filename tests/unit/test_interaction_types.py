"""Tests for the nestid <-> name lookup."""

import nestpy
import pytest
from hypothesis import given
from hypothesis import strategies as st

from ember.interaction_types import (
    ION_NESTID,
    NEST_INTERACTION_NAMES,
    interaction_name_from_nestid,
    nestid_from_interaction_name,
)


def test_names_match_nestpy():
    nestpy_names = {int(value): name for name, value in nestpy.interactions.__members__.items()}
    assert nestpy_names == NEST_INTERACTION_NAMES


@pytest.mark.parametrize(
    "nestid, mass_number, charge_number, name",
    [
        (0, 0, 0, "NR"),
        (7, 0, 0, "gammaRay"),
        (8, 0, 0, "beta"),
        (11, 0, 0, "Kr83m"),
        (6, 4, 2, "ion(A=4,Z=2)"),
        (6, 206, 82, "ion(A=206,Z=82)"),
    ],
)
def test_known_names(nestid, mass_number, charge_number, name):
    assert interaction_name_from_nestid(nestid, mass_number, charge_number) == name
    assert nestid_from_interaction_name(name) == (nestid, mass_number, charge_number)


def test_mass_and_charge_are_ignored_for_non_ions():
    assert interaction_name_from_nestid(8, 4, 2) == "beta"


@pytest.mark.parametrize("nestid", [-1, 18, 100])
def test_unknown_nestid_raises(nestid):
    with pytest.raises(ValueError):
        interaction_name_from_nestid(nestid)


@pytest.mark.parametrize("mass_number, charge_number", [(0, 2), (4, 0), (-4, 2)])
def test_ion_without_mass_or_charge_raises(mass_number, charge_number):
    with pytest.raises(ValueError):
        interaction_name_from_nestid(ION_NESTID, mass_number, charge_number)


@pytest.mark.parametrize("name", ["electron", "ion", "ion(A=4)", "Beta", ""])
def test_unknown_name_raises(name):
    with pytest.raises(ValueError):
        nestid_from_interaction_name(name)


@given(nestid=st.sampled_from(sorted(set(NEST_INTERACTION_NAMES) - {ION_NESTID})))
def test_round_trip_non_ions(nestid):
    name = interaction_name_from_nestid(nestid)
    assert nestid_from_interaction_name(name) == (nestid, 0, 0)


@given(mass_number=st.integers(1, 300), charge_number=st.integers(1, 120))
def test_round_trip_ions(mass_number, charge_number):
    name = interaction_name_from_nestid(ION_NESTID, mass_number, charge_number)
    assert nestid_from_interaction_name(name) == (ION_NESTID, mass_number, charge_number)
