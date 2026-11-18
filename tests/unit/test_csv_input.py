"""Tests for the CsvInput plugin and its helper functions."""

import logging

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from instruction_helpers import COLUMNS, make_instructions, write_csv

from ember.common import dynamic_chunking
from ember.plugins.micro_physics.csv_input import (
    CsvInput,
    chunk_boundaries,
    event_start_times,
    load_instructions,
)

W_VALUE = 13.7


def single_row(**changes) -> pd.DataFrame:
    row = make_instructions(1).iloc[0].to_dict()
    row.update(changes)
    return pd.DataFrame([row], columns=COLUMNS)


def csv_input_context(input_file: str, **config) -> strax.Context:
    return strax.Context(
        storage=[], register=[CsvInput], config={"input_file": input_file, **config}
    )


# --- load_instructions -------------------------------------------------------------------


def test_load_instructions_copies_columns(tmp_path):
    data_frame = make_instructions(5)
    instructions = load_instructions(write_csv(data_frame, tmp_path), W_VALUE)
    np.testing.assert_allclose(instructions["x"], data_frame["x"], rtol=1e-6)
    np.testing.assert_array_equal(instructions["input_photons"], -1)
    assert not np.any(instructions["ed_is_estimated"])


def test_missing_column_raises(tmp_path):
    path = write_csv(make_instructions(3).drop(columns=["nestid"]), tmp_path)
    with pytest.raises(ValueError, match="nestid"):
        load_instructions(path, W_VALUE)


def test_empty_file_raises(tmp_path):
    path = write_csv(pd.DataFrame(columns=COLUMNS), tmp_path)
    with pytest.raises(ValueError, match="no interactions"):
        load_instructions(path, W_VALUE)


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"photons": 10.5}, "integers"),
        ({"nestid": 17}, "nestid"),
        ({"nestid": -1}, "nestid"),
        ({"nestid": 6, "A": 0, "Z": 2}, "Ions"),
        ({"photons": -2}, "photons"),
        ({"ed": -5.0}, "'ed'"),
        ({"ed": -1.0}, "need photons or electrons"),
        ({"ed": -1.0, "nestid": 0, "photons": 10}, "NR"),
        ({"ed": -1.0, "nestid": 6, "A": 4, "Z": 2, "photons": 10}, "ion"),
    ],
)
def test_invalid_rows_raise(tmp_path, changes, message):
    path = write_csv(single_row(**changes), tmp_path)
    with pytest.raises(ValueError, match=message):
        load_instructions(path, W_VALUE)


@pytest.mark.parametrize(
    "photons, electrons, expected_photons, expected_electrons",
    [(100, 50, 100, 50), (100, -1, 100, 0), (-1, 50, 0, 50)],
)
def test_energy_is_estimated_for_electronic_recoils(
    tmp_path, photons, electrons, expected_photons, expected_electrons
):
    path = write_csv(single_row(ed=-1.0, photons=photons, electrons=electrons), tmp_path)
    instructions = load_instructions(path, W_VALUE)
    assert instructions["input_photons"][0] == expected_photons
    assert instructions["input_electrons"][0] == expected_electrons
    expected_energy = W_VALUE * 1e-3 * (expected_photons + expected_electrons)
    assert instructions["ed"][0] == pytest.approx(expected_energy, rel=1e-6)
    assert instructions["ed_is_estimated"][0]


def test_given_quanta_with_given_energy_are_kept(tmp_path):
    instructions = load_instructions(write_csv(single_row(photons=7, electrons=-1), tmp_path), 13.7)
    assert instructions["ed"][0] == pytest.approx(40.0)
    assert instructions["input_photons"][0] == 7
    assert instructions["input_electrons"][0] == -1


def test_low_energy_warning(tmp_path, caplog):
    path = write_csv(single_row(ed=0.01), tmp_path)
    with caplog.at_level(logging.WARNING, logger="ember.CsvInput"):
        load_instructions(path, W_VALUE)
    assert "gas-phase" in caplog.text


# --- event_start_times --------------------------------------------------------------------


def test_source_rate_zero_gives_zero_offsets():
    offsets = event_start_times(np.array([3, 1, 3]), np.random.default_rng(0), 0, False)
    np.testing.assert_array_equal(offsets, [0, 0, 0])


def test_fixed_spacing_known_values():
    offsets = event_start_times(np.array([2, 0, 1, 0]), np.random.default_rng(0), 10, True)
    np.testing.assert_array_equal(offsets, [200_000_000, 0, 100_000_000, 0])


def test_negative_source_rate_raises():
    with pytest.raises(ValueError):
        event_start_times(np.array([0]), np.random.default_rng(0), -1, False)


@settings(deadline=None, max_examples=100)
@given(
    event_ids=arrays(np.int32, st.integers(1, 100), elements=st.integers(0, 30)),
    source_rate=st.floats(0.1, 1e4),
    fixed_event_spacing=st.booleans(),
    seed=st.integers(0, 2**32 - 1),
)
def test_event_start_times_properties(event_ids, source_rate, fixed_event_spacing, seed):
    offsets = event_start_times(
        event_ids, np.random.default_rng(seed), source_rate, fixed_event_spacing
    )
    unique_event_ids = np.unique(event_ids)
    start_per_event = np.array([offsets[event_ids == i][0] for i in unique_event_ids])
    for event_id, start in zip(unique_event_ids, start_per_event):
        assert np.all(offsets[event_ids == event_id] == start)
    assert np.all(np.diff(start_per_event) >= 0)
    assert np.all(start_per_event >= 0)
    assert np.all(start_per_event <= len(unique_event_ids) / source_rate * 1e9)
    if fixed_event_spacing:
        expected = (np.arange(len(unique_event_ids)) / (source_rate / 1e9)).astype(np.int64)
        np.testing.assert_array_equal(start_per_event, expected)


# --- chunk_boundaries ----------------------------------------------------------------------


def test_chunk_boundaries_known_case():
    times = np.array([10, 20, 1000, 1010], dtype=np.int64)
    chunk_index = np.array([0, 0, 1, 1])
    boundaries = chunk_boundaries(times, chunk_index, first_chunk_left=5, last_chunk_length=100)
    np.testing.assert_array_equal(boundaries, [5, 20 + int(0.75 * 980), 1010 + int(0.75 * 1080)])


def test_first_boundary_is_not_negative():
    times = np.array([3, 4], dtype=np.int64)
    boundaries = chunk_boundaries(
        times, np.array([0, 0]), first_chunk_left=1e6, last_chunk_length=1
    )
    assert boundaries[0] == 0


@settings(deadline=None, max_examples=200)
@given(
    times=arrays(np.int64, st.integers(1, 200), elements=st.integers(0, 10**9)),
    scale=st.integers(1, 10**8),
    n_min=st.integers(1, 20),
    first_chunk_left=st.integers(0, 10**7),
    last_chunk_length=st.integers(1, 10**8),
)
def test_every_interaction_lies_inside_its_chunk(
    times, scale, n_min, first_chunk_left, last_chunk_length
):
    times = np.sort(times)
    chunk_index = dynamic_chunking(times, scale, n_min)
    boundaries = chunk_boundaries(times, chunk_index, first_chunk_left, last_chunk_length)
    assert len(boundaries) == len(np.unique(chunk_index)) + 1
    assert boundaries[0] >= 0
    assert np.all(np.diff(boundaries) > 0)
    assert np.all(times >= boundaries[chunk_index])
    assert np.all(times < boundaries[chunk_index + 1])


# --- the plugin ----------------------------------------------------------------------------


def test_plugin_returns_all_interactions_sorted(beta_csv):
    result = csv_input_context(beta_csv).get_array("000000", "clustered_interactions")
    assert len(result) == 50
    assert np.all(np.diff(result["time"]) >= 0)
    np.testing.assert_array_equal(np.sort(result["cluster_id"]), np.arange(1, 51))
    np.testing.assert_array_equal(result["time"], result["endtime"])


def test_plugin_is_reproducible(beta_csv):
    first = csv_input_context(beta_csv).get_array("000000", "clustered_interactions")
    second = csv_input_context(beta_csv).get_array("000000", "clustered_interactions")
    np.testing.assert_array_equal(first, second)


def test_plugin_keeps_spacing_within_events(tmp_path):
    data_frame = make_instructions(20)
    data_frame["eventid"] = np.repeat(np.arange(10), 2)
    data_frame["t"] = np.tile([0, 154], 10)
    result = csv_input_context(write_csv(data_frame, tmp_path)).get_array(
        "000000", "clustered_interactions"
    )
    for event_id in range(10):
        event = result[result["eventid"] == event_id]
        assert np.diff(event["time"])[0] == 154


def test_plugin_creates_several_chunks(tmp_path):
    path = write_csv(make_instructions(40), tmp_path)
    context = csv_input_context(path, n_interactions_per_chunk=10, separation_scale=1e6)
    chunks = list(context.get_iter("000000", "clustered_interactions"))
    assert len(chunks) >= 2
    assert sum(len(chunk.data) for chunk in chunks) == 40
    for chunk in chunks:
        assert np.all(chunk.data["time"] >= chunk.start)
        assert np.all(chunk.data["endtime"] <= chunk.end)


def test_plugin_uses_absolute_times_with_source_rate_zero(tmp_path):
    data_frame = make_instructions(5)
    data_frame["t"] = [500, 100, 300, 200, 400]
    path = write_csv(data_frame, tmp_path)
    result = csv_input_context(path, source_rate=0).get_array("000000", "clustered_interactions")
    np.testing.assert_array_equal(result["time"], [100, 200, 300, 400, 500])


def test_plugin_rejects_negative_times(tmp_path):
    data_frame = make_instructions(2)
    data_frame["t"] = [-10, 5]
    path = write_csv(data_frame, tmp_path)
    with pytest.raises(Exception, match="must not be negative"):
        csv_input_context(path, source_rate=0).get_array("000000", "clustered_interactions")
