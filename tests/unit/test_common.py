"""Tests for ember.common."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from ember.common import dynamic_chunking, stable_argsort, stable_sort


def test_dynamic_chunking_known_case():
    data = np.array([0, 1, 2, 100, 101, 300], dtype=np.int64)
    np.testing.assert_array_equal(dynamic_chunking(data, 10, 1), [0, 0, 0, 1, 1, 2])
    np.testing.assert_array_equal(dynamic_chunking(data, 10, 4), [0, 0, 0, 0, 0, 1])


def test_dynamic_chunking_keeps_input_order():
    data = np.array([300, 0, 100], dtype=np.int64)
    np.testing.assert_array_equal(dynamic_chunking(data, 10, 1), [2, 0, 1])


def test_stable_sort_keeps_order_of_equal_keys():
    records = np.array([(1, 0), (0, 1), (1, 2), (0, 3)], dtype=[("key", int), ("index", int)])
    result = stable_sort(records, order="key")
    np.testing.assert_array_equal(result["index"], [1, 3, 0, 2])
    np.testing.assert_array_equal(stable_argsort(records["key"]), [1, 3, 0, 2])


@settings(deadline=None, max_examples=200)
@given(
    data=arrays(np.int64, st.integers(1, 200), elements=st.integers(0, 10_000)),
    scale=st.integers(0, 500),
    n_min=st.integers(1, 50),
)
def test_dynamic_chunking_invariants(data, scale, n_min):
    chunk_index = dynamic_chunking(data, scale, n_min)
    order = np.argsort(data, kind="mergesort")
    sorted_data = data[order]
    sorted_index = chunk_index[order]

    assert sorted_index[0] == 0
    assert np.all(np.diff(sorted_index) >= 0)
    assert np.all(np.diff(sorted_index) <= 1)

    new_chunk_starts = np.flatnonzero(np.diff(sorted_index)) + 1
    gaps = sorted_data[new_chunk_starts] - sorted_data[new_chunk_starts - 1]
    assert np.all(gaps > scale)

    chunk_sizes = np.bincount(sorted_index)
    assert np.all(chunk_sizes[:-1] >= n_min)


@pytest.mark.parametrize("n_min", [1, 3])
def test_dynamic_chunking_single_value(n_min):
    np.testing.assert_array_equal(dynamic_chunking(np.array([5], dtype=np.int64), 1, n_min), [0])
