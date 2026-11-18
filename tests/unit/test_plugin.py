"""Tests for the seeding modes of EmberBasePlugin."""

import numpy as np
import pytest
import strax

from ember.plugin import EmberBasePlugin


class RandomNumbers(EmberBasePlugin):
    """Minimal source plugin returning three random numbers in a single chunk."""

    __version__ = "0.0.1"
    depends_on = tuple()
    provides = "random_numbers"
    save_when = strax.SaveWhen.NEVER
    dtype = strax.time_fields + [(("Random number", "value"), np.float64)]

    def compute(self, chunk_i):
        data = np.zeros(3, dtype=self.dtype)
        data["time"] = np.arange(3)
        data["endtime"] = data["time"] + 1
        data["value"] = self.rng.random(3)
        return self.chunk(start=0, end=3, data=data)

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i == 0


def random_values(config: dict, run_id: str = "000000") -> np.ndarray:
    context = strax.Context(storage=[], register=[RandomNumbers], config=config)
    return context.get_array(run_id, "random_numbers")["value"]


def test_deterministic_seed_is_reproducible():
    np.testing.assert_array_equal(random_values({}), random_values({}))


def test_deterministic_seed_depends_on_run_id():
    assert not np.array_equal(random_values({}, "000000"), random_values({}, "000001"))


def test_user_defined_seed_is_reproducible():
    config = {"deterministic_seed": False, "user_defined_random_seed": 42}
    expected = np.random.default_rng(42).random(3)
    np.testing.assert_array_equal(random_values(config), expected)


def test_seed_from_os_differs():
    config = {"deterministic_seed": False}
    assert not np.array_equal(random_values(config), random_values(config))


@pytest.mark.parametrize("seed", [-1, 0, 1.5])
def test_invalid_user_defined_seed_raises(seed):
    config = {"deterministic_seed": False, "user_defined_random_seed": seed}
    with pytest.raises(Exception, match="positive integer"):
        random_values(config)
