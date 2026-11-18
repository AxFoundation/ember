"""Shared fixtures for the ember unit tests."""

import pytest
from instruction_helpers import make_instructions, write_csv

import ember  # noqa: F401  (registers the protocols)


@pytest.fixture
def beta_csv(tmp_path) -> str:
    return write_csv(make_instructions(50), tmp_path)
