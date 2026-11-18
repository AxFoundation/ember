"""Skip markers for tests that need optional packages."""

import importlib.util

import pytest

abstrax_is_installed = importlib.util.find_spec("abstrax") is not None

requires_abstrax = pytest.mark.skipif(
    not abstrax_is_installed, reason="needs the optional package abstrax"
)
