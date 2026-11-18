"""Custom URLConfig protocols for ember.

All protocols resolve to local resources only (no database, no network).

Detector values
---------------
``detector://key`` returns ``key`` from ``ember/config/detector.yaml``. ember does not need abstrax.
abstrax registers a protocol with the same name, and ember's file contains all keys of abstrax's
detector file with the same values (marked ``[shared]``) plus ember's simulation settings. The URL
strings, and therefore the lineages, are the same in both packages, and their plugins can be
registered in one context. Importing ember makes ``detector://`` read ember's file whatever the
import order (:func:`activate_detector_protocol`).

Registered protocols
--------------------
- ``detector://key``        Look up ``key`` in ``ember/config/detector.yaml``.
- ``ember_map://name``      Load an interpolating map from ``ember/dummy_maps/`` (or from any path
                            if ``name`` is absolute). See :mod:`ember.maps`.
- ``ember_pmt_to_pe://``    Per-channel ADC-to-PE conversion factors built from ``detector.yaml``
                            (``pmt_to_pe`` for every channel, 0 for ``disabled_pmt_channels``).
- ``ember_table://name``    Load a CSV table (e.g. SPE spectra) from ``ember/dummy_maps/`` (or
                            from any path if ``name`` is absolute) as a ``pandas.DataFrame``.
- ``ember_file://name``     Load a ``.json``, ``.json.gz`` or ``.npz`` file (e.g. the SPE pulse
                            shape, noise or afterpulse inputs) from ``ember/dummy_maps/`` (or from
                            any path if ``name`` is absolute).

All names except ``detector`` start with ``ember_`` so they can never collide with protocols
registered by abstrax (``pkg``, ``yaml``, ...) or straxen (``json``, ``resource``, ``take``, ...).
"""

from __future__ import annotations

import gzip
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml
from strax import URLConfig

PACKAGE_ROOT = Path(__file__).resolve().parent
DETECTOR_CONFIG_FILE = PACKAGE_ROOT / "config" / "detector.yaml"
DUMMY_MAP_DIRECTORY = PACKAGE_ROOT / "dummy_maps"
DETECTOR_PROTOCOL = "detector"
ABSTRAX_PROTOCOL_MODULE = "abstrax.url_protocols"


def _register(protocol: str) -> Callable:
    """Register a URLConfig protocol unless the name already exists.

    The guard keeps re-imports (e.g. notebook autoreload) from raising.
    """

    def decorator(function: Callable) -> Callable:
        if protocol not in URLConfig._LOOKUP:
            URLConfig.register(protocol)(function)
        return function

    return decorator


@lru_cache(maxsize=None)
def load_detector_config(path: Path) -> dict[str, Any]:
    """Read a detector yaml file. The result is cached per path.

    :param path: path of the detector yaml file.
    :return: all entries of the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detector config not found at {path}.")
    with path.open() as config_file:
        return yaml.safe_load(config_file)


def lookup_detector_value(path: Path, key: str) -> Any:
    """Return ``key`` from the detector yaml file at ``path``.

    :param path: path of the detector yaml file.
    :param key: name of the entry.
    :return: the configured value.
    """
    config = load_detector_config(Path(path))
    if key not in config:
        raise KeyError(
            f"Detector config {path} has no entry '{key}'. Available keys: {sorted(config)}"
        )
    return config[key]


def detector_protocol(key: str, **_) -> Any:
    """Resolve ``detector://key`` to the value of ``key`` in ``ember/config/detector.yaml``.

    :param key: name of the entry.
    :return: the configured value.
    """
    return lookup_detector_value(DETECTOR_CONFIG_FILE, key)


def activate_detector_protocol() -> None:
    """Make ``detector://`` read ember's detector file, whatever the import order.

    - Nothing registered ``detector`` yet: register :func:`detector_protocol`. abstrax, if imported
      later, keeps this registration (its registration skips existing names).
    - abstrax registered it before: point abstrax's protocol to ember's file.
    - Any other package registered it: raise, because the values could differ from ember's.

    Call it before any plugin is set up: strax may cache evaluated URLs.
    """
    registered_function = URLConfig._LOOKUP.get(DETECTOR_PROTOCOL)
    if registered_function is None:
        URLConfig.register(DETECTOR_PROTOCOL)(detector_protocol)
    elif registered_function is detector_protocol:
        return
    elif getattr(registered_function, "__module__", None) == ABSTRAX_PROTOCOL_MODULE:
        sys.modules[ABSTRAX_PROTOCOL_MODULE].use_detector_config_file(DETECTOR_CONFIG_FILE)
    else:
        raise RuntimeError(
            f"The URLConfig protocol '{DETECTOR_PROTOCOL}' is already registered by "
            f"{getattr(registered_function, '__module__', registered_function)}. ember can only "
            "share it with abstrax."
        )


activate_detector_protocol()


@_register("ember_map")
def ember_map_protocol(name: str, method: str = "linear", **_) -> Any:
    """Resolve ``ember_map://name?method=linear`` to a :class:`ember.maps.RegularGridMap`.

    :param name: map file (``.json``, ``.json.gz`` or fuse's ``.pkl``) in ``ember/dummy_maps`` or an
        absolute path.
    :param method: interpolation method, ``linear`` (default) or ``nearest``.
    :return: the map, a callable taking positions of shape ``(n, n_dimensions)``.
    """
    from ember.maps import load_map

    return load_map(resolve_map_path(name), method=method)


def resolve_map_path(name: str) -> Path:
    """Return the path of a map file, relative to ``ember/dummy_maps`` unless absolute.

    :param name: file name in ``ember/dummy_maps`` or an absolute path.
    :return: the path of the map file.
    """
    path = Path(name)
    if not path.is_absolute():
        path = DUMMY_MAP_DIRECTORY / path
    if not path.exists():
        raise FileNotFoundError(f"Map file {path} does not exist.")
    return path


@_register("ember_pmt_to_pe")
def ember_pmt_to_pe_protocol(_: str = "", **__) -> np.ndarray:
    """Resolve ``ember_pmt_to_pe://`` to the per-channel ADC-to-PE factors of ``detector.yaml``.

    :return: array of length ``n_tpc_pmts``, 0 for switched-off channels.
    """
    return pmt_to_pe_per_channel(
        n_tpc_pmts=lookup_detector_value(DETECTOR_CONFIG_FILE, "n_tpc_pmts"),
        pmt_to_pe=lookup_detector_value(DETECTOR_CONFIG_FILE, "pmt_to_pe"),
        disabled_channels=lookup_detector_value(DETECTOR_CONFIG_FILE, "disabled_pmt_channels"),
    )


def pmt_to_pe_per_channel(
    n_tpc_pmts: int, pmt_to_pe: float, disabled_channels: list[int]
) -> np.ndarray:
    """Build the per-channel ADC-to-PE factors.

    :param n_tpc_pmts: number of channels.
    :param pmt_to_pe: ADC-to-PE factor of every active channel.
    :param disabled_channels: channels that are switched off; they get the factor 0.
    :return: float64 array of length ``n_tpc_pmts``.
    """
    disabled_channels = np.asarray(disabled_channels, dtype=int)
    if np.any((disabled_channels < 0) | (disabled_channels >= n_tpc_pmts)):
        raise ValueError(
            f"Disabled channels must lie in [0, {n_tpc_pmts}), got {disabled_channels}."
        )
    to_pe = np.full(int(n_tpc_pmts), float(pmt_to_pe))
    to_pe[disabled_channels] = 0.0
    return to_pe


@_register("ember_table")
def ember_table_protocol(name: str, **_) -> pd.DataFrame:
    """Resolve ``ember_table://name`` to a ``pandas.DataFrame``.

    :param name: CSV file in ``ember/dummy_maps`` or an absolute path.
    :return: the table. No column is used as index.
    """
    return pd.read_csv(resolve_map_path(name))


@_register("ember_file")
def ember_file_protocol(name: str, **_) -> Any:
    """Resolve ``ember_file://name`` to the content of a ``.json``, ``.json.gz`` or ``.npz`` file.

    :param name: file in ``ember/dummy_maps`` or an absolute path.
    :return: the decoded JSON object, or for ``.npz`` a dict of arrays.
    """
    return load_file(resolve_map_path(name))


def load_file(path: Path) -> Any:
    """Load a ``.json``, ``.json.gz`` or ``.npz`` file.

    :param path: path of the file.
    :return: the decoded JSON object, or for ``.npz`` a dict of arrays (the file is closed).
    """
    name = path.name
    if name.endswith(".json.gz"):
        with gzip.open(path, "rt") as json_file:
            return json.load(json_file)
    if name.endswith(".json"):
        with open(path) as json_file:
            return json.load(json_file)
    if name.endswith(".npz"):
        with np.load(path) as arrays:
            return {key: arrays[key] for key in arrays.files}
    raise ValueError(f"Unsupported file type of {path}; use .json, .json.gz or .npz.")
