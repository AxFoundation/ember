"""Interpolating maps on a regular grid, without straxen.

A map is a dictionary in the format used by fuse and straxen for regular grids::

    {
        "coordinate_system": [["r", [0.0, 66.4, 2]], ["z", [-154.6555, 0.0, 2]]],
        "map": [[23.0, 23.0], [23.0, 23.0]],
        "name": "...",
        "description": "..."
    }

Each axis is ``[name, [min, max, n_points]]`` with evenly spaced grid points
``np.linspace(min, max, n_points)``. A map has the shape ``(n_1, ..., n_d)`` for a scalar map or
``(n_1, ..., n_d, *value_shape)`` for a vector map (e.g. one value per PMT channel).

A file can hold several maps on the same grid: every key that is not a metadata key is a map
(e.g. ``"top"`` and ``"bottom"``). Select one with ``my_map(positions, map_name="top")``. The
default name is ``"map"``.

Supported files:

- ``.json`` and ``.json.gz``: the dictionary above.
- ``.pkl``: fuse's pickled dictionary, where ``map`` may be compressed
  (``"compressed": (compressor, dtype, shape)``, using strax's compressors) and quantized
  (``"quantized": step``, the stored integers times ``step`` give the values). Pickle files can run
  code when loaded, so only load files from sources you trust.

The interpolation uses ``scipy.interpolate.RegularGridInterpolator``. It gives the same results
and speed as ``straxen.InterpolatingMap(method="RegularGridInterpolator")``. Positions outside
the grid are moved onto the grid boundary first, i.e. the map is continued with its boundary
values instead of being extrapolated.

For per-channel 2D maps (e.g. the S2 pattern map P(c | x, y)), ``RegularGridMap.draw_channels``
draws channels for many photons without building the interpolated ``(n, n_channels)`` array: a
numba kernel interpolates one row at a time, accumulates its cumulative distribution and draws the
photons of that row from it. It uses the same interpolation as ``__call__``.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numba
import numpy as np
import strax
from scipy.interpolate import RegularGridInterpolator

METADATA_KEYS = (
    "coordinate_system",
    "name",
    "description",
    "timestamp",
    "compressed",
    "quantized",
)

MAX_INT16_CHANNELS = int(np.iinfo(np.int16).max) + 1
NO_INVALID_ROW = -1


def decode_map_values(map_data: dict) -> dict:
    """Return a copy of the map dictionary with ``map`` decompressed and dequantized.

    :param map_data: map dictionary, possibly with ``compressed`` and ``quantized`` keys.
    :return: new dictionary without ``compressed`` and ``quantized``.
    """
    decoded = dict(map_data)
    if "compressed" in decoded:
        compressor, dtype, shape = decoded.pop("compressed")
        raw_bytes = strax.io.COMPRESSORS[compressor]["decompress"](decoded["map"])
        decoded["map"] = np.frombuffer(raw_bytes, dtype=dtype).reshape(*shape)
    if "quantized" in decoded:
        step = decoded.pop("quantized")
        decoded["map"] = step * np.asarray(decoded["map"], dtype=np.float64)
    return decoded


def encode_map_values(
    map_data: dict, quantized: float = 1e-6, integer_dtype: type = np.uint32
) -> dict:
    """Return a copy of the map dictionary with ``map`` quantized and zstd-compressed (fuse format).

    :param map_data: map dictionary with non-negative values in ``map``.
    :param quantized: quantization step.
    :param integer_dtype: unsigned integer type of the stored values.
    :return: new dictionary with ``map`` as bytes and the keys ``compressed`` and ``quantized``.
    """
    values = np.asarray(map_data["map"], dtype=np.float64)
    if np.any(values < 0):
        raise ValueError("Only non-negative maps can be quantized.")
    integers = np.round(values / quantized)
    if integers.max(initial=0) > np.iinfo(integer_dtype).max:
        raise ValueError(f"Values too large for {integer_dtype.__name__} with step {quantized}.")
    integers = integers.astype(integer_dtype)
    encoded = dict(map_data)
    encoded["map"] = strax.io.COMPRESSORS["zstd"]["compress"](integers.tobytes())
    encoded["compressed"] = ("zstd", integer_dtype, integers.shape)
    encoded["quantized"] = quantized
    return encoded


class RegularGridMap:
    """Callable map interpolating values on a regular grid.

    :param map_data: dictionary with ``coordinate_system`` and one or more maps (see module
        docstring). Compressed and quantized maps are decoded.
    :param method: ``"linear"`` or ``"nearest"``.
    """

    def __init__(self, map_data: dict, method: str = "linear"):
        map_data = decode_map_values(map_data)
        self.method = method
        self.name = map_data.get("name", "")
        self.description = map_data.get("description", "")
        self.coordinate_system = [
            [axis_name, [float(lower), float(upper), int(n_points)]]
            for axis_name, (lower, upper, n_points) in map_data["coordinate_system"]
        ]
        self.axis_names = [axis_name for axis_name, _ in self.coordinate_system]
        self.grid = [
            np.linspace(lower, upper, n_points)
            for _, (lower, upper, n_points) in self.coordinate_system
        ]
        self.lower_bounds = np.array([axis[0] for axis in self.grid])
        self.upper_bounds = np.array([axis[-1] for axis in self.grid])

        self.map_names = sorted(key for key in map_data if key not in METADATA_KEYS)
        if not self.map_names:
            raise ValueError(f"Map '{self.name}' contains no map values.")
        self.maps = {}
        self.interpolators = {}
        grid_shape = tuple(len(axis) for axis in self.grid)
        for map_name in self.map_names:
            values = np.asarray(map_data[map_name], dtype=np.float64)
            if values.shape[: len(grid_shape)] != grid_shape:
                raise ValueError(
                    f"Map '{self.name}/{map_name}' has shape {values.shape}, which does not match "
                    f"the grid shape {grid_shape}."
                )
            self.maps[map_name] = values
            self.interpolators[map_name] = RegularGridInterpolator(self.grid, values, method=method)

    @property
    def n_dimensions(self) -> int:
        """Number of coordinates of a position."""
        return len(self.grid)

    @property
    def values(self) -> np.ndarray:
        """Values of the default map ``"map"``."""
        return self.maps["map"]

    def __call__(self, positions: np.ndarray, map_name: str = "map") -> np.ndarray:
        """Evaluate the map.

        :param positions: array of shape ``(n, n_dimensions)``.
        :param map_name: name of the map in the file, ``"map"`` by default.
        :return: array of shape ``(n,)`` for scalar maps or ``(n, *value_shape)``.
        """
        self._check_map_name(map_name)
        return self.interpolators[map_name](self.clipped_positions(positions))

    def _check_map_name(self, map_name: str) -> None:
        if map_name not in self.interpolators:
            raise KeyError(
                f"Map '{self.name}' has no map '{map_name}'. Available: {self.map_names}"
            )

    def clipped_positions(self, positions: np.ndarray) -> np.ndarray:
        """Return the positions as float64, moved onto the grid boundary where outside.

        :param positions: array of shape ``(n, n_dimensions)``.
        :return: clipped positions, shape ``(n, n_dimensions)``.
        """
        positions = np.atleast_2d(np.asarray(positions, dtype=np.float64))
        if positions.ndim != 2 or positions.shape[1] != self.n_dimensions:
            raise ValueError(
                f"Map '{self.name}' expects positions with {self.n_dimensions} coordinates "
                f"{self.axis_names}, got shape {positions.shape}."
            )
        return np.clip(positions, self.lower_bounds, self.upper_bounds)

    def grid_cells(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the grid cell of every position and the position inside the cell.

        Uses the same convention as scipy's ``RegularGridInterpolator``: the lower index is in
        ``[0, n_points - 2]`` and the fraction ``(x - grid[i]) / (grid[i + 1] - grid[i])`` is in
        ``[0, 1]``. For ``method="nearest"`` the fraction is rounded to 0 or 1 (0.5 rounds to 0, as
        in scipy). Axes with a single grid point give index 0 and fraction 0. Positions are
        clipped to the grid first.

        :param positions: array of shape ``(n, n_dimensions)``.
        :return: lower indices (int64) and fractions (float64), both of shape
            ``(n, n_dimensions)``.
        """
        clipped_positions = self.clipped_positions(positions)
        lower_indices = np.zeros(clipped_positions.shape, dtype=np.int64)
        fractions = np.zeros(clipped_positions.shape, dtype=np.float64)
        for axis_index, axis in enumerate(self.grid):
            if len(axis) == 1:
                continue
            coordinates = clipped_positions[:, axis_index]
            indices = np.searchsorted(axis, coordinates, side="right") - 1
            indices = np.clip(indices, 0, len(axis) - 2)
            axis_fractions = (coordinates - axis[indices]) / (axis[indices + 1] - axis[indices])
            lower_indices[:, axis_index] = indices
            fractions[:, axis_index] = np.clip(axis_fractions, 0.0, 1.0)
        if self.method == "nearest":
            fractions = (fractions > 0.5).astype(np.float64)
        return lower_indices, fractions

    def draw_channels(
        self,
        positions: np.ndarray,
        n_photons: np.ndarray,
        rng: np.random.Generator,
        map_name: str = "map",
    ) -> np.ndarray:
        """Draw ``n_photons[i]`` channels from the per-channel map at ``positions[i]``.

        The map must be 2D with one value per channel, shape ``(n_x, n_y, n_channels)``, with
        non-negative values (they are normalized per position). The result is the same as
        interpolating ``self(positions, map_name)``, normalizing every row and drawing from it,
        but the interpolated rows are never stored: the work and memory per photon do not depend
        on the number of rows. A channel with value 0 is never drawn.

        The random numbers are drawn as ``rng.random(n_photons.sum())``, one per photon in the
        order of the rows.

        :param positions: array of shape ``(n, 2)``.
        :param n_photons: number of photons per position, shape ``(n,)``.
        :param rng: random number generator.
        :param map_name: name of the map in the file, ``"map"`` by default.
        :return: channel of each photon, row by row, as int16.
        """
        self._check_map_name(map_name)
        values = self.maps[map_name]
        if self.n_dimensions != 2 or values.ndim != 3:
            raise ValueError(
                f"draw_channels needs a 2D map with one value per channel, but map "
                f"'{self.name}/{map_name}' has {self.n_dimensions} coordinates and values of "
                f"shape {values.shape}."
            )
        n_channels = values.shape[-1]
        if n_channels > MAX_INT16_CHANNELS:
            raise ValueError(f"At most {MAX_INT16_CHANNELS} channels fit into int16.")
        n_photons = np.asarray(n_photons, dtype=np.int64)
        if n_photons.ndim != 1:
            raise ValueError(f"n_photons must be 1D, got shape {n_photons.shape}.")
        if np.any(n_photons < 0):
            raise ValueError("n_photons must not be negative.")
        lower_indices, fractions = self.grid_cells(positions)
        if len(lower_indices) != len(n_photons):
            raise ValueError(
                f"Got {len(lower_indices)} positions but {len(n_photons)} photon numbers."
            )

        uniforms = rng.random(int(n_photons.sum()))
        channels = np.zeros(len(uniforms), dtype=np.int16)
        invalid_row = draw_channels_bilinear(
            np.ascontiguousarray(values, dtype=np.float64),
            np.ascontiguousarray(lower_indices[:, 0]),
            np.ascontiguousarray(lower_indices[:, 1]),
            np.ascontiguousarray(fractions[:, 0]),
            np.ascontiguousarray(fractions[:, 1]),
            n_photons,
            uniforms,
            channels,
        )
        if invalid_row != NO_INVALID_ROW:
            raise ValueError(
                f"Map '{self.name}/{map_name}' has no light or negative values at position "
                f"{np.atleast_2d(positions)[invalid_row]}."
            )
        return channels

    def with_new_values(self, values: np.ndarray, map_name: str = "map") -> RegularGridMap:
        """Return a new map on the same grid with other values.

        :param values: values on the grid, shape ``(n_1, ..., n_d, *value_shape)``.
        :param map_name: name of the new map.
        :return: the new map, with the same interpolation method.
        """
        map_data = {
            "coordinate_system": self.coordinate_system,
            "name": self.name,
            "description": self.description,
            map_name: values,
        }
        return RegularGridMap(map_data, method=self.method)

    def with_masked_channels(self, channel_is_active: np.ndarray) -> RegularGridMap:
        """Return a copy of a per-channel map with the inactive channels set to 0.

        :param channel_is_active: boolean array with one entry per channel (last map axis).
        :return: the masked map.
        """
        channel_is_active = np.asarray(channel_is_active, dtype=bool)
        if self.values.shape[-1] != len(channel_is_active):
            raise ValueError(
                f"Map '{self.name}' has {self.values.shape[-1]} channels, but the channel mask has "
                f"{len(channel_is_active)} entries."
            )
        masked_values = self.values.copy()
        masked_values[..., ~channel_is_active] = 0.0
        return self.with_new_values(masked_values)

    def summed_over_channels(self) -> RegularGridMap:
        """Return the scalar map of the sum over the last axis (e.g. the total LCE).

        :return: the summed map.
        """
        return self.with_new_values(self.values.sum(axis=-1))


def load_map_data(path: str | Path) -> dict:
    """Read a map dictionary from a ``.json``, ``.json.gz`` or ``.pkl`` file.

    :param path: path of the map file.
    :return: the (possibly still compressed) map dictionary.
    """
    path = Path(path)
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt") as map_file:
            return json.load(map_file)
    if path.suffix == ".json":
        with path.open("r") as map_file:
            return json.load(map_file)
    if path.suffix == ".pkl":
        with path.open("rb") as map_file:
            return pickle.load(map_file)
    raise ValueError(f"Unsupported map file format: {path.name}. Use .json, .json.gz or .pkl.")


def save_map_data(map_data: dict, path: str | Path) -> None:
    """Write a map dictionary to a ``.json``, ``.json.gz`` or ``.pkl`` file.

    :param map_data: map dictionary; numpy arrays are converted to lists for JSON files.
    :param path: path of the map file.
    """
    path = Path(path)
    if path.suffix == ".pkl":
        with path.open("wb") as map_file:
            pickle.dump(map_data, map_file)
        return
    json_data = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in map_data.items()
    }
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "wt") as map_file:
            json.dump(json_data, map_file)
    elif path.suffix == ".json":
        with path.open("w") as map_file:
            json.dump(json_data, map_file)
    else:
        raise ValueError(f"Unsupported map file format: {path.name}. Use .json, .json.gz or .pkl.")


def load_map(path: str | Path, method: str = "linear") -> RegularGridMap:
    """Load a map from a ``.json``, ``.json.gz`` or ``.pkl`` file.

    :param path: path of the map file.
    :param method: interpolation method, ``"linear"`` or ``"nearest"``.
    :return: the map.
    """
    return RegularGridMap(load_map_data(path), method=method)


@numba.njit(cache=True)
def draw_channels_bilinear(
    values: np.ndarray,
    lower_x: np.ndarray,
    lower_y: np.ndarray,
    fraction_x: np.ndarray,
    fraction_y: np.ndarray,
    n_photons: np.ndarray,
    uniforms: np.ndarray,
    channels: np.ndarray,
) -> int:
    """Draw channels from a bilinearly interpolated per-channel 2D map, one row at a time.

    For every row: interpolate the channel values from the four corners of its grid cell,
    accumulate them into a normalized cumulative distribution (CDF) and give each of its photons
    the first channel whose CDF value is larger than the photon's uniform random number. Rows
    without photons are skipped. A channel with value 0 is never drawn.

    :param values: map values, shape ``(n_x, n_y, n_channels)``.
    :param lower_x: lower grid index per row along x (``RegularGridMap.grid_cells``).
    :param lower_y: lower grid index per row along y.
    :param fraction_x: position inside the cell along x, in ``[0, 1]``.
    :param fraction_y: position inside the cell along y, in ``[0, 1]``.
    :param n_photons: number of photons per row.
    :param uniforms: one random number in ``[0, 1)`` per photon, row by row.
    :param channels: output, one channel per photon (filled in place).
    :return: ``NO_INVALID_ROW`` (-1), or the first row whose values sum to 0 or are negative
        (its photons are not drawn).
    """
    n_x, n_y, n_channels = values.shape
    cdf = np.empty(n_channels, dtype=np.float64)
    photon_index = 0
    for row in range(len(n_photons)):
        if n_photons[row] == 0:
            continue
        x_low = lower_x[row]
        y_low = lower_y[row]
        x_high = min(x_low + 1, n_x - 1)
        y_high = min(y_low + 1, n_y - 1)
        weight_low_low = (1.0 - fraction_x[row]) * (1.0 - fraction_y[row])
        weight_high_low = fraction_x[row] * (1.0 - fraction_y[row])
        weight_low_high = (1.0 - fraction_x[row]) * fraction_y[row]
        weight_high_high = fraction_x[row] * fraction_y[row]

        total = 0.0
        last_active_channel = -1
        for channel in range(n_channels):
            value = (
                weight_low_low * values[x_low, y_low, channel]
                + weight_high_low * values[x_high, y_low, channel]
                + weight_low_high * values[x_low, y_high, channel]
                + weight_high_high * values[x_high, y_high, channel]
            )
            if value < 0:
                return row
            if value > 0:
                last_active_channel = channel
            total += value
            cdf[channel] = total
        if total <= 0:
            return row
        for channel in range(n_channels):
            cdf[channel] /= total

        for _ in range(n_photons[row]):
            drawn_channel = np.searchsorted(cdf, uniforms[photon_index], side="right")
            channels[photon_index] = min(drawn_channel, last_active_channel)
            photon_index += 1
    return NO_INVALID_ROW
