"""Electron transport in xenon: drift velocities, diffusion constants and drift-path averages.

This module is used by the map generator notebooks (``map_generators/dummy_field_maps.ipynb`` and
``map_generators/dummy_s2_timing_maps.ipynb``) and by the tests. The plugins do not call it: they
read the maps that are built with it.

Drift velocities come from NEST (nestpy). The diffusion constants are not bound in nestpy 2.1.3, so
the default parametrizations of NEST (``NESTcalc::GetDiffTran_Liquid`` and
``NESTcalc::GetDiffLong_Liquid``, NEST master, ``highFieldModel = false``) are vendored here:

* transverse: ``D_T = 37.368 · E^0.093452 · exp(-8.1651e-5 · E)`` cm²/s, EXO-200, arXiv:1609.04467.
* longitudinal: ``D_L = 66.35 · E^-0.24855 + 36.85 · exp(-E / 35.661)`` cm²/s, NEST's default
  low-field model (LZ 2023).

``E`` is the drift field in V/cm. All functions return ember units: cm, ns.

Drift-path averages
-------------------
The electron drift plugins use fuse's equations, which assume one value per interaction position:

* drift time below the gate ``t = |z| / v̄``,
* time spread ``σ_t² = 2 D_L t / v̄²``,
* transverse spread ``σ_xy² = 2 D_T t``.

For a non-uniform field the correct values are integrals along the drift path from ``z`` up to the
gate (a straight vertical line, field distortion is not simulated):

* ``t = ∫ dz / v``,
* ``σ_t² = ∫ 2 D_L / v³ dz``,
* ``σ_xy² = ∫ 2 D_T / v dz``.

:func:`path_averaged_drift_properties` returns the effective values ``v̄``, ``D_L,eff`` and
``D_T,eff`` that turn fuse's equations into these integrals. For a constant field they are the local
values.
"""

from __future__ import annotations

import nestpy
import numpy as np
from scipy.integrate import cumulative_trapezoid

MM_PER_US_TO_CM_PER_NS = 1e-4
CM2_PER_S_TO_CM2_PER_NS = 1e-9
XENON_MOLAR_MASS = 131.293  # g/mol
AVOGADRO_CONSTANT = 6.02214076e23  # 1/mol
TOWNSEND = 1e-17  # V cm²

DRIFT_PROPERTY_MAP_NAMES = (
    "drift_speed_map",
    "diffusion_longitudinal_map",
    "diffusion_transverse_map",
)

_NEST_CALCULATOR = nestpy.NESTcalc(nestpy.detectors.VDetector())


def drift_velocity_liquid(
    field: np.ndarray, temperature: float, pressure: float, density: float
) -> np.ndarray:
    """Electron drift velocity in liquid xenon from NEST.

    :param field: drift field [V/cm].
    :param temperature: liquid temperature [K].
    :param pressure: gas pressure [bar].
    :param density: liquid density [g/cm³].
    :return: drift velocity [cm/ns], same shape as ``field``.
    """
    field = np.asarray(field, dtype=np.float64)
    velocity = _NEST_CALCULATOR.calculate_drift_velocity_liquid(
        temperature, pressure, density, field.ravel()
    )
    return np.asarray(velocity, dtype=np.float64).reshape(field.shape) * MM_PER_US_TO_CM_PER_NS


def drift_velocity_gas(
    field: np.ndarray, temperature: float, pressure: float, density: float
) -> np.ndarray:
    """Electron drift velocity in gaseous xenon from NEST.

    :param field: electric field in the gas [V/cm].
    :param temperature: gas temperature [K].
    :param pressure: gas pressure [bar].
    :param density: gas density [g/cm³].
    :return: drift velocity [cm/ns], same shape as ``field``.
    """
    field = np.asarray(field, dtype=np.float64)
    velocity = _NEST_CALCULATOR.calculate_drift_velocity_gas(
        temperature, pressure, density, field.ravel()
    )
    return np.asarray(velocity, dtype=np.float64).reshape(field.shape) * MM_PER_US_TO_CM_PER_NS


def transverse_diffusion_constant_liquid(field: np.ndarray) -> np.ndarray:
    """Transverse electron diffusion constant in liquid xenon (NEST default, EXO-200).

    :param field: drift field [V/cm].
    :return: diffusion constant [cm²/ns].
    """
    field = np.asarray(field, dtype=np.float64)
    diffusion = 37.368 * field**0.093452 * np.exp(-8.1651e-5 * field)
    return diffusion * CM2_PER_S_TO_CM2_PER_NS


def longitudinal_diffusion_constant_liquid(field: np.ndarray) -> np.ndarray:
    """Longitudinal electron diffusion constant in liquid xenon (NEST default low-field model).

    :param field: drift field [V/cm].
    :return: diffusion constant [cm²/ns].
    """
    field = np.asarray(field, dtype=np.float64)
    diffusion = 66.35 * field**-0.24855 + 36.85 * np.exp(-field / 35.661)
    return diffusion * CM2_PER_S_TO_CM2_PER_NS


def path_averaged_drift_properties(
    z_values: np.ndarray,
    field_values: np.ndarray,
    z_gate: float,
    temperature: float,
    pressure: float,
    density: float,
) -> dict:
    """Effective drift velocity and diffusion constants for drifting from ``z`` up to the gate.

    The field is interpolated linearly between the given points. Positions at or above the gate get
    the local values at the gate (the limit for a vanishing drift length).

    :param z_values: increasing z positions [cm] along one vertical line; the gate must lie inside.
    :param field_values: drift field [V/cm] at ``z_values``.
    :param z_gate: z position of the gate [cm].
    :param temperature: liquid temperature [K].
    :param pressure: gas pressure [bar].
    :param density: liquid density [g/cm³].
    :return: dictionary with the arrays ``drift_speed_map`` [cm/ns], ``diffusion_longitudinal_map``
        and ``diffusion_transverse_map`` [cm²/ns], one value per entry of ``z_values``.
    """
    z_values = np.asarray(z_values, dtype=np.float64)
    field_values = np.asarray(field_values, dtype=np.float64)
    if z_values.shape != field_values.shape or z_values.ndim != 1:
        raise ValueError("z_values and field_values must be 1D arrays of the same length.")
    if np.any(np.diff(z_values) <= 0):
        raise ValueError("z_values must be strictly increasing.")
    if not z_values[0] < z_gate <= z_values[-1]:
        raise ValueError(f"The gate (z = {z_gate} cm) must lie inside the z range of the values.")

    z_with_gate = np.union1d(z_values, [z_gate])
    field_with_gate = np.interp(z_with_gate, z_values, field_values)
    velocity = drift_velocity_liquid(field_with_gate, temperature, pressure, density)
    longitudinal = longitudinal_diffusion_constant_liquid(field_with_gate)
    transverse = transverse_diffusion_constant_liquid(field_with_gate)

    def integral_up_to_gate(integrand: np.ndarray) -> np.ndarray:
        cumulative = cumulative_trapezoid(integrand, z_with_gate, initial=0.0)
        gate_value = np.interp(z_gate, z_with_gate, cumulative)
        return gate_value - np.interp(z_values, z_with_gate, cumulative)

    drift_time = integral_up_to_gate(1.0 / velocity)
    time_spread_squared_half = integral_up_to_gate(longitudinal / velocity**3)
    transverse_spread_squared_half = integral_up_to_gate(transverse / velocity)

    gate_index = np.searchsorted(z_with_gate, z_gate)
    is_below_gate = z_values < z_gate
    drift_length = z_gate - z_values

    drift_speed = np.full(len(z_values), velocity[gate_index])
    diffusion_longitudinal = np.full(len(z_values), longitudinal[gate_index])
    diffusion_transverse = np.full(len(z_values), transverse[gate_index])

    time_below = drift_time[is_below_gate]
    drift_speed[is_below_gate] = drift_length[is_below_gate] / time_below
    diffusion_longitudinal[is_below_gate] = (
        drift_speed[is_below_gate] ** 2 * time_spread_squared_half[is_below_gate] / time_below
    )
    diffusion_transverse[is_below_gate] = transverse_spread_squared_half[is_below_gate] / time_below

    return {
        "drift_speed_map": drift_speed,
        "diffusion_longitudinal_map": diffusion_longitudinal,
        "diffusion_transverse_map": diffusion_transverse,
    }


def build_drift_property_map(
    field_map,
    r_values: np.ndarray,
    z_values: np.ndarray,
    z_gate: float,
    temperature: float,
    pressure: float,
    density: float,
) -> dict:
    """Build the (r, z) map of effective drift velocity and diffusion constants.

    :param field_map: callable map ``field_map(positions)`` with positions ``(r, z)`` [cm] that
        returns the drift field [V/cm], e.g. a :class:`ember.maps.RegularGridMap`.
    :param r_values: evenly spaced radii [cm] of the new map.
    :param z_values: evenly spaced, increasing z positions [cm] of the new map; the gate must lie
        inside.
    :param z_gate: z position of the gate [cm].
    :param temperature: liquid temperature [K].
    :param pressure: gas pressure [bar].
    :param density: liquid density [g/cm³].
    :return: map dictionary with ``coordinate_system`` (r, z) and the maps
        ``drift_speed_map`` [cm/ns], ``diffusion_longitudinal_map`` and
        ``diffusion_transverse_map`` [cm²/ns], each of shape ``(n_r, n_z)``.
    """
    r_values = np.asarray(r_values, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)
    for axis_name, axis_values in (("r", r_values), ("z", z_values)):
        if len(axis_values) < 2 or not np.allclose(
            axis_values, np.linspace(axis_values[0], axis_values[-1], len(axis_values))
        ):
            raise ValueError(f"The {axis_name} values must be evenly spaced with >= 2 points.")

    maps = {name: np.zeros((len(r_values), len(z_values))) for name in DRIFT_PROPERTY_MAP_NAMES}
    for r_index, r in enumerate(r_values):
        positions = np.column_stack([np.full_like(z_values, r), z_values])
        properties = path_averaged_drift_properties(
            z_values, field_map(positions), z_gate, temperature, pressure, density
        )
        for name in DRIFT_PROPERTY_MAP_NAMES:
            maps[name][r_index] = properties[name]

    return {
        "coordinate_system": [
            ["r", [float(r_values[0]), float(r_values[-1]), len(r_values)]],
            ["z", [float(z_values[0]), float(z_values[-1]), len(z_values)]],
        ],
        **maps,
    }


def drift_time_above_gate(
    z_liquid_level: float,
    z_gate: float,
    extraction_field: float,
    temperature: float,
    pressure: float,
    density: float,
) -> float:
    """Drift time from the gate to the liquid surface in a constant extraction field.

    :param z_liquid_level: z position of the liquid surface [cm].
    :param z_gate: z position of the gate [cm].
    :param extraction_field: field in the liquid between gate and surface [V/cm].
    :param temperature: liquid temperature [K].
    :param pressure: gas pressure [bar].
    :param density: liquid density [g/cm³].
    :return: drift time [ns].
    """
    velocity = drift_velocity_liquid(extraction_field, temperature, pressure, density)
    return float((z_liquid_level - z_gate) / velocity)


def electroluminescence_duration(
    gas_gap: float, field_gas: float, temperature: float, pressure: float, density_gas: float
) -> float:
    """Time an extracted electron needs to cross a uniform-field gas gap.

    :param gas_gap: distance between the liquid surface and the anode [cm].
    :param field_gas: electric field in the gas [V/cm].
    :param temperature: gas temperature [K].
    :param pressure: gas pressure [bar].
    :param density_gas: gas density [g/cm³].
    :return: crossing time [ns].
    """
    velocity = drift_velocity_gas(field_gas, temperature, pressure, density_gas)
    return float(gas_gap / velocity)


def xenon_number_density(density: float) -> float:
    """Number density of xenon atoms.

    :param density: xenon density [g/cm³].
    :return: atoms per cm³.
    """
    return density / XENON_MOLAR_MASS * AVOGADRO_CONSTANT


def electroluminescence_yield_per_length(field: np.ndarray, number_density: float) -> np.ndarray:
    """Electroluminescence photons emitted by one electron per cm of path in xenon gas.

    Linear fit of C.M.B. Monteiro et al., JINST 2 (2007) P05001, arXiv:physics/0702142, eq. 8:
    ``Y/N = 0.140 · E/N - 0.474`` in 10⁻¹⁷ photons cm² per electron and atom, with ``E/N`` in Td.
    Below the threshold ``E/N = 3.39 Td`` no light is emitted. (fuse uses the same fit in the
    room-temperature form ``Y/p ∝ E/p - 0.8 kV/(cm bar)``.)

    :param field: electric field [V/cm].
    :param number_density: xenon atoms per cm³.
    :return: photons per electron per cm, same shape as ``field``.
    """
    reduced_field = np.asarray(field, dtype=np.float64) / number_density / TOWNSEND
    reduced_yield = np.clip(0.140 * reduced_field - 0.474, 0.0, None)
    return reduced_yield * 1e-17 * number_density


def anode_field_profile(
    distance_to_anode: np.ndarray, field_far: float, anode_wire_pitch: float | None = None
) -> np.ndarray:
    """Electric field along the path of an electron from the liquid surface to the anode.

    * Mesh (``anode_wire_pitch=None``): ideal parallel plates, the field is uniform up to the anode
      plane.
    * Wire grid: far from the wires the field is uniform. Each wire collects the field lines of one
      wire pitch ``s``, so close to the wire the field grows like that of a charged line,
      ``E(r) = E_far · s / (2π r)``. Both agree at the domination distance ``r = s / (2π)``.

    :param distance_to_anode: distance to the anode plane (mesh) or to the wire axis (wires) [cm].
    :param field_far: uniform field in the gas far from the anode [V/cm].
    :param anode_wire_pitch: distance between the anode wires [cm], ``None`` for a mesh.
    :return: field [V/cm], same shape as ``distance_to_anode``.
    """
    distance_to_anode = np.asarray(distance_to_anode, dtype=np.float64)
    if anode_wire_pitch is None:
        return np.full(distance_to_anode.shape, float(field_far))
    domination_distance = anode_wire_pitch / (2 * np.pi)
    return field_far * np.maximum(1.0, domination_distance / distance_to_anode)


def build_luminescence_timing_map(
    gas_gap: float,
    field_gas: float,
    temperature: float,
    pressure: float,
    density_gas: float,
    anode_wire_pitch: float | None = None,
    anode_wire_radius: float | None = None,
    number_of_steps: int = 20_000,
    number_of_u_points: int = 1001,
) -> dict:
    """Inverse CDF of the electroluminescence emission time of one extracted electron.

    The electron moves on a straight line from the liquid surface to the anode (mesh: the anode
    plane, wires: the surface of a wire). Along the path the field follows
    :func:`anode_field_profile`, the drift velocity comes from NEST and the light per path length
    from :func:`electroluminescence_yield_per_length`. The emission time distribution is the light
    emitted per time step. For a mesh (uniform field) it is uniform between 0 and the crossing time.
    Diffusion in the gas is not included.

    :param gas_gap: distance between the liquid surface and the anode plane or wire axis [cm].
    :param field_gas: uniform field in the gas far from the anode [V/cm].
    :param temperature: gas temperature [K].
    :param pressure: gas pressure [bar].
    :param density_gas: gas density [g/cm³].
    :param anode_wire_pitch: distance between the anode wires [cm], ``None`` for a mesh.
    :param anode_wire_radius: radius of the anode wires [cm], ``None`` for a mesh.
    :param number_of_steps: number of steps along the path.
    :param number_of_u_points: number of points of the inverse CDF in u.
    :return: map dictionary with ``coordinate_system`` (u) and ``map`` (emission time in ns after
        the extraction of the electron).
    """
    if (anode_wire_pitch is None) != (anode_wire_radius is None):
        raise ValueError("Give both anode_wire_pitch and anode_wire_radius, or neither (mesh).")
    path_end = 0.0
    if anode_wire_pitch is not None:
        if not 0 < anode_wire_radius < anode_wire_pitch / (2 * np.pi):
            raise ValueError("The wire radius must be positive and below pitch / (2 pi).")
        path_end = anode_wire_radius
    if not path_end < gas_gap:
        raise ValueError("The gas gap must be larger than the anode wire radius.")

    distance_edges = np.linspace(gas_gap, path_end, number_of_steps + 1)
    step_length = (gas_gap - path_end) / number_of_steps
    distance_centres = 0.5 * (distance_edges[1:] + distance_edges[:-1])
    field = anode_field_profile(distance_centres, field_gas, anode_wire_pitch)
    velocity = drift_velocity_gas(field, temperature, pressure, density_gas)
    photons_per_step = (
        electroluminescence_yield_per_length(field, xenon_number_density(density_gas)) * step_length
    )
    if photons_per_step.sum() <= 0:
        raise ValueError("The field in the gas is below the electroluminescence threshold.")

    time_edges = np.concatenate([[0.0], np.cumsum(step_length / velocity)])
    cumulative_photons = np.concatenate([[0.0], np.cumsum(photons_per_step)])
    cdf_edges = cumulative_photons / cumulative_photons[-1]

    end_of_emission = np.argmax(cdf_edges == cdf_edges[-1])
    cdf_edges = cdf_edges[: end_of_emission + 1]
    time_edges = time_edges[: end_of_emission + 1]
    is_last_of_flat_part = np.append(np.diff(cdf_edges) > 0, True)

    u_values = np.linspace(0.0, 1.0, number_of_u_points)
    emission_times = np.interp(
        u_values, cdf_edges[is_last_of_flat_part], time_edges[is_last_of_flat_part]
    )
    return {
        "coordinate_system": [["u", [0.0, 1.0, number_of_u_points]]],
        "map": emission_times,
    }
