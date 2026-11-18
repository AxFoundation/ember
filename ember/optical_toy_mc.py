"""Optical toy Monte Carlo of photon transport in a cylindrical dual-phase xenon TPC.

The model is a strongly simplified version of a Geant4 optical simulation. It is used to build the
dummy S1 and S2 optical propagation timing maps (and, optionally, an S1 light collection estimate).
S1 photons start in the liquid, S2 photons in the gas between the liquid surface and the anode.
Photons are emitted isotropically and tracked until they are absorbed or reach a photocathode:

* liquid xenon: Rayleigh scattering (angular distribution 1 + cos²θ) and absorption,
* gaseous xenon (between liquid surface and top PMTs): absorption only,
* PTFE walls: diffuse (Lambertian) reflection with a reflectivity for liquid and gas,
* liquid surface: Fresnel reflection and refraction, total internal reflection,
* PMT arrays: planes, a photon hits a photocathode with the probability given by the photocathode
  coverage of the array, otherwise it hits PTFE between the PMTs.

The arrival delay of a photon is ``(L_liquid * n_liquid + L_gas) / c``.

Not modelled: electrodes, PMT window optics, the specular part of PTFE reflection and scattering in
the gas. Optical parameters and their meaning follow K. Böse, PhD thesis, Heidelberg University
(2026), Chapter 4.
"""

from dataclasses import dataclass

import numba
import numpy as np
import strax

SPEED_OF_LIGHT_IN_VACUUM = 29.9792458  # cm/ns

OUTCOME_ABSORBED = 0
OUTCOME_TOP = 1
OUTCOME_BOTTOM = 2

PHOTON_DTYPE = np.dtype(
    [
        ("outcome", np.int8),
        ("arrival_time", np.float64),
        ("path_liquid", np.float64),
        ("path_gas", np.float64),
        ("ptfe_reflections_liquid", np.int32),
        ("ptfe_reflections_gas", np.int32),
    ]
)

_SURFACE_OFFSET = 1e-6  # cm, moves a photon off a surface after a reflection or refraction


@dataclass(frozen=True)
class TpcOptics:
    """Geometry and optical properties of a cylindrical dual-phase TPC.

    Lengths in cm. z points upwards; the bottom PMTs, the liquid surface and the top PMTs are
    horizontal planes.
    """

    tpc_radius: float
    z_bottom_pmts: float
    z_liquid_level: float
    z_top_pmts: float
    lxe_refractive_index: float
    lxe_rayleigh_scattering_length: float
    lxe_absorption_length: float
    gxe_absorption_length: float
    ptfe_reflectivity_liquid: float
    ptfe_reflectivity_gas: float
    photocathode_coverage_top: float
    photocathode_coverage_bottom: float

    @classmethod
    def from_detector_config(cls) -> "TpcOptics":
        """Read the geometry and optical properties from ember's ``detector.yaml``.

        The photocathode coverage of an array is ``n_PMTs * (d / 2)² / R²`` with the photocathode
        diameter ``d`` and the TPC radius ``R``.

        :return: the optics of the configured detector.
        """

        def read(key: str):
            return strax.URLConfig.evaluate_dry(f"detector://{key}")

        tpc_radius = float(read("tpc_radius"))
        photocathode_area_fraction = (read("pmt_photocathode_diameter") / 2) ** 2 / tpc_radius**2
        return cls(
            tpc_radius=tpc_radius,
            z_bottom_pmts=float(read("z_bottom_pmts")),
            z_liquid_level=float(read("z_liquid_level")),
            z_top_pmts=float(read("z_top_pmts")),
            lxe_refractive_index=float(read("lxe_refractive_index")),
            lxe_rayleigh_scattering_length=float(read("lxe_rayleigh_scattering_length")),
            lxe_absorption_length=float(read("lxe_absorption_length")),
            gxe_absorption_length=float(read("gxe_absorption_length")),
            ptfe_reflectivity_liquid=float(read("ptfe_reflectivity_liquid")),
            ptfe_reflectivity_gas=float(read("ptfe_reflectivity_gas")),
            photocathode_coverage_top=read("n_top_pmts") * photocathode_area_fraction,
            photocathode_coverage_bottom=read("n_bottom_pmts") * photocathode_area_fraction,
        )

    def direct_flight_time(self, z: np.ndarray, array_name: str) -> np.ndarray:
        """Shortest possible delay: vertical flight from z to the top or bottom array.

        :param z: emission depths in the liquid in cm.
        :param array_name: ``"top"`` or ``"bottom"``.
        :return: delays in ns.
        """
        light_speed_in_liquid = SPEED_OF_LIGHT_IN_VACUUM / self.lxe_refractive_index
        if array_name == "bottom":
            return (np.asarray(z) - self.z_bottom_pmts) / light_speed_in_liquid
        return (self.z_liquid_level - np.asarray(z)) / light_speed_in_liquid + (
            self.z_top_pmts - self.z_liquid_level
        ) / SPEED_OF_LIGHT_IN_VACUUM


@numba.njit(cache=True)
def seed_random_generator(seed: int) -> None:
    """Seed numba's random number generator, which is separate from numpy's.

    :param seed: seed.
    """
    np.random.seed(seed)


@numba.njit(cache=True)
def isotropic_direction() -> tuple:
    """Draw a direction uniformly on the unit sphere.

    :return: unit vector (dx, dy, dz).
    """
    cos_theta = 2 * np.random.random() - 1
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = 2 * np.pi * np.random.random()
    return sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta


@numba.njit(cache=True)
def rayleigh_direction(dx: float, dy: float, dz: float) -> tuple:
    """Draw the direction after Rayleigh scattering, probability ∝ 1 + cos²θ to the old direction.

    :param dx: x component of the old unit direction.
    :param dy: y component of the old unit direction.
    :param dz: z component of the old unit direction.
    :return: new unit vector (dx, dy, dz).
    """
    while True:
        new_dx, new_dy, new_dz = isotropic_direction()
        cos_angle = new_dx * dx + new_dy * dy + new_dz * dz
        if 2 * np.random.random() < 1 + cos_angle**2:
            return new_dx, new_dy, new_dz


@numba.njit(cache=True)
def lambertian_direction(normal_x: float, normal_y: float, normal_z: float) -> tuple:
    """Draw a diffusely reflected direction, probability ∝ cos θ to the surface normal.

    :param normal_x: x component of the unit normal, pointing into the volume.
    :param normal_y: y component of the unit normal.
    :param normal_z: z component of the unit normal.
    :return: new unit vector (dx, dy, dz).
    """
    while True:
        new_dx, new_dy, new_dz = isotropic_direction()
        cos_angle = new_dx * normal_x + new_dy * normal_y + new_dz * normal_z
        if cos_angle > 0 and np.random.random() < cos_angle:
            return new_dx, new_dy, new_dz


@numba.njit(cache=True)
def fresnel_reflectance(cos_incident: float, n_from: float, n_to: float) -> tuple:
    """Reflectance of unpolarized light at a flat interface (Böse, App. B.1).

    :param cos_incident: cosine of the angle of incidence to the surface normal.
    :param n_from: refractive index of the medium the photon comes from.
    :param n_to: refractive index of the medium behind the interface.
    :return: (reflectance, cosine of the refraction angle); reflectance 1 and cosine 0 for total
        internal reflection.
    """
    sin_incident = np.sqrt(max(0.0, 1 - cos_incident**2))
    sin_transmitted = n_from / n_to * sin_incident
    if sin_transmitted >= 1:
        return 1.0, 0.0
    cos_transmitted = np.sqrt(1 - sin_transmitted**2)
    amplitude_s = (n_from * cos_incident - n_to * cos_transmitted) / (
        n_from * cos_incident + n_to * cos_transmitted
    )
    amplitude_p = (n_to * cos_incident - n_from * cos_transmitted) / (
        n_to * cos_incident + n_from * cos_transmitted
    )
    return 0.5 * (amplitude_s**2 + amplitude_p**2), cos_transmitted


@numba.njit(cache=True)
def distance_to_wall(x: float, y: float, dx: float, dy: float, radius: float) -> float:
    """Distance along the direction from a point inside the cylinder to its wall.

    :param x: x position in cm.
    :param y: y position in cm.
    :param dx: x component of the unit direction.
    :param dy: y component of the unit direction.
    :param radius: cylinder radius in cm.
    :return: distance in cm, infinite for vertical directions.
    """
    horizontal_squared = dx * dx + dy * dy
    if horizontal_squared < 1e-12:
        return np.inf
    half_b = x * dx + y * dy
    c = x * x + y * y - radius * radius
    return (-half_b + np.sqrt(max(0.0, half_b * half_b - horizontal_squared * c))) / (
        horizontal_squared
    )


@numba.njit(cache=True)
def _propagate_photon(x, y, z, optics, max_steps, result):
    (
        radius,
        z_bottom,
        z_liquid,
        z_top,
        n_liquid,
        scattering_length,
        absorption_length_liquid,
        absorption_length_gas,
        reflectivity_liquid,
        reflectivity_gas,
        coverage_top,
        coverage_bottom,
    ) = optics
    dx, dy, dz = isotropic_direction()
    is_in_liquid = z <= z_liquid
    result["outcome"] = OUTCOME_ABSORBED
    for _ in range(max_steps):
        to_wall = distance_to_wall(x, y, dx, dy, radius)
        lower_plane, upper_plane = (z_bottom, z_liquid) if is_in_liquid else (z_liquid, z_top)
        if dz > 0:
            to_plane = (upper_plane - z) / dz
        elif dz < 0:
            to_plane = (lower_plane - z) / dz
        else:
            to_plane = np.inf
        if is_in_liquid:
            to_scatter = -scattering_length * np.log(1 - np.random.random())
            to_absorption = -absorption_length_liquid * np.log(1 - np.random.random())
        else:
            to_scatter = np.inf
            to_absorption = -absorption_length_gas * np.log(1 - np.random.random())
        step = min(to_wall, to_plane, to_scatter, to_absorption)
        x += step * dx
        y += step * dy
        z += step * dz
        if is_in_liquid:
            result["path_liquid"] += step
        else:
            result["path_gas"] += step

        if step == to_absorption:
            return
        if step == to_scatter:
            dx, dy, dz = rayleigh_direction(dx, dy, dz)
            continue
        if step == to_wall:
            reflectivity = reflectivity_liquid if is_in_liquid else reflectivity_gas
            if np.random.random() >= reflectivity:
                return
            if is_in_liquid:
                result["ptfe_reflections_liquid"] += 1
            else:
                result["ptfe_reflections_gas"] += 1
            r = np.sqrt(x * x + y * y)
            x *= (radius - _SURFACE_OFFSET) / r
            y *= (radius - _SURFACE_OFFSET) / r
            dx, dy, dz = lambertian_direction(-x / r, -y / r, 0.0)
            continue
        if is_in_liquid and dz < 0:
            z = z_bottom + _SURFACE_OFFSET
            if np.random.random() < coverage_bottom:
                result["outcome"] = OUTCOME_BOTTOM
                return
            if np.random.random() >= reflectivity_liquid:
                return
            result["ptfe_reflections_liquid"] += 1
            dx, dy, dz = lambertian_direction(0.0, 0.0, 1.0)
            continue
        if not is_in_liquid and dz > 0:
            z = z_top - _SURFACE_OFFSET
            if np.random.random() < coverage_top:
                result["outcome"] = OUTCOME_TOP
                return
            if np.random.random() >= reflectivity_gas:
                return
            result["ptfe_reflections_gas"] += 1
            dx, dy, dz = lambertian_direction(0.0, 0.0, -1.0)
            continue

        n_from, n_to = (n_liquid, 1.0) if is_in_liquid else (1.0, n_liquid)
        reflectance, cos_transmitted = fresnel_reflectance(abs(dz), n_from, n_to)
        if np.random.random() < reflectance:
            dz = -dz
        else:
            index_ratio = n_from / n_to
            dx *= index_ratio
            dy *= index_ratio
            dz = np.sign(dz) * cos_transmitted
            is_in_liquid = not is_in_liquid
        z = z_liquid + np.sign(dz) * _SURFACE_OFFSET


@numba.njit(cache=True)
def _simulate_photons(start_positions, optics, max_steps, photons):
    for index in range(len(start_positions)):
        _propagate_photon(
            start_positions[index, 0],
            start_positions[index, 1],
            start_positions[index, 2],
            optics,
            max_steps,
            photons[index],
        )
        photons[index]["arrival_time"] = (
            photons[index]["path_liquid"] * optics[4] + photons[index]["path_gas"]
        ) / SPEED_OF_LIGHT_IN_VACUUM


def simulate_photons(
    start_positions: np.ndarray, optics: TpcOptics, seed: int, max_steps: int = 100_000
) -> np.ndarray:
    """Track one photon from each start position until it is absorbed or detected.

    :param start_positions: positions (x, y, z) in cm, shape (n, 3), inside the TPC.
    :param optics: geometry and optical properties.
    :param seed: seed of the random number generator.
    :param max_steps: photons still alive after this many steps count as absorbed.
    :return: structured array with ``PHOTON_DTYPE``: outcome (``OUTCOME_*``), arrival time in ns,
        path lengths in liquid and gas in cm, number of PTFE reflections in liquid and gas.
    """
    start_positions = np.ascontiguousarray(start_positions, dtype=np.float64)
    if start_positions.ndim != 2 or start_positions.shape[1] != 3:
        raise ValueError(f"start_positions must have shape (n, 3), got {start_positions.shape}")
    radii = np.hypot(start_positions[:, 0], start_positions[:, 1])
    is_inside = (
        (radii < optics.tpc_radius)
        & (start_positions[:, 2] >= optics.z_bottom_pmts)
        & (start_positions[:, 2] <= optics.z_top_pmts)
    )
    if not np.all(is_inside):
        raise ValueError("All start positions must lie inside the TPC")
    optics_values = (
        optics.tpc_radius,
        optics.z_bottom_pmts,
        optics.z_liquid_level,
        optics.z_top_pmts,
        optics.lxe_refractive_index,
        optics.lxe_rayleigh_scattering_length,
        optics.lxe_absorption_length,
        optics.gxe_absorption_length,
        optics.ptfe_reflectivity_liquid,
        optics.ptfe_reflectivity_gas,
        optics.photocathode_coverage_top,
        optics.photocathode_coverage_bottom,
    )
    photons = np.zeros(len(start_positions), dtype=PHOTON_DTYPE)
    seed_random_generator(seed)
    _simulate_photons(start_positions, optics_values, max_steps, photons)
    return photons


def draw_positions_in_disk(
    z: float, number_of_positions: int, radius: float, rng: np.random.Generator
) -> np.ndarray:
    """Draw positions uniformly distributed over a horizontal disk.

    :param z: height of the disk in cm.
    :param number_of_positions: number of positions.
    :param radius: radius of the disk in cm.
    :param rng: random number generator.
    :return: positions (x, y, z) in cm, shape (number_of_positions, 3).
    """
    r = radius * np.sqrt(rng.random(number_of_positions))
    phi = 2 * np.pi * rng.random(number_of_positions)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), np.full(number_of_positions, z)])


def build_s1_timing_map(
    optics: TpcOptics,
    number_of_z_points: int,
    number_of_u_points: int,
    photons_per_slice: int,
    seed: int,
) -> dict:
    """Build the S1 optical propagation timing map from the toy Monte Carlo.

    For every z slice between the bottom PMTs and the liquid surface, photons are emitted uniformly
    over the cross section of the TPC. The map stores the quantiles of the arrival delay of the
    photons detected by each array (inverse cumulative distribution function).

    :param optics: geometry and optical properties.
    :param number_of_z_points: number of z slices.
    :param number_of_u_points: number of quantiles u in [0, 1].
    :param photons_per_slice: number of emitted photons per z slice.
    :param seed: seed; slice i uses seed + i.
    :return: map dictionary with ``coordinate_system`` (z, u), ``top``, ``bottom`` (delays in ns,
        shape (number_of_z_points, number_of_u_points)) and ``description``.
    """
    z_values = np.linspace(optics.z_bottom_pmts, optics.z_liquid_level, number_of_z_points)
    u_values = np.linspace(0, 1, number_of_u_points)
    delays = {"top": [], "bottom": []}
    for slice_index, z in enumerate(z_values):
        rng = np.random.default_rng(seed + slice_index)
        positions = draw_positions_in_disk(z, photons_per_slice, optics.tpc_radius, rng)
        photons = simulate_photons(positions, optics, seed + slice_index)
        for array_name, outcome in [("top", OUTCOME_TOP), ("bottom", OUTCOME_BOTTOM)]:
            arrival_times = photons["arrival_time"][photons["outcome"] == outcome]
            if len(arrival_times) < 2:
                raise ValueError(
                    f"Too few photons reach the {array_name} array at z = {z:.1f} cm, "
                    "increase photons_per_slice"
                )
            delays[array_name].append(np.quantile(arrival_times, u_values))
    return {
        "coordinate_system": [
            ["z", (float(z_values[0]), float(z_values[-1]), number_of_z_points)],
            ["u", (0.0, 1.0, number_of_u_points)],
        ],
        "top": np.array(delays["top"]),
        "bottom": np.array(delays["bottom"]),
        "description": (
            "S1 optical propagation delay from ember's optical toy Monte Carlo "
            f"({photons_per_slice} photons per z slice). {optics}"
        ),
    }


def draw_positions_in_layer(
    z_lower: float,
    z_upper: float,
    number_of_positions: int,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw positions uniformly distributed in a horizontal layer of the TPC.

    :param z_lower: lower end of the layer in cm.
    :param z_upper: upper end of the layer in cm.
    :param number_of_positions: number of positions.
    :param radius: radius of the layer in cm.
    :param rng: random number generator.
    :return: positions (x, y, z) in cm, shape (number_of_positions, 3).
    """
    if not z_lower < z_upper:
        raise ValueError(f"z_lower ({z_lower} cm) must be below z_upper ({z_upper} cm)")
    positions = draw_positions_in_disk(0.0, number_of_positions, radius, rng)
    positions[:, 2] = z_lower + (z_upper - z_lower) * rng.random(number_of_positions)
    return positions


def simulate_s2_photons(
    optics: TpcOptics, z_anode: float, number_of_photons: int, seed: int
) -> np.ndarray:
    """Track S2 photons emitted uniformly in the electroluminescence gap.

    The photons start uniformly over the cross section of the TPC and between the liquid surface
    and the anode.

    :param optics: geometry and optical properties.
    :param z_anode: z position of the anode in cm, the upper end of the gap.
    :param number_of_photons: number of emitted photons.
    :param seed: seed of the random number generator.
    :return: structured array with ``PHOTON_DTYPE``, see :func:`simulate_photons`.
    """
    rng = np.random.default_rng(seed)
    positions = draw_positions_in_layer(
        optics.z_liquid_level, z_anode, number_of_photons, optics.tpc_radius, rng
    )
    return simulate_photons(positions, optics, seed)


def build_s2_timing_map(photons: np.ndarray, number_of_u_points: int) -> dict:
    """Build the S2 optical propagation timing map from simulated S2 photons.

    The map stores the quantiles of the arrival delay of the photons detected by each array
    (inverse cumulative distribution function), as fuse's ``s2_time_spline``.

    :param photons: photons from :func:`simulate_s2_photons`.
    :param number_of_u_points: number of quantiles u in [0, 1].
    :return: map dictionary with ``coordinate_system`` (u), ``top`` and ``bottom`` (delays in ns,
        shape (number_of_u_points,)).
    """
    u_values = np.linspace(0, 1, number_of_u_points)
    timing_map = {"coordinate_system": [["u", (0.0, 1.0, number_of_u_points)]]}
    for array_name, outcome in [("top", OUTCOME_TOP), ("bottom", OUTCOME_BOTTOM)]:
        arrival_times = photons["arrival_time"][photons["outcome"] == outcome]
        if len(arrival_times) < 2:
            raise ValueError(
                f"Too few photons reach the {array_name} array, increase the number of photons"
            )
        timing_map[array_name] = np.quantile(arrival_times, u_values)
    return timing_map
