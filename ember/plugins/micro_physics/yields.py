"""Number of photons, electrons and excitons per interaction from NEST.

Ported from fuse's ``NestYields`` to the nestpy >= 2.1 API (``nestpy.interactions``,
``nestpy.detectors.VDetector``, ``er_parameters`` / ``nr_parameters``).

Quanta given in the input file replace the NEST values (see :mod:`.csv_input` for the rules).
Excitons are always computed. If the number of photons is given, the excitons are scaled to it
with the mean exciton-to-photon ratio of NEST at the energy, field and type of the interaction.
"""

import contextlib
import os
from typing import Iterator

import nestpy
import numpy as np
import strax

from ...dtypes import quanta_fields
from ...interaction_types import ION_NESTID
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()
__all__.extend(["NEST_WIDTH_PARAMETER_INDEX"])

NOT_GIVEN = -1
KR83M_NESTID = 11
KR83M_MAX_TIME_SEPARATION = np.iinfo(np.int8).max  # ns, as fuse ("infinity" for int8)
KR83M_ENERGY_TOLERANCE = 1.0  # keV

# Meaning of the entries of NEST's NRERWidthsParam (NEST master, default_NRERWidthsParam).
NEST_WIDTH_PARAMETER_INDEX = {
    "fano_ions_NR": 0,
    "fano_excitons_NR": 1,
    "A_NR": 2,
    "xi_NR": 3,
    "omega_NR": 4,
    "skewness_NR": 5,
    "fano_ER": 6,
    "A_ER": 7,
    "xi_ER": 8,
    "omega_ER": 9,
    "alpha_skewness_ER": 10,
    "fano_ions_NR_slope": 11,
    "fano_excitons_NR_slope": 12,
}

# The nestpy random generator is global. The seed is set and locked for each chunk.
nest_rng = nestpy.RandomGen.rndm()


@contextlib.contextmanager
def suppress_c_stderr() -> Iterator[None]:
    """Silence messages that C++ code writes directly to the stderr file descriptor."""
    stderr_fd = 2
    saved_fd = os.dup(stderr_fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), stderr_fd)
            yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)


@export
class NestYields(EmberBasePlugin):
    """Calculate photons, electrons and excitons of each interaction with NEST."""

    __version__ = "0.0.1"

    depends_on = ("interactions_in_roi", "electric_field_values")
    provides = "quanta"
    data_kind = "interactions_in_roi"

    dtype = quanta_fields + strax.time_fields

    save_when = strax.SaveWhen.TARGET

    return_yields_only = strax.URLConfig(
        default=False,
        type=bool,
        help="Return the mean NEST yields instead of sampled quanta. Only for testing purposes.",
    )

    nest_width_parameters = strax.URLConfig(
        default={},
        type=dict,
        help="Modify NEST's NRERWidthsParam, e.g. {'fano_ER': -0.0015}. "
        f"Available names: {sorted(NEST_WIDTH_PARAMETER_INDEX)}",
    )

    nest_er_yields_parameters = strax.URLConfig(
        default=-1,
        type=(int, list),
        help="List of 10 NEST ER yield parameters (use -1 entries for the NEST default), "
        "or -1 for all defaults",
    )

    nest_nr_yields_parameters = strax.URLConfig(
        default=-1,
        type=(int, list),
        help="List of 12 NEST NR yield parameters, or -1 for the NEST defaults",
    )

    fix_gamma_yield_field = strax.URLConfig(
        default=-1.0,
        type=float,
        help="Field [V/cm] used for the NEST gamma yields. Only used if > 0.",
    )

    def setup(self):
        super().setup()

        if self.seed is not None:
            self.short_seed = int(repr(self.seed)[-8:])
        else:
            self.short_seed = int(self.rng.integers(1, 10**8))
        self.log.debug(f"Generating nest random numbers starting with seed {self.short_seed}")

        with suppress_c_stderr():
            self.nc = nestpy.NESTcalc(nestpy.detectors.VDetector())

        self.width_parameters = self.updated_width_parameters()
        self.er_yields_parameters = self.yields_parameter_list(
            self.nest_er_yields_parameters, self.nc.default_er_parameters
        )
        self.nr_yields_parameters = self.yields_parameter_list(
            self.nest_nr_yields_parameters, self.nc.default_nr_parameters
        )
        self.vectorized_get_quanta = np.vectorize(self.get_quanta)

    def updated_width_parameters(self) -> list:
        """Return NEST's default width parameters with the user changes applied."""
        width_parameters = list(self.nc.default_nr_er_width_parameters)
        for name, value in self.nest_width_parameters.items():
            if name not in NEST_WIDTH_PARAMETER_INDEX:
                raise ValueError(
                    f"Unknown NEST width parameter {name}. "
                    f"Available parameters: {sorted(NEST_WIDTH_PARAMETER_INDEX)}"
                )
            width_parameters[NEST_WIDTH_PARAMETER_INDEX[name]] = float(value)
        return width_parameters

    @staticmethod
    def yields_parameter_list(setting, default: list) -> list:
        """Return the yield parameters, ``-1`` meaning NEST's defaults."""
        if isinstance(setting, list):
            if len(setting) != len(default):
                raise ValueError(f"Expected {len(default)} yield parameters, got {len(setting)}.")
            return [float(value) for value in setting]
        if setting == -1:
            return list(default)
        raise ValueError("Not supported setting for the NEST yield parameters!")

    def compute(self, interactions_in_roi):
        result = np.zeros(len(interactions_in_roi), dtype=self.dtype)
        result["time"] = interactions_in_roi["time"]
        result["endtime"] = interactions_in_roi["endtime"]
        if len(interactions_in_roi) == 0:
            return result

        nest_rng.set_seed(self.short_seed)
        nest_rng.lock_seed()
        self.short_seed += 1
        try:
            photons, electrons, excitons = self.vectorized_get_quanta(
                interactions_in_roi["ed"],
                interactions_in_roi["nestid"],
                interactions_in_roi["e_field"],
                interactions_in_roi["A"],
                interactions_in_roi["Z"],
                interactions_in_roi["xe_density"],
                interactions_in_roi["input_photons"],
                interactions_in_roi["input_electrons"],
            )
        finally:
            nest_rng.unlock_seed()

        result["photons"] = photons
        result["electrons"] = electrons
        result["excitons"] = excitons
        return result

    def get_quanta(
        self,
        energy: float,
        nestid: int,
        e_field: float,
        mass_number: int,
        charge_number: int,
        density: float,
        input_photons: int,
        input_electrons: int,
    ) -> tuple:
        """Return (photons, electrons, excitons) of one interaction.

        Given quanta (>= 0) replace the NEST values. If the photons are given, the excitons are
        scaled to them with NEST's mean exciton-to-photon ratio.
        """
        if energy <= 0:
            return max(input_photons, 0), max(input_electrons, 0), 0

        yields_result = self.get_yields_from_nest(
            energy, nestid, e_field, mass_number, charge_number, density
        )
        if self.return_yields_only:
            photons = yields_result.PhotonYield
            electrons = yields_result.ElectronYield
            excitons = mean_excitons(yields_result)
        else:
            quanta = self.nc.GetQuanta(
                yields_result, density=density, nr_er_width_parameters=self.width_parameters
            )
            photons, electrons, excitons = quanta.photons, quanta.electrons, quanta.excitons

        if input_electrons != NOT_GIVEN:
            electrons = input_electrons
        if input_photons != NOT_GIVEN:
            photons = input_photons
            excitons = scaled_excitons(yields_result, photons)
        return photons, electrons, excitons

    def get_yields_from_nest(
        self,
        energy: float,
        nestid: int,
        e_field: float,
        mass_number: int,
        charge_number: int,
        density: float,
    ):
        """Return NEST's mean yields (``nestpy.YieldResult``) for one interaction."""
        if nestid == KR83M_NESTID:
            for line_energy in (32.1, 9.4):
                if abs(energy - line_energy) < KR83M_ENERGY_TOLERANCE:
                    energy = line_energy
            mass_number, charge_number = KR83M_MAX_TIME_SEPARATION, 0
        elif nestid != ION_NESTID:
            charge_number = 0

        if nestid <= 5 and energy > 2e2:
            self.log.warning(
                f"Energy deposit of {energy} keV beyond NEST validity for NR (200 keV)"
            )
        if nestid in (7, 8) and energy > 3e3:
            self.log.warning(f"Energy deposit of {energy} keV beyond NEST validity for ER (3 MeV)")

        if nestid == 7 and self.fix_gamma_yield_field > 0:
            e_field = self.fix_gamma_yield_field

        if e_field < 0:
            raise ValueError(
                f"Negative electric field {e_field} V/cm not allowed "
                "(no error will be raised by NEST)."
            )

        return self.nc.GetYields(
            interaction=nestpy.interactions(int(nestid)),
            energy=float(energy),
            density=float(density),
            drift_field=float(e_field),
            A=float(mass_number),
            Z=float(charge_number),
            nr_parameters=self.nr_yields_parameters,
            er_parameters=self.er_yields_parameters,
        )


def mean_excitons(yields_result) -> float:
    """Mean number of excitons from NEST's mean yields.

    The mean number of quanta ``Nq = photons + electrons`` splits into ions and excitons with
    ``Nex / Ni = ExcitonRatio``.
    """
    number_of_quanta = yields_result.PhotonYield + yields_result.ElectronYield
    exciton_ratio = yields_result.ExcitonRatio
    return number_of_quanta * exciton_ratio / (1.0 + exciton_ratio)


def scaled_excitons(yields_result, photons: int) -> int:
    """Number of excitons for a given number of photons, using NEST's mean ratio.

    The result never exceeds the number of photons (excitons decay into photons).
    """
    if yields_result.PhotonYield <= 0:
        return 0
    exciton_to_photon_ratio = min(mean_excitons(yields_result) / yields_result.PhotonYield, 1.0)
    return int(round(exciton_to_photon_ratio * photons))
