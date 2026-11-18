"""Lookup between NEST interaction ids (nestid, A, Z) and human-readable names.

ember input files identify the interaction type by its NEST id (``nestid``). The mass number
``A`` and charge number ``Z`` are only used by NEST for ions (``nestid = 6``, e.g. alphas with
``A = 4, Z = 2``). For all other types they are ignored and set to 0.

Examples::

    interaction_name_from_nestid(8)          -> "beta"
    interaction_name_from_nestid(6, 4, 2)    -> "ion(A=4,Z=2)"
    nestid_from_interaction_name("gammaRay") -> (7, 0, 0)
    nestid_from_interaction_name("ion(A=4,Z=2)") -> (6, 4, 2)
"""

import re

# Same ids and names as nestpy.interactions (NEST INTERACTION_TYPE).
NEST_INTERACTION_NAMES = {
    0: "NR",
    1: "WIMP",
    2: "B8",
    3: "DD",
    4: "AmBe",
    5: "Cf",
    6: "ion",
    7: "gammaRay",
    8: "beta",
    9: "CH3T",
    10: "C14",
    11: "Kr83m",
    12: "ppSolar",
    13: "atmNu",
    14: "fullGamma",
    15: "fullGamma_PE",
    16: "fullGamma_Compton_PP",
    17: "NoneType",
}
ION_NESTID = 6
MAX_NUCLEAR_RECOIL_NESTID = 5

_ION_NAME_PATTERN = re.compile(r"^ion\(A=(\d+),Z=(\d+)\)$")


def interaction_name_from_nestid(nestid: int, mass_number: int = 0, charge_number: int = 0) -> str:
    """Return the human-readable name of a NEST interaction type.

    :param nestid: NEST interaction id (0 to 17).
    :param mass_number: mass number A, only used for ions (nestid 6).
    :param charge_number: charge number Z, only used for ions (nestid 6).
    :return: e.g. ``"beta"`` or ``"ion(A=4,Z=2)"``.
    """
    nestid = int(nestid)
    if nestid not in NEST_INTERACTION_NAMES:
        raise ValueError(f"Unknown NEST interaction id {nestid}.")
    if nestid != ION_NESTID:
        return NEST_INTERACTION_NAMES[nestid]
    if mass_number <= 0 or charge_number <= 0:
        raise ValueError(f"Ions need A > 0 and Z > 0, got A={mass_number}, Z={charge_number}.")
    return f"ion(A={int(mass_number)},Z={int(charge_number)})"


def nestid_from_interaction_name(name: str) -> tuple[int, int, int]:
    """Return ``(nestid, A, Z)`` for a human-readable interaction name.

    :param name: a NEST interaction name such as ``"beta"``, or ``"ion(A=4,Z=2)"`` for ions.
    :return: tuple of NEST id, mass number and charge number (A = Z = 0 for non-ions).
    """
    ion_match = _ION_NAME_PATTERN.match(name)
    if ion_match:
        return ION_NESTID, int(ion_match.group(1)), int(ion_match.group(2))
    for nestid, interaction_name in NEST_INTERACTION_NAMES.items():
        if name == interaction_name and nestid != ION_NESTID:
            return nestid, 0, 0
    raise ValueError(
        f"Unknown interaction name '{name}'. Use one of "
        f"{sorted(set(NEST_INTERACTION_NAMES.values()) - {'ion'})} or 'ion(A=<A>,Z=<Z>)'."
    )
