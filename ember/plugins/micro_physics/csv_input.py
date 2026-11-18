"""Source plugin reading interaction instructions from a CSV file.

Ported from fuse's ``ChunkCsvInput`` (detector physics) and ``ChunkInput`` (event-time modes).

CSV format, one row per interaction::

    eventid, t, x, y, z, ed, nestid, A, Z, photons, electrons

* ``eventid``: rows with the same id belong to one event and keep their time spacing.
* ``t`` [ns]: time relative to the event start, or absolute time if ``source_rate = 0``.
* ``x, y, z`` [cm]: position, gate at ``z = 0``.
* ``ed`` [keV]: deposited energy, ``-1`` if not given.
* ``nestid``: NEST interaction type, see :mod:`ember.interaction_types`.
* ``A, Z``: mass and charge number, only used for ions (``nestid = 6``).
* ``photons, electrons``: number of quanta, ``-1`` if NEST shall compute them.

Rules for ``-1`` entries (checked by :func:`load_instructions`):

* ``ed >= 0``: NEST computes every quantum given as ``-1``; given quanta replace the NEST values.
* ``ed = -1``: the quanta are used as given and a ``-1`` quantum becomes 0. For electronic
  recoils the energy is estimated as ``W * (photons + electrons)``. Both quanta ``-1``, or
  ``nestid <= 6`` (nuclear recoils and ions), raise an error.

The input file is not part of the lineage (``input_file`` is not tracked, as in fuse). Use one run
id per CSV file: a second file simulated with the same run id into the same storage folder gets the
same lineage, and strax loads the cached data of the first file instead.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
import strax

from ...common import dynamic_chunking, stable_argsort, stable_sort
from ...dtypes import clustered_interactions_fields
from ...interaction_types import ION_NESTID, NEST_INTERACTION_NAMES
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()
__all__.extend(["REQUIRED_COLUMNS", "NEST_GAS_TIMING_ENERGY_THRESHOLD"])

log = logging.getLogger("ember.CsvInput")

REQUIRED_COLUMNS = [
    "eventid",
    "t",
    "x",
    "y",
    "z",
    "ed",
    "nestid",
    "A",
    "Z",
    "photons",
    "electrons",
]
NOT_GIVEN = -1
MAX_SUPPORTED_NESTID = 16
NEST_GAS_TIMING_ENERGY_THRESHOLD = 0.0137  # keV, NEST: W_DEFAULT * 0.001


@export
class CsvInput(EmberBasePlugin):
    """Read interactions from a CSV file and return them in time-ordered chunks.

    Event times are assigned in one of three modes:

    * ``source_rate > 0``: uniformly random in ``U(0, N / source_rate)``.
    * ``source_rate > 0`` and ``fixed_event_spacing``: events are ``1 / source_rate`` apart.
    * ``source_rate = 0``: the ``t`` column is used as absolute time.
    """

    __version__ = "0.0.1"

    depends_on: Tuple = tuple()
    provides = "clustered_interactions"
    data_kind = "clustered_interactions"

    save_when = strax.SaveWhen.TARGET

    source_done = False

    dtype = clustered_interactions_fields + strax.time_fields

    input_file = strax.URLConfig(
        track=False,
        infer_type=False,
        help="CSV file to read. Not part of the lineage: use one run id per CSV file",
    )

    separation_scale = strax.URLConfig(
        default=1e8,
        type=(int, float),
        help="Start a new chunk when the previous interaction is separated by this time [ns]",
    )

    source_rate = strax.URLConfig(
        default=1,
        type=(int, float),
        help="Source rate used to generate event times [1/s]. "
        "Use source_rate = 0 to use the times from the input file.",
    )

    fixed_event_spacing = strax.URLConfig(
        default=False,
        type=bool,
        help="If True, events are spaced with a fixed time difference of 1/source_rate",
    )

    n_interactions_per_chunk = strax.URLConfig(
        default=1e5,
        type=(int, float),
        help="Minimum number of interactions per chunk",
    )

    first_chunk_left = strax.URLConfig(
        default=1e6,
        type=(int, float),
        help="Time left of the first interaction for the start of the first chunk [ns]",
    )

    last_chunk_length = strax.URLConfig(
        default=1e8,
        type=(int, float),
        help="Time length added after the last interaction for the end of the last chunk [ns]",
    )

    w_value = strax.URLConfig(
        default="detector://w_value_ev",
        type=(int, float),
        help="Energy per quantum [eV], used to estimate the energy from given quanta",
    )

    def setup(self):
        super().setup()
        if self.separation_scale < 1:
            raise ValueError("separation_scale must be at least 1 ns.")
        instructions = load_instructions(self.input_file, self.w_value)
        instructions["time"] = (
            event_start_times(
                instructions["eventid"], self.rng, self.source_rate, self.fixed_event_spacing
            )
            + instructions["t"]
        )
        if np.any(instructions["time"] < 0):
            raise ValueError(
                "Interaction times must not be negative. With source_rate = 0, the 't' column "
                "holds absolute times; otherwise 't' must not be earlier than the event start."
            )
        instructions["endtime"] = instructions["time"]
        instructions = instructions[stable_argsort(instructions["time"])]
        instructions["cluster_id"] = np.arange(1, len(instructions) + 1)

        chunk_index = dynamic_chunking(
            instructions["time"], self.separation_scale, int(self.n_interactions_per_chunk)
        )
        self.chunk_bounds = chunk_boundaries(
            instructions["time"], chunk_index, self.first_chunk_left, self.last_chunk_length
        )
        self.chunks = [instructions[chunk_index == i] for i in np.unique(chunk_index)]
        log.debug(f"Simulating {len(instructions)} interactions in {len(self.chunks)} chunks.")

    def compute(self, chunk_i):
        data = self.chunks[chunk_i]
        self.source_done = chunk_i == len(self.chunks) - 1
        return self.chunk(
            start=self.chunk_bounds[chunk_i], end=self.chunk_bounds[chunk_i + 1], data=data
        )

    def source_finished(self):
        return self.source_done

    def is_ready(self, chunk_i):
        """Overwritten to mimic an online input plugin (as in fuse).

        Alternates between False (check source_finished) and True (build next chunk).
        """
        if "ready" not in self.__dict__:
            self.ready = False
        self.ready ^= True
        return self.ready


def load_instructions(input_file: str, w_value: float) -> np.ndarray:
    """Read and validate a CSV instruction file.

    :param input_file: path of the CSV file.
    :param w_value: energy per quantum [eV], used to estimate energies with ``ed = -1``.
    :return: structured array with the ``clustered_interactions`` dtype (times not yet set).
    """
    data_frame = pd.read_csv(input_file)
    missing_columns = set(REQUIRED_COLUMNS) - set(data_frame.columns)
    if missing_columns:
        raise ValueError(f"Not all needed columns provided! {sorted(missing_columns)} are missing.")
    if len(data_frame) == 0:
        raise ValueError(f"The input file {input_file} contains no interactions.")

    validate_instructions(data_frame)

    instructions = np.zeros(
        len(data_frame), dtype=clustered_interactions_fields + strax.time_fields
    )
    for column in ["eventid", "t", "x", "y", "z", "ed", "nestid", "A", "Z"]:
        instructions[column] = data_frame[column].to_numpy()
    instructions["input_photons"] = data_frame["photons"].to_numpy()
    instructions["input_electrons"] = data_frame["electrons"].to_numpy()

    estimate_missing_energies(instructions, w_value)
    warn_about_gas_timing(instructions["ed"])
    return instructions


def validate_instructions(data_frame: pd.DataFrame) -> None:
    """Raise a ValueError if the instructions violate the input rules.

    :param data_frame: content of the CSV file.
    """
    for column in ["eventid", "t", "nestid", "A", "Z", "photons", "electrons"]:
        values = data_frame[column].to_numpy(dtype=np.float64)
        if np.any(values != np.round(values)):
            raise ValueError(f"Column '{column}' must contain integers.")

    nestid = data_frame["nestid"].to_numpy()
    if np.any((nestid < 0) | (nestid > MAX_SUPPORTED_NESTID)):
        raise ValueError(f"nestid must be between 0 and {MAX_SUPPORTED_NESTID}.")

    is_ion = nestid == ION_NESTID
    ion_without_nucleus = is_ion & ((data_frame["A"] <= 0) | (data_frame["Z"] <= 0)).to_numpy()
    if np.any(ion_without_nucleus):
        raise ValueError("Ions (nestid 6) need A > 0 and Z > 0.")

    for column in ["photons", "electrons"]:
        if np.any(data_frame[column] < NOT_GIVEN):
            raise ValueError(f"'{column}' must be >= 0, or -1 if not given.")

    energy = data_frame["ed"].to_numpy()
    energy_is_missing = energy == NOT_GIVEN
    if np.any((energy < 0) & ~energy_is_missing):
        raise ValueError("'ed' must be >= 0, or -1 if not given.")

    no_quanta_given = (data_frame["photons"] == NOT_GIVEN) & (data_frame["electrons"] == NOT_GIVEN)
    if np.any(energy_is_missing & no_quanta_given.to_numpy()):
        raise ValueError("Rows with ed = -1 need photons or electrons (or both) to be given.")

    energy_needed = nestid <= ION_NESTID
    if np.any(energy_is_missing & energy_needed):
        names = sorted(
            {NEST_INTERACTION_NAMES[i] for i in nestid[energy_is_missing & energy_needed]}
        )
        raise ValueError(
            f"Rows with ed = -1 are only possible for electronic recoils, not for {names}: "
            "NEST needs the energy for nuclear recoils and ions."
        )


def estimate_missing_energies(instructions: np.ndarray, w_value: float) -> None:
    """Fill ``ed = -1`` rows with ``W * (photons + electrons)``, in place.

    Quanta given as -1 in these rows are set to 0.

    :param instructions: structured array with ``ed``, ``input_photons``, ``input_electrons``.
    :param w_value: energy per quantum [eV].
    """
    energy_is_missing = instructions["ed"] == NOT_GIVEN
    for field in ["input_photons", "input_electrons"]:
        quantum_is_missing = energy_is_missing & (instructions[field] == NOT_GIVEN)
        instructions[field][quantum_is_missing] = 0
    number_of_quanta = instructions["input_photons"] + instructions["input_electrons"]
    instructions["ed"][energy_is_missing] = w_value * 1e-3 * number_of_quanta[energy_is_missing]
    instructions["ed_is_estimated"] = energy_is_missing


def warn_about_gas_timing(energy: np.ndarray) -> None:
    """Warn if energies are so small that NEST's photon timing uses gas-phase values.

    :param energy: deposited energies [keV].
    """
    number_of_low_energies = int(np.sum(energy < NEST_GAS_TIMING_ENERGY_THRESHOLD))
    if number_of_low_energies > 0:
        log.warning(
            f"{number_of_low_energies} interactions have ed < "
            f"{NEST_GAS_TIMING_ENERGY_THRESHOLD} keV. For these NEST assumes gas-phase "
            "scintillation timing (singlet 5.18 ns, triplet 100.1 ns) in the S1 simulation."
        )


def event_start_times(
    event_ids: np.ndarray,
    rng: np.random.Generator,
    source_rate: float,
    fixed_event_spacing: bool,
) -> np.ndarray:
    """Return the start time of the event of each interaction.

    :param event_ids: event id of each interaction (any order).
    :param rng: random number generator.
    :param source_rate: events per second; 0 means times are taken from the input file.
    :param fixed_event_spacing: if True, events are exactly ``1 / source_rate`` apart.
    :return: event start time [ns] for each interaction, int64.
    """
    if source_rate < 0:
        raise ValueError("Source rate cannot be negative!")
    if source_rate == 0:
        return np.zeros(len(event_ids), dtype=np.int64)

    unique_event_ids, event_index = np.unique(event_ids, return_inverse=True)
    number_of_events = len(unique_event_ids)
    event_rate_per_ns = source_rate / 1e9
    if fixed_event_spacing:
        start_times = np.arange(number_of_events) / event_rate_per_ns
    else:
        start_times = stable_sort(
            rng.uniform(low=0, high=number_of_events / event_rate_per_ns, size=number_of_events)
        )
    return start_times.astype(np.int64)[event_index]


def chunk_boundaries(
    times: np.ndarray,
    chunk_index: np.ndarray,
    first_chunk_left: float,
    last_chunk_length: float,
    chunk_delay_fraction: float = 0.75,
) -> np.ndarray:
    """Return the chunk boundaries for time-sorted interactions (as in fuse).

    A boundary is placed after ``chunk_delay_fraction`` of the gap between two chunks. The first
    chunk starts ``first_chunk_left`` before the first interaction (but not before 0), the last
    chunk ends ``last_chunk_length`` after the last gap.

    :param times: sorted interaction times [ns].
    :param chunk_index: chunk index of each interaction (from :func:`dynamic_chunking`).
    :param first_chunk_left: margin before the first interaction [ns].
    :param last_chunk_length: margin after the last interaction [ns].
    :param chunk_delay_fraction: position of the boundary within a gap.
    :return: array of ``n_chunks + 1`` boundaries, int64.
    """
    unique_chunks = np.unique(chunk_index)
    chunk_start = np.array([times[chunk_index == i][0] for i in unique_chunks])
    chunk_end = np.array([times[chunk_index == i][-1] for i in unique_chunks])
    first_boundary = max(chunk_start[0] - np.int64(first_chunk_left), 0)

    if len(unique_chunks) == 1:
        log.warning("Only one chunk created! Rate too high or too few events?")
        return np.array([first_boundary, chunk_end[0] + np.int64(last_chunk_length)], np.int64)

    gap_length = chunk_start[1:] - chunk_end[:-1]
    gap_length = np.append(gap_length, gap_length[-1] + np.int64(last_chunk_length))
    later_boundaries = chunk_end + (chunk_delay_fraction * gap_length).astype(np.int64)
    return np.append(first_boundary, later_boundaries).astype(np.int64)
