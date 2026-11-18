"""Extraction of the electrons into the gas phase (ported from fuse's ``ElectronExtraction``).

Every electron at the liquid surface is extracted with the probability
``electron_extraction_yield`` (a Bernoulli trial per electron, which is the same as
``Binomial(N_e, p)`` per cluster). An extracted electron leaves the liquid after an exponentially
distributed trapping delay with mean ``electron_trapping_time`` (Henning's thesis, eq. 4.7, 4.8).

Deviations from fuse:

* Only the scalar extraction probability is ported; fuse can also compute it from an S2 correction
  map and a single electron gain map.
* The output is sorted by time: the trapping delays change the order of the electrons.
* Invalid configurations (probability outside [0, 1], negative trapping time) raise a
  ``ValueError``.
* An error is raised if an electron would end up behind the end of the chunk.
"""

import numpy as np
import strax

from ...dtypes import individual_electron_fields
from ...plugin import EmberBasePlugin

export, __all__ = strax.exporter()


@export
class ElectronExtraction(EmberBasePlugin):
    """Simulate the extraction of the electrons from the liquid into the gas phase."""

    __version__ = "0.0.1"

    depends_on = "electrons_at_interface"
    provides = "extracted_electrons"
    data_kind = "individual_electrons"

    save_when = strax.SaveWhen.TARGET

    dtype = individual_electron_fields + strax.time_fields

    electron_extraction_yield = strax.URLConfig(
        default="detector://electron_extraction_efficiency",
        type=(int, float),
        help="Probability that an electron at the liquid surface is extracted into the gas",
    )

    electron_trapping_time = strax.URLConfig(
        default="detector://electron_trapping_time",
        type=(int, float),
        help="Mean delay of the extraction at the liquid surface [ns]",
    )

    def setup(self):
        super().setup()
        if not 0 <= self.electron_extraction_yield <= 1:
            raise ValueError(
                f"electron_extraction_yield must be in [0, 1], got {self.electron_extraction_yield}"
            )
        if self.electron_trapping_time < 0:
            raise ValueError(
                f"electron_trapping_time must not be negative, got {self.electron_trapping_time}"
            )

    def compute(self, individual_electrons, start, end):
        is_extracted = self.rng.random(len(individual_electrons)) < self.electron_extraction_yield
        extracted = individual_electrons[is_extracted]
        if len(extracted) == 0:
            return np.zeros(0, dtype=self.dtype)

        result = np.zeros(len(extracted), dtype=self.dtype)
        result["time"] = self.extraction_delay(extracted["time"])
        result["endtime"] = result["time"]
        result["x_interface"] = extracted["x_interface"]
        result["y_interface"] = extracted["y_interface"]
        result["cluster_id"] = extracted["cluster_id"]
        result = strax.sort_by_time(result)

        if result["time"].max() >= end:
            raise ValueError(
                f"Extracted electrons up to {result['time'].max() - end + 1} ns behind the chunk "
                "end. Increase the separation_scale of the input plugin."
            )
        return result

    def extraction_delay(self, electron_times: np.ndarray) -> np.ndarray:
        """Add the trapping delay at the liquid surface.

        :param electron_times: arrival times at the liquid surface [ns].
        :return: extraction times [ns], int64.
        """
        delays = self.rng.exponential(float(self.electron_trapping_time), len(electron_times))
        return electron_times + delays.astype(np.int64)
