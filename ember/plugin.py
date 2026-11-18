"""Base plugin for all ember plugins (port of fuse's ``FuseBasePlugin``)."""

import logging

import numpy as np
import strax

logging.basicConfig(handlers=[logging.StreamHandler()])

export, __all__ = strax.exporter()


@export
class EmberBasePlugin(strax.Plugin):
    """Base plugin providing logging and a reproducible random number generator.

    Three seeding modes are supported:

    * ``deterministic_seed=True`` (default): the seed is derived from the run id and the
      lineage of the plugin, so the same input and configuration give the same output.
    * ``deterministic_seed=False`` and ``user_defined_random_seed`` set: fixed user seed.
    * ``deterministic_seed=False`` and no user seed: the seed is pulled from the OS.

    The config names and defaults are identical to fuse's ``FuseBasePlugin``, so ember plugins
    can be registered in a fuse context without config conflicts.
    """

    rechunk_on_save = False

    input_timeout = 900

    debug = strax.URLConfig(
        default=False,
        type=bool,
        track=False,
        help="Show debug informations",
    )

    deterministic_seed = strax.URLConfig(
        default=True,
        type=bool,
        help="Set the random seed from lineage and run_id, or pull the seed from the OS",
    )

    user_defined_random_seed = strax.URLConfig(
        default=None,
        help="Define the random seed manually. You need to set deterministic_seed to False",
    )

    def setup(self):
        super().setup()

        log = logging.getLogger(f"{self.__class__.__name__}")

        if self.debug:
            log.setLevel("DEBUG")
            log.debug(f"Running {self.__class__.__name__} version {self.__version__} in debug mode")
        else:
            log.setLevel("INFO")

        if self.deterministic_seed:
            if self.user_defined_random_seed is not None:
                log.warning(
                    "deterministic_seed is set to True. "
                    "The provided user_defined_random_seed will not be used!"
                )
            hash_string = strax.deterministic_hash((self.run_id, self.lineage))
            self.seed = int(hash_string.encode().hex(), 16)
            self.rng = np.random.default_rng(seed=self.seed)
            log.debug(f"Generating random numbers from deterministic seed {self.seed}")
        elif self.user_defined_random_seed is not None:
            seed_is_valid = (
                isinstance(self.user_defined_random_seed, int) and self.user_defined_random_seed > 0
            )
            if not seed_is_valid:
                raise ValueError("user_defined_random_seed must be a positive integer!")
            self.seed = self.user_defined_random_seed
            self.rng = np.random.default_rng(self.user_defined_random_seed)
            log.info(f"Generating random numbers with user defined seed {self.seed}")
        else:
            self.seed = None
            self.rng = np.random.default_rng()
            log.debug("Generating random numbers with seed pulled from OS")


@export
class EmberBaseDownChunkingPlugin(strax.DownChunkingPlugin, EmberBasePlugin):
    """Base plugin for ember plugins that split their output into several chunks.

    Port of fuse's ``FuseBaseDownChunkingPlugin``: a ``strax.DownChunkingPlugin`` (``compute``
    yields ``strax.Chunk`` objects, runs serially) with the seeding and logging of
    :class:`EmberBasePlugin`.
    """


@export
class EmberVerticalMergerPlugin(EmberBasePlugin):
    """Base plugin that concatenates its inputs along the first axis and sorts them by time.

    Port of fuse's ``VerticalMergerPlugin``. All inputs must have the same dtype.
    """

    save_when = strax.SaveWhen.TARGET

    def infer_dtype(self):
        incoming_dtypes = [self.deps[name].dtype_for(name) for name in sorted(self.depends_on)]
        if any(dtype != incoming_dtypes[0] for dtype in incoming_dtypes[1:]):
            raise ValueError(f"{self.__class__.__name__} can only merge data with the same dtype!")
        return incoming_dtypes[0]

    def compute(self, **kwargs):
        merged_data = np.concatenate([kwargs[data_kind] for data_kind in kwargs])
        return strax.sort_by_time(merged_data)
