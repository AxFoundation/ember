"""End-to-end performance benchmark of the ember simulation chain.

Run it from the command line::

    python -m ember.benchmark --events 100 1000
    python -m ember.benchmark --events 1000 --with-abstrax --output results.json

For each number of events, beta instructions (default 40 keV) are generated uniformly in the TPC
and simulated from the CSV file up to ``raw_records`` (with ``--with-abstrax`` further up to
abstrax's ``event_info``). Two modes are measured:

* ``stages``: every plugin is made as target in dependency order. Each stage reads its inputs from
  the data stored by the previous stages, so its time includes loading them. This gives the time,
  peak memory, number of rows and size on disk per plugin.
* ``total``: one ``make`` of the final target in a fresh folder. This is the real end-to-end time
  and peak memory; only the data types strax saves by default are written.

The peak memory (resident set size, RSS) is read from ``/proc/self/status`` (Linux) and reset before
every measurement through ``/proc/self/clear_refs``. On other systems the process-wide peak is
reported, which cannot be reset and therefore only grows.

By default a warm-up run with a few events is done first, so numba compilation and the loading of
maps are not part of the measured times.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import strax

import ember  # noqa: F401  (activates ember's detector file for detector://)
from ember.contexts import ember_context

RUN_ID = "000000"
EMBER_FINAL_TARGET = "raw_records"
ABSTRAX_FINAL_TARGET = "event_info"
BETA_NESTID = 8
WARMUP_NUMBER_OF_EVENTS = 10
INSTRUCTION_COLUMNS = [
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
PROC_STATUS_FILE = Path("/proc/self/status")
PROC_CLEAR_REFS_FILE = Path("/proc/self/clear_refs")
BYTES_PER_MB = 1024**2
PACKAGES_TO_REPORT = ["ember", "abstrax", "strax", "numba", "numpy", "scipy", "nestpy", "pandas"]


@dataclass
class StageResult:
    """Measurement of one stage (one plugin, or the full chain in mode ``total``).

    :param name: plugin class name, or ``total``.
    :param data_types: data types made in this stage.
    :param wall_time_s: wall-clock time of the ``make`` call [s].
    :param rss_at_start_mb: resident memory when the stage starts [MB].
    :param peak_rss_mb: peak resident memory during the stage [MB].
    :param n_rows: rows per data type (``None`` if the data type was not stored).
    :param nbytes_in_memory: uncompressed size per data type [bytes].
    :param nbytes_on_disk: compressed size on disk per data type [bytes].
    :param n_chunks: number of chunks per data type.
    """

    name: str
    data_types: Tuple[str, ...]
    wall_time_s: float
    rss_at_start_mb: float
    peak_rss_mb: float
    n_rows: Dict[str, Optional[int]] = field(default_factory=dict)
    nbytes_in_memory: Dict[str, Optional[int]] = field(default_factory=dict)
    nbytes_on_disk: Dict[str, Optional[int]] = field(default_factory=dict)
    n_chunks: Dict[str, Optional[int]] = field(default_factory=dict)


def make_beta_instructions(
    number_of_events: int,
    tpc_radius: float,
    tpc_length: float,
    energy: float = 40.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Single-interaction beta events, uniformly distributed in the TPC volume.

    :param number_of_events: number of events (one interaction each).
    :param tpc_radius: radius of the TPC [cm].
    :param tpc_length: distance from the gate (z = 0) to the cathode [cm].
    :param energy: deposited energy per event [keV].
    :param seed: seed of the random generator.
    :return: instructions with the columns of ember's CSV format.
    """
    if number_of_events < 1:
        raise ValueError("number_of_events must be at least 1.")
    rng = np.random.default_rng(seed)
    radius = tpc_radius * np.sqrt(rng.uniform(0, 1, number_of_events))
    angle = rng.uniform(-np.pi, np.pi, number_of_events)
    return pd.DataFrame(
        {
            "eventid": np.arange(number_of_events),
            "t": np.zeros(number_of_events, dtype=int),
            "x": radius * np.cos(angle),
            "y": radius * np.sin(angle),
            "z": -tpc_length * rng.uniform(0, 1, number_of_events),
            "ed": np.full(number_of_events, float(energy)),
            "nestid": np.full(number_of_events, BETA_NESTID),
            "A": np.zeros(number_of_events, dtype=int),
            "Z": np.zeros(number_of_events, dtype=int),
            "photons": np.full(number_of_events, -1),
            "electrons": np.full(number_of_events, -1),
        },
        columns=INSTRUCTION_COLUMNS,
    )


def write_beta_csv(
    number_of_events: int, folder: Path, energy: float = 40.0, seed: int = 0
) -> Path:
    """Write beta instructions with the TPC dimensions of the active detector file.

    :param number_of_events: number of events.
    :param folder: folder of the CSV file.
    :param energy: deposited energy per event [keV].
    :param seed: seed of the random generator.
    :return: path of the CSV file.
    """
    instructions = make_beta_instructions(
        number_of_events,
        tpc_radius=float(strax.URLConfig.evaluate_dry("detector://tpc_radius")),
        tpc_length=float(strax.URLConfig.evaluate_dry("detector://tpc_length")),
        energy=energy,
        seed=seed,
    )
    path = Path(folder) / f"beta_{energy:g}keV_{number_of_events}.csv"
    instructions.to_csv(path, index=False)
    return path


def build_context(output_folder: Path, csv_file: Path, with_abstrax: bool = False) -> strax.Context:
    """ember context reading ``csv_file``, optionally with abstrax's plugins registered.

    :param output_folder: storage folder.
    :param csv_file: CSV instruction file.
    :param with_abstrax: register abstrax's plugins to process the ``raw_records`` further. Needs
        the optional package abstrax.
    :return: the context.
    """
    context = ember_context(output_folder=output_folder)
    context.set_config({"input_file": str(csv_file)})
    if with_abstrax:
        try:
            from abstrax.contexts import ABSTRAX_PLUGINS
        except ImportError as err:
            raise ImportError(
                "with_abstrax needs the optional package abstrax: pip install 'ember[abstrax]'"
            ) from err

        context.register(ABSTRAX_PLUGINS)
        context.set_config({"gain_model": "ember_pmt_to_pe://"})
    return context


def order_stages(context: strax.Context, run_id: str, final_target: str) -> List[Tuple[str, ...]]:
    """Plugins needed for ``final_target``, in dependency order.

    :param context: the context.
    :param run_id: run id.
    :param final_target: last data type of the chain.
    :return: one tuple of provided data types per plugin; every plugin comes after the plugins
        it depends on, and the last one provides ``final_target``.
    """
    ordered_stages: List[Tuple[str, ...]] = []
    visited_data_types: set = set()

    def visit(data_type: str) -> None:
        if data_type in visited_data_types:
            return
        plugin = context.get_single_plugin(run_id, data_type)
        provided_data_types = strax.to_str_tuple(plugin.provides)
        visited_data_types.update(provided_data_types)
        for dependency in strax.to_str_tuple(plugin.depends_on):
            visit(dependency)
        ordered_stages.append(provided_data_types)

    visit(final_target)
    return ordered_stages


def peak_rss_is_resettable() -> bool:
    """Whether the peak memory can be reset (Linux ``/proc`` interface).

    :return: True on Linux with ``/proc/self/clear_refs`` writable.
    """
    if not (PROC_STATUS_FILE.exists() and PROC_CLEAR_REFS_FILE.exists()):
        return False
    try:
        PROC_CLEAR_REFS_FILE.write_text("5")
    except OSError:
        return False
    return True


def reset_peak_rss() -> None:
    """Reset the peak memory of this process to its current memory (Linux only, else no-op)."""
    try:
        PROC_CLEAR_REFS_FILE.write_text("5")
    except OSError:
        pass


def read_current_rss_mb() -> float:
    """Current resident memory of this process [MB], 0 if unknown (no ``/proc``).

    :return: VmRSS from ``/proc/self/status``.
    """
    if PROC_STATUS_FILE.exists():
        match = re.search(r"VmRSS:\s+(\d+)\s+kB", PROC_STATUS_FILE.read_text())
        if match:
            return int(match.group(1)) * 1024 / BYTES_PER_MB
    return 0.0


def read_peak_rss_mb() -> float:
    """Peak resident memory of this process since the last reset [MB].

    :return: VmHWM from ``/proc/self/status``, or the process-wide maximum from ``resource``.
    """
    if PROC_STATUS_FILE.exists():
        match = re.search(r"VmHWM:\s+(\d+)\s+kB", PROC_STATUS_FILE.read_text())
        if match:
            return int(match.group(1)) * 1024 / BYTES_PER_MB
    max_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return max_rss_kb * 1024 / BYTES_PER_MB


def measure(action: Callable[[], None]) -> Tuple[float, float, float]:
    """Run ``action`` and measure its wall time and memory.

    :param action: function without arguments.
    :return: wall time [s], resident memory at the start [MB] and peak resident memory [MB].
    """
    reset_peak_rss()
    rss_at_start_mb = read_current_rss_mb()
    start = time.perf_counter()
    action()
    wall_time_s = time.perf_counter() - start
    return wall_time_s, rss_at_start_mb, read_peak_rss_mb()


def folder_size(folder: Path) -> int:
    """Total size of the files in ``folder`` [bytes].

    :param folder: folder.
    :return: sum of the file sizes.
    """
    return sum(path.stat().st_size for path in Path(folder).rglob("*") if path.is_file())


def describe_stored_data(
    context: strax.Context, run_id: str, data_type: str, output_folder: Path
) -> Dict[str, Optional[int]]:
    """Rows, chunks and sizes of stored data, all ``None`` if the data type is not stored.

    :param context: the context.
    :param run_id: run id.
    :param data_type: data type.
    :param output_folder: storage folder of the context.
    :return: dict with ``n_rows``, ``n_chunks``, ``nbytes_in_memory`` and ``nbytes_on_disk``.
    """
    if not context.is_stored(run_id, data_type):
        return {"n_rows": None, "n_chunks": None, "nbytes_in_memory": None, "nbytes_on_disk": None}
    chunks = context.get_metadata(run_id, data_type)["chunks"]
    key = context.key_for(run_id, data_type)
    data_folder = Path(output_folder) / f"{run_id}-{data_type}-{key.lineage_hash}"
    return {
        "n_rows": int(sum(chunk["n"] for chunk in chunks)),
        "n_chunks": len(chunks),
        "nbytes_in_memory": int(sum(chunk["nbytes"] for chunk in chunks)),
        "nbytes_on_disk": folder_size(data_folder) if data_folder.exists() else None,
    }


def add_stored_data(
    result: StageResult, context: strax.Context, run_id: str, output_folder: Path
) -> StageResult:
    """Fill the rows and sizes of ``result.data_types`` from the storage.

    :param result: stage result to fill.
    :param context: the context.
    :param run_id: run id.
    :param output_folder: storage folder of the context.
    :return: the same result.
    """
    for data_type in result.data_types:
        description = describe_stored_data(context, run_id, data_type, output_folder)
        result.n_rows[data_type] = description["n_rows"]
        result.n_chunks[data_type] = description["n_chunks"]
        result.nbytes_in_memory[data_type] = description["nbytes_in_memory"]
        result.nbytes_on_disk[data_type] = description["nbytes_on_disk"]
    return result


def make_each(context: strax.Context, run_id: str, data_types: Sequence[str]) -> None:
    """Make the data types one after the other.

    strax cannot make outputs of different data kinds in one call. For a multi-output plugin the
    first call stores all outputs that strax saves by default, and the next calls load them.

    :param context: the context.
    :param run_id: run id.
    :param data_types: data types to make.
    """
    for data_type in data_types:
        context.make(run_id, data_type, progress_bar=False)


def run_stages(
    context: strax.Context, run_id: str, final_target: str, output_folder: Path
) -> List[StageResult]:
    """Make every plugin of the chain as target, in dependency order.

    :param context: context with an empty storage folder.
    :param run_id: run id.
    :param final_target: last data type of the chain.
    :param output_folder: storage folder of the context.
    :return: one result per plugin.
    """
    results = []
    for data_types in order_stages(context, run_id, final_target):
        plugin_name = type(context.get_single_plugin(run_id, data_types[0])).__name__
        measurement = measure(lambda: make_each(context, run_id, data_types))
        result = StageResult(plugin_name, data_types, *measurement)
        results.append(add_stored_data(result, context, run_id, output_folder))
    return results


def run_total(
    context: strax.Context, run_id: str, final_target: str, output_folder: Path
) -> StageResult:
    """Make ``final_target`` in one go.

    :param context: context with an empty storage folder.
    :param run_id: run id.
    :param final_target: last data type of the chain.
    :param output_folder: storage folder of the context.
    :return: result of the full chain (rows and sizes of ``final_target``).
    """
    measurement = measure(lambda: context.make(run_id, final_target, progress_bar=False))
    result = StageResult("total", (final_target,), *measurement)
    return add_stored_data(result, context, run_id, output_folder)


def benchmark_csv(
    csv_file: Path,
    work_folder: Path,
    modes: Sequence[str] = ("stages", "total"),
    with_abstrax: bool = False,
) -> Dict[str, object]:
    """Benchmark the chain for one CSV file.

    :param csv_file: CSV instruction file.
    :param work_folder: folder for the storage folders of the modes (must be empty or new).
    :param modes: ``stages`` and/or ``total``.
    :param with_abstrax: continue with abstrax up to ``event_info``.
    :return: dict with the number of events and interactions and the results per mode.
    """
    unknown_modes = set(modes) - {"stages", "total"}
    if unknown_modes:
        raise ValueError(f"Unknown modes {sorted(unknown_modes)}, use 'stages' and/or 'total'.")
    final_target = ABSTRAX_FINAL_TARGET if with_abstrax else EMBER_FINAL_TARGET
    instructions = pd.read_csv(csv_file)
    summary: Dict[str, object] = {
        "csv_file": str(csv_file),
        "number_of_events": int(instructions["eventid"].nunique()),
        "number_of_interactions": len(instructions),
        "final_target": final_target,
    }
    if "stages" in modes:
        output_folder = Path(work_folder) / "stages"
        context = build_context(output_folder, csv_file, with_abstrax)
        summary["stages"] = run_stages(context, RUN_ID, final_target, output_folder)
    if "total" in modes:
        output_folder = Path(work_folder) / "total"
        context = build_context(output_folder, csv_file, with_abstrax)
        summary["total"] = run_total(context, RUN_ID, final_target, output_folder)
    return summary


def run_warmup(work_folder: Path, with_abstrax: bool) -> float:
    """Simulate a few events so that numba compiles and the maps are loaded.

    :param work_folder: folder for the warm-up data (deleted afterwards).
    :param with_abstrax: also warm up abstrax's plugins.
    :return: wall time of the warm-up [s].
    """
    warmup_folder = Path(work_folder) / "warmup"
    warmup_folder.mkdir(parents=True)
    csv_file = write_beta_csv(WARMUP_NUMBER_OF_EVENTS, warmup_folder)
    start = time.perf_counter()
    benchmark_csv(csv_file, warmup_folder, modes=("total",), with_abstrax=with_abstrax)
    wall_time_s = time.perf_counter() - start
    shutil.rmtree(warmup_folder)
    return wall_time_s


def collect_environment() -> Dict[str, object]:
    """Machine and package versions, to compare benchmarks run on different setups.

    :return: dict with platform, CPU count, memory and package versions.
    """
    package_versions = {}
    for package in PACKAGES_TO_REPORT:
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = None
    total_memory_gb = None
    meminfo_file = Path("/proc/meminfo")
    if meminfo_file.exists():
        match = re.search(r"MemTotal:\s+(\d+)\s+kB", meminfo_file.read_text())
        if match:
            total_memory_gb = round(int(match.group(1)) / 1024**2, 1)
    return {
        "date": datetime.now().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "total_memory_gb": total_memory_gb,
        "peak_rss_is_resettable": peak_rss_is_resettable(),
        "packages": package_versions,
    }


def format_size_mb(nbytes: Optional[int]) -> str:
    """Bytes as MB with one decimal, ``-`` if unknown.

    :param nbytes: size in bytes or None.
    :return: formatted size.
    """
    return "-" if nbytes is None else f"{nbytes / BYTES_PER_MB:.1f}"


def format_rows(n_rows: Optional[int]) -> str:
    """Number of rows with thousands separators, ``-`` if unknown.

    :param n_rows: number of rows or None.
    :return: formatted number.
    """
    return "-" if n_rows is None else f"{n_rows:,}"


def format_stages_table(stages: Sequence[StageResult]) -> str:
    """Markdown table of the stage results, one row per data type.

    :param stages: results of mode ``stages``.
    :return: the table.
    """
    lines = [
        "| plugin | data type | time [s] | peak RSS [MB] | peak - start [MB] | rows | chunks "
        "| memory [MB] | disk [MB] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in stages:
        for index, data_type in enumerate(stage.data_types):
            is_first_data_type = index == 0
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
                    stage.name if is_first_data_type else "",
                    data_type,
                    f"{stage.wall_time_s:.2f}" if is_first_data_type else "",
                    f"{stage.peak_rss_mb:.0f}" if is_first_data_type else "",
                    (
                        f"{stage.peak_rss_mb - stage.rss_at_start_mb:.0f}"
                        if is_first_data_type
                        else ""
                    ),
                    format_rows(stage.n_rows.get(data_type)),
                    format_rows(stage.n_chunks.get(data_type)),
                    format_size_mb(stage.nbytes_in_memory.get(data_type)),
                    format_size_mb(stage.nbytes_on_disk.get(data_type)),
                )
            )
    total_time_s = sum(stage.wall_time_s for stage in stages)
    max_peak_rss_mb = max(stage.peak_rss_mb for stage in stages)
    lines.append(
        f"| **sum** | | **{total_time_s:.2f}** | **{max_peak_rss_mb:.0f}** (max) | | | | | |"
    )
    return "\n".join(lines)


def format_total_table(summaries: Sequence[Dict[str, object]]) -> str:
    """Markdown table of the ``total`` results, one row per number of events.

    :param summaries: outputs of :func:`benchmark_csv` containing mode ``total``.
    :return: the table.
    """
    lines = [
        "| events | interactions | final target | time [s] | events/s | peak RSS [MB] "
        "| peak - start [MB] | rows | disk [MB] |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        total = summary["total"]
        final_target = summary["final_target"]
        events_per_second = summary["number_of_events"] / total.wall_time_s
        lines.append(
            f"| {summary['number_of_events']:,} | {summary['number_of_interactions']:,} "
            f"| {final_target} | {total.wall_time_s:.2f} | {events_per_second:.1f} "
            f"| {total.peak_rss_mb:.0f} | {total.peak_rss_mb - total.rss_at_start_mb:.0f} "
            f"| {format_rows(total.n_rows.get(final_target))} "
            f"| {format_size_mb(total.nbytes_on_disk.get(final_target))} |"
        )
    return "\n".join(lines)


def format_report(summaries: Sequence[Dict[str, object]], environment: Dict[str, object]) -> str:
    """Markdown report of all benchmark results.

    :param summaries: outputs of :func:`benchmark_csv`.
    :param environment: output of :func:`collect_environment`.
    :return: the report.
    """
    packages = ", ".join(
        f"{name} {version}" for name, version in environment["packages"].items() if version
    )
    sections = [
        "# ember end-to-end benchmark",
        f"{environment['date']}, {environment['platform']}, {environment['cpu_count']} CPUs, "
        f"{environment['total_memory_gb']} GB RAM, Python {environment['python']}",
        packages,
    ]
    if not environment["peak_rss_is_resettable"]:
        sections.append("Peak RSS cannot be reset on this system: values are process-wide maxima.")
    if environment.get("warmup_time_s") is not None:
        sections.append(
            f"Warm-up ({WARMUP_NUMBER_OF_EVENTS} events): {environment['warmup_time_s']:.1f} s"
        )
    summaries_with_total = [summary for summary in summaries if "total" in summary]
    if summaries_with_total:
        sections += [
            "## Total (one make of the final target)",
            format_total_table(summaries_with_total),
        ]
    for summary in summaries:
        if "stages" in summary:
            sections += [
                f"## Stages, {summary['number_of_events']:,} events "
                f"({summary['number_of_interactions']:,} interactions)",
                format_stages_table(summary["stages"]),
            ]
    return "\n\n".join(sections) + "\n"


def summaries_to_json(
    summaries: Sequence[Dict[str, object]], environment: Dict[str, object]
) -> Dict[str, object]:
    """JSON-serialisable form of the results.

    :param summaries: outputs of :func:`benchmark_csv`.
    :param environment: output of :func:`collect_environment`.
    :return: dict with ``environment`` and ``results``.
    """
    results = []
    for summary in summaries:
        entry = dict(summary)
        if "stages" in entry:
            entry["stages"] = [asdict(stage) for stage in entry["stages"]]
        if "total" in entry:
            entry["total"] = asdict(entry["total"])
        results.append(entry)
    return {"environment": environment, "results": results}


def parse_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Command line options.

    :param argv: arguments (default: ``sys.argv``).
    :return: parsed options.
    """
    parser = argparse.ArgumentParser(
        prog="python -m ember.benchmark",
        description="End-to-end performance benchmark of ember (CSV -> raw_records).",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--events",
        type=int,
        nargs="+",
        default=[100, 1000],
        help="Numbers of beta events to simulate (default: 100 1000)",
    )
    input_group.add_argument("--csv", type=Path, nargs="+", help="Use these CSV files instead")
    parser.add_argument("--energy", type=float, default=40.0, help="Beta energy [keV]")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the generated events")
    parser.add_argument(
        "--modes", nargs="+", default=["stages", "total"], choices=["stages", "total"]
    )
    parser.add_argument(
        "--with-abstrax", action="store_true", help="Process further with abstrax to event_info"
    )
    parser.add_argument("--no-warmup", action="store_true", help="Include numba compilation")
    parser.add_argument(
        "--work-dir", type=Path, default=None, help="Folder for the data (default: temporary)"
    )
    parser.add_argument("--keep-data", action="store_true", help="Do not delete the data")
    parser.add_argument("--output", type=Path, default=None, help="Write the results as JSON")
    parser.add_argument("--report", type=Path, default=None, help="Write the Markdown report")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
    """Run the benchmark from the command line.

    :param argv: arguments (default: ``sys.argv``).
    :return: outputs of :func:`benchmark_csv`, one per input.
    """
    arguments = parse_arguments(argv)
    work_dir = arguments.work_dir or Path(tempfile.mkdtemp(prefix="ember_benchmark_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        environment = collect_environment()
        environment["warmup_time_s"] = (
            None if arguments.no_warmup else run_warmup(work_dir, arguments.with_abstrax)
        )
        if arguments.csv:
            csv_files = arguments.csv
        else:
            csv_folder = work_dir / "instructions"
            csv_folder.mkdir(exist_ok=True)
            csv_files = [
                write_beta_csv(number_of_events, csv_folder, arguments.energy, arguments.seed)
                for number_of_events in arguments.events
            ]
        summaries = []
        for csv_file in csv_files:
            run_folder = work_dir / Path(csv_file).stem
            summaries.append(
                benchmark_csv(csv_file, run_folder, arguments.modes, arguments.with_abstrax)
            )
        report = format_report(summaries, environment)
        print(report)
        if arguments.report:
            arguments.report.write_text(report)
        if arguments.output:
            arguments.output.write_text(
                json.dumps(summaries_to_json(summaries, environment), indent=2)
            )
    finally:
        if arguments.keep_data:
            print(f"Data kept in {work_dir}")
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
    return summaries


if __name__ == "__main__":
    main()
