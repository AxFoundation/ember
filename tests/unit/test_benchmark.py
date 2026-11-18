"""Tests of the end-to-end benchmark (ember.benchmark)."""

import json
import time

import numpy as np
import pandas as pd
import pytest
import strax
from hypothesis import given, settings
from hypothesis import strategies as st_hyp
from optional_packages import requires_abstrax

from ember import benchmark

NUMBER_OF_TEST_EVENTS = 5


@pytest.fixture(scope="module")
def benchmark_summary(tmp_path_factory) -> dict:
    work_folder = tmp_path_factory.mktemp("benchmark")
    csv_file = benchmark.write_beta_csv(NUMBER_OF_TEST_EVENTS, work_folder)
    return benchmark.benchmark_csv(csv_file, work_folder / "run")


def test_beta_instructions_known_case():
    instructions = benchmark.make_beta_instructions(100, tpc_radius=60.0, tpc_length=140.0)
    assert list(instructions.columns) == benchmark.INSTRUCTION_COLUMNS
    assert len(instructions) == 100
    assert np.array_equal(instructions["eventid"], np.arange(100))
    assert (instructions["ed"] == 40.0).all()
    assert (instructions["nestid"] == benchmark.BETA_NESTID).all()
    assert (instructions[["photons", "electrons"]] == -1).all().all()


def test_beta_instructions_are_reproducible():
    first = benchmark.make_beta_instructions(20, tpc_radius=60.0, tpc_length=140.0, seed=3)
    second = benchmark.make_beta_instructions(20, tpc_radius=60.0, tpc_length=140.0, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_beta_instructions_need_events():
    with pytest.raises(ValueError):
        benchmark.make_beta_instructions(0, tpc_radius=60.0, tpc_length=140.0)


@settings(deadline=None, max_examples=50)
@given(
    number_of_events=st_hyp.integers(1, 500),
    tpc_radius=st_hyp.floats(1.0, 500.0),
    tpc_length=st_hyp.floats(1.0, 500.0),
    energy=st_hyp.floats(0.1, 1e4),
    seed=st_hyp.integers(0, 2**32 - 1),
)
def test_beta_instructions_stay_inside_the_tpc(
    number_of_events, tpc_radius, tpc_length, energy, seed
):
    instructions = benchmark.make_beta_instructions(
        number_of_events, tpc_radius, tpc_length, energy, seed
    )
    radius = np.hypot(instructions["x"], instructions["y"])
    assert len(instructions) == number_of_events
    assert (radius <= tpc_radius * (1 + 1e-12)).all()
    assert ((instructions["z"] <= 0) & (instructions["z"] >= -tpc_length)).all()
    assert np.allclose(instructions["ed"], energy)


def test_beta_csv_uses_the_detector_dimensions(tmp_path):
    csv_file = benchmark.write_beta_csv(200, tmp_path, energy=10.0)
    instructions = pd.read_csv(csv_file)
    tpc_radius = float(strax.URLConfig.evaluate_dry("detector://tpc_radius"))
    tpc_length = float(strax.URLConfig.evaluate_dry("detector://tpc_length"))
    assert len(instructions) == 200
    assert (np.hypot(instructions["x"], instructions["y"]) <= tpc_radius).all()
    assert (instructions["z"] >= -tpc_length).all()
    assert (instructions["ed"] == 10.0).all()


@pytest.mark.parametrize(
    "with_abstrax, final_target",
    [
        (False, benchmark.EMBER_FINAL_TARGET),
        pytest.param(True, benchmark.ABSTRAX_FINAL_TARGET, marks=requires_abstrax),
    ],
)
def test_stages_are_in_dependency_order(tmp_path, with_abstrax, final_target):
    context = benchmark.build_context(tmp_path, benchmark.write_beta_csv(1, tmp_path), with_abstrax)
    stages = benchmark.order_stages(context, benchmark.RUN_ID, final_target)
    provided_data_types = [data_type for stage in stages for data_type in stage]
    assert stages[0] == ("clustered_interactions",)
    assert final_target in stages[-1]
    assert len(provided_data_types) == len(set(provided_data_types))
    for index, stage in enumerate(stages):
        made_before = {data_type for earlier in stages[:index] for data_type in earlier}
        plugin = context.get_single_plugin(benchmark.RUN_ID, stage[0])
        assert set(strax.to_str_tuple(plugin.depends_on)) <= made_before


def test_multi_output_plugins_form_one_stage(tmp_path):
    context = benchmark.build_context(tmp_path, benchmark.write_beta_csv(1, tmp_path))
    stages = benchmark.order_stages(context, benchmark.RUN_ID, benchmark.EMBER_FINAL_TARGET)
    assert set(("s2_photons", "s2_photons_sum")) in [set(stage) for stage in stages]


@pytest.mark.skipif(not benchmark.peak_rss_is_resettable(), reason="needs Linux /proc")
def test_peak_memory_sees_a_large_allocation():
    allocation_mb = 200

    def allocate():
        array = np.ones(allocation_mb * benchmark.BYTES_PER_MB // 8)
        del array

    wall_time_s, rss_at_start_mb, peak_rss_mb = benchmark.measure(allocate)
    assert wall_time_s > 0
    assert peak_rss_mb - rss_at_start_mb >= 0.9 * allocation_mb


@pytest.mark.skipif(not benchmark.peak_rss_is_resettable(), reason="needs Linux /proc")
def test_reset_lowers_the_peak_memory():
    array = np.ones(200 * benchmark.BYTES_PER_MB // 8)
    del array
    peak_before_reset_mb = benchmark.read_peak_rss_mb()
    benchmark.reset_peak_rss()
    assert benchmark.read_peak_rss_mb() < peak_before_reset_mb
    assert benchmark.read_current_rss_mb() <= benchmark.read_peak_rss_mb()


def test_measure_returns_the_wall_time():
    wall_time_s, _, _ = benchmark.measure(lambda: time.sleep(0.05))
    assert 0.05 <= wall_time_s < 1.0


def test_folder_size(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.bin").write_bytes(b"x" * 10)
    (tmp_path / "sub" / "b.bin").write_bytes(b"x" * 32)
    assert benchmark.folder_size(tmp_path) == 42


def test_benchmark_reports_every_plugin(benchmark_summary):
    stages = benchmark_summary["stages"]
    names = [stage.name for stage in stages]
    assert benchmark_summary["number_of_events"] == NUMBER_OF_TEST_EVENTS
    assert benchmark_summary["final_target"] == benchmark.EMBER_FINAL_TARGET
    assert names[0] == "CsvInput"
    assert names[-1] == "PMTResponseAndDAQ"
    assert len(names) == len(set(names))
    for stage in stages:
        assert stage.wall_time_s > 0
        assert stage.peak_rss_mb >= stage.rss_at_start_mb > 0


def test_benchmark_reports_the_stored_data(benchmark_summary):
    raw_records_stage = benchmark_summary["stages"][-1]
    assert raw_records_stage.n_rows["raw_records"] > 0
    assert raw_records_stage.n_chunks["raw_records"] >= 1
    assert raw_records_stage.nbytes_on_disk["raw_records"] > 0
    assert raw_records_stage.nbytes_in_memory["raw_records"] > 0


def test_total_and_stages_give_the_same_raw_records(benchmark_summary):
    total = benchmark_summary["total"]
    raw_records_stage = benchmark_summary["stages"][-1]
    assert total.name == "total"
    assert total.n_rows["raw_records"] == raw_records_stage.n_rows["raw_records"]


def test_report_and_json(benchmark_summary):
    environment = benchmark.collect_environment()
    environment["warmup_time_s"] = 1.0
    report = benchmark.format_report([benchmark_summary], environment)
    assert "## Total" in report
    assert "PMTResponseAndDAQ" in report
    assert "Warm-up" in report
    results = json.loads(json.dumps(benchmark.summaries_to_json([benchmark_summary], environment)))
    assert results["results"][0]["total"]["data_types"] == ["raw_records"]
    assert results["environment"]["packages"]["ember"] is not None


def test_unknown_mode_raises(tmp_path):
    csv_file = benchmark.write_beta_csv(1, tmp_path)
    with pytest.raises(ValueError, match="Unknown modes"):
        benchmark.benchmark_csv(csv_file, tmp_path / "run", modes=("fast",))


@pytest.mark.parametrize("value, expected", [(None, "-"), (0, "0"), (1234567, "1,234,567")])
def test_format_rows(value, expected):
    assert benchmark.format_rows(value) == expected


@pytest.mark.parametrize("value, expected", [(None, "-"), (0, "0.0"), (3 * 1024**2, "3.0")])
def test_format_size_mb(value, expected):
    assert benchmark.format_size_mb(value) == expected


def test_main_writes_results_and_removes_the_data(tmp_path, capsys):
    work_dir = tmp_path / "work"
    output_file = tmp_path / "results.json"
    report_file = tmp_path / "report.md"
    summaries = benchmark.main(
        [
            "--events", "3",
            "--modes", "total",
            "--no-warmup",
            "--work-dir", str(work_dir),
            "--output", str(output_file),
            "--report", str(report_file),
        ]
    )  # fmt: skip
    assert len(summaries) == 1
    assert not work_dir.exists()
    assert json.loads(output_file.read_text())["results"][0]["number_of_events"] == 3
    assert report_file.read_text() in capsys.readouterr().out


def test_main_keeps_the_data_on_request(tmp_path):
    work_dir = tmp_path / "work"
    benchmark.main(
        [
            "--events",
            "2",
            "--modes",
            "total",
            "--no-warmup",
            "--keep-data",
            "--work-dir",
            str(work_dir),
        ]
    )
    assert any(work_dir.rglob("*raw_records*"))
