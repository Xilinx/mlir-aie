# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The SA placer component checks, with the real aie-opt; no NPU required."""

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / "utils/component_checks"
SMALL = ROOT / "test/place-tiles/sa_placer/test_sa_effort.mlir"
AIE_OPT = shutil.which("aie-opt")
POSIX_SWEEP = pytest.mark.skipif(
    sys.platform == "win32",
    reason="sa_placer.py measures each seed with os.wait4, which is POSIX-only",
)

# What aie2_mobilenet_iron.py printed on an npu2, verbatim, at batch 1 and 16.
MOBILENET_OUTPUT = """\
NPU time     (avg/min/max us): 399.9 / 384.4 / 420.2   [median 399.6, MAD 13.3, p95 416.1]
End-to-end   (avg/min/max us): 499.7 / 480.9 / 533.8   [median 498.4, MAD 13.2, p95 519.2]
max_difference: 2
PASS!
"""
MOBILENET_B16_OUTPUT = """\
NPU time     (avg/min/max us): 1343.6 / 1303.7 / 1475.0   [median 1343.7, MAD 22.9, p95 1380.9]
End-to-end   (avg/min/max us): 1681.5 / 1431.5 / 2075.9   [median 1737.1, MAD 220.9, p95 1985.1]
max_difference: 2
PASS!
"""


def load(name):
    spec = importlib.util.spec_from_file_location(name, CHECKS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def aie_opt():
    assert AIE_OPT, "aie-opt is not on PATH"
    return AIE_OPT


@pytest.fixture(scope="module")
def hw():
    return load("sa_placer_hw")


def sweep(tmp_path, *args):
    out, meta = tmp_path / "perf.json", tmp_path / "meta.json"
    result = subprocess.run(
        [sys.executable, CHECKS / "sa_placer.py", *map(str, args)]
        + ["--out", out, "--meta", meta],
        capture_output=True,
        text=True,
    )
    return result, json.loads(out.read_text()), json.loads(meta.read_text())


@POSIX_SWEEP
def test_the_sweep_records_every_seed_of_every_fixture(aie_opt, tmp_path):
    result, rows, meta = sweep(
        tmp_path, "--aie-opt", aie_opt, "--seeds", 2, "--fixture", SMALL
    )
    assert result.returncode == 0, result.stdout + result.stderr
    by_name = {r["name"]: r for r in rows}
    assert sorted(by_name) == [
        f"test_sa_effort/{m}"
        for m in (
            "cpu_ms_max",
            "cpu_ms_mean",
            "failed_seeds",
            "final_cost_max",
            "final_cost_mean",
            "peak_rss_mb_max",
        )
    ]
    assert by_name["test_sa_effort/failed_seeds"]["value"] == 0
    assert by_name["test_sa_effort/final_cost_max"] == {
        "name": "test_sa_effort/final_cost_max",
        "unit": "cost",
        "value": 12,
    }
    assert by_name["test_sa_effort/cpu_ms_max"]["value"] > 0
    assert meta["measurement_sane"] is True and meta["exitstatus"] == 0
    assert meta["failed"] == [] and meta["seeds"] == 2 and meta["effort"] == 1.0
    assert meta["provenance"].split(" | ")[1].startswith("fixtures ")
    assert [(d["fixture"], d["seed"], d["passed"]) for d in meta["detail"]] == [
        ("test_sa_effort", 1, True),
        ("test_sa_effort", 2, True),
    ]
    assert max(d["final_cost"] for d in meta["detail"]) == 12
    assert all(d["cpu_ms"] > 0 and d["peak_rss_mb"] > 0 for d in meta["detail"])
    assert [line.split()[:3] for line in result.stdout.splitlines()[1:]] == [
        ["test_sa_effort", "1", "yes"],
        ["test_sa_effort", "2", "yes"],
    ]


@POSIX_SWEEP
def test_a_seed_that_does_not_place_fails_the_sweep(aie_opt, tmp_path):
    bad = tmp_path / "unplaceable.mlir"
    bad.write_text("module { this is not MLIR }\n")
    result, rows, meta = sweep(
        tmp_path,
        "--aie-opt",
        aie_opt,
        "--seeds",
        1,
        "--fixture",
        SMALL,
        "--fixture",
        bad,
    )
    assert result.returncode == 1
    assert meta["failed"] == ["unplaceable/seed1"] and meta["exitstatus"] == 1
    by_name = {r["name"]: r["value"] for r in rows}
    assert by_name["test_sa_effort/failed_seeds"] == 0
    assert by_name["unplaceable/failed_seeds"] == 1
    # A failed seed has no cost to average.
    assert "unplaceable/final_cost_mean" not in by_name
    assert {d["fixture"]: (d["passed"], d["final_cost"]) for d in meta["detail"]} == {
        "test_sa_effort": (True, 12),
        "unplaceable": (False, None),
    }
    table = [line.split()[:3] for line in result.stdout.splitlines()]
    assert ["unplaceable", "1", "NO"] in table
    # The table carries aie-opt's error, so a regression is debuggable from the log.
    assert "custom op 'this' is unknown" in result.stdout


def test_the_fixture_digest_follows_the_fixtures_content(tmp_path):
    sa_placer = load("sa_placer")
    copy = tmp_path / SMALL.name
    copy.write_text(SMALL.read_text())
    same = sa_placer.fixtures_digest([str(SMALL)])
    assert sa_placer.fixtures_digest([str(copy)]) == same
    copy.write_text(SMALL.read_text() + "\n")
    assert sa_placer.fixtures_digest([str(copy)]) != same


@pytest.mark.parametrize("seeds", ["0", "-3", "two"])
def test_the_sweep_refuses_a_seed_count_that_is_not_positive(seeds, tmp_path):
    result = subprocess.run(
        [sys.executable, CHECKS / "sa_placer.py", "--aie-opt", "aie-opt"]
        + ["--seeds", seeds, "--out", tmp_path / "p", "--meta", tmp_path / "m"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "--seeds" in result.stderr
    assert not (tmp_path / "p").exists()


def test_the_mobilenet_fixture_is_still_valid_and_unplaced(aie_opt):
    fixture = CHECKS / "fixtures/mobilenet.mlir"
    result = subprocess.run(
        [aie_opt, fixture], capture_output=True, text=True, check=True
    )
    assert "aie.logical_tile" in result.stdout and "aie.tile(" not in result.stdout


def test_a_mobilenet_run_reports_its_times_per_image(hw):
    assert hw.parse_run(MOBILENET_B16_OUTPUT, 0, 16) == {
        "passed": True,
        "launch_us": 1303.7,
        "us_per_image": 81.48,
        "us_per_image_range": "median 83.98, MAD 1.43, p95 86.31",
        "e2e_us_per_image": 89.47,
        "e2e_range": "median 108.57, MAD 13.81, p95 124.07",
    }


@pytest.mark.parametrize(
    "out, returncode",
    [
        (MOBILENET_OUTPUT, 1),
        (MOBILENET_OUTPUT.replace("PASS!", "FAIL!"), 0),
        (MOBILENET_OUTPUT.replace("NPU time ", "NPU wall time "), 0),
    ],
)
def test_a_mobilenet_run_fails_without_a_pass_or_a_latency(hw, out, returncode):
    assert not hw.parse_run(out, returncode, 1)["passed"]


def test_only_passing_runs_chart_their_times(hw):
    runs = [
        {"seed": seed, "batch": batch, **hw.parse_run(out, 0, batch), "compile_s": s}
        for seed, batch, out, s in [
            (3, 1, MOBILENET_OUTPUT, 41.2),
            (3, 16, MOBILENET_B16_OUTPUT, 43.0),
            # A wrong answer at batch 1 leaves seed 2 with nothing to
            # stream against.
            (2, 1, MOBILENET_OUTPUT.replace("PASS!", "FAIL!"), 40.9),
            (2, 16, MOBILENET_B16_OUTPUT, 42.5),
            (7, 16, "", None),
        ]
    ]
    placements = [
        {"seed": 3, "placement": "0123456789ab", "placement_cost": 300},
        {"seed": 2, "placement": "fedcba987654", "placement_cost": None},
    ]
    rows = hw.aggregate(runs, placements)
    assert {r["name"]: r["value"] for r in rows} == {
        "mobilenet/failed_seeds": 2,
        "mobilenet/seed=3/placement_cost": 300,
        "mobilenet/seed=3/batch=1/us_per_image": 384.4,
        "mobilenet/seed=3/batch=1/e2e_us_per_image": 480.9,
        "mobilenet/seed=3/batch=1/compile_s": 41.2,
        "mobilenet/seed=3/batch=16/us_per_image": 81.48,
        "mobilenet/seed=3/batch=16/streaming_us": 61.29,
        "mobilenet/seed=3/batch=16/e2e_us_per_image": 89.47,
        "mobilenet/seed=3/batch=16/compile_s": 43.0,
        "mobilenet/seed=2/batch=1/compile_s": 40.9,
        "mobilenet/seed=2/batch=16/us_per_image": 81.48,
        "mobilenet/seed=2/batch=16/e2e_us_per_image": 89.47,
        "mobilenet/seed=2/batch=16/compile_s": 42.5,
    }
    ranges = {r["name"]: r.get("range") for r in rows}
    assert ranges["mobilenet/seed=3/placement_cost"] == "placement 0123456789ab"
    assert (
        ranges["mobilenet/seed=3/batch=16/streaming_us"]
        == "from b1 384.4 and b16 1303.7"
    )
    assert (
        ranges["mobilenet/seed=3/batch=16/us_per_image"]
        == "median 83.98, MAD 1.43, p95 86.31"
    )


@pytest.mark.parametrize("batches", ["0", "-3", "two"])
def test_the_hardware_check_refuses_a_batch_that_is_not_positive(batches, tmp_path):
    result = subprocess.run(
        [sys.executable, CHECKS / "sa_placer_hw.py", "--batches", "1", batches]
        + ["--out", tmp_path / "p", "--meta", tmp_path / "m"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "--batches" in result.stderr
    assert not (tmp_path / "p").exists()


def test_a_replayed_placement_matches_placing_the_design(hw, aie_opt):
    placed = subprocess.run(
        [
            aie_opt,
            "--aie-place-tiles=placer=sa_placer sa-seed=1 sa-effort=1.0",
            SMALL,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    digest = hw.placement_of(placed)
    assert digest and len(digest) == 12
    assert hw.replay_placement(aie_opt, str(SMALL), 1) == (digest, 12)
    assert hw.placement_of(SMALL.read_text()) is None

    runs = [
        {"seed": 1, "batch": batch, "placement": placement}
        for batch, placement in [(1, digest), (4, digest), (16, "0" * 12), (64, None)]
    ]
    assert hw.place_seed(runs, aie_opt, str(SMALL)) == {
        "seed": 1,
        "placement": digest,
        "placement_cost": 12,
        "replay_matches": True,
        "placed_apart": [16],
    }
    # A cost is only recorded for the placement the compile made.
    assert hw.place_seed(runs[2:], aie_opt, str(SMALL))["placement_cost"] is None


def test_a_placement_that_fails_to_replay_has_no_cost(hw, aie_opt, tmp_path):
    bad = tmp_path / "bad.mlir"
    bad.write_text("not MLIR\n")
    assert hw.replay_placement(aie_opt, str(bad), 1) == (None, None)
