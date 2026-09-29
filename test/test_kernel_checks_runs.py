# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Record a run's meta.json on the publication branch; standard library only."""

import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "utils/kernel_checks/runs_index.py"

META = {
    "preflight": {
        "npu": "npu1",
        "arch": "aie2",
        "device": "RyzenAI-npu1",
        "pmode": "performance",
    },
    "provenance": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | xrt 2.20.0 | host bench-1 | device RyzenAI-npu1 | pmode performance",
    "measurement_sane": True,
    "exitstatus": 1,
    "n_rows": 780,
    "failed": [
        "test_kernels_perf.py::test_kernel_perf[mul_sized/4096x16/bfloat16/llama-prefill-ffn]"
    ],
    "cases": {
        "swiglu/1024x256/bfloat16": {"cycles": {"min": 2875, "truncated": True}},
        "add/1024x16/bfloat16": {"cycles": {"min": 78, "truncated": False}},
    },
}
CATALOGUE = {
    "kernels": [
        {
            "builds": ["add"],
            "passed": 2,
            "failed": [],
            "timed": 2,
            "timing_failed": [],
            "untimed": [],
        },
        {
            "builds": ["zero"],
            "passed": 4,
            "failed": [],
            "timed": 0,
            "timing_failed": [],
            "untimed": ["zero/64/int32"],
        },
        {
            "builds": ["mm"],
            "passed": 0,
            "failed": ["mm/64/i8"],
            "timed": 0,
            "timing_failed": ["mm/32/i8"],
        },
        {"builds": ["cascade_mm"], "passed": 0, "failed": [], "timed": 0},
        {"builds": [], "passed": 0, "failed": [], "timed": 0},
    ]
}


def run(tmp_path, run_id, meta=META, catalogue=CATALOGUE, sha="d53582d3e0f9"):
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    args = [sys.executable, SCRIPT, "--npu", "npu1", "--meta", tmp_path / "meta.json"]
    if catalogue is not None:
        (tmp_path / "catalogue.json").write_text(json.dumps(catalogue))
        args += ["--catalogue", tmp_path / "catalogue.json"]
    args += ["--run-id", run_id, "--run-url", f"https://example.com/runs/{run_id}"]
    args += ["--out-dir", tmp_path / "out"]
    subprocess.run(args, check=True, env={**os.environ, "GITHUB_SHA": sha})
    return (
        json.loads((tmp_path / "out/runs.json").read_text()),
        json.loads((tmp_path / "out/latest.json").read_text()),
    )


def test_a_run_is_summarized_and_kept_in_full(tmp_path):
    index, latest = run(tmp_path, "100")
    assert index["npu"] == "npu1"
    (entry,) = index["runs"]
    assert entry["id"] == "100"
    assert entry["url"] == "https://example.com/runs/100"
    assert entry["commit"] == "d53582d3e0f9"
    assert entry["date"].endswith("+00:00")
    assert entry["pmode"] == "performance"
    assert entry["device"] == "RyzenAI-npu1"
    assert entry["provenance"]["peano"] == "22.0.0+0006955e"
    assert entry["provenance"]["host"] == "bench-1"
    assert entry["provenance"]["xrt"] == "2.20.0"
    assert entry["sane"] is True and entry["published"] is True
    assert entry["n_rows"] == 780 and entry["exitstatus"] == 1
    assert entry["failed"] == META["failed"]
    assert entry["truncated"] == ["swiglu/1024x256/bfloat16"]
    assert entry["cases"] == {
        "passed": 6,
        "failed": 1,
        "timed": 2,
        "timing_failed": 1,
        "untimed": 1,
    }
    assert entry["kernels"] == {"offered": 4, "checked": 3}
    # latest.json is the whole meta, with the run named.
    assert latest["cases"] == META["cases"]
    assert (latest["id"], latest["url"], latest["commit"]) == (
        "100",
        "https://example.com/runs/100",
        "d53582d3e0f9",
    )


def test_runs_append_and_a_rerun_replaces_its_entry(tmp_path):
    run(tmp_path, "100")
    run(tmp_path, "101", sha="abcdef123456")
    index, _ = run(tmp_path, "100", meta=dict(META, n_rows=5))
    assert [r["id"] for r in index["runs"]] == ["101", "100"]
    assert index["runs"][-1]["n_rows"] == 5


def test_an_unpublished_leg_is_recorded_as_such(tmp_path):
    meta = {
        "preflight": {"npu": "npu1", "pmode": "default"},
        "provenance": "commit abc | pmode default",
        "measurement_sane": False,
        "exitstatus": 1,
        "n_rows": 0,
        "failed": ["test_kernels_perf.py::test_measurement_is_sane"],
        "correctness_error": "correctness report contains no tests",
    }
    index, _ = run(tmp_path, "7", meta=meta, catalogue=None)
    (entry,) = index["runs"]
    assert entry["sane"] is False and entry["published"] is False
    assert entry["pmode"] == "default"
    assert entry["truncated"] == []
    assert entry["correctness_error"] == "correctness report contains no tests"
    assert "cases" not in entry and "kernels" not in entry


def test_the_index_is_capped(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("runs_index", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "runs.json"
    for i in range(module.MAX_RUNS + 3):
        module.record(
            path, {}, npu="npu2", run_id=str(i), run_url="", commit="", date=""
        )
    index = json.loads(path.read_text())
    assert len(index["runs"]) == module.MAX_RUNS
    assert index["runs"][0]["id"] == "3"
    assert module.provenance_fields("a 1 | b two words | c") == {
        "a": "1",
        "b": "two words",
    }
