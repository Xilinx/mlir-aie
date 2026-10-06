# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Publish run records, history and the migration from data.js; standard library only."""

import datetime
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils/kernel_checks/publish.py"


@pytest.fixture(scope="module")
def publish():
    spec = importlib.util.spec_from_file_location("publish", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    "n_rows": 4,
    "failed": ["test_kernels_perf.py::test_kernel_perf[mm/64/i8]"],
    "cases": {
        "swiglu/1024x256/bfloat16": {"cycles": {"min": 2875, "truncated": True}},
        "add/1024x16/bfloat16": {"cycles": {"min": 78, "truncated": False}},
    },
}
ROWS = [
    {
        "name": "add/1024x16/bfloat16/cycles",
        "unit": "cycles",
        "value": 78,
        "range": "median 87 max 131 n=16",
        "extra": "x",
    },
    {
        "name": "add/1024x16/bfloat16/npu_us",
        "unit": "us",
        "value": 193.5,
        "range": "± 1.6; min 172.8 max 202.2 n=50",
        "extra": "x",
    },
    {
        "name": "add/1024x16/bfloat16/core_elf_bytes",
        "unit": "bytes",
        "value": 4096,
        "extra": "x",
    },
    {
        "name": "swiglu/1024x256/bfloat16/cycles",
        "unit": "cycles",
        "value": 2875,
        "range": "median 2939 max 2990 n=84; truncated",
        "extra": "x",
    },
]
CATALOGUE = {
    "kernels": [
        {
            "builds": ["add"],
            "passed": 2,
            "failed": [],
            "timed": 2,
            "timing_failed": [],
            "untimed": [],
            "flaky": ["add/1024x16/bfloat16"],
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
        {"builds": [], "passed": 0, "failed": [], "timed": 0},
    ]
}


def results_dir(tmp_path, name="results", meta=META, rows=ROWS, catalogue=CATALOGUE):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps(meta))
    if rows is not None:
        (d / "perf.json").write_text(json.dumps(rows))
    if catalogue is not None:
        (d / "catalogue.json").write_text(json.dumps(catalogue))
    return d


def run_cli(args, sha="d53582d3e0f9"):
    subprocess.run(
        [sys.executable, SCRIPT, *map(str, args)],
        check=True,
        env={**os.environ, "GITHUB_SHA": sha},
        capture_output=True,
    )


def test_a_perf_run_is_recorded_summarized_and_charted(publish, tmp_path):
    out = tmp_path / "npu1"
    results = results_dir(tmp_path)
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            results,
            "--run-id",
            "100",
            "--run-url",
            "https://example.com/runs/100",
            "--out",
            out,
            "--commit-message",
            "Fix add\n\nThe body is not kept.",
            "--commit-date",
            "2026-09-28T20:24:43+00:00",
            "--date",
            "2026-09-29T06:35:00+00:00",
        ]
    )
    record = json.loads((out / "runs/100.json").read_text())
    assert record["target"] == "npu1"
    assert record["id"] == "100" and record["url"] == "https://example.com/runs/100"
    assert record["date"] == "2026-09-29T06:35:00+00:00"
    assert record["commit"] == {
        "id": "d53582d3e0f9",
        "url": "https://github.com/Xilinx/mlir-aie/commit/d53582d3e0f9",
        "message": "Fix add",
        "timestamp": "2026-09-28T20:24:43+00:00",
    }
    assert record["pmode"] == "performance" and record["device"] == "Phoenix"
    assert record["device_raw"] == "RyzenAI-npu1" and record["schema"] == 1
    assert record["provenance"]["host"] == "bench-1"
    assert record["sane"] and record["published"] and record["n_rows"] == 4
    assert record["failed"] == META["failed"]
    assert record["truncated"] == ["swiglu/1024x256/bfloat16"]
    assert record["cases"] == {
        "passed": 6,
        "failed": 1,
        "timed": 2,
        "timing_failed": 1,
        "untimed": 1,
        "flaky": 1,
    }
    assert record["kernels"] == {"offered": 3, "checked": 3}
    assert record["rows"]["add/1024x16/bfloat16"]["cycles"] == {
        "value": 78,
        "unit": "cycles",
        "range": "median 87 max 131 n=16",
    }
    assert record["rows"]["add/1024x16/bfloat16"]["core_elf_bytes"] == {
        "value": 4096,
        "unit": "bytes",
    }
    # extra is provenance, kept once per record, not per row.
    assert "extra" not in json.dumps(record["rows"])

    index = json.loads((out / "runs.json").read_text())
    assert index["target"] == "npu1"
    (entry,) = index["runs"]
    assert entry["id"] == "100" and "rows" not in entry
    latest = json.loads((out / "latest.json").read_text())
    assert latest == record

    cycles = json.loads((out / "history/cycles.json").read_text())
    assert cycles["metric"] == "cycles" and cycles["unit"] == "cycles"
    assert [r["id"] for r in cycles["runs"]] == ["100"]
    assert cycles["runs"][0]["provenance"] == {
        "peano": "22.0.0+0006955e",
        "kernels": "0857407c3322",
        "xrt": "2.20.0",
        "host": "bench-1",
        "device": "RyzenAI-npu1",
    }
    assert cycles["series"] == {
        "add/1024x16/bfloat16": {"values": [78], "ranges": ["median 87 max 131 n=16"]},
        "swiglu/1024x256/bfloat16": {
            "values": [2875],
            "ranges": ["median 2939 max 2990 n=84; truncated"],
        },
    }
    elf = json.loads((out / "history/core_elf_bytes.json").read_text())
    assert elf["series"] == {"add/1024x16/bfloat16": {"values": [4096]}}
    assert sorted(p.stem for p in (out / "history").glob("*.json")) == [
        "core_elf_bytes",
        "cycles",
        "npu_us",
    ]


def test_a_refused_run_records_why_and_no_sanity_result(publish, tmp_path):
    refused = "power mode is performance, required 'turbo'"
    meta = {"refused": refused, "exitstatus": 1, "n_rows": 0, "failed": []}
    record = publish.record_perf(
        results_dir(tmp_path, meta=meta, rows=None), target="npu1", run={"id": "1"}
    )
    assert record["refused"] == refused
    assert record["sane"] is None and not record["published"]


def test_runs_accumulate_gaps_stay_and_unsane_runs_publish_nothing(publish, tmp_path):
    out = tmp_path / "npu1"
    a = results_dir(tmp_path, "a")
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            a,
            "--run-id",
            "1",
            "--out",
            out,
            "--date",
            "2026-09-27T06:00:00+00:00",
        ]
    )
    # The second night drops swiglu and adds relu.
    rows = [r for r in ROWS if not r["name"].startswith("swiglu")] + [
        {"name": "relu/1024/bf16/cycles", "unit": "cycles", "value": 5, "extra": "x"},
    ]
    b = results_dir(tmp_path, "b", rows=rows)
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            b,
            "--run-id",
            "2",
            "--out",
            out,
            "--date",
            "2026-09-28T06:00:00+00:00",
        ]
    )
    # The third failed its sanity check: recorded, no rows, not the latest baseline.
    c = results_dir(
        tmp_path, "c", meta=dict(META, measurement_sane=False, n_rows=0), rows=None
    )
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            c,
            "--run-id",
            "3",
            "--out",
            out,
            "--date",
            "2026-09-29T06:00:00+00:00",
        ]
    )

    index = json.loads((out / "runs.json").read_text())
    assert [(r["id"], r["published"]) for r in index["runs"]] == [
        ("1", True),
        ("2", True),
        ("3", False),
    ]
    assert index["runs"][2]["sane"] is False
    assert json.loads((out / "latest.json").read_text())["id"] == "2"
    cycles = json.loads((out / "history/cycles.json").read_text())
    assert [r["id"] for r in cycles["runs"]] == ["1", "2"]
    assert cycles["series"]["swiglu/1024x256/bfloat16"]["values"] == [2875, None]
    assert cycles["series"]["relu/1024/bf16"] == {"values": [None, 5]}
    assert cycles["series"]["add/1024x16/bfloat16"]["values"] == [78, 78]

    # A rerun of run 2 replaces its record.
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            a,
            "--run-id",
            "2",
            "--out",
            out,
            "--date",
            "2026-09-28T07:00:00+00:00",
        ]
    )
    index = json.loads((out / "runs.json").read_text())
    assert [r["id"] for r in index["runs"]] == ["1", "2", "3"]
    assert index["runs"][1]["date"] == "2026-09-28T07:00:00+00:00"
    assert len(list((out / "runs").glob("*.json"))) == 3


DATA_JS = """window.BENCHMARK_DATA = {
  "lastUpdate": 1790663759370,
  "repoUrl": "https://github.com/Xilinx/mlir-aie",
  "entries": {
    "aie_kernels (npu1, default)": [
      {
        "commit": {"author": {"name": "E"}, "id": "d53582d3e0f9f8a2b77695bbf4abbd7766d5584a",
                   "message": "Single-core Kernel Optimizations and Tooling (#3801)\\n\\nbody",
                   "timestamp": "2026-09-28T20:24:43Z", "url": "https://github.com/Xilinx/mlir-aie/commit/d53582d3e0f9"},
        "date": 1790632656742, "tool": "customSmallerIsBetter",
        "benches": [
          {"name": "passthrough/2048x16/int32/cycles", "value": 264, "range": "median 264 max 264 n=16", "unit": "cycles",
           "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"},
          {"name": "swiglu/1024x256/bfloat16/cycles", "value": 2875, "range": "median 2939 max 2990 n=84; truncated", "unit": "cycles",
           "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"}
        ]
      },
      {
        "commit": {"id": "d53582d3e0f9f8a2b77695bbf4abbd7766d5584a", "message": "Single-core Kernel Optimizations and Tooling (#3801)",
                   "timestamp": "2026-09-28T20:24:43Z", "url": "https://github.com/Xilinx/mlir-aie/commit/d53582d3e0f9"},
        "date": 1790663757417, "tool": "customSmallerIsBetter",
        "benches": [
          {"name": "passthrough/2048x16/int32/cycles", "value": 264, "range": "median 264 max 264 n=16", "unit": "cycles",
           "extra": "commit d53582d3e0 | peano 22.0.0+0006955e | kernels 0857407c3322 | device RyzenAI-npu1 | pmode default"}
        ]
      }
    ]
  }
};
"""


def test_migration_turns_every_entry_into_a_run_and_removes_the_old_files(
    publish, tmp_path
):
    out = tmp_path / "npu1"
    out.mkdir()
    (out / "data.js").write_text(DATA_JS)
    (out / "index.html").write_text("<html>window.BENCHMARK_DATA</html>")
    run_cli(["migrate", "--out", out])
    assert not (out / "data.js").exists() and not (out / "index.html").exists()
    index = json.loads((out / "runs.json").read_text())
    assert [r["id"] for r in index["runs"]] == [
        "bench-1790632656742",
        "bench-1790663757417",
    ]
    first = json.loads((out / "runs/bench-1790632656742.json").read_text())
    assert first["date"] == "2026-09-28T21:57:36+00:00"
    assert first["pmode"] == "default" and first["device"] == "Phoenix"
    assert first["device_raw"] == "RyzenAI-npu1" and first["schema"] == 1
    assert first["provenance"]["peano"] == "22.0.0+0006955e"
    assert first["commit"]["id"].startswith("d53582d3e0f9")
    assert (
        first["commit"]["message"]
        == "Single-core Kernel Optimizations and Tooling (#3801)"
    )
    assert first["migrated_from"] == "aie_kernels (npu1, default)"
    assert first["truncated"] == ["swiglu/1024x256/bfloat16"]
    assert first["published"] and first["url"] == ""
    cycles = json.loads((out / "history/cycles.json").read_text())
    assert cycles["series"]["swiglu/1024x256/bfloat16"]["values"] == [2875, None]
    assert cycles["runs"][1]["pmode"] == "default"
    # Migrating again changes nothing; a publish after it keeps the history.
    run_cli(["migrate", "--out", out])
    assert len(index["runs"]) == 2
    results = results_dir(tmp_path)
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            results,
            "--run-id",
            "5",
            "--out",
            out,
            "--date",
            "2026-09-30T06:00:00+00:00",
        ]
    )
    cycles = json.loads((out / "history/cycles.json").read_text())
    assert [r["id"] for r in cycles["runs"]] == [
        "bench-1790632656742",
        "bench-1790663757417",
        "5",
    ]
    assert cycles["series"]["add/1024x16/bfloat16"]["values"] == [None, None, 78]


def test_old_runs_thin_to_one_a_week(publish):
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    records = [
        {"id": str(i), "date": publish.iso(now - datetime.timedelta(days=i))}
        for i in range(0, 200)
    ]
    kept = publish.prune(records, now)
    ids = [int(r["id"]) for r in kept]
    assert ids == sorted(ids, reverse=True)
    assert all(i in ids for i in range(publish.KEEP_DAYS))
    old = [i for i in ids if i >= publish.KEEP_DAYS]
    # About one per week beyond the window, the newest of each.
    assert 14 <= len(old) <= 17
    weeks = {
        publish.parse_date(r["date"]).isocalendar()[:2]
        for r in kept
        if int(r["id"]) >= publish.KEEP_DAYS
    }
    assert len(weeks) == len(old)
    assert kept[-1]["id"] == "0"


def test_rebuild_drops_pruned_run_files_and_stale_histories(publish, tmp_path):
    out = tmp_path / "npu2"
    (out / "runs").mkdir(parents=True)
    (out / "history").mkdir()
    (out / "history/gone.json").write_text("{}")
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    for i, days in enumerate((0, 1, 100, 101)):
        record = {
            "target": "npu2",
            "id": f"r{i}",
            "url": "",
            "commit": {},
            "date": publish.iso(now - datetime.timedelta(days=days)),
            "pmode": "performance",
            "provenance": {},
            "sane": True,
            "published": True,
            "n_rows": 1,
            "failed": [],
            "truncated": [],
            "rows": {"add/1/bf16": {"cycles": {"value": i, "unit": "cycles"}}},
        }
        (out / "runs" / f"r{i}.json").write_text(json.dumps(record))
    result = publish.rebuild(out, now)
    # Days 100 and 101 are one ISO week: the newer stays.
    assert result == {"runs": 3, "published": 3, "metrics": ["cycles"]}
    assert sorted(p.stem for p in (out / "runs").glob("*.json")) == ["r0", "r1", "r2"]
    assert not (out / "history/gone.json").exists()
    cycles = json.loads((out / "history/cycles.json").read_text())
    assert cycles["series"]["add/1/bf16"]["values"] == [2, 1, 0]


def test_a_retired_power_mode_is_dropped_on_request(publish, tmp_path):
    out = tmp_path / "npu1"
    out.mkdir()
    (out / "data.js").write_text(DATA_JS)
    run_cli(["migrate", "--out", out])
    results = results_dir(tmp_path)
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            results,
            "--run-id",
            "5",
            "--out",
            out,
            "--date",
            "2026-09-30T06:00:00+00:00",
        ]
    )
    run_cli(["rebuild", "--out", out, "--drop-pmode", "default"])
    index = json.loads((out / "runs.json").read_text())
    assert [(r["id"], r["pmode"]) for r in index["runs"]] == [("5", "performance")]
    assert sorted(p.name for p in (out / "runs").iterdir()) == ["5.json"]
    cycles = json.loads((out / "history/cycles.json").read_text())
    assert [r["id"] for r in cycles["runs"]] == ["5"]
    assert "passthrough/2048x16/int32" not in cycles["series"]


def test_backfill_dates_runs_by_their_start_and_keeps_one_power_mode(publish, tmp_path):
    out = tmp_path / "npu1"
    out.mkdir()
    (out / "data.js").write_text(DATA_JS)
    run_cli(["migrate", "--out", out])
    turbo = dict(META, preflight=dict(META["preflight"], pmode="turbo"))
    for run_id, date in (("41", "2026-09-30T09:12:00+00:00"), ("42", "")):
        args = ["perf", "--target", "npu1", "--run-id", run_id, "--out", out]
        args += ["--event", "", "--date", date] if date else ["--event", ""]
        run_cli(args + ["--results", results_dir(tmp_path, run_id, meta=turbo)])
    refused = {
        "preflight": {"pmode": "default"},
        "refused": "power mode is default, required 'turbo'",
        "failed": [],
    }
    run_cli(
        ["perf", "--target", "npu1", "--run-id", "43", "--out", out, "--event", ""]
        + ["--results", results_dir(tmp_path, "43", meta=refused, rows=None)]
    )
    runs = tmp_path / "runs.jsonl"
    runs.write_text(
        '{"id":41,"event":"schedule","run_started_at":"2026-09-30T06:00:05Z"}\n'
        '{"id":42,"event":"workflow_dispatch","run_started_at":"2026-10-01T06:00:07Z"}\n'
        '{"id":43,"event":"schedule","run_started_at":"2026-10-02T06:00:09Z"}\n'
        '{"id":99,"event":"schedule","run_started_at":"2026-10-03T06:00:00Z"}\n'
    )
    backfill = ["backfill", "--out", out, "--runs", runs, "--keep-pmode", "turbo"]
    run_cli(backfill)
    index = json.loads((out / "runs.json").read_text())
    assert [(r["id"], r["date"], r["event"], r["pmode"]) for r in index["runs"]] == [
        ("41", "2026-09-30T06:00:05+00:00", "schedule", "turbo"),
        ("42", "2026-10-01T06:00:07+00:00", "workflow_dispatch", "turbo"),
        ("43", "2026-10-02T06:00:09+00:00", "schedule", "default"),
    ]
    # The default-mode numbers migrated from data.js are gone from the charts.
    cycles = json.loads((out / "history/cycles.json").read_text())
    assert [r["id"] for r in cycles["runs"]] == ["41", "42"]
    assert "passthrough/2048x16/int32" not in cycles["series"]
    # The dispatch is charted; the nightly before it stays the baseline.
    assert json.loads((out / "latest.json").read_text())["id"] == "41"
    files = {p.name: p.read_text() for p in (out / "runs").iterdir()}
    run_cli(backfill)
    assert {p.name: p.read_text() for p in (out / "runs").iterdir()} == files
    assert publish.backfill(out, publish.read_runs(runs)) == 0


@pytest.mark.parametrize(
    "raw,part",
    [
        ("RyzenAI-npu1", "Phoenix"),
        ("NPU Phoenix", "Phoenix"),
        ("NPU Strix", "Strix"),
        ("RyzenAI-npu4", "Strix"),
        ("NPU Strix Halo", "Strix Halo"),
        ("RyzenAI-npu5", "Strix Halo"),
        ("NPU Krackan 1", "Krackan"),
        ("NPU Gorgon Point", "Gorgon Point"),
        ("Something New", "Something New"),
        (None, None),
    ],
)
def test_device_names_become_parts(publish, raw, part):
    assert publish.device_part(raw) == part


def test_every_file_carries_the_schema_and_a_newer_one_is_refused(publish, tmp_path):
    out = tmp_path / "npu1"
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            results_dir(tmp_path),
            "--run-id",
            "1",
            "--out",
            out,
            "--date",
            "2026-09-29T06:00:00+00:00",
        ]
    )
    files = [out / "runs.json", out / "latest.json", out / "runs/1.json"]
    files += list((out / "history").glob("*.json"))
    assert all(json.loads(f.read_text())["schema"] == publish.SCHEMA for f in files)
    index = json.loads((out / "runs.json").read_text())
    (out / "runs.json").write_text(json.dumps(dict(index, schema=publish.SCHEMA + 1)))
    result = subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "perf",
            "--target",
            "npu1",
            "--results",
            str(results_dir(tmp_path)),
            "--run-id",
            "2",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "NewerSchema" in result.stderr
    assert not (out / "runs/2.json").exists()
    # A newer record among the runs stops a rebuild too.
    (out / "runs.json").write_text(json.dumps(index))
    record = json.loads((out / "runs/1.json").read_text())
    (out / "runs/1.json").write_text(
        json.dumps(dict(record, schema=publish.SCHEMA + 1))
    )
    with pytest.raises(publish.NewerSchema):
        publish.rebuild(out)


@pytest.mark.parametrize(
    "old, new",
    [
        ("sa_placer/fail_count", "test_sa_effort/failed_seeds"),
        ("sa_placer/mean_final_cost", "test_sa_effort/final_cost_mean"),
        ("sa_placer/max_final_cost", "test_sa_effort/final_cost_max"),
        ("sa_placer/hw_fail_count", "mobilenet/failed_seeds"),
        (
            "sa_placer/hw_seed7_latency_us",
            "mobilenet/seed=7/batch=1/us_per_image",
        ),
        ("sa_placer/mean_wall_time_ms", None),
        ("sa_placer/max_peak_rss_mb", None),
        ("add/1024x16/bfloat16/cycles", "add/1024x16/bfloat16/cycles"),
    ],
)
def test_component_rows_take_their_current_names(publish, old, new):
    assert publish._current_name(old) == new


def test_a_component_check_migrates_and_keeps_its_redirect(publish, tmp_path):
    out = tmp_path / "sa-placer-hw"
    out.mkdir()
    legacy = {
        "entries": {
            "SA placer hardware check (mobilenet, npu2)": [
                {
                    "commit": {"id": "18ca6c1cbf4c", "message": "Old"},
                    "date": 1790924974800,
                    "benches": [
                        {
                            "name": "sa_placer/hw_fail_count",
                            "value": 0,
                            "unit": "seeds",
                        },
                        {
                            "name": "sa_placer/hw_seed3_latency_us",
                            "value": 295.7,
                            "unit": "us",
                        },
                    ],
                }
            ]
        }
    }
    (out / "data.js").write_text(f"window.BENCHMARK_DATA = {json.dumps(legacy)};")
    run_cli(["redirect", "--out", out, "--to", "../../kernel-checks/#view=components"])
    page = (out / "index.html").read_text()
    assert '<meta http-equiv="refresh" content="0; url=../../kernel-checks/' in page
    run_cli(["migrate", "--out", out])
    assert (out / "index.html").read_text() == page
    (record,) = [json.loads(p.read_text()) for p in (out / "runs").glob("*.json")]
    assert record["target"] == "sa-placer-hw" and record["pmode"] is None
    assert record["rows"] == {
        "mobilenet": {"failed_seeds": {"value": 0, "unit": "seeds"}},
        "mobilenet/seed=3/batch=1": {"us_per_image": {"value": 295.7, "unit": "us"}},
    }
    index = json.loads((out / "runs.json").read_text())
    assert index["metrics"] == ["failed_seeds", "us_per_image"]

    meta = {
        "preflight": {"pmode": "turbo", "device": "NPU Strix Halo"},
        "provenance": "commit 0123456789 | peano 22.0.0+0006955e | pmode turbo",
        "measurement_sane": True,
        "detail": [
            {"seed": 3, "batch": 16, "passed": True, "us_per_image": 81.48},
            {"seed": 2, "batch": 16, "passed": False, "us_per_image": None},
        ],
        "failed": ["mobilenet/seed=2/batch=16"],
        "exitstatus": 1,
    }
    rows = [
        {"name": "mobilenet/failed_seeds", "unit": "seeds", "value": 1},
        {
            "name": "mobilenet/seed=3/placement_cost",
            "unit": "cost",
            "value": 300,
            "range": "placement 0123456789ab",
        },
    ]
    results = results_dir(tmp_path, meta=meta, rows=rows, catalogue=None)
    run_cli(
        [
            "perf",
            "--target",
            "sa-placer-hw",
            "--results",
            results,
            "--run-id",
            "9",
            "--out",
            out,
            "--date",
            "2026-10-03T06:30:00+00:00",
        ]
    )
    record = json.loads((out / "runs/9.json").read_text())
    assert record["failed"] == ["mobilenet/seed=2/batch=16"] and record["published"]
    assert record["detail"] == meta["detail"]
    assert record["device"] == "Strix Halo" and record["pmode"] == "turbo"
    index = json.loads((out / "runs.json").read_text())
    assert index["metrics"] == ["failed_seeds", "placement_cost", "us_per_image"]
    assert all("detail" not in run for run in index["runs"])
    failed = json.loads((out / "history/failed_seeds.json").read_text())
    assert failed["target"] == "sa-placer-hw"
    assert failed["series"] == {"mobilenet": {"values": [0, 1]}}
    cost = json.loads((out / "history/placement_cost.json").read_text())
    assert cost["series"]["mobilenet/seed=3"]["ranges"] == [
        None,
        "placement 0123456789ab",
    ]


def test_an_unknown_target_is_refused(tmp_path):
    result = subprocess.run(
        [sys.executable, SCRIPT, "perf", "--target", "sa-placer-cpu"]
        + ["--results", tmp_path, "--run-id", "1", "--out", tmp_path],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2 and "invalid choice" in result.stderr


def test_a_redirect_keeps_the_links_own_view(publish, tmp_path):
    publish.redirect(tmp_path, "../dashboard/#view=night")
    page = (tmp_path / "index.html").read_text()
    # A link with its own #view lands on it; a bare one on the default.
    assert '<script>location.replace("../dashboard/" + (location.hash || "#view=night"));</script>' in page
    assert '<meta http-equiv="refresh" content="0; url=../dashboard/#view=night">' in page
