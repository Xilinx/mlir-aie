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
        "name": "add/1024x16/bfloat16/dispatch_us",
        "unit": "us",
        "value": 193.5,
        "range": "± 1.6; min 172.8 max 202.2 n=50",
        "extra": "x",
    },
    {
        "name": "add/1024x16/bfloat16/npu_us",
        "unit": "us",
        "value": 101.2,
        "range": "± 1.6; min 80.5 max 109.9 n=50",
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
    assert record["sane"] and record["published"] and record["n_rows"] == 5
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
    # The part each run measured on, so a chart can keep parts apart.
    assert cycles["runs"][0]["device"] == "Phoenix"
    assert cycles["runs"][0]["provenance"] == {
        "peano": "22.0.0+0006955e",
        "kernels": "0857407c3322",
        "xrt": "2.20.0",
        "host": "bench-1",
        "device": "RyzenAI-npu1",
        # Dated before DECODER_FIXED.
        "measurement": "1",
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
        "dispatch_us",
        "npu_us",
    ]


def test_a_timed_case_is_charted_with_its_worst_error_per_dtype(publish, tmp_path):
    bf16 = dict(
        output=0,
        dtype="bfloat16",
        reference="float64",
        modeled=False,
        max_ulp=1,
        not_correctly_rounded=3,
        n=1024,
    )
    f32 = dict(bf16, output=1, dtype="float32", max_ulp=40, not_correctly_rounded=1)
    errors = {
        "add/1024x16/bfloat16/random/s0": {"passed": True, "outputs": [bf16, f32]},
        "add/1024x16/bfloat16/large/s1": {
            "passed": True,
            "outputs": [dict(bf16, max_ulp=2, not_correctly_rounded=1, modeled=True)],
        },
        # All-zero data is right or wrong as one: left out of the pool.
        "add/1024x16/bfloat16/zeros/s0": {
            "passed": True,
            "outputs": [dict(bf16, max_ulp=9, not_correctly_rounded=1024)],
        },
        # An untimed case has no row to stand beside.
        "relu/1024/bf16/random/s0": {"passed": True, "outputs": [bf16]},
    }
    results = results_dir(tmp_path)
    (results / "error.json").write_text(json.dumps(errors))
    out = tmp_path / "npu1"
    run_cli(
        [
            "perf",
            "--target",
            "npu1",
            "--results",
            results,
            "--run-id",
            "1",
            "--out",
            out,
        ]
    )
    record = json.loads((out / "runs/1.json").read_text())
    add = record["rows"]["add/1024x16/bfloat16"]
    modeled = (
        "vs the float64 correctly rounded reference; the check judges this build"
        " by a model of the kernel"
    )
    # A float32 output's ulps never mix with a bfloat16 one's.
    assert add["max_ulp_bfloat16"] == {"value": 2, "unit": "ulp", "range": modeled}
    assert add["not_correctly_rounded_bfloat16"]["value"] == 0.195
    assert add["max_ulp_float32"] == {
        "value": 40,
        "unit": "ulp",
        "range": "vs the float64 correctly rounded reference",
    }
    assert add["not_correctly_rounded_float32"]["value"] == 0.098
    assert "relu/1024/bf16" not in record["rows"]
    ulps = json.loads((out / "history/max_ulp_bfloat16.json").read_text())
    assert ulps["series"]["add/1024x16/bfloat16"]["values"] == [2]


def test_each_timed_case_names_its_kernel_build(publish, tmp_path):
    cases = {
        "add/1024x16/bfloat16": {"build": "add_bf16-0123456789ab"},
        # A case that recorded no row has no build to name.
        "relu/1024x16/bfloat16": {"build": "relu_bf16-ba9876543210"},
    }
    record = publish.record_perf(
        results_dir(tmp_path, meta={**META, "cases": cases}),
        target="npu1",
        run={"id": "1"},
    )
    assert record["builds"] == {"add/1024x16/bfloat16": "add_bf16-0123456789ab"}
    assert "builds" not in publish.summary(record)


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


def test_a_thinned_week_keeps_its_published_nightly(publish):
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)

    def run(rid, days, **extra):
        return {
            "id": rid,
            "date": publish.iso(now - datetime.timedelta(days=days)),
            **extra,
        }

    # One old ISO week: a published nightly, then a newer refused night and
    # a newer published run started by hand. The week's charted point is the
    # nightly's: neither newer run may stand in for it.
    week = publish.parse_date(publish.iso(now - datetime.timedelta(days=120)))
    monday = 120 + week.weekday()
    records = [
        run("nightly", monday - 1, published=True, rows={"a": {}}),
        run("refused", monday - 2, published=False, rows={}),
        run(
            "manual",
            monday - 3,
            published=True,
            rows={"a": {}},
            event="workflow_dispatch",
        ),
    ]
    assert [r["id"] for r in publish.prune(records, now)] == ["nightly"]
    # A week with no published run keeps its newest.
    only_refused = [
        run("r1", monday - 1, published=False),
        run("r2", monday - 2, published=False),
    ]
    assert [r["id"] for r in publish.prune(only_refused, now)] == ["r2"]


def test_runs_past_a_year_of_weeks_are_dropped(publish):
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    horizon = publish.KEEP_DAYS + 7 * publish.KEEP_WEEKS
    records = [
        {"id": str(d), "date": publish.iso(now - datetime.timedelta(days=d))}
        for d in (0, horizon - 10, horizon + 10, horizon + 400)
    ]
    assert [r["id"] for r in publish.prune(records, now)] == [str(horizon - 10), "0"]


def test_old_records_shed_their_detail_and_retired_cases_leave_the_history(
    publish, tmp_path
):
    out = tmp_path / "sa-placer-hw"
    (out / "runs").mkdir(parents=True)
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    # Twenty nights, oldest first (one 200 days ago, then the last 19): case
    # "kept" every night; "retired" only in the first five; "gap" every night
    # but the last three.
    days = [200, *range(18, -1, -1)]
    for i, d in enumerate(days):
        rows = {"kept": {"cost": {"value": i, "unit": "cost"}}}
        if i < 5:
            rows["retired"] = {"cost": {"value": i, "unit": "cost"}}
        if i < len(days) - 3:
            rows["gap"] = {"cost": {"value": i, "unit": "cost"}}
        record = {
            "target": "sa-placer-hw",
            "id": f"r{i:02}",
            "url": "",
            "commit": {},
            "date": publish.iso(now - datetime.timedelta(days=d)),
            "pmode": "turbo",
            "provenance": {},
            "sane": True,
            "published": True,
            "n_rows": 1,
            "failed": [],
            "truncated": [],
            "rows": rows,
            "detail": [{"seed": 3}],
            "builds": {"x": 1},
        }
        (out / "runs" / f"r{i:02}.json").write_text(json.dumps(record))
    publish.rebuild(out, now)
    # The 200-day-old record sheds what only the latest runs are read for;
    # a recent one keeps it.
    old = json.loads((out / "runs/r00.json").read_text())
    assert "detail" not in old and "builds" not in old and old["rows"]
    assert "detail" in json.loads((out / "runs/r19.json").read_text())
    # A case not measured in the last RETIRED_AFTER runs leaves the history;
    # one missing a few nights stays.
    cost = json.loads((out / "history/cost.json").read_text())
    assert sorted(cost["series"]) == ["gap", "kept"]


@pytest.mark.parametrize(
    "changes, selected, needed",
    [
        (
            [
                {"event": "schedule"},
                {"event": "schedule"},
                {"event": "workflow_dispatch"},
                {"event": "workflow_dispatch"},
            ],
            "2",
            {"1", "2"},
        ),
        ([{}, {"pmode": "performance"}, {}], "3", {"1", "3"}),
        ([{}, {"device": "Phoenix"}, {}], "3", {"1", "3"}),
        ([{"device": "Phoenix"}, {"device": "Strix Halo"}, {}], "3", {"2", "3"}),
        ([{}, {}, {"published": False, "rows": {}}], "2", {"1", "2"}),
        (
            [{"event": "workflow_dispatch"} for _ in range(4)],
            "4",
            {"4"},
        ),
        (
            [{}, {"event": "workflow_dispatch"}, {}],
            "3",
            {"1", "3"},
        ),
    ],
    ids=[
        "manual-after",
        "power",
        "same-device",
        "new-device",
        "refused",
        "manual-only",
        "manual-before",
    ],
)
def test_old_detail_follows_the_pages_selected_comparison(
    publish, tmp_path, changes, selected, needed
):
    out = tmp_path / "sa-placer-hw"
    (out / "runs").mkdir(parents=True)
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    for i, change in enumerate(changes, 1):
        record = hardware_record(
            str(i),
            publish.iso(now - datetime.timedelta(days=160 - 7 * i)),
            "full",
            builds={"kernel": "built"},
            **change,
        )
        (out / "runs" / f"{i}.json").write_text(json.dumps(record))

    publish.rebuild(out, now)
    latest = json.loads((out / "latest.json").read_text())
    assert latest["id"] == selected
    assert latest["detail"] and latest["builds"]
    for i in range(1, len(changes) + 1):
        record = json.loads((out / "runs" / f"{i}.json").read_text())
        assert ("detail" in record) == (str(i) in needed)
        assert ("builds" in record) == (str(i) in needed)
    before = {p: p.read_text() for p in out.rglob("*.json")}
    publish.rebuild(out, now)
    assert {p: p.read_text() for p in out.rglob("*.json")} == before


@pytest.mark.parametrize(
    "middle, latest, previous",
    [
        ({}, {}, "2"),
        ({"pmode": "performance"}, {}, "1"),
        ({"device": "Phoenix"}, {}, "1"),
        ({"device": "Phoenix"}, {"device": "Strix Halo"}, "2"),
    ],
    ids=["same-week", "power", "same-device", "new-device"],
)
def test_comparison_survives_weekly_thinning_and_the_run_limit(
    publish, tmp_path, middle, latest, previous
):
    out = tmp_path / "sa-placer-hw"
    (out / "runs").mkdir(parents=True)
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    max_runs = 6
    records = [
        hardware_record(
            str(i),
            publish.iso(now - datetime.timedelta(days=161 - i)),
            "full",
            event="schedule",
            builds={"kernel": "built"},
            **change,
        )
        for i, change in enumerate(({}, middle, latest), 1)
    ]
    assert len({publish.parse_date(r["date"]).isocalendar()[:2] for r in records}) == 1
    records += [
        hardware_record(
            str(i),
            publish.iso(now - datetime.timedelta(days=24 - i)),
            "full",
            event="workflow_dispatch" if i % 2 else "schedule",
            published=bool(i % 2),
            **({} if i % 2 else {"rows": {}}),
        )
        for i in range(4, 24)
    ]
    # File enumeration order is not chronology.
    for r in reversed(records):
        (out / "runs" / f"{r['id']}.json").write_text(json.dumps(r))
    publish.rebuild(out, now, max_runs=max_runs)

    selected = json.loads((out / "latest.json").read_text())
    assert selected["id"] == "3" and selected["detail"] and selected["builds"]
    index = json.loads((out / "runs.json").read_text())["runs"]
    assert [r["id"] for r in index] == [previous, "3", "20", "21", "22", "23"]
    assert len(index) == max_runs
    assert [r["date"] for r in index] == sorted(r["date"] for r in index)
    for run_id in (previous, "3"):
        detail = json.loads((out / "runs" / f"{run_id}.json").read_text())
        assert detail["detail"] and detail["builds"]
    assert len(list((out / "runs").glob("*.json"))) == max_runs
    before = {p: p.read_text() for p in out.rglob("*.json")}
    publish.rebuild(out, now, max_runs=max_runs)
    assert {p: p.read_text() for p in out.rglob("*.json")} == before


def test_a_run_limit_below_the_comparison_keeps_only_the_comparison(publish, tmp_path):
    out = tmp_path / "sa-placer-hw"
    (out / "runs").mkdir(parents=True)
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    records = [
        hardware_record(
            str(i),
            publish.iso(now - datetime.timedelta(days=10 - i)),
            "full",
            event="schedule" if i < 3 else "workflow_dispatch",
            published=i < 3,
            **({} if i < 3 else {"rows": {}}),
        )
        for i in range(1, 8)
    ]
    for r in records:
        (out / "runs" / f"{r['id']}.json").write_text(json.dumps(r))
    with pytest.raises(ValueError, match="max_runs must be at least 1"):
        publish.rebuild(out, now, max_runs=0)
    assert len(list((out / "runs").glob("*.json"))) == len(records)
    publish.rebuild(out, now, max_runs=1)
    index = json.loads((out / "runs.json").read_text())["runs"]
    assert [r["id"] for r in index] == ["1", "2"]
    assert sorted(p.stem for p in (out / "runs").glob("*.json")) == ["1", "2"]


def test_comparison_protection_does_not_extend_the_retention_horizon(publish, tmp_path):
    out = tmp_path / "sa-placer-hw"
    (out / "runs").mkdir(parents=True)
    now = datetime.datetime(2026, 9, 29, tzinfo=datetime.timezone.utc)
    horizon = publish.KEEP_DAYS + 7 * publish.KEEP_WEEKS
    for i, days in enumerate((horizon + 1, horizon, 1), 1):
        record = hardware_record(
            str(i),
            publish.iso(now - datetime.timedelta(days=days)),
            "full",
            event="workflow_dispatch" if i == 3 else "schedule",
        )
        (out / "runs" / f"{i}.json").write_text(json.dumps(record))
    publish.rebuild(out, now)
    assert json.loads((out / "latest.json").read_text())["id"] == "3"
    assert [p.stem for p in (out / "runs").glob("*.json")] == ["3"]


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
    assert record["provenance"]["dispatch"] == "static"
    index = json.loads((out / "runs.json").read_text())
    assert index["metrics"] == ["failed_seeds", "us_per_image"]

    meta = {
        "preflight": {"pmode": "turbo", "device": "NPU Strix Halo"},
        "provenance": "commit 0123456789 | peano 22.0.0+0006955e | pmode turbo"
        " | dispatch dynamic",
        "efforts": {"default": 0.05, "full": 1.0},
        "iters": 20,
        "measurement_sane": True,
        "detail": [
            {
                "seed": 3,
                "batch": 16,
                "effort_label": "default",
                "passed": True,
                "us_per_image": 81.48,
            },
            {
                "seed": 2,
                "batch": 16,
                "effort_label": "default",
                "passed": False,
                "us_per_image": None,
            },
        ],
        "failed": ["mobilenet/seed=2/batch=16/effort=default"],
        "exitstatus": 1,
    }
    rows = [
        {"name": "mobilenet/failed_seeds", "unit": "seeds", "value": 1},
        {
            "name": "mobilenet/seed=3/effort=default/placement_cost",
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
    assert record["failed"] == ["mobilenet/seed=2/batch=16/effort=default"]
    assert record["published"]
    assert record["detail"] == meta["detail"]
    # The page reads the efforts the check ran from the record.
    assert record["efforts"] == meta["efforts"] and record["iters"] == 20
    # The raw device name is where the page keeps each part's times apart.
    assert record["provenance"]["device"] == "NPU Strix Halo"
    assert record["device"] == "Strix Halo" and record["pmode"] == "turbo"
    index = json.loads((out / "runs.json").read_text())
    assert index["metrics"] == ["failed_seeds", "placement_cost", "us_per_image"]
    assert all("detail" not in run for run in index["runs"])
    failed = json.loads((out / "history/failed_seeds.json").read_text())
    assert failed["target"] == "sa-placer-hw"
    assert failed["series"] == {"mobilenet": {"values": [0, 1]}}
    cost = json.loads((out / "history/placement_cost.json").read_text())
    assert cost["series"]["mobilenet/seed=3/effort=default"]["ranges"] == [
        None,
        "placement 0123456789ab",
    ]


def hardware_record(run_id, date, effort=None, **extra):
    suffix = f"/effort={effort}" if effort else ""
    return {
        "schema": 1,
        "target": "sa-placer-hw",
        "id": run_id,
        "url": "",
        "commit": {},
        "date": date,
        "pmode": "turbo",
        "device": "Strix",
        "provenance": {"dispatch": "dynamic"} if effort else {},
        "sane": True,
        "published": True,
        "n_rows": 3,
        "exitstatus": 1,
        "failed": [f"mobilenet/seed=7/batch=64{suffix}"],
        "truncated": [],
        "rows": {
            "mobilenet": {"failed_seeds": {"value": 1, "unit": "seeds"}},
            f"mobilenet/seed=3{suffix}": {
                "placement_cost": {"value": 300, "unit": "cost"},
                **({"compile_s": {"value": 27.1, "unit": "s"}} if effort else {}),
            },
            **{
                f"mobilenet/seed=3/batch={b}{suffix}": {
                    "us_per_image": {"value": us, "unit": "us"},
                    **({} if effort else {"compile_s": {"value": s, "unit": "s"}}),
                }
                for b, us, s in ((1, 292.1, 28.0), (16, 81.5, 27.7))
            },
        },
        "detail": [
            {
                "seed": 3,
                "batch": 16,
                "passed": True,
                "us_per_image": 81.5,
                **({"effort_label": effort} if effort else {}),
            }
        ],
        **extra,
    }


def test_a_static_hardware_run_keeps_its_times_and_gives_full_effort_its_compile(
    publish, tmp_path
):
    out = tmp_path / "sa-placer-hw"
    (out / "runs").mkdir(parents=True)
    old = hardware_record("old", "2026-10-16T06:52:54+00:00")
    new = hardware_record(
        "new", "2026-10-18T06:52:54+00:00", "full", event="workflow_dispatch"
    )
    (out / "runs/old.json").write_text(json.dumps(old))
    current = json.dumps(new, indent=2)
    (out / "runs/new.json").write_text(current)
    now = datetime.datetime(2026, 10, 19, tzinfo=datetime.timezone.utc)
    publish.rebuild(out, now)

    upgraded = json.loads((out / "runs/old.json").read_text())
    assert upgraded["provenance"] == {"dispatch": "static"}
    # The placement and the one compile a seed takes now continue at full
    # effort; the static design's per-batch times keep their own ids.
    assert upgraded["rows"] == {
        "mobilenet": {"failed_seeds": {"value": 1, "unit": "seeds"}},
        "mobilenet/seed=3/effort=full": {
            "placement_cost": {"value": 300, "unit": "cost"},
            "compile_s": {"value": 28.0, "unit": "s"},
        },
        "mobilenet/seed=3/batch=1": {"us_per_image": {"value": 292.1, "unit": "us"}},
        "mobilenet/seed=3/batch=16": {"us_per_image": {"value": 81.5, "unit": "us"}},
    }
    assert upgraded["failed"] == ["mobilenet/seed=7/batch=64/effort=full"]
    assert upgraded["detail"][0]["effort_label"] == "full"
    assert (out / "runs/new.json").read_text() == current
    latest = json.loads((out / "latest.json").read_text())
    assert latest["id"] == "old"
    per_image = json.loads((out / "history/us_per_image.json").read_text())
    assert per_image["series"] == {
        "mobilenet/seed=3/batch=1": {"values": [292.1, None]},
        "mobilenet/seed=3/batch=1/effort=full": {"values": [None, 292.1]},
        "mobilenet/seed=3/batch=16": {"values": [81.5, None]},
        "mobilenet/seed=3/batch=16/effort=full": {"values": [None, 81.5]},
    }
    compile_s = json.loads((out / "history/compile_s.json").read_text())
    assert compile_s["series"] == {
        "mobilenet/seed=3/effort=full": {"values": [28.0, 27.1]}
    }
    assert [r["provenance"]["dispatch"] for r in compile_s["runs"]] == [
        "static",
        "dynamic",
    ]

    before = {p.name: p.read_text() for p in out.rglob("*.json")}
    publish.rebuild(out, now)
    assert {p.name: p.read_text() for p in out.rglob("*.json")} == before


@pytest.mark.parametrize("target", ["sa-placer", "llama-chain-hw", "yolo26n-hw"])
def test_only_the_hardware_check_has_its_cases_upgraded(publish, target):
    record = {**hardware_record("r", "2026-10-16T06:52:54+00:00"), "target": target}
    assert not publish.upgrade(record)
    assert record == {
        **hardware_record("r", "2026-10-16T06:52:54+00:00"),
        "target": target,
    }


def kernel_record(target, date, cycles):
    return {
        "target": target,
        "id": "r",
        "date": date,
        "provenance": {"peano": "22.0.0+e1"},
        "rows": {
            case: {
                "cycles": {"value": cycles, "unit": "cycles"},
                "cycles_per_kop": {"value": 1.5, "unit": "cycles"},
                "dispatch_us": {"value": 3378.0, "unit": "us"},
            }
            for case in sorted(KERNEL_CASES)
        },
    }


KERNEL_CASES = {
    # Over two timer wraps per call before the 10-02 decoder fix.
    "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16",
    # Under them: decoded right either way.
    "bn_conv2dk3_dw/3360x8/uint8/input_channels=120/input_width=28/output_channels=120",
}


def test_cycles_the_old_trace_decoder_understated_are_dropped(publish):
    record = kernel_record("npu2", "2026-10-01T06:16:12+00:00", 293058)
    assert publish.upgrade(record)
    assert record["provenance"] == {"peano": "22.0.0+e1", "measurement": "1"}
    wrapped, unwrapped = sorted(KERNEL_CASES)
    assert record["rows"][wrapped] == {"dispatch_us": {"value": 3378.0, "unit": "us"}}
    assert set(record["rows"][unwrapped]) == {"cycles", "cycles_per_kop", "dispatch_us"}
    assert not publish.upgrade(record)

    # From the fix on, and on npu1, every point stays.
    for target, date in (
        ("npu2", "2026-10-02T06:16:12+00:00"),
        ("npu1", "2026-09-30T06:16:12+00:00"),
    ):
        record = kernel_record(target, date, 301)
        assert publish.upgrade(record)
        assert all(len(cells) == 3 for cells in record["rows"].values())
    assert record["provenance"]["measurement"] == "1"
    # A record that names its measurement keeps it.
    record = kernel_record("npu2", "2026-09-30T06:16:12+00:00", 293058)
    record["provenance"]["measurement"] = "3"
    assert not publish.upgrade(record)


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
    assert (
        '<script>location.replace("../dashboard/" + (location.hash || "#view=night"));</script>'
        in page
    )
    assert (
        '<meta http-equiv="refresh" content="0; url=../dashboard/#view=night">' in page
    )


def test_a_host_time_from_before_the_split_is_its_dispatch_time(publish):
    old = kernel_record("npu1", "2026-10-20T06:16:12+00:00", 900)
    old["provenance"]["measurement"] = "2"
    for cells in old["rows"].values():
        cells["npu_us"] = cells.pop("dispatch_us")
    assert publish.upgrade(old)
    for cells in old["rows"].values():
        assert "npu_us" not in cells
        assert cells["dispatch_us"] == {"value": 3378.0, "unit": "us"}
    assert not publish.upgrade(old)

    new = kernel_record("npu1", "2026-10-22T06:16:12+00:00", 900)
    new["provenance"]["measurement"] = "2"
    for cells in new["rows"].values():
        cells["npu_us"] = {"value": 3290.5, "unit": "us"}
    assert not publish.upgrade(new)
    for cells in new["rows"].values():
        assert cells["npu_us"]["value"] == 3290.5
