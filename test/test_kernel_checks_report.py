# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Build the PR report from a run's artifacts and the published baselines on disk."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils/kernel_checks/pr_report.py"
EXTRA = "commit 41dcf3cd6a | peano {} | kernels c86c05864e | pmode performance"


def _rows(values, peano):
    return [
        {"name": name, "unit": unit, "value": value, "extra": EXTRA.format(peano)}
        for name, (unit, value) in values.items()
    ]


def _record(values, peano="22.0.0+old", target="npu2"):
    """Return a publish.py record: the last nightly's rows, by case and metric."""
    rows = {}
    for name, (unit, value) in values.items():
        case, metric = name.rsplit("/", 1)
        rows.setdefault(case, {})[metric] = {"value": value, "unit": unit}
    return {
        "kind": "perf" if target.startswith("npu") else "static",
        "target": target,
        "id": "77",
        "commit": {"id": "0123456789ab", "url": "https://example.com/c/0123456"},
        "date": "2026-09-28T06:00:00+00:00",
        "provenance": {"commit": "0123456789", "peano": peano},
        "published": True,
        "rows": rows,
    }


def _junit(cases):
    tests = "".join(
        f'<testcase name="test_kernel_extensive[{name}]">{body}</testcase>'
        for name, body in cases
    )
    return f"<testsuites><testsuite>{tests}</testsuite></testsuites>"


@pytest.fixture
def report(tmp_path):
    def run(results, baselines=None):
        for leg, files in results.items():
            d = tmp_path / "results" / leg
            d.mkdir(parents=True)
            for name, content in files.items():
                text = content if isinstance(content, str) else json.dumps(content)
                (d / name).write_text(text)
        for leg, record in (baselines or {}).items():
            base = tmp_path / "baselines" / leg
            base.mkdir(parents=True)
            (base / "latest.json").write_text(json.dumps(record))
        (tmp_path / "results").mkdir(exist_ok=True)
        out = tmp_path / "report.md"
        subprocess.run(
            [sys.executable, SCRIPT, "--results", tmp_path / "results"]
            + ["--baselines", tmp_path / "baselines", "--out", out]
            + ["--run-url", "https://example.com/run/1"],
            check=True,
        )
        return out.read_text()

    return run


META = {"preflight": {"npu": "npu2", "pmode": "performance"}, "failed": []}


def test_failures_regressions_and_coverage(report):
    nightly = {
        "relu/1024/bf16/cycles": ("cycles", 1000),
        "relu/1024/bf16/core_elf_bytes": ("bytes", 4096),
        "relu/1024/bf16/npu_us": ("us", 100),
        "gelu/1024/bf16/cycles": ("cycles", 2000),
        "gone/64/i8/cycles": ("cycles", 10),
    }
    run = {
        "relu/1024/bf16/cycles": ("cycles", 1100),
        "relu/1024/bf16/cycles_per_kop": ("cycles/1k-ops", 90),
        "relu/1024/bf16/core_elf_bytes": ("bytes", 4096),
        "relu/1024/bf16/npu_us": ("us", 150),
        "gelu/1024/bf16/cycles": ("cycles", 1500),
        "fresh/64/i8/cycles": ("cycles", 5),
    }
    junit = _junit(
        [
            ("relu/1024/bf16/random/s0", ""),
            ("mm/64x64/i8/random/s0", '<failure message="x.cc:3: error: bad" />'),
            ("mm/64x64/i8/random/s1", ""),
            ("mm/64x64/i8/ones/s0", "<skipped />"),
        ]
    )
    meta = dict(META, failed=["test_kernels_perf.py::test_kernel_perf[conv/8/i8]"])
    text = report(
        {
            "npu2": {
                "meta.json": meta,
                "perf.json": _rows(run, "22.0.0+new"),
                "correctness.xml": junit,
            }
        },
        {"npu2": _record(nightly)},
    )
    assert text.startswith("<!-- kernel-checks-report -->\n")
    assert "## Kernel checks: 2 failing, 1 regressed" in text
    assert "on hardware, `cycles` 2% or `core_elf_bytes` 2% or more worse" in text
    assert "static checks" not in text
    assert "| npu2 | 22.0.0+new (nightly: 22.0.0+old) | [0123456]" in text
    assert "| `mm/64x64/i8` | 1 of 2 | bad |" in text
    assert "| `conv/8/i8` | timing run | failed in the timing run" in text
    assert (
        "| `relu/1024/bf16` | cycles | 1,000 cycles | 1,100 cycles | +10.0% |" in text
    )
    assert (
        "| `gelu/1024/bf16` | cycles | 2,000 cycles | 1,500 cycles | -25.0% |" in text
    )
    assert "| `relu/1024/bf16` | npu_us | 100 us | 150 us | +50.0% |" in text
    assert "cycles_per_kop" not in text
    assert "<summary>In the nightly but not measured here (1)</summary>" in text
    assert "| npu2 | `gone/64/i8` |" in text
    assert "<summary>Measured here, not in the nightly (1)</summary>" in text


def test_clean_run_and_missing_leg(report):
    values = {"relu/1024/bf16/cycles": ("cycles", 1000)}
    text = report(
        {
            "npu1": {},
            "npu2": {"meta.json": META, "perf.json": _rows(values, "22.0.0+a")},
        },
        {"npu2": _record(values, peano="22.0.0+a")},
    )
    assert "## Kernel checks: all passed, no regressions" in text
    assert "| npu1 | ? | — | no results |" in text
    assert (
        "| npu2 | 22.0.0+a | [0123456](https://example.com/c/0123456) | 1 | 0 | 0 | 0 |"
        in text
    )
    assert "### " not in text and "<details>" not in text


def test_static_legs_gate_on_any_increase(report):
    nightly = {
        "softmax/unpipelined_loops": ("loops", 0),
        "softmax/loop/softmax_bf16/for.body/II": ("cycles", 4),
        "softmax/pm_bytes": ("bytes", 2000),
        "gelu/pm_bytes": ("bytes", 1000),
        "gelu/libcalls": ("calls", 2),
    }
    run = {
        "softmax/unpipelined_loops": ("loops", 1),
        "softmax/loop/softmax_bf16/for.body/II": ("cycles", 5),
        "softmax/pm_bytes": ("bytes", 2020),
        "gelu/pm_bytes": ("bytes", 1040),
        "gelu/libcalls": ("calls", 1),
    }
    extra = "commit 41dcf3cd6a | peano 22.0.0+new | target aie2p"
    rows = [
        {"name": n, "unit": u, "value": v, "extra": extra} for n, (u, v) in run.items()
    ]
    text = report(
        {
            "aie2p": {
                "static.json": [r for r in rows if not r["name"].endswith("pm_bytes")],
                "static-pm.json": [r for r in rows if r["name"].endswith("pm_bytes")],
                "static-meta.json": {
                    "kernels": {"softmax": {}, "gelu": {}},
                    "failed": ["tanh: clang exited 1"],
                },
            }
        },
        {"aie2p": _record(nightly, target="aie2p")},
    )
    assert "## Kernel checks: 1 failing, 3 regressed" in text
    assert "any increase in `II`, `not_zol`, `unpipelined_loops`" in text
    assert "`pm_bytes` 3% or more" in text
    assert "on hardware" not in text
    assert "| aie2p | 22.0.0+new (nightly: 22.0.0+old) | [0123456]" in text
    assert "| aie2p | `tanh` | compile | clang exited 1 |" in text
    assert "| `softmax` | unpipelined_loops | 0 loops | 1 loops | from 0 |" in text
    assert (
        "| `softmax/loop/softmax_bf16/for.body` | II | 4 cycles | 5 cycles | +25.0% |"
        in text
    )
    assert "| `gelu` | pm_bytes | 1.0 KiB | 1.0 KiB | +4.0% |" in text
    # 1% of program memory is under its slack; one call fewer is an improvement.
    assert "| `softmax` | pm_bytes |" not in text
    assert "<summary>Improved (1)</summary>" in text
    assert "| `gelu` | libcalls | 2 calls | 1 calls | -50.0% |" in text


def test_no_baseline(report):
    values = {"relu/1024/bf16/cycles": ("cycles", 1000)}
    text = report({"npu2": {"meta.json": META, "perf.json": _rows(values, "a")}})
    assert "| npu2 | a | none cached | 1 | 0 | — | — |" in text


def test_a_baseline_without_rows_is_no_baseline(report):
    values = {"relu/1024/bf16/cycles": ("cycles", 1000)}
    empty = dict(_record({}), published=False)
    text = report(
        {"npu2": {"meta.json": META, "perf.json": _rows(values, "a")}}, {"npu2": empty}
    )
    assert "| npu2 | a | none cached | 1 | 0 | — | — |" in text


def test_no_results(report):
    assert "## Kernel checks: no leg produced results" in report({})
