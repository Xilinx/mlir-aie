# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Build the PR report from a run's artifacts and a nightly series on disk."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils/kernel_checks/pr_report.py"
EXTRA = "commit 41dcf3cd6a | peano {} | kernels c86c05864e | pmode default"


def _rows(values, peano):
    return [
        {"name": name, "unit": unit, "value": value, "extra": EXTRA.format(peano)}
        for name, (unit, value) in values.items()
    ]


def _series(values, peano="22.0.0+old", suite="aie_kernels (npu2, default)"):
    commit = {"id": "0123456789ab", "url": "https://example.com/c/0123456"}
    entry = {"commit": commit, "date": 1, "benches": _rows(values, peano)}
    return {"entries": {suite: [entry]}}


def _junit(cases):
    tests = "".join(
        f'<testcase name="test_kernel_extensive[{name}]">{body}</testcase>'
        for name, body in cases
    )
    return f"<testsuites><testsuite>{tests}</testsuite></testsuites>"


@pytest.fixture
def report(tmp_path):
    def run(results, series=None):
        for npu, files in results.items():
            leg = tmp_path / "results" / npu
            leg.mkdir(parents=True)
            for name, content in files.items():
                text = content if isinstance(content, str) else json.dumps(content)
                (leg / name).write_text(text)
        if series is not None:
            base = tmp_path / "baselines/npu2"
            base.mkdir(parents=True)
            (base / "series.json").write_text(json.dumps(series))
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


META = {"preflight": {"npu": "npu2", "pmode": "default"}, "failed": []}


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
        _series(nightly),
    )
    assert text.startswith("<!-- kernel-checks-report -->\n")
    assert "## Kernel checks: 2 failing, 1 regressed" in text
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
        _series(values, peano="22.0.0+a"),
    )
    assert "## Kernel checks: all passed, no regressions" in text
    assert "| npu1 | ? | — | no results |" in text
    assert (
        "| npu2 | 22.0.0+a | [0123456](https://example.com/c/0123456) | 1 | 0 | 0 | 0 |"
        in text
    )
    assert "### " not in text and "<details>" not in text


def test_power_mode_suite_is_preferred(report):
    values = {"relu/1024/bf16/cycles": ("cycles", 1000)}
    series = _series({"relu/1024/bf16/cycles": ("cycles", 500)})
    series["entries"]["aie_kernels (npu2, turbo)"] = _series(values)["entries"][
        "aie_kernels (npu2, default)"
    ]
    series["entries"]["aie_kernels (npu2, turbo)"][0]["date"] = 2
    text = report(
        {"npu2": {"meta.json": META, "perf.json": _rows(values, "a")}}, series
    )
    assert "1 regressed" in text


def test_no_baseline(report):
    values = {"relu/1024/bf16/cycles": ("cycles", 1000)}
    text = report({"npu2": {"meta.json": META, "perf.json": _rows(values, "a")}})
    assert "| npu2 | a | none cached | 1 | 0 | — | — |" in text


def test_no_results(report):
    assert "## Kernel checks: no NPU produced results" in report({})
