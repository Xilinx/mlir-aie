# test_kernel_catalogue.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""The results page's catalogue lists every factory with its NPU's verdicts (no NPU)."""

import json
import sys
from pathlib import Path

from aie.iron import kernels

sys.path.insert(0, str(Path(__file__).parents[2] / "utils" / "kernel_checks"))
import catalogue  # noqa: E402

JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="test_kernels_e2e" name="test_kernel_extensive[gelu/1024x16/bfloat16/random/s0]"/>
<testcase classname="test_kernels_e2e" name="test_kernel_extensive[gelu/1024x16/bfloat16/zeros/s1]"/>
<testcase classname="test_kernels_e2e" name="test_kernel_extensive[gelu/1024x256/bfloat16/random/s0]"/>
<testcase classname="test_kernels_e2e" name="test_kernel_extensive[gelu/1024x256/bfloat16/large/s2]"><failure message="mismatch"/></testcase>
<testcase classname="test_kernels_e2e" name="test_kernel_extensive[softmax/1024x16/bfloat16/random/s0]"><skipped message="npu2 only"/></testcase>
<testcase classname="test_kernels_e2e" name="test_kernel_smoke[tanh/1024x16/bfloat16]"/>
</testsuite></testsuites>
"""

PERF = [
    {"name": "gelu/1024x16/bfloat16/npu_us", "unit": "us", "value": 270.0},
    {"name": "gelu/1024x16/bfloat16/cycles", "unit": "cycles", "value": 9000},
    {"name": "gelu/1024x256/bfloat16/npu_us", "unit": "us", "value": 2100.0},
]


def _write(tmp_path, argv):
    out = tmp_path / "catalogue.json"
    assert catalogue.main([*argv, "--out", str(out)]) == 0
    result = json.loads(out.read_text())
    return result, {k["factory"]: k for k in result["kernels"]}


def test_sweep_and_perf_verdicts(tmp_path):
    (tmp_path / "correctness.xml").write_text(JUNIT)
    (tmp_path / "perf.json").write_text(json.dumps(PERF))
    result, rows = _write(
        tmp_path,
        [
            "--npu",
            "npu1",
            "--correctness",
            str(tmp_path / "correctness.xml"),
            "--perf",
            str(tmp_path / "perf.json"),
        ],
    )
    assert result["arch"] == "aie2" and result["swept"]
    # One failing input or seed fails its case; a skipped case is not a pass.
    assert rows["gelu"]["passed"] == 1
    assert rows["gelu"]["failed"] == ["gelu/1024x256/bfloat16"]
    assert rows["gelu"]["timed"] == 2
    assert rows["softmax"]["passed"] == 0 and rows["softmax"]["failed"] == []
    assert rows["tanh"]["passed"] == 0
    assert rows["gelu"]["family"] == "activation"
    assert rows["gelu"]["sources"] == ["activation/gelu.cc"]


def test_every_factory_is_listed_with_its_sources(tmp_path):
    for npu, arch in [("npu1", "aie2"), ("npu2", "aie2p")]:
        result, rows = _write(tmp_path, ["--npu", npu])
        assert result["arch"] == arch and not result["swept"]
        assert list(rows) == list(kernels.factories())
        for row in rows.values():
            assert row["summary"] and "``" not in row["summary"]
            assert bool(row["sources"]) == bool(row["builds"])
            assert row["passed"] == row["timed"] == 0 and row["failed"] == []
        assert bool(rows["mm_bfp"]["builds"]) == (arch == "aie2p")
