# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Assemble a catalogue column from a run's artifacts; no aie package needed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "utils/kernel_checks"


@pytest.fixture(scope="module")
def catalogue():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(
            "catalogue", SCRIPTS / "catalogue.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _junit(cases):
    tests = "".join(
        f'<testcase name="test_kernel_extensive[{name}]">{body}</testcase>'
        for name, body in cases
    )
    return f"<testsuites><testsuite>{tests}</testsuite></testsuites>"


FACTORIES = [
    {
        "factory": "softmax",
        "family": "activation",
        "summary": "Softmax",
        "sources": ["activation/softmax.cc"],
        "builds": ["softmax", "softmax/dtype=bfloat16"],
    },
    {
        "factory": "zero",
        "family": "zero",
        "summary": "Zero",
        "sources": ["common/zero.h"],
        "builds": ["zero"],
    },
    {
        "factory": "cascade_mm",
        "family": "linalg",
        "summary": "Cascade",
        "sources": ["linalg/mm.cc"],
        "builds": ["cascade_mm"],
    },
    {
        "factory": "exp2f_vec",
        "family": "activation",
        "summary": "Only on npu2",
        "sources": [],
        "builds": [],
    },
]


def test_rows_attribute_every_case_outcome(catalogue, tmp_path):
    correctness = tmp_path / "correctness.xml"
    correctness.write_text(
        _junit(
            [
                ("softmax/1024x16/bfloat16/random/s0", ""),
                ("softmax/1024x16/bfloat16/large/s0", ""),
                ("softmax/64x16/bfloat16/random/s0", ""),
                ("softmax/2048x16/bfloat16/random/s0", '<failure message="bad" />'),
                ("softmax/2048x16/bfloat16/random/s1", ""),
                ("zero/64/int32/random/s0", ""),
                ("zero/64/bfloat16/random/s0", "<skipped />"),
            ]
        )
    )
    perf = tmp_path / "perf.json"
    perf.write_text(
        json.dumps(
            [
                {"name": "softmax/1024x16/bfloat16/cycles", "value": 1},
                {"name": "softmax/1024x16/bfloat16/npu_us", "value": 2},
            ]
        )
    )
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "failed": [
                    "test_kernels_perf.py::test_kernel_perf[softmax/64x16/bfloat16]",
                    "test_kernels_e2e.py::test_kernel_extensive[softmax/2048x16/bfloat16/random/s0]",
                    "test_kernels_perf.py::test_measurement_is_sane",
                ]
            }
        )
    )
    declared = {
        "softmax": {
            "softmax/1024x16/bfloat16": True,
            "softmax/64x16/bfloat16": True,
            "softmax/2048x16/bfloat16": True,
        },
        "zero": {"zero/64/int32": False, "zero/64/bfloat16": False},
    }
    passed, failed = catalogue.swept(correctness)
    rows = catalogue.rows(
        FACTORIES,
        passed,
        failed,
        catalogue.timed(perf),
        catalogue.timing_failed(meta),
        declared,
    )
    by_name = {row["factory"]: row for row in rows}

    softmax = by_name["softmax"]
    assert softmax["passed"] == 2
    assert softmax["failed"] == ["softmax/2048x16/bfloat16"]
    assert softmax["timed"] == 1
    assert softmax["timing_failed"] == ["softmax/64x16/bfloat16"]
    assert softmax["untimed"] == []
    assert "reason" not in softmax

    zero = by_name["zero"]
    assert (zero["passed"], zero["timed"]) == (1, 0)
    assert zero["untimed"] == ["zero/64/int32"]  # the skipped case is not "passed"
    assert "reason" not in zero

    cascade = by_name["cascade_mm"]
    assert (cascade["passed"], cascade["timed"], cascade["failed"]) == (0, 0, [])
    assert cascade["reason"] == catalogue.NO_CASE

    # Not built on this NPU: nothing to explain.
    assert "reason" not in by_name["exp2f_vec"]
    assert [row["factory"] for row in rows] == [f["factory"] for f in FACTORIES]


def test_rows_without_artifacts(catalogue):
    rows = catalogue.rows(FACTORIES, {}, {}, {}, {}, {})
    assert all(
        (r["passed"], r["failed"], r["timed"], r["timing_failed"], r["untimed"])
        == (0, [], 0, [], [])
        for r in rows
    )
    assert [r.get("reason") for r in rows] == [catalogue.NO_CASE] * 3 + [None]


def test_timing_failures_ignore_the_sweep_and_the_sanity_test(catalogue, tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "failed": [
                    "test_kernels_e2e.py::test_kernel_extensive[mm/64x64/i8/ones/s0]",
                    "test_kernels_perf.py::test_measurement_is_sane",
                    "test_kernels_perf.py::test_kernel_perf[mm/64x32x64x4/bfloat16_float32/b_col_maj=True]",
                ]
            }
        )
    )
    assert catalogue.timing_failed(meta) == {
        "mm": {"mm/64x32x64x4/bfloat16_float32/b_col_maj=True"}
    }
    meta.write_text("{}")
    assert catalogue.timing_failed(meta) == {}
