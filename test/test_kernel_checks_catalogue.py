# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Catalogue behavior with synthetic factories, independent of the kernel tree."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

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


FACTORIES = [
    {"factory": "synthetic", "builds": ["synthetic"]},
    {"factory": "uncased", "builds": ["uncased"], "why": "needs an external input"},
    {"factory": "unavailable", "builds": []},
]


def test_rows_aggregate_variants_and_attribute_outcomes(catalogue, tmp_path):
    cases = [
        ("synthetic/1/i8/random/s0", ""),
        ("synthetic/1/i8/ones/s0", ""),
        ("synthetic/2/i8/random/s0", ""),
        ("synthetic/3/i8/random/s0", '<failure message="bad" />'),
        ("synthetic/3/i8/random/s1", ""),
        ("synthetic/4/i8/random/s0", "<skipped />"),
        ("synthetic/5/i8/random/s0", ""),
    ]
    correctness = tmp_path / "correctness.xml"
    correctness.write_text(
        "<testsuite>"
        + "".join(
            f'<testcase name="test_kernel_extensive[{name}]">{body}</testcase>'
            for name, body in cases
        )
        + "</testsuite>"
    )
    perf = tmp_path / "perf.json"
    perf.write_text(
        json.dumps(
            [
                {"name": "synthetic/1/i8/cycles", "value": 1},
                {"name": "synthetic/1/i8/npu_us", "value": 2},
            ]
        )
    )
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "failed": [
                    "test_kernel_perf[synthetic/2/i8]",
                    "test_kernel_extensive[synthetic/3/i8/random/s0]",
                    "test_measurement_is_sane",
                ]
            }
        )
    )
    passed, failed = catalogue.swept(correctness)
    declared = {"synthetic": {f"synthetic/{i}/i8": i != 5 for i in range(1, 6)}}
    rows = catalogue.rows(
        FACTORIES,
        passed,
        failed,
        catalogue.timed(perf),
        catalogue.timing_failed(meta),
        declared,
    )
    by_name = {row["factory"]: row for row in rows}
    measured = by_name["synthetic"]
    assert measured["passed"] == 3
    assert measured["failed"] == ["synthetic/3/i8"]
    assert measured["timed"] == 1
    assert measured["timing_failed"] == ["synthetic/2/i8"]
    assert measured["untimed"] == ["synthetic/5/i8"]
    assert "reason" not in measured
    assert by_name["uncased"]["reason"] == "needs an external input"
    assert "reason" not in by_name["unavailable"]


def test_rows_without_artifacts_distinguish_built_and_unavailable(catalogue):
    rows = catalogue.rows(FACTORIES, {}, {}, {}, {}, {})
    assert all(
        (r["passed"], r["failed"], r["timed"], r["timing_failed"], r["untimed"])
        == (0, [], 0, [], [])
        for r in rows
    )
    assert [r.get("reason") for r in rows] == [
        catalogue.NO_CASE,
        "needs an external input",
        None,
    ]


@pytest.mark.parametrize(
    "factory,reason",
    [
        (NS(contract=NS(unsupported="external result", trace=None)), "external result"),
        (
            NS(
                contract=NS(
                    unsupported=None, trace=NS(shape="none", reason="runs once")
                )
            ),
            "runs once",
        ),
        (
            NS(
                contract=NS(unsupported=None, trace=NS(shape="whole_call", reason=None))
            ),
            None,
        ),
        (NS(contract=None), None),
    ],
)
def test_uncased_reason_comes_from_contract(catalogue, factory, reason):
    assert catalogue.why_uncased(factory) == reason
