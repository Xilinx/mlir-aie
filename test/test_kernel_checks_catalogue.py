# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""How the catalogue attributes a run's outcomes to each factory."""

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
                    "test_pairs::test_pair[shape]",
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


@pytest.mark.parametrize("outcome", ["", "failure", "error", "skipped"])
def test_dedicated_pair_outcome_is_attributed_to_both_factories(
    catalogue, tmp_path, outcome
):
    correctness = tmp_path / "correctness.xml"
    correctness.write_text(
        '<testsuite><testcase name="test_pair">'
        '<properties><property name="kernel_check" value="get" />'
        '<property name="kernel_check" value="put" /></properties>'
        + (f'<{outcome} message="bad" />' if outcome else "")
        + "</testcase></testsuite>"
    )
    passed, failed = catalogue.swept(correctness)
    declared = {}
    for case, variant, _ in catalogue.pr_report.sweep(
        correctness, include_skipped=True
    ):
        assert variant == "dedicated"
        declared.setdefault(case.split("/", 1)[0], {})[case] = False
    rows = catalogue.rows(
        [{"factory": name, "builds": [name]} for name in ("get", "put")],
        passed,
        failed,
        {},
        {},
        declared,
    )
    for row in rows:
        case = f"{row['factory']}/test_pair"
        assert row["passed"] == (1 if not outcome else 0)
        assert row["failed"] == ([case] if outcome in ("failure", "error") else [])
        assert row["timed"] == 0
        assert row["untimed"] == ([case] if not outcome else [])
        assert "reason" not in row
