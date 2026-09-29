# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Compare synthetic run artifacts with published baselines; no NPU needed."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "utils/kernel_checks/pr_report.py"
META = {"preflight": {"npu": "npu2", "pmode": "performance"}, "failed": []}


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("pr_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rows(values):
    rows = []
    for name, (unit, value) in values.items():
        value, span = value if isinstance(value, tuple) else (value, None)
        row = {"name": name, "unit": unit, "value": value}
        if span:
            row["range"] = span
        rows.append(row)
    return rows


def _record(values, **overrides):
    rows = {}
    for row in _rows(values):
        case, metric = row.pop("name").rsplit("/", 1)
        rows.setdefault(case, {})[metric] = row
    return {
        "id": "baseline-run",
        "pmode": "performance",
        "rows": rows,
        **overrides,
    }


@pytest.fixture
def read_leg(report, tmp_path):
    def read(values, baseline=None, meta=META, cases=(), run_id="current-run"):
        (tmp_path / "perf.json").write_text(json.dumps(_rows(values)))
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        tests = "".join(
            f'<testcase name="test_kernel_extensive[{name}]">{body}</testcase>'
            for name, body in cases
        )
        (tmp_path / "correctness.xml").write_text(f"<testsuite>{tests}</testsuite>")
        latest = tmp_path / "latest.json"
        if baseline is not None:
            latest.write_text(json.dumps(baseline))
        else:
            latest.unlink(missing_ok=True)
        return report.read_leg("npu2", tmp_path, latest, run_id=run_id)

    return read


def test_comparison_classifies_changes_and_coverage(read_leg):
    nightly = {
        "slower/1/i8/cycles": ("cycles", 1000),
        "larger/1/i8/kernel_object_bytes": ("bytes", 1000),
        "faster/1/i8/cycles": ("cycles", 2000),
        "slower/1/i8/npu_us": ("us", 100),
        "slower/1/i8/cycles_per_kop": ("cycles/1k-ops", 10),
        "gone/1/i8/cycles": ("cycles", 10),
        "failed/1/i8/cycles": ("cycles", 10),
    }
    current = {
        "slower/1/i8/cycles": ("cycles", 1100),
        "larger/1/i8/kernel_object_bytes": ("bytes", 1100),
        "faster/1/i8/cycles": ("cycles", 1500),
        "slower/1/i8/npu_us": ("us", 150),
        "slower/1/i8/cycles_per_kop": ("cycles/1k-ops", 90),
        "fresh/1/i8/cycles": ("cycles", 5),
    }
    leg = read_leg(
        current,
        _record(nightly),
        meta=dict(META, failed=["test_kernel_perf[failed/1/i8]"]),
    )
    assert {(c.case, c.metric) for c in leg.regressed} == {
        ("slower/1/i8", "cycles"),
        ("larger/1/i8", "kernel_object_bytes"),
    }
    assert [(c.case, c.ratio) for c in leg.improved] == [("faster/1/i8", -0.25)]
    assert [(c.metric, c.before, c.after) for c in leg.other] == [("npu_us", 100, 150)]
    assert leg.unmeasured == ["gone/1/i8"]
    assert leg.new == ["fresh/1/i8"]


def test_failures_combine_sweep_and_timing_without_counting_skips(read_leg):
    leg = read_leg(
        {},
        cases=[
            ("synthetic/1/i8/random/s0", '<failure message="x.cc:3: error: bad" />'),
            ("synthetic/1/i8/random/s1", ""),
            ("synthetic/1/i8/ones/s0", "<skipped />"),
        ],
        meta=dict(
            META,
            failed=[
                "test_kernel_extensive[synthetic/1/i8/random/s0]",
                "test_kernel_perf[timing/1/i8]",
            ],
        ),
    )
    failures = {f.case: f for f in leg.failures}
    assert set(failures) == {"synthetic/1/i8", "timing/1/i8"}
    assert failures["synthetic/1/i8"].total == 2
    assert failures["synthetic/1/i8"].failed == ["random/s0"]
    assert failures["synthetic/1/i8"].reason == "bad"
    assert failures["timing/1/i8"].failed == ["timing run"]


@pytest.mark.parametrize(
    "before_mad,after_mad,after,changed",
    [
        (1, 1, 120, True),
        (10, 1, 120, False),
        (1, 7, 120, False),
        (None, None, 120, True),
        (0.1, 0.1, 105, False),
    ],
)
def test_host_timing_change_must_exceed_threshold_and_noise(
    read_leg, before_mad, after_mad, after, changed
):
    def values(value, mad):
        span = None if mad is None else f"\u00b1 {mad}; min 1 max 2 n=50"
        return {"synthetic/1/i8/npu_us": ("us", (value, span))}

    leg = read_leg(values(after, after_mad), _record(values(100, before_mad)))
    assert bool(leg.other) is changed
    assert not leg.regressed


@pytest.mark.parametrize("record_kind", ["absent", "empty", "own-run", "new-schema"])
def test_unusable_baseline_does_not_produce_comparisons(read_leg, record_kind):
    values = {"synthetic/1/i8/cycles": ("cycles", 100)}
    baseline = {
        "absent": None,
        "empty": _record({}),
        "own-run": _record(values, id="current-run"),
        "new-schema": _record(values, schema=2),
    }[record_kind]
    leg = read_leg({"synthetic/1/i8/cycles": ("cycles", 150)}, baseline)
    assert leg.measured and leg.cases == 1
    assert not (leg.regressed or leg.improved or leg.other or leg.new)


@pytest.mark.parametrize(
    "current_mode,baseline_mode,compared",
    [
        ("performance", "performance", True),
        ("turbo", "turbo", True),
        ("turbo", "performance", False),
        ("performance", None, False),
        (None, "performance", False),
        (None, None, False),
        ("", "", False),
    ],
)
def test_baseline_requires_matching_known_power_modes(
    read_leg, current_mode, baseline_mode, compared
):
    baseline = _record({"synthetic/1/i8/cycles": ("cycles", 100)})
    preflight = {}
    if current_mode is not None:
        preflight["pmode"] = current_mode
    if baseline_mode is None:
        baseline.pop("pmode")
    else:
        baseline["pmode"] = baseline_mode
    leg = read_leg(
        {"synthetic/1/i8/cycles": ("cycles", 150)},
        baseline,
        meta={"preflight": preflight},
    )
    assert bool(leg.regressed) is compared


def test_cli_writes_report(tmp_path):
    results = tmp_path / "results/npu2"
    results.mkdir(parents=True)
    (results / "meta.json").write_text(json.dumps(META))
    (results / "perf.json").write_text(
        json.dumps(_rows({"synthetic/1/i8/cycles": ("cycles", 100)}))
    )
    out = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            SCRIPT,
            "--results",
            results.parent,
            "--baselines",
            tmp_path / "baselines",
            "--out",
            out,
            "--run-id",
            "current-run",
        ],
        check=True,
    )
    text = out.read_text()
    assert "<!-- kernel-checks-report -->" in text
    assert "all passed, no regressions" in text
    assert "none cached" in text
