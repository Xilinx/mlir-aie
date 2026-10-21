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
META = {
    "preflight": {"npu": "npu2", "pmode": "performance", "device": "NPU Strix"},
    "failed": [],
}


@pytest.fixture(scope="module")
def report():
    # pr_report imports publish, its neighbour, as it does when run.
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("pr_report", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
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
        "device": "Strix",
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
    assert failures["timing/1/i8"].reason == "failed in the timing run; see perf.log"
    assert leg.inputs == 2


def test_timing_failure_reason_and_coverage_reach_the_comment(read_leg, report):
    nodeid = "test_kernel_perf[timing/1/i8]"
    leg = read_leg(
        {},
        cases=[("timing/1/i8/random/s0", "")],
        meta=dict(
            META,
            failed=[nodeid],
            reasons={
                nodeid: "AssertionError: timing/1/i8: "
                "wrong on run 62 (right on the first)"
            },
            runs_per_case=62,
        ),
    )
    assert leg.failures[0].reason == "wrong on run 62 (right on the first)"
    assert leg.runs == 62
    comment = report.render([leg])
    assert "Checked 1 inputs per NPU (random and edge) and each timed case" in comment
    assert "after 62 runs in a row" in comment
    assert "`seeds` or `iters`" in comment
    assert "Checked" not in report.render([report.Leg("npu2", measured=False)])


def test_dedicated_checks_report_failures_without_crediting_unmarked_tests(
    report, tmp_path
):
    (tmp_path / "correctness.xml").write_text(
        '<testsuite><testcase classname="test_pairs" name="test_pair[shape]">'
        '<properties><property name="kernel_check" value="get" />'
        '<property name="kernel_check" value="put" />'
        '<property name="kernel_check" value="put" /></properties>'
        '<failure message="wrong answer" /></testcase>'
        '<testcase name="test_unrelated" /></testsuite>'
    )
    (tmp_path / "meta.json").write_text(
        json.dumps({"failed": ["test_pairs::test_pair[shape]"]})
    )
    failures, swept, inputs = report.failures(tmp_path)
    assert inputs == 0  # a dedicated test is not a random or edge input
    assert swept == {"get/test_pair[shape]", "put/test_pair[shape]"}
    assert {f.case for f in failures} == swept
    for failure in failures:
        assert failure.failed == ["dedicated"]
        assert failure.total == 1
        assert failure.reason == "wrong answer"


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
    preflight = {"device": "NPU Strix"}
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
    assert bool(leg.uncompared) is not compared
    if current_mode and baseline_mode and not compared:
        assert leg.uncompared == (
            f"nightly in {baseline_mode} mode, this run in {current_mode}: "
            "not compared"
        )


def test_a_refused_run_reports_one_reason_not_a_failure_per_case(read_leg, report):
    refusal = "power mode is default, required 'turbo'"
    leg = read_leg(
        {},
        _record({"synthetic/1/i8/cycles": ("cycles", 100)}),
        meta={
            "preflight": {"pmode": "default"},
            "refused": refusal,
            "failed": [f"test_kernel_perf[synthetic/{i}/i8]" for i in range(3)],
        },
        cases=[("synthetic/1/i8/random/s0", "")],
    )
    assert leg.failures == []
    assert leg.uncompared == f"not timed: {refusal}"
    comment = report.render([leg])
    assert "all passed, not compared with a nightly, npu2 not timed" in comment
    assert "Not compared with a nightly on main; the table says why." in comment
    assert f"| npu2 | ? | ? | not timed: {refusal} | 1 | 0 | \u2014 |" in comment


def test_host_time_is_compared_only_on_the_same_part(read_leg, report):
    values = {
        "synthetic/1/i8/cycles": ("cycles", 150),
        "synthetic/1/i8/npu_us": ("us", 200.0),
    }
    baseline = _record(
        {
            "synthetic/1/i8/cycles": ("cycles", 100),
            "synthetic/1/i8/npu_us": ("us", 100.0),
        }
    )
    same = read_leg(values, baseline)
    assert [c.metric for c in same.regressed + same.other] == ["cycles", "npu_us"]
    assert same.other_part == 0

    other = read_leg(values, dict(baseline, device="Krackan"))
    assert [c.metric for c in other.regressed] == ["cycles"]
    assert other.other == [] and other.other_part == 1
    comment = report.render([other])
    assert "| npu2 | Strix (nightly: Krackan) |" in comment
    assert "this run is on Strix, the nightly on Krackan, so 1 host-timed rows" in (
        comment
    )


@pytest.mark.parametrize(
    "before, after, listed", [(0.13, 0.2, False), (1.0, 1.6, True)]
)
def test_a_compile_time_must_also_move_its_floor(read_leg, before, after, listed):
    leg = read_leg(
        {"mobilenet/compile_place_s": ("s", after)},
        _record({"mobilenet/compile_place_s": ("s", before)}),
    )
    assert bool(leg.other) is listed


@pytest.mark.parametrize(
    "metric, unit, before, after, where",
    [
        # Gated on any rise of a whole ulp, from zero too.
        ("max_ulp_bfloat16", "ulp", 0, 1, "regressed"),
        ("max_ulp_bfloat16", "ulp", 5, 4, "improved"),
        ("not_correctly_rounded_float32", "%", 1.0, 1.05, None),
        # A count that takes abs 1 is judged from zero; one thresholds.json
        # doesn't name is not.
        ("failed_seeds", "seeds", 0, 2, "other"),
        ("an_unnamed_count", "seeds", 0, 2, None),
        # Shown, never judged.
        ("ort_matches", "images", 8, 6, None),
    ],
)
def test_each_metric_is_judged_by_its_own_rule(
    read_leg, report, metric, unit, before, after, where
):
    leg = read_leg(
        {f"c/1/i8/{metric}": (unit, after)},
        _record({f"c/1/i8/{metric}": (unit, before)}),
    )
    moved = {
        name: [c.metric for c in getattr(leg, name)]
        for name in ("regressed", "improved", "other")
    }
    assert moved == {name: [metric] if name == where else [] for name in moved}
    if where and not before:
        assert f"+{after} {unit}" in report.render([leg])


def test_a_test_that_passed_on_a_retry_is_listed(read_leg, report):
    retried = '<properties><property name="reruns" value="1" /></properties>'
    leg = read_leg(
        {},
        cases=[
            ("synthetic/1/i8/random/s0", ""),
            ("synthetic/1/i8/random/s0", retried),
            ("synthetic/2/i8/random/s0", ""),
            ("synthetic/2/i8/random/s0", retried + '<failure message="bad" />'),
        ],
        meta=dict(
            META,
            failed=["perf.py::test_kernel_perf[synthetic/4/i8]"],
            reruns={
                "perf.py::test_kernel_perf[synthetic/3/i8]": 1,
                "perf.py::test_kernel_perf[synthetic/4/i8]": 1,
            },
        ),
    )
    assert leg.flaky == [
        ("::test_kernel_extensive[synthetic/1/i8/random/s0]", 1),
        ("perf.py::test_kernel_perf[synthetic/3/i8]", 1),
    ]
    assert leg.inputs == 2
    assert [f.total for f in leg.failures if f.case == "synthetic/2/i8"] == [1]
    comment = report.render([leg])
    assert "2 passed on a retry" in comment
    assert "Passed only on a retry (2)" in comment
    assert "`perf.py::test_kernel_perf[synthetic/3/i8]` | 1 |" in comment


@pytest.mark.parametrize("commit", ["abcdef0123", "fedcba9876"])
def test_a_run_off_the_nightlys_commit_links_the_range(read_leg, report, commit):
    nightly = {"id": "abcdef0123", "url": "https://example.com/abcdef0"}
    leg = read_leg(
        {"synthetic/1/i8/cycles": ("cycles", 100)},
        _record({"synthetic/1/i8/cycles": ("cycles", 100)}, commit=nightly),
    )
    comment = report.render([leg], commit=commit)
    link = f"[abcdef0...fedcba9]({report.REPO}/compare/abcdef0123...fedcba9876)"
    assert (link in comment) is (commit != nightly["id"])
    assert ("different commit than the nightly" in comment) is (commit != nightly["id"])


@pytest.mark.parametrize("published", [False, True])
def test_cli_writes_report(tmp_path, published):
    results = tmp_path / "results/npu2"
    results.mkdir(parents=True)
    (results / "meta.json").write_text(json.dumps(META))
    (results / "perf.json").write_text(
        json.dumps(_rows({"synthetic/1/i8/cycles": ("cycles", 100)}))
    )
    if published:
        nightly = tmp_path / "baselines/npu2"
        nightly.mkdir(parents=True)
        commit = {"id": "abcdef0123", "url": "https://example.com/abcdef0"}
        (nightly / "latest.json").write_text(
            json.dumps(
                _record({"synthetic/1/i8/cycles": ("cycles", 90)}, commit=commit)
            )
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
            "--commit",
            "fedcba9876",
        ],
        check=True,
    )
    text = out.read_text()
    assert "<!-- kernel-checks-report -->" in text
    if published:
        assert "0 failing, 1 regressed" in text
        assert "Compared with the last nightly on main." in text
        assert "| [abcdef0](https://example.com/abcdef0) |" in text
        assert "compare/abcdef0123...fedcba9876" in text
    else:
        assert "all passed, not compared with a nightly" in text
        assert "Not compared with a nightly on main" in text
        assert "| no nightly to compare with |" in text
        assert "compare/" not in text
