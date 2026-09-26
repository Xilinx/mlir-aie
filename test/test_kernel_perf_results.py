# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-only tests of the performance checks' publication gates."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def hooks():
    path = Path(__file__).parent / "python/npu/conftest.py"
    spec = importlib.util.spec_from_file_location("npu_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def finish(hooks, tmp_path):
    def run(meta, correctness=None, exitstatus=0, failed=(), sanity_selected=True):
        options = {
            "--perf-out": str(tmp_path / "perf.json"),
            "--perf-meta": str(tmp_path / "meta.json"),
            "--correctness-results": correctness,
        }
        rows = [
            {"name": f"{case}/{metric}", "unit": "cycles", "value": 100}
            for case in ("softmax/1024x16/bfloat16", "relu/1024x16/bfloat16")
            for metric in ("cycles", "cycles_per_kop")
        ]
        reporter = SimpleNamespace(
            stats={"failed": [SimpleNamespace(nodeid=name) for name in failed]}
        )
        config = SimpleNamespace(
            _perf_rows=rows,
            _perf_meta=meta,
            getoption=options.get,
            pluginmanager=SimpleNamespace(get_plugin=lambda _: reporter),
        )
        selected = ["test_kernel_perf[softmax/1024x16/bfloat16]"]
        if sanity_selected:
            selected.append("test_measurement_is_sane")
        items = [SimpleNamespace(name=name) for name in selected]
        hooks.pytest_sessionfinish(
            SimpleNamespace(config=config, items=items), exitstatus
        )
        out = tmp_path / "perf.json"
        return (
            json.loads(out.read_text()) if out.exists() else None,
            json.loads((tmp_path / "meta.json").read_text()),
        )

    return run


@pytest.mark.parametrize(
    "meta,exitstatus,publishes",
    [
        ({"preflight": {}, "measurement_sane": True}, 0, True),
        ({"preflight": {}, "measurement_sane": True}, 1, True),
        ({"preflight": {}, "measurement_sane": True}, 2, False),
        ({"preflight": {}, "measurement_sane": False}, 1, False),
        ({"preflight": {}}, 0, False),
        ({"preflight": {}, "measurement_sane": 1}, 0, False),
        ({"measurement_sane": True}, 1, False),
    ],
)
def test_publication_requires_explicit_sanity_success(
    finish, meta, exitstatus, publishes
):
    rows, _ = finish(meta, exitstatus=exitstatus)
    assert bool(rows) == publishes


@pytest.mark.parametrize(
    "meta,publishes",
    [
        ({"preflight": {}}, True),
        ({"preflight": {}, "measurement_sane": False}, False),
        ({}, False),
    ],
)
def test_deselected_sanity_check_still_publishes(finish, meta, publishes):
    """``-k '[case]'`` deselects the sanity check; its rows are still wanted."""
    rows, written = finish(meta, sanity_selected=False)
    assert bool(rows) == publishes
    assert written["measurement_sane"] is meta.get("measurement_sane")


def report(tmp_path, tests):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    for name, outcome in tests:
        test = ET.SubElement(suite, "testcase", classname="test_kernels_e2e", name=name)
        if outcome:
            ET.SubElement(test, outcome)
    path = tmp_path / "correctness.xml"
    ET.ElementTree(root).write(path)
    return str(path)


@pytest.mark.parametrize("outcome", ["failure", "error"])
def test_edge_failure_excludes_all_case_metrics(finish, tmp_path, outcome):
    softmax = "test_kernel_extensive[softmax/1024x16/bfloat16"
    relu = "test_kernel_extensive[relu/1024x16/bfloat16"
    path = report(
        tmp_path,
        [
            (f"{softmax}/random/s0]", None),
            (f"{softmax}/large/s0]", outcome),
            (f"{softmax}/random/s1]", None),
            (f"{relu}/random/s0]", None),
            (f"{relu}/zero/s0]", None),
        ],
    )
    rows, meta = finish(
        {"preflight": {}, "measurement_sane": True},
        path,
        exitstatus=1,
        failed=["test_perf_failure"],
    )
    assert len(rows) == meta["n_rows"] == 2
    assert all(row["name"].startswith("relu/") for row in rows)
    assert meta["failed"] == [
        f"test_kernels_e2e::{softmax}/large/s0]",
        "test_perf_failure",
    ]


def test_skipped_or_absent_cases_are_not_published(finish, tmp_path):
    path = report(
        tmp_path,
        [("test_kernel_extensive[softmax/1024x16/bfloat16/random/s0]", "skipped")],
    )
    rows, meta = finish({"preflight": {}, "measurement_sane": True}, path)
    assert rows is None
    assert meta["n_rows"] == 0


@pytest.mark.parametrize("content", [None, "<testsuites>", "<testsuites/>"])
def test_missing_or_invalid_report_withholds_results(finish, tmp_path, content):
    path = tmp_path / "correctness.xml"
    if content is not None:
        path.write_text(content)
    rows, meta = finish({"preflight": {}, "measurement_sane": True}, str(path))
    assert rows is None
    assert "correctness_error" in meta


def test_unmapped_correctness_error_withholds_results(finish, tmp_path):
    path = report(tmp_path, [("test_kernels_e2e", "error")])
    rows, meta = finish({"preflight": {}, "measurement_sane": True}, path)
    assert rows is None
    assert "unmapped correctness failure" in meta["correctness_error"]


def test_case_names_with_nested_options_are_preserved(hooks, tmp_path):
    case = "mm/64x32x64x4/bfloat16_float32/b_col_maj=True"
    path = report(tmp_path, [(f"test_kernel_extensive[{case}/random/s2]", None)])
    assert hooks._checked_cases(path) == ({case}, [])
