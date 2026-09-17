# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-only publication regression tests; no compiled aie package required."""

import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def workflow(name):
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def benchmark_steps(job):
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("benchmark-action/")
    ]


@pytest.mark.parametrize(
    "filename,compute,static",
    [
        ("benchmarkKernels.yml", "bench", "false"),
        ("staticKernelChecks.yml", "static", "true"),
    ],
)
def test_parallel_compute_has_one_main_only_publisher(filename, compute, static):
    config = workflow(filename)
    assert set(config["on"]) == {"workflow_dispatch"}
    assert config["permissions"]["contents"] == "read"
    job = config["jobs"][compute]
    assert len(job["strategy"]["matrix"]["include"]) == 2
    if "concurrency" in job:
        assert "${{ matrix." in job["concurrency"]["group"]
        assert job["concurrency"]["cancel-in-progress"] == "false"
        assert job["concurrency"]["queue"] == "max"
    for step in job["steps"]:
        assert "git push" not in step.get("run", "")
        assert "git show" not in step.get("run", "")
        assert not step.get("uses", "").startswith("actions/cache/save@")
    for step in benchmark_steps(job):
        assert step["with"]["save-data-file"] == "false"
        assert "external-data-json-path" in step["with"]
    publisher = config["jobs"]["publish"]
    assert publisher["needs"] == compute
    assert "github.ref == 'refs/heads/main'" in publisher["if"]
    assert "github.event_name != 'pull_request'" in publisher["if"]
    assert publisher["uses"] == "./.github/workflows/publishKernelResults.yml"
    assert publisher["with"]["static"] == static
    assert "strategy" not in publisher
    assert "concurrency" not in publisher
    assert publisher["permissions"]["contents"] == "write"


def test_publishers_share_one_branch_lock_and_push_one_complete_batch():
    publisher = workflow("publishKernelResults.yml")["jobs"]["publish"]
    docs = workflow("generateDocs.yml")
    assert (
        publisher["concurrency"]
        == docs["concurrency"]
        == {
            "group": "gh-pages-publish",
            "cancel-in-progress": "false",
            "queue": "max",
        }
    )
    assert "strategy" not in publisher
    assert "github.ref == 'refs/heads/main'" in publisher["if"]
    steps = publisher["steps"]
    records = benchmark_steps(publisher)
    assert len(records) == 4
    for step in records:
        options = step["with"]
        assert options["auto-push"] == "false"
        assert options["skip-fetch-gh-pages"] == "true"
        assert options["gh-pages-branch"] == "gh-pages"
        assert options["comment-on-alert"] == "false"
    assert [step["if"] for step in records[2:]] == ["inputs.static"] * 2
    pushes = [step for step in steps if "git push" in step.get("run", "")]
    assert len(pushes) == 1
    push_index = steps.index(pushes[0])
    fetch_index = next(
        i for i, step in enumerate(steps) if "git fetch" in step.get("run", "")
    )
    assert all(fetch_index < steps.index(step) < push_index for step in records)
    baseline_index = next(
        i for i, step in enumerate(steps) if "git show" in step.get("run", "")
    )
    assert baseline_index > push_index
    caches = [
        step for step in steps if step.get("uses", "").startswith("actions/cache/save@")
    ]
    assert len(caches) == 2
    assert all(step["with"]["path"] == "baseline" for step in caches)
    assert all(steps.index(step) > baseline_index for step in caches)
    downloads = [
        step
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert [step["with"]["path"] for step in downloads] == [
        "results/npu1",
        "results/npu2",
    ]


def test_dispatch_filter_is_passed_as_data_not_shell_source():
    job = workflow("benchmarkKernels.yml")["jobs"]["bench"]
    step = next(step for step in job["steps"] if step.get("id") == "bench")
    assert step["env"]["ONLY"] == "${{ github.event.inputs.only }}"
    assert "${{ github.event.inputs.only }}" not in step["run"]
    assert '${ONLY:+-k "$ONLY"}' in step["run"]


def test_static_compiles_fused_sources_for_the_matrix_device():
    job = workflow("staticKernelChecks.yml")["jobs"]["static"]
    step = next(
        step
        for step in job["steps"]
        if "test_kernels_compile.py" in step.get("run", "")
    )
    assert "test/python/npu/test_fused_mm_compile.py" in step["run"]
    assert "-m extensive" in step["run"]
    assert step["run"].count("python -m pytest") == 1
    assert step["env"]["KERNEL_TEST_DEVICE"] == "${{ matrix.device }}"


def test_static_checks_include_q4nx_reference_tests():
    job = workflow("staticKernelChecks.yml")["jobs"]["static"]
    assert any(
        "test/python/test_q4nx_dequant.py" in step.get("run", "")
        for step in job["steps"]
    )


def test_docs_cleanup_preserves_benchmark_history():
    steps = workflow("generateDocs.yml")["jobs"]["build-docs"]["steps"]
    cleanup = next(
        step["run"] for step in steps if "git ls-files -z" in step.get("run", "")
    )
    filters = cleanup.split("git ls-files -z", 1)[1].split("| xargs", 1)[0]
    result = subprocess.run(
        ["bash", "-c", "cat " + filters.rstrip().rstrip("\\")],
        input=b"bench/npu1/data.js\0bench/static/aie2/data.js\0dev/index.html\0legacy.html\0",
        capture_output=True,
        check=True,
    )
    assert result.stdout == b"legacy.html\0"
