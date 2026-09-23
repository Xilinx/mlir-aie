# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-only publication regression tests; no compiled aie package required."""

import os
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
    ],
)
def test_parallel_compute_has_one_main_only_publisher(filename, compute, static):
    config = workflow(filename)
    assert set(config["on"]) == {"workflow_dispatch", "schedule", "pull_request"}
    # A change to the workflow that runs the checks must itself run them.
    assert filename in " ".join(config["on"]["pull_request"]["paths"]) or (
        filename == "benchmarkKernels.yml"
    )
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
        # Every record writes whichever branch the caller named, so a
        # rehearsal cannot land half its series on the real one.
        assert options["gh-pages-branch"] == "${{ inputs.branch }}"
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


def test_benchmark_preflight_sets_memlock_and_reuses_one_examine():
    job = workflow("benchmarkKernels.yml")["jobs"]["bench"]
    step = next(step for step in job["steps"] if step.get("id") == "preflight")
    run = step["run"]
    assert "sudo prlimit -lunlimited --pid $$" in run
    assert run.index("sudo prlimit -lunlimited --pid $$") < run.index(
        "XRT_SMI=$(command -v xrt-smi"
    )
    assert "XRT_SMI=$(command -v xrt-smi || command -v xrt-smi.exe)" in run
    assert 'EXAMINE=$("$XRT_SMI" examine)' in run
    assert "printf '%s\\n' \"$EXAMINE\"" in run
    assert "BDF=$(printf '%s\\n' \"$EXAMINE\"" in run
    assert 'sudo "$XRT_SMI" configure -d "$BDF" --pmode "$BENCH_PMODE"' in run
    assert '"$XRT_SMI" examine -d "$BDF" --report platform' in run
    assert "xrt-smi examine | grep -oE" not in run


@pytest.mark.parametrize(
    "filename,compute",
    [("benchmarkKernels.yml", "bench")],
)
def test_source_builds_initialize_submodules(filename, compute):
    steps = workflow(filename)["jobs"][compute]["steps"]
    checkout = next(
        step for step in steps if step.get("uses", "").startswith("actions/checkout@")
    )
    assert checkout["with"]["submodules"] in ("true", "recursive")
    build = next(
        step for step in steps if "build-mlir-aie-from-wheels.sh" in step.get("run", "")
    )
    assert steps.index(checkout) < steps.index(build)


@pytest.mark.parametrize("step_id", ["correctness", "bench"])
def test_benchmark_uses_built_package(step_id):
    steps = workflow("benchmarkKernels.yml")["jobs"]["bench"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == step_id)
    assert (
        run.index("sudo prlimit -lunlimited --pid $$")
        < run.index("source aie-venv/bin/activate")
        < run.index("source utils/env_setup.sh mlir_aie")
        < run.index("python -m pytest")
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


def test_rehearsing_the_publish_cannot_touch_the_real_series():
    """The branch override has to be inert everywhere but the branch itself.

    A rehearsal publishes to a scratch branch to prove the path works. Two
    things would let it reach past that: the baseline caches are restored by
    key prefix, so a rehearsal writing one would hand its numbers to the next
    PR comparison; and the lock is what keeps this workflow from pushing the
    branch while the docs workflow is pushing it, so it has to stay a name
    both can agree on rather than one derived from the input.
    """
    config = workflow("publishKernelResults.yml")
    branch = config["on"]["workflow_call"]["inputs"]["branch"]
    assert branch["default"] == "gh-pages"
    assert branch["required"] == "false"

    job = config["jobs"]["publish"]
    assert job["concurrency"]["group"] == "gh-pages-publish"

    saves = [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/cache/save@")
    ]
    assert len(saves) == 2
    for step in saves:
        assert step["if"] == "inputs.branch == 'gh-pages'"
