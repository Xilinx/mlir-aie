# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Execute workflow shell behavior on synthetic inputs; no NPU required."""

import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def workflow(name):
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def matrix_pairs(job):
    return {
        (entry["python_version"], entry["ENABLE_RTTI"])
        for entry in job["strategy"]["matrix"]["include"]
    }


def test_ryzen_wheel_publish_workflow_gates_on_both_os_builds():
    publish = workflow("buildRyzenWheels.yml")
    assert "pull_request" not in publish["on"]

    jobs = publish["jobs"]
    assert (
        jobs["linux-wheels"]["uses"] == "./.github/workflows/buildRyzenWheelsLinux.yml"
    )
    assert (
        jobs["windows-wheels"]["uses"]
        == "./.github/workflows/buildRyzenWheelsWindows.yml"
    )
    assert set(jobs["publish"]["needs"]) == {"linux-wheels", "windows-wheels"}


def test_ryzen_linux_npu_smoke_depends_only_on_its_wheel():
    jobs = workflow("buildRyzenWheelsLinux.yml")["jobs"]

    assert jobs["build-linux-npu-wheel"]["env"] == {
        "BUILD_PYTHON_VERSION": "3.12",
        "ENABLE_RTTI": "ON",
    }
    assert ("3.12", "ON") not in matrix_pairs(jobs["build-linux-wheels"])
    assert jobs["smoke-test-npu"]["needs"] == "build-linux-npu-wheel"


def test_ryzen_windows_smoke_tests_are_per_wheel():
    jobs = workflow("buildRyzenWheelsWindows.yml")["jobs"]

    assert jobs["build-windows-npu-wheel"]["env"] == {
        "BUILD_PYTHON_VERSION": "3.13",
        "ENABLE_RTTI": "ON",
    }
    assert ("3.13", "ON") not in matrix_pairs(jobs["build-windows"])
    assert "smoke-test-wheels-windows" not in jobs
    assert jobs["smoke-test-wheels-npu-windows"]["needs"] == "build-windows-npu-wheel"


def test_baseline_cache_key_is_unique_per_attempt_and_restorable():
    publish = workflow("publishKernelResults.yml")["jobs"]["publish"]["steps"]
    report = workflow("nightlyKernelChecks.yml")["jobs"]["report"]["steps"]
    saved = [
        step["with"]
        for step in publish
        if step.get("uses", "").startswith("actions/cache/save@")
    ]
    restored = [
        step["with"]
        for step in report
        if step.get("uses", "").startswith("actions/cache/restore@")
    ]
    assert saved and restored
    for save in saved:
        assert "${{ github.run_id }}" in save["key"]
        assert "${{ github.run_attempt }}" in save["key"]
        assert any(
            save["key"].startswith(prefix) and save["path"] == restore["path"]
            for restore in restored
            for prefix in restore["restore-keys"].splitlines()
            if prefix
        )


@pytest.mark.parametrize("only", ["", "softmax", "softmax and not large", "$(false)"])
def test_dispatch_filter_keeps_sanity_and_preserves_shell_quoting(only, tmp_path):
    steps = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "perf")
    launcher = "MLIR_AIE_NPU_TEST=1 python utils/run_pytest.py"
    command = run[run.index(launcher) :].split("2>&1", 1)[0]
    result = subprocess.run(
        ["bash", "-eu", "-c", 'python() { printf "%s\\n" "$@"; }\n' + command],
        cwd=tmp_path,
        env={**os.environ, "ONLY": only, "REQUIRED_PMODE": "any"},
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    if only:
        assert args[-2:] == ["-k", f"({only}) or test_measurement_is_sane"]
    else:
        assert "-k" not in args
    assert args[args.index("--pmode") + 1] == "any"


# Synthetic release listing: duplicate platform wheels and a rebuilt commit.
NIGHTLY_PAGE = """
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026010301+aaaaaaaa-py3-none-manylinux_2_28_x86_64.whl">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026010301+aaaaaaaa-py3-none-win_amd64.whl">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026010101+bbbbbbbb-py3-none-win_amd64.whl">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026010201+bbbbbbbb-py3-none-win_amd64.whl">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026010101+bbbbbbbc-py3-none-win_amd64.whl">
"""
PINNED = "llvm-aie==22.0.0.2026010301+aaaaaaaa"
WHEEL = "https://example.com/llvm_aie-1.0-py3-none-any.whl"


def run_peano_step(peano, tmp_path):
    step = next(
        step
        for step in workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
        if "PEANO" in step.get("env", {})
    )
    (tmp_path / "aie-venv/bin").mkdir(parents=True, exist_ok=True)
    (tmp_path / "aie-venv/bin/activate").write_text("")
    (tmp_path / "page.html").write_text(NIGHTLY_PAGE)
    stubs = (
        "curl() { cat page.html; }\n"
        'python() { if [ "$3" = show ]; then echo "Version: stub"; '
        'else printf "%s\\n" "$@" > pip-args; fi; }\n'
    )
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", stubs + step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            "PEANO": peano,
            "NIGHTLY": step["env"]["NIGHTLY"],
            "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        },
        capture_output=True,
        text=True,
    )
    args = tmp_path / "pip-args"
    return result, args.read_text().splitlines() if args.exists() else None


@pytest.mark.parametrize(
    "peano,spec",
    [
        ("22.0.0.2026010301+aaaaaaaa", PINNED),
        ("aaaaaaa", PINNED),
        ("a" * 40, PINNED),
        ("bbbbbbbb", "llvm-aie==22.0.0.2026010201+bbbbbbbb"),
        (WHEEL, WHEEL),
    ],
)
def test_dispatch_installs_the_requested_peano(peano, spec, tmp_path):
    result, args = run_peano_step(peano, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert args[-1] == spec
    assert "--force-reinstall" in args


@pytest.mark.parametrize(
    "peano",
    ["deadbeef", "bbbbbbb", "$(false)", "1.0; true", "http://example.com/x.whl"],
)
def test_dispatch_rejects_an_unresolvable_peano(peano, tmp_path):
    result, args = run_peano_step(peano, tmp_path)
    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert args is None


def run_step(run, cwd):
    output = cwd / "github_output"
    output.write_text("")
    subprocess.run(
        ["bash", "-eo", "pipefail", "-c", run],
        cwd=cwd,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def write_meta(path, pmode):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"preflight": {"pmode": pmode}}))


def test_only_turbo_mode_is_published(tmp_path):
    publisher = workflow("publishKernelResults.yml")["jobs"]["publish"]
    run = next(step["run"] for step in publisher["steps"] if step.get("id") == "pmode")
    write_meta(tmp_path / "results/npu1/meta.json", "turbo")
    write_meta(tmp_path / "results/npu2/meta.json", "performance")
    # Failed legs may have metadata but no measurements.
    assert run_step(run, tmp_path) == {}
    (tmp_path / "results/npu1/perf.json").write_text("[]")
    assert run_step(run, tmp_path) == {"npu1": "turbo"}
    (tmp_path / "results/npu2/perf.json").write_text("[]")
    with pytest.raises(subprocess.CalledProcessError):
        run_step(run, tmp_path)
    write_meta(tmp_path / "results/npu2/meta.json", "turbo")
    assert run_step(run, tmp_path) == {"npu1": "turbo", "npu2": "turbo"}
    write_meta(tmp_path / "results/npu2/meta.json", None)
    with pytest.raises(subprocess.CalledProcessError):
        run_step(run, tmp_path)


def test_report_uses_restored_baseline_and_updates_summary(tmp_path):
    steps = workflow("nightlyKernelChecks.yml")["jobs"]["report"]["steps"]
    results = tmp_path / "results/npu2"
    results.mkdir(parents=True)
    write_meta(results / "meta.json", "performance")
    (results / "perf.json").write_text(
        json.dumps([{"name": "synthetic/1/i8/cycles", "unit": "cycles", "value": 150}])
    )
    baseline = {
        "id": "baseline-run",
        "pmode": "performance",
        "rows": {"synthetic/1/i8": {"cycles": {"unit": "cycles", "value": 100}}},
    }
    (tmp_path / "utils").symlink_to(WORKFLOWS.parents[1] / "utils")
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "example/synthetic",
        "GITHUB_RUN_ID": "7",
    }
    for step in steps:
        if step.get("uses", "").startswith("actions/cache/restore@"):
            if "npu2-" in step["with"]["key"]:
                (tmp_path / "baseline").mkdir()
                (tmp_path / "baseline/latest.json").write_text(json.dumps(baseline))
        elif "run" in step and "gh api" not in step["run"]:
            subprocess.run(
                ["bash", "-eo", "pipefail", "-c", step["run"]],
                cwd=tmp_path,
                env=env,
                check=True,
            )
    text = summary.read_text()
    assert text == (tmp_path / "report.md").read_text()
    assert "1 regressed" in text
    assert "synthetic/1/i8" in text
    assert "https://github.com/example/synthetic/actions/runs/7" in text
