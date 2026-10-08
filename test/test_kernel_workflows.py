# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Execute workflow shell behavior on synthetic inputs; no NPU required."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def workflow(name):
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )


# Resolve bash in PATH order: a bare "bash" argv[0] goes through CreateProcess's
# search, which checks System32 before PATH and so finds Windows' WSL launcher
# stub instead of Git for Windows' bash.
BASH = shutil.which("bash") or "bash"


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
        [BASH, "-eu", "-c", 'python() { printf "%s\\n" "$@"; }\n' + command],
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


@pytest.mark.parametrize("soak", [{}, {"ITERS": "5000"}, {"SEEDS": "50"}])
def test_a_soak_does_not_retry_a_failure(soak, tmp_path):
    """The runner retries a failed NPU test, which hides a failure that
    comes and goes: the kind a soak is run to find."""
    steps = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "perf")
    launcher = "MLIR_AIE_NPU_TEST=1 python utils/run_pytest.py"
    command = run[run.index(launcher) :].split("2>&1", 1)[0]
    result = subprocess.run(
        [BASH, "-eu", "-c", 'python() { printf "%s\\n" "$@"; }\n' + command],
        cwd=tmp_path,
        env={**os.environ, "ONLY": "", "REQUIRED_PMODE": "any", **soak},
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    # run_pytest.py puts its own --reruns first; the last one wins.
    assert ("--reruns" in args) == bool(soak)
    if soak:
        assert args[len(args) - 1 - args[::-1].index("--reruns") + 1] == "0"


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
        [BASH, "-eo", "pipefail", "-c", stubs + step["run"]],
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
        [BASH, "-eo", "pipefail", "-c", run],
        cwd=cwd,
        env={**os.environ, "GITHUB_OUTPUT": str(output)},
        check=True,
    )
    return dict(line.split("=", 1) for line in output.read_text().splitlines())


def write_meta(path, pmode):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"preflight": {"pmode": pmode}}))


@pytest.mark.parametrize("pmode", ["performance", None])
def test_only_turbo_mode_is_published_and_the_other_npu_still_is(pmode, tmp_path):
    publisher = workflow("publishKernelResults.yml")["jobs"]["publish"]
    run = next(
        step["run"]
        for step in publisher["steps"]
        if step.get("name") == "Read power modes"
    )
    write_meta(tmp_path / "results/npu1/meta.json", "turbo")
    write_meta(tmp_path / "results/npu2/meta.json", pmode)
    # Failed legs may have metadata but no measurements.
    run_step(run, tmp_path)
    for npu in ("npu1", "npu2"):
        (tmp_path / f"results/{npu}/perf.json").write_text("[]")
    run_step(run, tmp_path)
    assert (tmp_path / "results/npu1/perf.json").exists()
    assert not (tmp_path / "results/npu2/perf.json").exists()
    assert (tmp_path / "results/npu2/meta.json").exists()


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
                [BASH, "-eo", "pipefail", "-c", step["run"]],
                cwd=tmp_path,
                env=env,
                check=True,
            )
    text = summary.read_text()
    assert text == (tmp_path / "report.md").read_text()
    assert "1 regressed" in text
    assert "synthetic/1/i8" in text
    assert "https://github.com/example/synthetic/actions/runs/7" in text


def component_steps(job):
    return workflow("nightlyComponentChecks.yml")["jobs"][job]["steps"]


@pytest.mark.parametrize(
    "seeds, expected", [("", "20"), ("5", "5"), ("5; $(false)", "5; $(false)")]
)
def test_the_sweep_takes_its_seed_count_from_the_environment(seeds, expected, tmp_path):
    run = next(
        step["run"]
        for step in component_steps("seed-sweep")
        if step.get("name") == "Run the seed sweep"
    )
    (tmp_path / "aie-venv/bin").mkdir(parents=True)
    (tmp_path / "aie-venv/bin/activate").write_text("")
    result = subprocess.run(
        [BASH, "-eo", "pipefail", "-c", 'python() { printf "%s\\n" "$@"; }\n' + run],
        cwd=tmp_path,
        env={**os.environ, "SEEDS": seeds},
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    assert args[args.index("--seeds") + 1] == expected


def evaluate(expression, context):
    """Evaluate the subset of GitHub's expression syntax the workflow uses."""
    text = expression.strip()
    if text.startswith("${{"):
        text = text[3:-2]
    text = text.replace("!cancelled()", "True")
    text = re.sub(
        r"\b(github|inputs|env)\.(\w+)", lambda m: repr(context[m.group(0)]), text
    )
    text = text.replace("&&", " and ").replace("||", " or ")
    return eval(re.sub(r"!(?!=)", " not ", text), {})


@pytest.mark.parametrize(
    "event, ref",
    [
        ("schedule", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/topic"),
        ("pull_request", "refs/pull/1/merge"),
    ],
)
@pytest.mark.parametrize("seeds", ["", "5"])
def test_a_run_that_publishes_requires_the_power_mode(event, ref, seeds):
    jobs = workflow("nightlyComponentChecks.yml")["jobs"]
    context = {
        "github.event_name": event,
        "github.ref": ref,
        "inputs.seeds": seeds,
        "env.PERF_PMODE": workflow("nightlyComponentChecks.yml")["env"]["PERF_PMODE"],
    }
    publishes = evaluate(jobs["publish"]["if"], context)
    assert publishes == (
        event != "pull_request" and ref == "refs/heads/main" and not seeds
    )
    step = next(
        step
        for step in jobs["hw-check"]["steps"]
        if step.get("name") == "Run mobilenet at each seed and batch"
    )
    if not evaluate(jobs["hw-check"]["if"], context):
        assert event == "pull_request"
        return
    required = evaluate(step["env"]["REQUIRED_PMODE"], context)
    assert required == ("turbo" if publishes else "any")


def test_every_path_that_triggers_a_pull_request_run_exists():
    root = WORKFLOWS.parents[1]
    paths = workflow("nightlyComponentChecks.yml")["on"]["pull_request"]["paths"]
    for path in paths:
        assert list(root.glob(path)), path


def git(cwd, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


LEGACY_SWEEP = {
    "entries": {
        "SA placer seed sweep": [
            {
                "commit": {"id": "18ca6c1cbf4c", "message": "Old", "url": ""},
                "date": 1790924970143,
                "benches": [
                    {"name": "sa_placer/fail_count", "value": 0, "unit": "seeds"},
                    {
                        "name": "sa_placer/mean_wall_time_ms",
                        "value": 1259,
                        "unit": "ms",
                    },
                    {"name": "sa_placer/mean_final_cost", "value": 12, "unit": "cost"},
                ],
            }
        ]
    }
}


def test_publishing_migrates_records_and_leaves_redirects(tmp_path):
    """The Publish step, run on a local repository with a gh-pages branch as
    the old component checks left it."""
    run = next(
        step["run"]
        for step in component_steps("publish")
        if step.get("name") == "Publish results"
    )
    repo = tmp_path / "repo"
    (repo / "utils/kernel_checks").mkdir(parents=True)
    for name in ("index.html", "publish.py"):
        (repo / "utils/kernel_checks" / name).write_bytes(
            (WORKFLOWS.parents[1] / "utils/kernel_checks" / name).read_bytes()
        )
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "main")
    git(repo, "switch", "-q", "--orphan", "gh-pages")
    old = repo / "component-checks/sa-placer"
    old.mkdir(parents=True)
    (old / "data.js").write_text(
        "window.BENCHMARK_DATA = " + json.dumps(LEGACY_SWEEP) + ";\n"
    )
    (old / "index.html").write_text("<script>window.BENCHMARK_DATA</script>")
    (repo / "component-checks/index.html").write_text("<p>old index</p>")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "pages")
    git(repo, "switch", "-q", "main")

    results = repo / "results/sa-placer"
    results.mkdir(parents=True)
    (results / "meta.json").write_text(
        json.dumps(
            {
                "preflight": {},
                "provenance": "commit 0123456789 | fixtures abcdef012345",
                "measurement_sane": True,
                "failed": [],
                "exitstatus": 0,
            }
        )
    )
    (results / "perf.json").write_text(
        json.dumps(
            [{"name": "test_sa_effort/failed_seeds", "unit": "seeds", "value": 0}]
        )
    )
    temp = tmp_path / "runner-temp"
    temp.mkdir()
    subprocess.run(
        [BASH, "-eo", "pipefail", "-c", run],
        cwd=repo,
        env={
            **os.environ,
            "RUNNER_TEMP": str(temp),
            "GITHUB_RUN_ID": "42",
            "GITHUB_SHA": "0123456789abcdef",
            "RUN_URL": "https://example.com/runs/42",
            "RUN_STARTED_AT": "2026-10-03T06:30:00Z",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
        },
        check=True,
        capture_output=True,
    )
    assert git(repo, "branch", "--show-current").strip() == "main"
    files = set(git(repo, "ls-tree", "-r", "--name-only", "gh-pages").split())
    assert "component-checks/sa-placer/data.js" not in files
    assert {
        "kernel-checks/index.html",
        "component-checks/index.html",
        "component-checks/sa-placer/index.html",
        "component-checks/sa-placer/runs/42.json",
        "component-checks/sa-placer/runs/bench-1790924970143.json",
        "component-checks/sa-placer/runs.json",
        "component-checks/sa-placer/history/failed_seeds.json",
    } <= files
    # The hardware check produced nothing and had no page to redirect.
    assert not any(f.startswith("component-checks/sa-placer-hw/") for f in files)

    def show(path):
        return git(repo, "show", f"gh-pages:{path}")

    assert 'url=../kernel-checks/#view=components"' in show(
        "component-checks/index.html"
    )
    assert 'url=../../kernel-checks/#view=components"' in show(
        "component-checks/sa-placer/index.html"
    )
    index = json.loads(show("component-checks/sa-placer/runs.json"))
    assert index["target"] == "sa-placer"
    assert [r["id"] for r in index["runs"]] == ["bench-1790924970143", "42"]
    assert index["metrics"] == ["failed_seeds", "final_cost_mean"]
    record = json.loads(show("component-checks/sa-placer/runs/42.json"))
    assert record["date"] == "2026-10-03T06:30:00+00:00"
    assert record["provenance"]["fixtures"] == "abcdef012345"
    history = json.loads(show("component-checks/sa-placer/history/failed_seeds.json"))
    assert history["series"] == {"test_sa_effort": {"values": [0, 0]}}
    assert history["runs"][1]["provenance"] == {"fixtures": "abcdef012345"}


def test_a_hosted_job_runs_the_page_tests_instead_of_skipping_them():
    job = workflow("buildAndTestPythons.yml")["jobs"]["build-repo"]
    assert job["runs-on"].startswith("ubuntu-")
    assert job["env"]["MLIR_AIE_REQUIRE_NODE"] == "1"
