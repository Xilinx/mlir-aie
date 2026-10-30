# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Execute workflow shell behavior on synthetic inputs; no NPU required."""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"


def workflow(name):
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load(
        (WORKFLOWS / name).read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )


@pytest.mark.parametrize("index", [0, 1])
def test_code_health_sanitizers_do_not_stop_after_one_failure(tmp_path, index):
    job = workflow("nightlyCodeHealth.yml")["jobs"]["sanitizer"]
    entry = job["strategy"]["matrix"]["include"][index]
    script = next(s["run"] for s in job["steps"] if s.get("name") == "Test")
    script = script.replace("${{ matrix.options }}", entry["options"]).replace(
        "${{ matrix.filter-out }}", entry["filter-out"]
    )
    result = subprocess.run(
        [BASH, "-e", "-c", 'lit() { printf "%s\\n" "$@"; return 1; }\n' + script],
        env={**os.environ, "RUNNER_TEMP": str(tmp_path)},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    args = result.stdout.splitlines()
    assert "--max-failures" not in args
    assert args[args.index("--filter-out") + 1] == entry["filter-out"]
    # The script joins with "/" whatever separator RUNNER_TEMP uses.
    assert args[args.index("-o") + 1] == f"{tmp_path}/lit/tests.json"
    assert "build/test" in args


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the step runs on a Linux runner and reads POSIX paths",
)
@pytest.mark.parametrize(
    "tidy_exit, crash, written",
    [
        (0, False, True),
        (123, False, True),
        (125, False, False),
        (1, False, False),
        (123, True, False),
    ],
)
def test_code_health_tidy_reports_only_a_finished_analysis(
    tmp_path, tidy_exit, crash, written
):
    job = workflow("nightlyCodeHealth.yml")["jobs"]["clang-tidy"]
    script = next(s["run"] for s in job["steps"] if s.get("name") == "clang-tidy")
    root = Path(__file__).resolve().parents[1]
    (tmp_path / "build").mkdir()
    (tmp_path / "build/compile_commands.json").write_text(
        json.dumps([{"file": str(tmp_path / "lib/A.cpp")}])
    )
    (tmp_path / "utils/kernel_checks").mkdir(parents=True)
    shutil.copy(
        root / "utils/kernel_checks/code_health.py", tmp_path / "utils/kernel_checks"
    )
    shutil.copy(root / ".pre-commit-config.yaml", tmp_path)
    tidy = tmp_path / "utils/run_clang_tidy.sh"
    tidy.write_text(
        "#!/usr/bin/env bash\n"
        f"echo \"$*\" > {json.dumps(str(tmp_path / 'args.txt'))}\n"
        + ("echo 'Stack dump:'\n" if crash else "")
        + f"exit {tidy_exit}\n"
    )
    tidy.chmod(0o755)
    py = Path(sys.executable).as_posix()
    result = subprocess.run(
        [BASH, "-e", "-c", f'python3() {{ "{py}" "$@"; }}\n' + script],
        cwd=tmp_path,
        env={
            **os.environ,
            "RUNNER_TEMP": str(tmp_path),
            "GITHUB_WORKSPACE": str(tmp_path),
        },
        capture_output=True,
        text=True,
    )
    assert (tmp_path / "args.txt").read_text().split() == ["lib/A.cpp"]
    assert (result.returncode == 0) == written, result.stderr
    assert (tmp_path / "clang-tidy.json").exists() == written


# Resolve bash in PATH order: a bare "bash" argv[0] goes through CreateProcess's
# search, which checks System32 before PATH and so finds Windows' WSL launcher
# stub instead of Git for Windows' bash.
BASH = shutil.which("bash") or "bash"


@pytest.mark.parametrize(
    "available",
    ["python3", "python3.14", "python3.13", "python3.12", "python3.11", None],
)
def test_lockfile_regeneration_finds_python_with_tomllib(available):
    script = (WORKFLOWS.parents[1] / "utils/regenerate_lockfiles.sh").read_text()
    setup = script.split("# Resolve repo root", 1)[0]
    interpreters = ["python3", "python3.14", "python3.13", "python3.12", "python3.11"]
    mocks = 'uv() { echo "uv 0.11.17"; }\n' + "\n".join(
        f"{name}() {{ return {0 if name == available else 1}; }}"
        for name in interpreters
    )
    result = subprocess.run(
        [BASH, "-c", mocks + "\n" + setup + '\nprintf "%s" "$PY"'],
        capture_output=True,
        text=True,
    )
    if available is None:
        assert result.returncode == 1
        assert "no python3.11+ (tomllib) found" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == available


@pytest.mark.parametrize("download_status", [0, 22])
def test_llvm_archive_download_handles_http_errors(download_status):
    run = next(
        step["run"]
        for step in workflow("mlirDistro.yml")["jobs"]["build"]["steps"]
        if step.get("name") == "Get LLVM"
    )
    run = re.sub(r"\$\{\{.*?\}\}", "abcdef01", run)
    mocks = f"""
curl() {{ printf '%s\\n' "$@"; return {download_status}; }}
unzip() {{ echo unzipped; }}
rm() {{ :; }}
mv() {{ :; }}
"""
    result = subprocess.run(
        [BASH, "-e", "-c", mocks + run], capture_output=True, text=True
    )
    assert result.returncode == download_status
    args = result.stdout.splitlines()
    assert "--fail" in args
    assert "--location" in args
    assert args[args.index("--retry") + 1] == "5"
    assert "--show-error" in args
    assert ("unzipped" in args) == (download_status == 0)


def matrix_pairs(job):
    return {
        (entry["python_version"], entry["ENABLE_RTTI"])
        for entry in job["strategy"]["matrix"]["include"]
    }


@pytest.mark.parametrize(
    "path",
    sorted(WORKFLOWS.glob("*.yml"))
    + [WORKFLOWS.parent / "actions/setup_base/action.yml"],
    ids=lambda path: path.stem if path.parent == WORKFLOWS else "setup_base",
)
def test_hosted_ubuntu_runners_and_conditions_use_2604(path):
    labels = re.findall(r"\bubuntu-(?:\d+\.\d+|latest)\b", path.read_text())
    assert all(label == "ubuntu-26.04" for label in labels)


@pytest.mark.parametrize(
    "path",
    sorted(WORKFLOWS.glob("*.yml"))
    + [WORKFLOWS.parent / "actions/setup_base/action.yml"],
    ids=lambda path: path.stem if path.parent == WORKFLOWS else "setup_base",
)
def test_disk_cleanup_disables_incompatible_package_removal(path):
    config = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    jobs = config["jobs"].values() if "jobs" in config else [config["runs"]]
    for job in jobs:
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("descriptinc/free-disk-space@"):
                assert step["with"]["large-packages"] == "false"
                assert step["with"]["tool-cache"] == "true"
                assert step["with"]["android"] == "true"
                assert step["with"]["dotnet"] == "true"
                assert step["with"]["haskell"] == "true"


def test_seed_sweep_provisions_python_before_creating_the_environment():
    steps = workflow("nightlyComponentChecks.yml")["jobs"]["seed-sweep"]["steps"]
    setup_index = next(
        i
        for i, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/setup-python@")
    )
    env_index = next(
        i for i, step in enumerate(steps) if "env_install.sh" in step.get("run", "")
    )
    assert steps[setup_index]["with"]["python-version"] == "3.12"
    assert setup_index < env_index


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


def test_report_reads_the_published_baseline_and_updates_summary(tmp_path):
    """The report job's steps, in a checkout whose origin has a gh-pages
    branch holding npu2's latest record and none for npu1."""
    steps = workflow("nightlyKernelChecks.yml")["jobs"]["report"]["steps"]
    pages = tmp_path / "pages"
    (pages / "kernel-checks/npu2").mkdir(parents=True)
    baseline = {
        "id": "baseline-run",
        "pmode": "performance",
        "commit": {"id": "abcdef0123", "url": "https://example.com/abcdef0"},
        "rows": {"synthetic/1/i8": {"cycles": {"unit": "cycles", "value": 100}}},
    }
    (pages / "kernel-checks/npu2/latest.json").write_text(json.dumps(baseline))
    git(pages, "init", "-q", "-b", "gh-pages")
    git(pages, "add", "-A")
    git(pages, "commit", "-q", "-m", "pages")

    checkout = tmp_path / "checkout"
    results = checkout / "results/npu2"
    results.mkdir(parents=True)
    write_meta(results / "meta.json", "performance")
    (results / "perf.json").write_text(
        json.dumps([{"name": "synthetic/1/i8/cycles", "unit": "cycles", "value": 150}])
    )
    (checkout / "utils").symlink_to(WORKFLOWS.parents[1] / "utils")
    git(checkout, "init", "-q")
    git(checkout, "remote", "add", "origin", pages.as_uri())
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "example/synthetic",
        "GITHUB_RUN_ID": "7",
        "COMMIT": "fedcba9876",
    }
    ran = []
    for step in steps:
        if "run" in step and "gh api" not in step["run"]:
            result = subprocess.run(
                [BASH, "-eo", "pipefail", "-c", step["run"]],
                cwd=checkout,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            ran.append(result.stdout)
    assert "::warning::no gh-pages:kernel-checks/npu1/latest.json" in "".join(ran)
    assert not (checkout / "baselines/npu1").exists()
    text = summary.read_text()
    assert text == (checkout / "report.md").read_text()
    assert "1 regressed" in text
    assert "| [abcdef0](https://example.com/abcdef0) |" in text
    assert "compare/abcdef0123...fedcba9876" in text
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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="the step runs on the Linux bench runner; Windows has no uid",
)
def test_the_hardware_checks_find_the_runners_user_manager(tmp_path):
    run = next(
        step["run"]
        for step in component_steps("hw-checks")
        if step.get("name") == "Point systemd-run --user at the runner's user manager"
    )
    github_env = tmp_path / "github_env"
    subprocess.run(
        [BASH, "-eo", "pipefail", "-c", run],
        env={**os.environ, "GITHUB_ENV": str(github_env)},
        check=True,
    )
    assert github_env.read_text() == f"XDG_RUNTIME_DIR=/run/user/{os.getuid()}\n"


def evaluate(expression, context):
    """Evaluate the subset of GitHub's expression syntax the workflow uses."""
    text = expression.strip()
    if text.startswith("${{"):
        text = text[3:-2]
    text = text.replace("!cancelled()", "True").replace("always()", "True")
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
    # Even after the hardware job times out, which GitHub reports as
    # cancelled: its checks' failed placeholders publish.
    assert jobs["publish"]["if"].startswith("${{ always() &&")
    designs = [
        step
        for step in jobs["hw-checks"]["steps"]
        if "REQUIRED_PMODE" in step.get("env", {})
    ]
    assert [step["id"] for step in designs] == ["mobilenet", "llama", "yolo26n"]
    # The hardware job waits on the runner it is given, which a pull request never asks for.
    assert jobs["hw-checks"]["needs"] == "runners"
    if not evaluate(jobs["runners"]["if"], context):
        assert event == "pull_request"
        return
    for step in designs:
        required = evaluate(step["env"]["REQUIRED_PMODE"], context)
        assert required == ("turbo" if publishes else "any")


def test_the_hardware_checks_install_nothing_mid_run():
    steps = component_steps("hw-checks")
    for step in steps:
        if step.get("name") != "Setup virtual environment / prerequisites":
            assert "pip install" not in step.get("run", ""), step["name"]
    requirements = (WORKFLOWS.parents[1] / "python/requirements_ml.txt").read_text()
    assert "\ntiktoken==" in requirements
    assert "--extras" in steps[1]["run"]


MODEL_INPUTS = (
    "allocator_seeds",
    "allocator_bankres_seeds",
    "router_seeds",
    "mutation_seeds",
    "first_seed",
)


@pytest.mark.parametrize(
    "event, ref",
    [
        ("schedule", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/main"),
        ("workflow_dispatch", "refs/heads/topic"),
    ],
)
@pytest.mark.parametrize("given", ["", *MODEL_INPUTS])
def test_model_based_tests_publish_only_an_unfiltered_run_on_main(event, ref, given):
    jobs = workflow("modelBasedTests.yml")["jobs"]
    context = {"github.ref": ref}
    context.update(
        {f"inputs.{name}": "7" if name == given else "" for name in MODEL_INPUTS}
    )
    assert set(
        workflow("modelBasedTests.yml")["on"]["workflow_dispatch"]["inputs"]
    ) == set(MODEL_INPUTS)
    assert evaluate(jobs["publish"]["if"], context) == (
        ref == "refs/heads/main" and not given
    )


def test_model_based_tests_collect_every_router_leg_and_no_other():
    jobs = workflow("modelBasedTests.yml")["jobs"]
    matrix = jobs["router"]["strategy"]["matrix"]
    leg = next(s for s in jobs["router"]["steps"] if s.get("env", {}).get("RUN"))
    assert 'report="results/router/reports/$DEVICE-$RUN.json"' in leg["run"]
    collect = next(s["run"] for s in jobs["collect"]["steps"] if "run" in s)
    devices = re.search(r"for device in ([^;]+); do", collect).group(1).split()
    runs = re.search(r"for run in ([^;]+); do", collect).group(1).split()
    assert devices == matrix["device"]
    assert runs == matrix["run"]
    assert set(jobs["collect"]["needs"]) == {"build", "router"}
    assert jobs["collect"]["if"] == "always()"


def test_model_based_tests_publish_both_checks_and_the_page(tmp_path):
    """modelBasedTests.yml's Publish step, on a local repository, with what
    model_based.py makes of the scripts' reports."""
    root = WORKFLOWS.parents[1]
    job = workflow("modelBasedTests.yml")["jobs"]["publish"]
    assert job["concurrency"] == {
        "group": "gh-pages-publish",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    run = next(s["run"] for s in job["steps"] if s.get("name") == "Publish results")
    repo = tmp_path / "repo"
    (repo / "utils/kernel_checks").mkdir(parents=True)
    for name in ("index.html", "publish.py"):
        (repo / "utils/kernel_checks" / name).write_bytes(
            (root / "utils/kernel_checks" / name).read_bytes()
        )
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "main")
    git(repo, "switch", "-q", "--orphan", "gh-pages")
    (repo / "kernel-checks").mkdir()
    (repo / "kernel-checks/index.html").write_text("redirect")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "pages")
    git(repo, "switch", "-q", "main")

    allocator = tmp_path / "allocator.json"
    allocator.write_text(
        json.dumps(
            {
                "properties": [{"name": "legality", "ok": True, "detail": ""}],
                "counts": {
                    "unsolved": 0,
                    "illegal": 0,
                    "avoidable_crossings": 0,
                    "bank_pair_sharing": 0.24,
                    "wall_s": 103.0,
                },
            }
        )
    )
    collector = root / "utils/component_checks/model_based.py"
    for check, report in (("allocator", allocator), ("router", tmp_path / "gone.json")):
        out = repo / "results" / check
        out.mkdir(parents=True)
        subprocess.run(
            [sys.executable, collector, check, "--report", report]
            + ["--out", out / "perf.json", "--meta", out / "meta.json"],
            capture_output=True,
            check=False,
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
            "RUN_STARTED_AT": "2026-10-10T03:17:00Z",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
        },
        check=True,
        capture_output=True,
    )
    assert git(repo, "branch", "--show-current").strip() == "main"
    files = set(git(repo, "ls-tree", "-r", "--name-only", "gh-pages").split())
    assert {
        "dashboard/index.html",
        "component-checks/allocator/runs/42.json",
        "component-checks/allocator/history/bank_pair_sharing.json",
        "component-checks/allocator/index.html",
        "component-checks/router/runs/42.json",
        "component-checks/router/index.html",
    } <= files

    record = json.loads(
        git(repo, "show", "gh-pages:component-checks/allocator/runs/42.json")
    )
    assert record["target"] == "allocator" and record["failed"] == []
    assert record["rows"]["allocator"]["bank_pair_sharing"]["value"] == 0.24
    # A router that wrote no report still publishes a record saying so.
    record = json.loads(
        git(repo, "show", "gh-pages:component-checks/router/runs/42.json")
    )
    assert record["failed"] == ["gone: no report"] and record["exitstatus"] == 1
    assert 'url=../../dashboard/#view=routing"' in git(
        repo, "show", "gh-pages:component-checks/router/index.html"
    )


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
        encoding="utf-8",
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
        "dashboard/index.html",
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

    # The dashboard is the page; every old address redirects to its view.
    assert show("dashboard/index.html") == (
        WORKFLOWS.parents[1] / "utils/kernel_checks/index.html"
    ).read_text(encoding="utf-8")
    assert 'url=../dashboard/#view=night"' in show("kernel-checks/index.html")
    assert 'url=../dashboard/#view=placement"' in show("component-checks/index.html")
    assert 'url=../../dashboard/#view=placement"' in show(
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


def test_the_docs_deploy_keeps_what_the_nightlies_publish(tmp_path):
    """The docs deploy clears the gh-pages root of everything mike does not
    own; the nightlies' directories must survive it."""
    steps = workflow("generateDocs.yml")["jobs"]["build-docs"]["steps"]
    run = next(s["run"] for s in steps if "git ls-files -z" in s.get("run", ""))
    clean = run[run.index("git ls-files -z") : run.index("cd -")]
    repo = tmp_path / "pages"
    kept = [
        "dashboard/index.html",
        "kernel-checks/index.html",
        "kernel-checks/npu1/runs.json",
        "component-checks/sa-placer/runs.json",
        "ci-health/status.json",
        "code-health/status.json",
        "dev/index.html",
        "index.html",
        "versions.json",
    ]
    for f in kept + ["bench/data.js", "old.html"]:
        (repo / f).parent.mkdir(parents=True, exist_ok=True)
        (repo / f).write_text("x")
    git(repo, "init", "-q")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "pages")
    subprocess.run([BASH, "-eo", "pipefail", "-c", clean], cwd=repo, check=True)
    left = set(git(repo, "ls-files").split())
    assert left == set(kept)


def test_ci_health_publishes_its_files_and_the_page(tmp_path):
    """publishCiHealth.yml's Publish step, on a local repository."""
    job = workflow("publishCiHealth.yml")["jobs"]["publish"]
    assert job["concurrency"] == {
        "group": "gh-pages-publish",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    assert job["permissions"] == {
        "contents": "write",
        "actions": "read",
        "checks": "read",
        "pull-requests": "read",
    }
    run = next(step["run"] for step in job["steps"] if step.get("name") == "Publish")
    repo = tmp_path / "repo"
    (repo / "utils/kernel_checks").mkdir(parents=True)
    page = WORKFLOWS.parents[1] / "utils/kernel_checks/index.html"
    (repo / "utils/kernel_checks/index.html").write_bytes(page.read_bytes())
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "main")
    git(repo, "switch", "-q", "--orphan", "gh-pages")
    (repo / "kernel-checks").mkdir()
    (repo / "kernel-checks/index.html").write_text("redirect")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "pages")
    git(repo, "switch", "-q", "main")
    temp = tmp_path / "runner-temp"
    (temp / "ci-health").mkdir(parents=True)
    (temp / "ci-health/status.json").write_text('{"schema": 1}')
    (temp / "ci-health/history.json").write_text('{"schema": 1, "points": []}')
    env = {
        **os.environ,
        "RUNNER_TEMP": str(temp),
        "GITHUB_SHA": "0123456789abcdef",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
    }
    subprocess.run(
        [BASH, "-eo", "pipefail", "-c", run],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    assert git(repo, "branch", "--show-current").strip() == "main"
    files = set(git(repo, "ls-tree", "-r", "--name-only", "gh-pages").split())
    assert files == {
        "kernel-checks/index.html",
        "ci-health/status.json",
        "ci-health/history.json",
        "dashboard/index.html",
    }
    assert (
        git(repo, "log", "-1", "--format=%s", "gh-pages").strip()
        == "Publish CI health for 0123456789"
    )
    # Nothing new: nothing committed.
    subprocess.run(
        [BASH, "-eo", "pipefail", "-c", run],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    assert git(repo, "rev-list", "--count", "gh-pages").strip() == "2"

    # The next publish reads the published files back: the history to extend
    # and the status whose judged runs on main need not be asked about again.
    read = next(
        step["run"]
        for step in job["steps"]
        if step.get("name") == "Read the workflows' runs and the pull requests"
    )
    result = subprocess.run(
        [BASH, "-eo", "pipefail", "-c", 'python3() { printf "%s\\n" "$@"; }\n' + read],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    args = result.stdout.splitlines()
    assert args[args.index("--previous") + 1] == f"{temp}/status.json"
    assert (temp / "status.json").read_text() == '{"schema": 1}'
    assert args[args.index("--history") + 1] == f"{temp}/history.json"


def pick_runner():
    script = WORKFLOWS.parents[1] / "utils" / "kernel_checks" / "pick_runner.py"
    spec = importlib.util.spec_from_file_location("pick_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "file, job, types",
    [
        ("nightlyKernelChecks.yml", "checks", "aie2-4col aie2p-8col"),
        ("nightlyComponentChecks.yml", "hw-checks", "aie2p-8col"),
    ],
)
def test_a_nightly_leg_runs_on_the_runner_picked_for_it(file, job, types):
    jobs = workflow(file)["jobs"]
    picker = jobs["runners"]
    assert picker["uses"] == "./.github/workflows/pickBenchRunner.yml"
    assert picker["with"]["runner_types"] == types
    assert picker["permissions"]["actions"] == "write"
    # A dispatch may name the role; the nightly and a pull request probe.
    assert picker["with"]["role"] == "${{ inputs.runner || 'auto' }}"
    assert workflow(file)["on"]["workflow_dispatch"]["inputs"]["runner"]["options"] == [
        "auto",
        "primary",
        "backup",
    ]
    leg = jobs[job]
    assert leg["needs"] == "runners"
    key = "matrix.runner_type" if job == "checks" else "'aie2p-8col'"
    assert leg["runs-on"] == "${{ fromJSON(needs.runners.outputs.runs_on)[%s] }}" % key
    # The picker's types are the legs' own.
    if job == "checks":
        legs = [m["runner_type"] for m in leg["strategy"]["matrix"]["include"]]
        assert " ".join(legs) == types


def test_the_probe_is_the_run_the_picker_looks_for():
    module = pick_runner()
    probe = workflow(module.PROBE)
    # Its run name, as GitHub renders it for one leg, is the title the picker finds.
    rendered = probe["run-name"].replace("${{ inputs.runner_type }}", "aie2p-8col")
    rendered = rendered.replace("${{ inputs.for_run }}", "123")
    assert rendered == module.probe_title("aie2p-8col", "123")
    job = probe["jobs"]["probe"]
    assert job["runs-on"] == [
        "self-hosted",
        "bench",
        "${{ inputs.runner_type }}",
        module.LABELS["primary"],
    ]
    options = probe["on"]["workflow_dispatch"]["inputs"]["runner_type"]["options"]
    assert options == ["aie2-4col", "aie2p-8col"]


def test_a_primary_that_takes_the_probe_keeps_the_leg():
    module = pick_runner()
    # Jobs as the API lists them: queued with no runner, then taken.
    queued = [{"name": "Probe (aie2-4col)", "status": "queued", "runner_name": None}]
    taken = [
        {"name": "Probe (aie2-4col)", "status": "in_progress", "runner_name": "aie1"}
    ]
    assert module.verdict([], 1, 240) is None
    assert module.verdict(queued, 239, 240) is None
    assert module.verdict(taken, 239, 240) == "primary"
    # Taken at the last moment still counts; never taken goes to the backup.
    assert module.verdict(taken, 300, 240) == "primary"
    assert module.verdict(queued, 240, 240) == "backup"
    assert module.verdict([], 240, 240) == "backup"


@pytest.mark.parametrize("role", ["primary", "backup"])
def test_a_dispatch_that_names_the_role_writes_its_labels(role, tmp_path):
    out = tmp_path / "output"
    env = {
        **os.environ,
        "GITHUB_OUTPUT": str(out),
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
    }
    script = WORKFLOWS.parents[1] / "utils" / "kernel_checks" / "pick_runner.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--runner-types",
            "aie2-4col",
            "aie2p-8col",
            "--run-id",
            "7",
            "--role",
            role,
        ],
        check=True,
        env=env,
    )
    name, value = out.read_text().strip().split("=", 1)
    assert name == "runs_on"
    assert json.loads(value) == {
        t: ["self-hosted", "bench", t, f"nightly-{role}"]
        for t in ["aie2-4col", "aie2p-8col"]
    }
    assert f"aie2p-8col: {role}" in (tmp_path / "summary").read_text()


def test_the_picker_step_runs_the_script_with_the_types_split():
    step = workflow("pickBenchRunner.yml")["jobs"]["pick"]["steps"][-1]
    assert step["id"] == "pick"
    assert 'read -ra types <<< "$TYPES"' in step["run"]
    assert '--runner-types "${types[@]}"' in step["run"]
    outputs = workflow("pickBenchRunner.yml")["on"]["workflow_call"]["outputs"]
    assert outputs["runs_on"]["value"] == "${{ jobs.pick.outputs.runs_on }}"


def test_code_health_publishes_its_files_and_the_page(tmp_path):
    """nightlyCodeHealth.yml's Publish step, on a local repository: its two
    files and the page, under the lock every gh-pages writer takes."""
    job = workflow("nightlyCodeHealth.yml")["jobs"]["publish"]
    assert job["concurrency"] == {
        "group": "gh-pages-publish",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    assert job["permissions"] == {"contents": "write", "security-events": "read"}
    # A failed check still publishes the others; a cancelled run does not.
    assert set(job["needs"]) == {"coverage", "sanitizer", "clang-tidy"}
    assert "!cancelled()" in job["if"] and "refs/heads/main" in job["if"]
    run = next(step["run"] for step in job["steps"] if step.get("name") == "Publish")
    repo = tmp_path / "repo"
    (repo / "utils/kernel_checks").mkdir(parents=True)
    page = WORKFLOWS.parents[1] / "utils/kernel_checks/index.html"
    (repo / "utils/kernel_checks/index.html").write_bytes(page.read_bytes())
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "main")
    git(repo, "switch", "-q", "--orphan", "gh-pages")
    (repo / "ci-health").mkdir()
    (repo / "ci-health/status.json").write_text("{}")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "pages")
    git(repo, "switch", "-q", "main")
    temp = tmp_path / "runner-temp"
    (temp / "code-health").mkdir(parents=True)
    (temp / "code-health/status.json").write_text('{"schema": 1}')
    (temp / "code-health/history.json").write_text('{"schema": 1, "points": []}')
    env = {
        **os.environ,
        "RUNNER_TEMP": str(temp),
        "GITHUB_SHA": "0123456789abcdef",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
    }
    subprocess.run(
        [BASH, "-eo", "pipefail", "-c", run],
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
    )
    assert git(repo, "branch", "--show-current").strip() == "main"
    files = set(git(repo, "ls-tree", "-r", "--name-only", "gh-pages").split())
    assert files == {
        "ci-health/status.json",
        "code-health/status.json",
        "code-health/history.json",
        "dashboard/index.html",
    }
    assert (
        git(repo, "log", "-1", "--format=%s", "gh-pages").strip()
        == "Publish code health for 0123456789"
    )
    # The coverage job is the coverage workflow itself, called.
    assert workflow("nightlyCodeHealth.yml")["jobs"]["coverage"]["uses"] == (
        "./.github/workflows/codeCoverage.yml"
    )
    assert "workflow_call" in workflow("codeCoverage.yml")["on"]
