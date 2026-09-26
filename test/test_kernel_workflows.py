# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-only publication regression tests; no compiled aie package required."""

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


def record_steps(job):
    return [
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("benchmark-action/")
    ]


@pytest.mark.parametrize(
    "filename,compute,static",
    [
        ("nightlyKernelChecks.yml", "checks", "false"),
    ],
)
def test_parallel_compute_has_one_main_only_publisher(filename, compute, static):
    config = workflow(filename)
    assert set(config["on"]) == {"workflow_dispatch", "schedule", "pull_request"}
    # A change to the workflow that runs the checks must itself run them.
    assert filename in " ".join(config["on"]["pull_request"]["paths"]) or (
        filename == "nightlyKernelChecks.yml"
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
    for step in record_steps(job):
        assert step["with"]["save-data-file"] == "false"
        assert "external-data-json-path" in step["with"]
    publisher = config["jobs"]["publish"]
    assert publisher["needs"] == compute
    assert "github.ref == 'refs/heads/main'" in publisher["if"]
    assert "github.event_name != 'pull_request'" in publisher["if"]
    assert publisher["if"].endswith("&& !inputs.only && !inputs.peano }}")
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
    records = record_steps(publisher)
    assert len(records) == 4
    for step in records:
        options = step["with"]
        assert options["auto-push"] == "false"
        assert options["skip-fetch-gh-pages"] == "true"
        assert options["gh-pages-branch"] == "gh-pages"
        assert options["comment-on-alert"] == "false"
    # Each NPU records only if its leg produced results, so one failed leg
    # cannot hold back the other's publication.
    assert [step["if"] for step in records] == [
        "hashFiles(format('results/npu1/{0}', env.RESULT_FILE)) != ''",
        "hashFiles(format('results/npu2/{0}', env.RESULT_FILE)) != ''",
        "inputs.static && hashFiles('results/npu1/static-pm.json') != ''",
        "inputs.static && hashFiles('results/npu2/static-pm.json') != ''",
    ]
    pushes = [step for step in steps if "git push" in step.get("run", "")]
    assert len(pushes) == 1
    push_index = steps.index(pushes[0])
    fetch_index = next(
        i for i, step in enumerate(steps) if "git fetch" in step.get("run", "")
    )
    assert all(fetch_index < steps.index(step) < push_index for step in records)
    page = next(step for step in steps if step.get("name") == "Install results page")
    assert fetch_index < steps.index(page) < push_index
    assert "utils/kernel_checks/index.html" in page["run"]
    assert (WORKFLOWS.parents[1] / "utils/kernel_checks/index.html").is_file()
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
    # `name` fails on a missing artifact; `pattern` skips it.
    assert all("name" not in step["with"] for step in downloads)
    assert all("pattern" in step["with"] for step in downloads)


def test_partial_results_are_timed_and_published():
    config = workflow("nightlyKernelChecks.yml")
    assert config["concurrency"]["cancel-in-progress"] == (
        "${{ github.event_name == 'pull_request' }}"
    )
    assert "github.event.pull_request.number" in config["concurrency"]["group"]
    steps = {step.get("id"): step for step in config["jobs"]["checks"]["steps"]}
    assert steps["perf"]["if"] == (
        "${{ !cancelled() && steps.preflight.outcome == 'success' }}"
    )
    assert "--junitxml=correctness.xml" in steps["correctness"]["run"]
    assert "--correctness-results correctness.xml" in steps["perf"]["run"]
    assert config["jobs"]["publish"]["if"].startswith("${{ !cancelled() && ")


def test_dispatch_filter_is_passed_as_data_not_shell_source():
    job = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]
    step = next(step for step in job["steps"] if step.get("id") == "perf")
    assert step["env"]["ONLY"] == "${{ inputs.only }}"
    assert "inputs.only" not in step["run"]
    assert '${ONLY:+-k "($ONLY) or test_measurement_is_sane"}' in step["run"]


@pytest.mark.parametrize("only", ["", "softmax", "softmax and not large", "$(false)"])
def test_dispatch_filter_keeps_sanity_and_preserves_shell_quoting(only, tmp_path):
    steps = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "perf")
    command = run[run.index("python -m pytest") :].split("2>&1", 1)[0]
    result = subprocess.run(
        ["bash", "-eu", "-c", 'python() { printf "%s\\n" "$@"; }\n' + command],
        cwd=tmp_path,
        env={**os.environ, "ONLY": only},
        capture_output=True,
        text=True,
        check=True,
    )
    args = result.stdout.splitlines()
    if only:
        assert args[-2:] == ["-k", f"({only}) or test_measurement_is_sane"]
    else:
        assert "-k" not in args


NIGHTLY_PAGE = """
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092401+6dc4d6dd-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" rel="nofollow">
  <span class="text-bold">llvm_aie-22.0.0.2026092401+6dc4d6dd-py3-none-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl</span>
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092401+6dc4d6dd-py3-none-win_amd64.whl" rel="nofollow">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092001+0006955e-py3-none-win_amd64.whl" rel="nofollow">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092101+0006955e-py3-none-win_amd64.whl" rel="nofollow">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026091901+0006955e-py3-none-win_amd64.whl" rel="nofollow">
<a href="/Xilinx/llvm-aie/releases/download/nightly/llvm_aie-22.0.0.2026092201+0006955f-py3-none-win_amd64.whl" rel="nofollow">
"""

PINNED = "llvm-aie==22.0.0.2026092401+6dc4d6dd"
WHEEL = "https://example.com/llvm_aie-1.0-py3-none-any.whl"


def run_peano_step(peano, tmp_path):
    step = next(
        step
        for step in workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
        if step.get("name") == "Install requested Peano"
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
        ("22.0.0.2026092401+6dc4d6dd", PINNED),
        ("6dc4d6dd", PINNED),
        ("6dc4d6d", PINNED),
        ("6dc4d6dd71558b553448df97254408ec3d800eb5", PINNED),
        # A commit rebuilt by several nightlies resolves to the newest wheel.
        ("0006955e", "llvm-aie==22.0.0.2026092101+0006955e"),
        (WHEEL, WHEEL),
    ],
)
def test_dispatch_installs_the_requested_peano(peano, spec, tmp_path):
    step = next(
        step
        for step in workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
        if step.get("name") == "Install requested Peano"
    )
    assert step["if"] == "${{ inputs.peano }}"
    assert step["env"]["PEANO"] == "${{ inputs.peano }}"
    assert "inputs.peano" not in step["run"]
    result, args = run_peano_step(peano, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert args[-1] == spec
    assert "--force-reinstall" in args
    assert (tmp_path / "summary").read_text() == "Peano: Version: stub\n"


@pytest.mark.parametrize(
    "peano",
    ["deadbeef", "0006955", "$(false)", "1.0; true", "http://example.com/x.whl"],
)
def test_dispatch_rejects_an_unresolvable_peano(peano, tmp_path):
    result, args = run_peano_step(peano, tmp_path)
    assert result.returncode == 1
    assert "::error::" in result.stdout
    assert args is None


def test_preflight_sets_memlock_and_reuses_one_examine():
    job = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]
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
    # Execute the complete command in the condition: sudoers can match arguments.
    configure = 'if sudo -n "$XRT_SMI" configure -d "$BDF" --pmode "$PERF_PMODE"; then'
    assert configure in run
    assert run.count('sudo -n "$XRT_SMI" configure') == 1
    assert "else\n" in run[run.index(configure) :]
    assert 'echo "::warning::Cannot set --pmode $PERF_PMODE' in run
    assert '"$XRT_SMI" examine -d "$BDF" --report platform' in run
    assert "xrt-smi examine | grep -oE" not in run


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
    path.write_text(json.dumps({"preflight": {"npu": "npu1", "pmode": pmode}}))


def test_each_power_mode_is_its_own_series(tmp_path):
    checks = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]
    steps = checks["steps"]
    run = next(step["run"] for step in steps if step.get("id") == "perf")
    assert "--pmode any" in run
    assert '--pmode "$PERF_PMODE"' not in run
    assert not record_steps(checks)

    publisher = workflow("publishKernelResults.yml")["jobs"]["publish"]
    read = next(step for step in publisher["steps"] if step.get("id") == "pmode")
    assert read["if"] == "${{ !inputs.static }}"
    write_meta(tmp_path / "results/npu1/meta.json", "performance")
    write_meta(tmp_path / "results/npu2/meta.json", "turbo")
    run = read["run"].replace("$RESULT_FILE", "perf.json")
    # A leg with meta but no results (its NPU checks failed) is skipped.
    assert run_step(run, tmp_path) == {}
    (tmp_path / "results/npu1/perf.json").write_text("[]")
    assert run_step(run, tmp_path) == {"npu1": "performance"}
    (tmp_path / "results/npu2/perf.json").write_text("[]")
    assert run_step(run, tmp_path) == {"npu1": "performance", "npu2": "turbo"}
    write_meta(tmp_path / "results/npu2/meta.json", None)
    with pytest.raises(subprocess.CalledProcessError):
        run_step(run, tmp_path)
    names = [step["with"]["name"] for step in record_steps(publisher)[:2]]
    for npu, name in zip(["npu1", "npu2"], names):
        assert (
            f"format('aie_kernels ({npu}, {{0}})', steps.pmode.outputs.{npu})" in name
        )


def test_unpublished_runs_report_against_the_published_baseline(tmp_path):
    config = workflow("nightlyKernelChecks.yml")
    assert "pull-requests" not in config["permissions"]
    report = config["jobs"]["report"]
    assert report["needs"] == "checks"
    assert "github.event_name != 'schedule'" in report["if"]
    assert "needs.checks.result != 'skipped'" in report["if"]
    assert report["permissions"] == {"contents": "read", "pull-requests": "write"}
    steps = report["steps"]
    downloads = [
        step["with"]
        for step in steps
        if step.get("uses", "").startswith("actions/download-artifact@")
    ]
    assert downloads == [
        {
            "pattern": f"kernel-checks-{npu}-${{{{ github.run_id }}}}",
            "path": f"results/{npu}",
        }
        for npu in ("npu1", "npu2")
    ]
    # Each restore reads what the publisher saved, under the path it saved.
    publisher = workflow("publishKernelResults.yml")["jobs"]["publish"]
    saved = [
        step["with"]
        for step in publisher["steps"]
        if step.get("uses", "").startswith("actions/cache/save@")
    ]
    restored = [
        step["with"]
        for step in steps
        if step.get("uses", "").startswith("actions/cache/restore@")
    ]
    for npu, save, restore in zip(("npu1", "npu2"), saved, restored):
        assert save["path"] == restore["path"] == "baseline"
        assert f"'kernel-checks-baseline-{npu}' }}}}-" in save["key"]
        assert restore["key"] == restore["restore-keys"]
        assert restore["key"] == f"kernel-checks-baseline-{npu}-"

    # Run the steps between the downloads and the comment, in order, on the
    # artifacts an npu2-only run leaves.
    (tmp_path / "results/npu2").mkdir(parents=True)
    (tmp_path / "results/npu2/perf.json").write_text(
        json.dumps([{"name": "relu/1024/bf16/cycles", "unit": "cycles", "value": 9}])
    )
    (tmp_path / "utils").symlink_to(WORKFLOWS.parents[1] / "utils")
    summary = tmp_path / "summary.md"
    env = {
        **os.environ,
        "GITHUB_STEP_SUMMARY": str(summary),
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_REPOSITORY": "Xilinx/mlir-aie",
        "GITHUB_RUN_ID": "7",
    }
    for step in steps:
        if step.get("uses", "").startswith("actions/cache/restore@"):
            if step["with"]["key"].endswith("npu2-"):
                (tmp_path / "baseline").mkdir()
                (tmp_path / "baseline/series.json").write_text('{"entries": {}}')
        elif "run" in step and "gh api" not in step["run"]:
            subprocess.run(
                ["bash", "-eo", "pipefail", "-c", step["run"]],
                cwd=tmp_path,
                env=env,
                check=True,
            )
    assert (tmp_path / "baselines/npu2/series.json").is_file()
    assert not (tmp_path / "baselines/npu1").exists()
    text = summary.read_text()
    assert text == (tmp_path / "report.md").read_text()
    assert "(https://github.com/Xilinx/mlir-aie/actions/runs/7)" in text
    assert "| npu2 | ? | none cached | 1 | 0 |" in text
    comment = next(step for step in steps if "gh api" in step.get("run", ""))
    assert comment["if"] == "github.event_name == 'pull_request'"
    marker = (WORKFLOWS.parents[1] / "utils/kernel_checks/pr_report.py").read_text()
    assert 'MARKER = "<!-- kernel-checks-report -->"' in marker
    assert 'startswith("<!-- kernel-checks-report -->")' in comment["run"]


def test_results_page_is_committed_to_the_publication_branch(tmp_path):
    steps = workflow("publishKernelResults.yml")["jobs"]["publish"]["steps"]
    page = next(step for step in steps if step.get("name") == "Install results page")
    assert page["if"] == "${{ !inputs.static }}"
    source = WORKFLOWS.parents[1] / "utils/kernel_checks/index.html"

    def git(*args):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    (tmp_path / "utils/kernel_checks").mkdir(parents=True)
    (tmp_path / "utils/kernel_checks/index.html").write_bytes(source.read_bytes())
    git("add", ".")
    git("commit", "-q", "-m", "main")
    git("switch", "-q", "--orphan", "gh-pages")
    git("commit", "-q", "--allow-empty", "-m", "pages")
    git("switch", "-q", "main")

    run = page["run"].replace("$RUNNER_TEMP", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    (tmp_path / "results/npu1").mkdir(parents=True)
    (tmp_path / "results/npu1/catalogue.json").write_text('{"npu": "npu1"}')
    for _ in range(2):  # The second run finds nothing to change.
        subprocess.run(
            ["bash", "-eo", "pipefail", "-c", run],
            cwd=tmp_path,
            env={k: v for k, v in os.environ.items() if k != "BRANCH"},
            check=True,
        )
        assert git("branch", "--show-current") == "main"
    assert (
        git("show", "gh-pages:kernel-checks/index.html") == source.read_text().strip()
    )
    assert (
        git("show", "gh-pages:kernel-checks/npu1/catalogue.json") == '{"npu": "npu1"}'
    )
    assert git("ls-tree", "-r", "--name-only", "gh-pages", "kernel-checks/npu2") == ""
    assert git("rev-list", "--count", "gh-pages") == "2"


@pytest.mark.parametrize(
    "filename,compute",
    [("nightlyKernelChecks.yml", "checks")],
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


@pytest.mark.parametrize("step_id", ["correctness", "perf"])
def test_checks_use_built_package(step_id):
    steps = workflow("nightlyKernelChecks.yml")["jobs"]["checks"]["steps"]
    run = next(step["run"] for step in steps if step.get("id") == step_id)
    assert (
        run.index("sudo prlimit -lunlimited --pid $$")
        < run.index("source aie-venv/bin/activate")
        < run.index("source utils/env_setup.sh mlir_aie")
        < run.index("python -m pytest")
    )


def test_docs_cleanup_preserves_kernel_checks_history():
    steps = workflow("generateDocs.yml")["jobs"]["build-docs"]["steps"]
    cleanup = next(
        step["run"] for step in steps if "git ls-files -z" in step.get("run", "")
    )
    filters = cleanup.split("git ls-files -z", 1)[1].split("| xargs", 1)[0]
    result = subprocess.run(
        ["bash", "-c", "cat " + filters.rstrip().rstrip("\\")],
        input=b"\0".join(
            [
                b"kernel-checks/npu1/data.js",
                b"kernel-checks/static/aie2/data.js",
                b"bench/npu1/index.html",
                b"dev/index.html",
                b"legacy.html",
                b"",
            ]
        ),
        capture_output=True,
        check=True,
    )
    assert result.stdout == b"legacy.html\0"
