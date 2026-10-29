# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""utils/kernel_checks/code_health.py against recorded sanitizer, clang-tidy and lcov output."""

import datetime
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "utils" / "kernel_checks"))

import code_health as ch  # noqa: E402

SRC = "/home/runner/work/mlir-aie/mlir-aie"

ASAN_OUT = f"""Exit Code: 1

Command Output (stderr):
--
=================================================================
==4242==ERROR: AddressSanitizer: heap-use-after-free on address 0x6020000000f0 at pc 0x55d1 bp 0x7ffd sp 0x7ffc
READ of size 8 at 0x6020000000f0 thread T0
    #0 0x55d1a in llvm::SmallVectorBase<unsigned int>::size() const {SRC}/mlir/include/llvm/ADT/SmallVector.h:98:32
    #1 0x55d1b in xilinx::AIE::Pathfinder::findPaths(int) {SRC}/lib/Dialect/AIE/Transforms/AIEPathFinder.cpp:412:18
    #2 0x55d1c in xilinx::AIE::AIEPathfinderPass::runOnOperation() {SRC}/lib/Dialect/AIE/Transforms/AIECreatePathFindFlows.cpp:88:5
SUMMARY: AddressSanitizer: heap-use-after-free {SRC}/lib/Dialect/AIE/Transforms/AIEPathFinder.cpp:412:18 in findPaths
==4242==ABORTING
"""

UBSAN_OUT = f"""Command Output (stderr):
{SRC}/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp:233:41: runtime error: shift exponent 32 is too large for 32-bit type 'uint32_t' (aka 'unsigned int')
    #0 0x7f in xilinx::AIEX::appendWriteBd(int) {SRC}/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp:233:41
SUMMARY: UndefinedBehaviorSanitizer: undefined-behavior {SRC}/lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp:233:41
"""

TSAN_OUT = f"""==================
WARNING: ThreadSanitizer: data race (pid=777)
  Write of size 8 at 0x7b0400000010 by thread T3:
    #0 xilinx::AIE::DeviceModel::cache(int) {SRC}/lib/Dialect/AIE/IR/AIETargetModel.cpp:57:12 (aie-opt+0x1234)
    #1 llvm::ThreadPool::run() /opt/mlir/llvm/lib/Support/ThreadPool.cpp:40:3 (aie-opt+0x5678)
  Previous read of size 8 at 0x7b0400000010 by thread T1:
    #0 xilinx::AIE::DeviceModel::cache(int) {SRC}/lib/Dialect/AIE/IR/AIETargetModel.cpp:51:9 (aie-opt+0x1200)
SUMMARY: ThreadSanitizer: data race {SRC}/lib/Dialect/AIE/IR/AIETargetModel.cpp:57:12 in xilinx::AIE::DeviceModel::cache(int)
==================
"""


def lit(*tests):
    return {
        "__version__": [18, 0, 0],
        "tests": [{"name": n, "code": c, "output": o} for n, c, o in tests],
    }


def test_one_bug_from_many_tests_is_one_finding():
    piece = ch.summarize_sanitizer(
        "asan-ubsan",
        [
            lit(
                ("AIE :: create-flows/a.mlir", "FAIL", ASAN_OUT),
                # The same bug, a line moved by an edit, from a test that expects
                # the tool to fail: still found, still the same finding.
                (
                    "AIE :: create-flows/b.mlir",
                    "PASS",
                    ASAN_OUT.replace("412:18", "415:18"),
                ),
                ("AIE :: npu-xrt/c.mlir", "FAIL", UBSAN_OUT),
                ("AIE :: slow/d.mlir", "TIMEOUT", "Reached timeout of 1800 seconds"),
                ("AIE :: x/e.mlir", "UNSUPPORTED", ""),
            )
        ],
        SRC,
    )
    assert piece["finding_count"] == 2
    asan = next(f for f in piece["findings"] if f["sanitizer"] == "asan")
    assert asan["kind"] == "heap-use-after-free"
    # The first frame in this repository's code, not the ADT header.
    assert asan["where"]["file"] == "lib/Dialect/AIE/Transforms/AIEPathFinder.cpp"
    assert asan["where"]["function"].startswith("xilinx::AIE::Pathfinder::findPaths")
    assert asan["test_count"] == 2
    assert SRC not in asan["excerpt"]
    ub = next(f for f in piece["findings"] if f["sanitizer"] == "ubsan")
    assert ub["kind"] == "shift exponent N is too large for N-bit type T (aka T)"
    assert ub["message"] == "shift exponent 32 is too large for 32-bit type 'uint32_t'"
    assert ub["where"] == {
        "file": "lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp",
        "line": 233,
        "function": "xilinx::AIEX::appendWriteBd(int)",
    }
    assert piece["failed_without_report"] == [
        {"test": "AIE :: slow/d.mlir", "code": "TIMEOUT"}
    ]
    assert piece["ran"] == 4 and piece["tests"] == 5


def test_tsan_names_the_race_in_our_code():
    piece = ch.summarize_sanitizer(
        "tsan", [lit(("AIE :: python-concurrency/p.py", "FAIL", TSAN_OUT))], SRC
    )
    (f,) = piece["findings"]
    assert (f["sanitizer"], f["kind"]) == ("tsan", "data race")
    assert f["where"]["file"] == "lib/Dialect/AIE/IR/AIETargetModel.cpp"


def test_a_finding_keeps_its_first_night():
    before = {
        "findings": [
            {"key": "asan|heap-use-after-free|lib/X.cpp|f", "since": "2026-10-20"}
        ]
    }
    piece = {
        "findings": [
            {"key": "asan|heap-use-after-free|lib/X.cpp|f"},
            {"key": "tsan|data race|lib/Y.cpp|g"},
        ]
    }
    ch.date_findings(piece, before, "2026-10-29")
    assert [(f["since"], f["new"]) for f in piece["findings"]] == [
        ("2026-10-20", False),
        ("2026-10-29", True),
    ]


TIDY_LOG = f"""{SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:10:3: error: use auto when initializing with a cast [modernize-use-auto,-warnings-as-errors]
{SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:10:3: error: use auto when initializing with a cast [modernize-use-auto,-warnings-as-errors]
{SRC}/lib/Dialect/AIE/Transforms/AIEFoo.cpp:5:1: error: variable 'x' is not initialized [cppcoreguidelines-init-variables,-warnings-as-errors]
{SRC}/lib/Dialect/AIE/Transforms/AIEFoo.cpp:9:1: error: redundant cast [readability-redundant-casting,-warnings-as-errors]
{SRC}/lib/Broken.cpp:1:10: error: 'gen.h.inc' file not found [clang-diagnostic-error]
/usr/include/c++/13/bits/stl.h:1:1: warning: something [bugprone-x]
"""


def test_tidy_splits_enabled_files_from_the_rest():
    files = [
        "lib/Dialect/AIE/IR/AIEDialect.cpp",
        "lib/Dialect/AIE/Transforms/AIEFoo.cpp",
        "lib/Clean.cpp",
        "lib/Broken.cpp",
    ]
    enabled = re.compile(r"(?x)^(lib/Dialect/AIE/IR/AIEDialect\.cpp)$")
    piece = ch.summarize_tidy(TIDY_LOG, files, enabled, SRC)
    assert piece["enabled_files"] == 1
    assert [(f["file"], f["check"]) for f in piece["enabled_findings"]] == [
        ("lib/Dialect/AIE/IR/AIEDialect.cpp", "modernize-use-auto")
    ]
    assert piece["outside_count"] == 2
    assert piece["outside_by_check"] == {
        "cppcoreguidelines-init-variables": 1,
        "readability-redundant-casting": 1,
    }
    assert piece["ready"] == ["lib/Clean.cpp"]
    assert piece["could_not_parse"] == ["lib/Broken.cpp"]


def test_the_enabled_list_is_read_from_the_pre_commit_hook():
    rx = ch.enabled_pattern(ROOT / ".pre-commit-config.yaml")
    assert rx.match("lib/Dialect/AIE/Transforms/AIEPathFinder.cpp")
    assert not rx.match("lib/Dialect/AIE/Transforms/NotARealFile.cpp")


LCOV = f"""SF:{SRC}/lib/Dialect/AIE/Transforms/AIEPathFinder.cpp
FNF:10
FNH:8
LF:200
LH:150
end_of_record
SF:{SRC}/lib/Targets/AIETargetNPU.cpp
FNF:4
FNH:4
LF:100
LH:90
end_of_record
SF:{SRC}/tools/aiecc/aiecc.cpp
LF:0
LH:0
end_of_record
"""


def test_coverage_by_area_and_what_lost_lines():
    files = ch.read_lcov(LCOV, SRC)
    assert files == {
        "lib/Dialect/AIE/Transforms/AIEPathFinder.cpp": [150, 200, 8, 10],
        "lib/Targets/AIETargetNPU.cpp": [90, 100, 4, 4],
    }
    previous = {"files": {"lib/Targets/AIETargetNPU.cpp": [95, 100, 4, 4]}}
    cov = ch.coverage_status(files, previous)
    assert cov["lines"] == {"hit": 240, "total": 300}
    # Most uncovered lines first.
    assert [a["area"] for a in cov["areas"]] == ["lib/Dialect/AIE", "lib/Targets"]
    assert cov["lost"] == [
        {
            "file": "lib/Targets/AIETargetNPU.cpp",
            "before": 95.0,
            "after": 90.0,
            "lines_lost": 5,
        }
    ]


def alert(n, severity, lang, path):
    return {
        "number": n,
        "rule": {
            "id": f"rule-{n}",
            "description": "d",
            "security_severity_level": severity,
        },
        "tool": {"name": "CodeQL"},
        "html_url": f"https://x/{n}",
        "created_at": "2026-10-28T00:00:00Z",
        "most_recent_instance": {
            "category": f"/language:{lang}",
            "location": {"path": path, "start_line": 3},
        },
    }


def test_code_scanning_says_what_is_new_and_fixed():
    pages = {
        1: [
            alert(7, "high", "c-cpp", "lib/a.cpp"),
            alert(3, None, "python", "python/b.py"),
        ]
    }
    get = lambda path: pages.get(int(re.search(r"[?&]page=(\d+)", path).group(1)), [])
    cs = ch.scanning_status(
        get, "Xilinx/mlir-aie", {"alerts": [{"number": 3}, {"number": 1}]}
    )
    assert cs["open"] == 2 and cs["new"] == 1 and cs["fixed"] == 1
    assert [(a["number"], a["language"], a["new"]) for a in cs["alerts"]] == [
        (7, "cpp", True),
        (3, "python", False),
    ]
    assert cs["by_severity"] == {"high": 1, "note": 1}
    # The first publish has nothing to call new.
    assert ch.scanning_status(get, "Xilinx/mlir-aie", None)["new"] == 0


@pytest.mark.parametrize("count", [ch.KEEP_ALERTS + 1, 2001])
def test_code_scanning_keeps_all_identities_for_churn(count):
    alerts = [alert(n, "high", "c-cpp", "lib/a.cpp") for n in range(count)]

    def get(path):
        page = int(re.search(r"[?&]page=(\d+)", path).group(1))
        return alerts[(page - 1) * 100 : page * 100]

    previous = ch.scanning_status(get, "Xilinx/mlir-aie", None)
    assert previous["open"] == count
    assert len(previous["alerts"]) == ch.KEEP_ALERTS
    assert previous["alert_ids"] == list(range(count))
    unchanged = ch.scanning_status(get, "Xilinx/mlir-aie", previous)
    assert unchanged["new"] == unchanged["fixed"] == 0
    # Alert zero was omitted from the display, but must still count as fixed.
    alerts.pop(0)
    alerts.append(alert(count, "high", "c-cpp", "lib/a.cpp"))
    changed = ch.scanning_status(get, "Xilinx/mlir-aie", unchanged)
    assert changed["new"] == changed["fixed"] == 1


@pytest.mark.parametrize("command", ["sanitizer", "tidy", "publish"])
@pytest.mark.parametrize("after_command", [False, True])
def test_source_root_option_on_each_command(tmp_path, command, after_command):
    log = tmp_path / "input"
    out = tmp_path / "out.json"
    files = tmp_path / "files.txt"
    files.write_text("lib/Dialect/AIE/IR/AIEDialect.cpp\n")
    if command == "sanitizer":
        log.write_text(json.dumps(lit(("AIE :: a.mlir", "FAIL", ASAN_OUT))))
        args = ["--name", "asan-ubsan", str(log)]
    elif command == "tidy":
        log.write_text(TIDY_LOG)
        args = ["--files", str(files), str(log)]
    else:
        args = []
    root = ["--source-root", SRC]
    argv = [command, *root] if after_command else [*root, command]
    assert ch.main([*argv, *args, "--out", str(out)]) == 0
    result = json.loads(
        (out / "status.json" if command == "publish" else out).read_text()
    )
    if command == "sanitizer":
        assert result["findings"][0]["where"]["file"].startswith("lib/")
    elif command == "tidy":
        assert result["enabled_files"] == 1


@pytest.mark.parametrize("contents", [None, "", "{broken"])
def test_missing_lit_report_does_not_publish_a_healthy_summary(tmp_path, contents):
    report = tmp_path / "report.json"
    if contents is not None:
        report.write_text(contents)
    out = tmp_path / "summary.json"
    assert ch.main(["sanitizer", "--name", "tsan", "--out", str(out), str(report)]) == 1
    assert not out.exists()


def test_sanitizer_uses_available_reports(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps(lit(("AIE :: a.mlir", "PASS", ""))))
    out = tmp_path / "summary.json"
    assert (
        ch.main(
            [
                "sanitizer",
                "--name",
                "tsan",
                "--out",
                str(out),
                str(tmp_path / "missing.json"),
                str(report),
            ]
        )
        == 0
    )
    assert json.loads(out.read_text())["ran"] == 1


def test_publish_writes_status_and_history(tmp_path):
    (tmp_path / "cov.lcov").write_text(LCOV)
    asan = ch.summarize_sanitizer(
        "asan-ubsan", [lit(("AIE :: a.mlir", "FAIL", ASAN_OUT))], SRC
    )
    (tmp_path / "asan.json").write_text(json.dumps(asan))
    now = datetime.datetime(2026, 10, 29, 9, tzinfo=datetime.timezone.utc)
    ch.main(
        [
            "--source-root",
            SRC,
            "publish",
            "--coverage",
            str(tmp_path / "cov.lcov"),
            "--piece",
            str(tmp_path / "asan.json"),
            "--piece",
            str(tmp_path / "gone.json"),
            "--out",
            str(tmp_path / "out"),
        ],
        now=now,
    )
    status = json.loads((tmp_path / "out/status.json").read_text())
    assert status["generated_at"] == "2026-10-29T09:00:00Z"
    # A job that left no summary is missing, not clean.
    assert status["tsan"] == {"check": "tsan", "missing": True}
    assert status["clang-tidy"]["missing"] and status["code_scanning"]["missing"]
    assert status["asan-ubsan"]["findings"][0]["since"] == "2026-10-29"
    (point,) = json.loads((tmp_path / "out/history.json").read_text())["points"]
    assert point["lines"] == 80.0 and point["asan-ubsan"]["by"] == {"asan": 1}
    assert "tsan" not in point


def test_the_nightly_publishes_what_the_page_reads():
    wf = (ROOT / ".github/workflows/nightlyCodeHealth.yml").read_text()
    assert "code_health.py publish" in wf and "code-health/" in wf
    assert "group: gh-pages-publish" in wf
    page = (ROOT / "utils/kernel_checks/index.html").read_text()
    assert "'../code-health'" in page
    listed = json.loads((ROOT / "utils/kernel_checks/ci_workflows.json").read_text())
    files = [w["file"] for k in ("critical", "workflows") for w in listed[k]]
    assert "nightlyCodeHealth.yml" in files and "codeCoverage.yml" not in files


def test_a_flood_of_findings_stays_within_the_budget(tmp_path):
    """However bad a night, status.json stays bounded; history.json keeps a
    year of points and no more."""
    frame = "xilinx::AIE::" + "Template<" * 400 + "int" + ">" * 400 + "::run(int)"
    out = "\n".join(
        f"==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x{i:x}\n"
        + "\n".join(
            f"    #{k} 0x1 in {frame}{i} {SRC}/lib/F{i}.cpp:{k}:1" for k in range(60)
        )
        for i in range(400)
    )
    piece = ch.summarize_sanitizer(
        "asan-ubsan",
        [lit(*[(f"AIE :: t{n}.mlir", "FAIL", out) for n in range(3)])],
        SRC,
    )
    assert piece["finding_count"] == 400 and len(piece["findings"]) == ch.KEEP_FINDINGS
    assert sum(bool(f["excerpt"]) for f in piece["findings"]) == ch.EXCERPTS_KEPT
    tidy = "".join(
        f"{SRC}/lib/F{i}.cpp:{i}:1: error: {'x' * 5000} [modernize-use-auto,-warnings-as-errors]\n"
        for i in range(5000)
    )
    tidy_piece = ch.summarize_tidy(
        tidy,
        [f"lib/F{i}.cpp" for i in range(5000)],
        re.compile(r"lib/F\d\d?\.cpp$"),
        SRC,
    )
    many = [alert(n, "high", "c-cpp", "lib/a.cpp") for n in range(1, 2001)]
    for a in many:
        a["rule"]["description"] = "d" * 5000
    pages = {k + 1: many[100 * k : 100 * (k + 1)] for k in range(20)}
    get = lambda path: pages.get(int(re.search(r"[?&]page=(\d+)", path).group(1)), [])
    scanning = ch.scanning_status(get, "Xilinx/mlir-aie", None)
    files = {f"lib/Dialect/X/F{i}.cpp": [i, 1000, 1, 10] for i in range(400)}
    status = ch.build_status(
        {
            "asan-ubsan": piece,
            "tsan": {**piece, "check": "tsan"},
            "clang-tidy": tidy_piece,
        },
        ch.coverage_status(files, None),
        scanning,
        None,
        datetime.datetime(2026, 10, 29, tzinfo=datetime.timezone.utc),
        {"id": 1},
    )
    size = len(json.dumps(status, separators=(",", ":")))
    assert size < ch.STATUS_BUDGET, size
    history = None
    for n in range(800):
        history = ch.update_history(history, {**ch.history_point(status), "run": n})
    assert len(history["points"]) == ch.HISTORY_POINTS
    assert len(json.dumps(history, separators=(",", ":"))) < ch.HISTORY_BUDGET
