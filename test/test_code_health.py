# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""utils/kernel_checks/code_health.py against recorded sanitizer, clang-tidy and lcov output."""

import datetime
import json
import re
import subprocess
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


# Two real ASan reports (clang 18, a double free and a free of a stack
# object), their frames moved into this repository's code.
ASAN_DOUBLE_FREE = f"""\
==715658==ERROR: AddressSanitizer: attempting double-free on 0x502000000010 in thread T0:
    #0 0x7982574fc4d8 in free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:52
    #1 0x5c1cf17741da in xilinx::AIE::release() {SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:2
    #2 0x79825702a1c9 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58

0x502000000010 is located 0 bytes inside of 8-byte region [0x502000000010,0x502000000018)
freed by thread T0 here:
    #0 0x7982574fc4d8 in free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:52
    #1 0x5c1cf17741ce in xilinx::AIE::release() {SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:2

SUMMARY: AddressSanitizer: double-free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:52 in free
==715658==ABORTING
"""
ASAN_BAD_FREE = f"""\
==715667==ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed: 0x7779a4500020 in thread T0
    #0 0x7982574fc4d8 in free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:52
    #1 0x5c1cf17741da in xilinx::AIE::release() {SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:2
    #2 0x79825702a1c9 in __libc_start_call_main ../sysdeps/nptl/libc_start_call_main.h:58

SUMMARY: AddressSanitizer: bad-free ../../../../src/libsanitizer/asan/asan_malloc_linux.cpp:52 in free
==715667==ABORTING
"""


def test_two_asan_bugs_in_one_function_stay_apart():
    piece = ch.summarize_sanitizer(
        "asan-ubsan",
        [
            lit(
                ("AIE :: a.mlir", "FAIL", ASAN_DOUBLE_FREE),
                ("AIE :: b.mlir", "FAIL", ASAN_BAD_FREE),
            )
        ],
        SRC,
    )
    assert sorted(f["kind"] for f in piece["findings"]) == ["bad-free", "double-free"]
    assert piece["finding_count"] == 2


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
    # The new one first.
    assert [(f["since"], f["new"]) for f in piece["findings"]] == [
        ("2026-10-29", True),
        ("2026-10-20", False),
    ]


def test_unnamed_namespace_functions_stay_apart():
    def at(func):
        return {
            "sanitizer": "asan",
            "kind": "heap-use-after-free",
            "where": {"file": "lib/X.cpp", "function": func},
        }

    a = ch.finding_key(at("(anonymous namespace)::appendSync(mlir::OpBuilder&)"))
    b = ch.finding_key(at("(anonymous namespace)::appendWait(int) const"))
    assert a != b
    assert a.endswith("|(anonymous namespace)::appendSync")


def test_a_test_reporting_one_bug_often_counts_once():
    reports = [(f"AIE :: t{n}.mlir", "FAIL", ASAN_OUT) for n in range(ch.KEEP_TESTS)]
    # Past the kept sample, the same tests report the same bug again.
    more = [(f"AIE :: t{n}.mlir", "FAIL", ASAN_OUT) for n in range(ch.KEEP_TESTS + 5)]
    piece = ch.summarize_sanitizer(
        "asan-ubsan", [lit(*reports), lit(*more), lit(*more)], SRC
    )
    (f,) = piece["findings"]
    assert f["test_count"] == ch.KEEP_TESTS + 5
    assert len(f["tests"]) == ch.KEEP_TESTS


def test_sanitizer_history_counts_every_finding():
    out = "\n".join(
        f"==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x{i:x}\n"
        f"    #0 0x1 in f{i}() {SRC}/lib/F{i}.cpp:1:1"
        for i in range(ch.KEEP_FINDINGS + 1)
    )
    # The least frequent kind sorts past the displayed findings.
    piece = ch.summarize_sanitizer(
        "asan-ubsan",
        [
            lit(
                *[(f"AIE :: t{n}.mlir", "FAIL", out) for n in range(2)],
                ("AIE :: u.mlir", "FAIL", UBSAN_OUT),
            )
        ],
        SRC,
    )
    assert all(f["sanitizer"] == "asan" for f in piece["findings"][: ch.KEEP_FINDINGS])
    status = ch.build_status(
        {"asan-ubsan": piece},
        None,
        None,
        None,
        datetime.datetime(2026, 10, 29, tzinfo=datetime.timezone.utc),
        {"id": 1},
    )
    by = ch.history_point(status)["asan-ubsan"]["by"]
    assert by == {"asan": ch.KEEP_FINDINGS + 1, "ubsan": 1}


def test_a_missing_night_keeps_the_first_nights():
    def night(day, pieces, previous):
        now = datetime.datetime(2026, 10, day, tzinfo=datetime.timezone.utc)
        return json.loads(
            json.dumps(ch.build_status(pieces, None, None, previous, now, {"id": day}))
        )

    def piece():
        return ch.summarize_sanitizer(
            "tsan", [lit(("AIE :: p.py", "FAIL", TSAN_OUT))], SRC
        )

    first = night(20, {"tsan": piece()}, None)
    assert first["tsan"]["findings"][0]["new"]
    gone = night(21, {}, first)
    assert gone["tsan"]["missing"] and "findings" not in gone["tsan"]
    gone = night(22, {}, gone)
    back = night(23, {"tsan": piece()}, gone)
    (f,) = back["tsan"]["findings"]
    assert (f["since"], f["new"]) == ("2026-10-20", False)


def test_a_new_finding_past_the_display_is_shown_and_counted():
    many = "\n".join(
        f"==1==ERROR: AddressSanitizer: heap-use-after-free on address 0x{i:x}\n"
        f"    #0 0x1 in f{i}() {SRC}/lib/F{i}.cpp:1:1"
        for i in range(ch.KEEP_FINDINGS + 1)
    )
    asan = [(f"AIE :: t{n}.mlir", "FAIL", many) for n in range(2)]

    def night(day, tests, previous):
        piece = ch.summarize_sanitizer("asan-ubsan", [lit(*tests)], SRC)
        now = datetime.datetime(2026, 10, day, tzinfo=datetime.timezone.utc)
        status = ch.build_status(
            {"asan-ubsan": piece}, None, None, previous, now, {"id": day}
        )
        return json.loads(json.dumps(status))["asan-ubsan"]

    first = night(20, asan, None)
    # Reached from one test, the new UBSan bug ranks below every ASan one.
    second = night(
        21, [*asan, ("AIE :: u.mlir", "FAIL", UBSAN_OUT)], {"asan-ubsan": first}
    )
    assert second["finding_count"] == ch.KEEP_FINDINGS + 2
    assert len(second["findings"]) == ch.KEEP_FINDINGS
    assert second["new_count"] == 1
    assert (second["findings"][0]["sanitizer"], second["findings"][0]["new"]) == (
        "ubsan",
        True,
    )
    assert second["findings"][0]["excerpt"]
    assert len(second["known"]) == ch.KEEP_FINDINGS + 2
    # Past the display, an ASan finding keeps its first night.
    assert (
        second["known"][
            ch.known_id(
                f"asan|heap-use-after-free|lib/F{ch.KEEP_FINDINGS}.cpp|f{ch.KEEP_FINDINGS}"
            )
        ]
        == "2026-10-20"
    )
    third = night(
        22, [*asan, ("AIE :: u.mlir", "FAIL", UBSAN_OUT)], {"asan-ubsan": second}
    )
    assert third["new_count"] == 0
    assert third["known"][ch.known_id(second["findings"][0]["key"])] == "2026-10-21"


def test_code_scanning_outage_keeps_the_alert_inventory():
    pages = {1: [alert(1, "high", "c-cpp", "lib/a.cpp")]}
    get = lambda path: pages.get(int(re.search(r"[?&]page=(\d+)", path).group(1)), [])
    now = datetime.datetime(2026, 10, 29, tzinfo=datetime.timezone.utc)
    first = ch.build_status(
        {}, None, ch.scanning_status(get, "r", None), None, now, {"id": 1}
    )
    down = ch.build_status({}, None, "HTTP 502", first, now, {"id": 2})
    assert down["code_scanning"] == {"error": "HTTP 502", "alert_ids": [1]}
    down = ch.build_status({}, None, "HTTP 502", down, now, {"id": 3})
    pages[1].append(alert(2, "high", "c-cpp", "lib/b.cpp"))
    back = ch.scanning_status(get, "r", down["code_scanning"])
    assert back["new"] == 1 and back["fixed"] == 0
    assert [a["number"] for a in back["alerts"] if a["new"]] == [2]


@pytest.mark.parametrize(
    "exit_code, log, written",
    [
        (0, "", True),
        (123, "lib/Clean.cpp:1:1: error: x [modernize-use-auto]", True),
        (1, "error: no compile_commands.json found", False),
        (125, "", False),
        (123, "Stack dump:\n0. Program arguments: clang-tidy", False),
    ],
)
def test_an_unfinished_clang_tidy_run_is_not_reported(
    tmp_path, exit_code, log, written
):
    files = tmp_path / "files.txt"
    files.write_text("lib/Clean.cpp\n")
    (tmp_path / "tidy.log").write_text(log)
    out = tmp_path / "tidy.json"
    argv = ["tidy", "--files", str(files), "--exit-code", str(exit_code)]
    rc = ch.main([*argv, "--out", str(out), str(tmp_path / "tidy.log")])
    assert (rc == 0) == written and out.exists() == written


TIDY_LOG = f"""{SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:10:3: error: use auto when initializing with a cast [modernize-use-auto,-warnings-as-errors]
{SRC}/lib/Dialect/AIE/IR/AIEDialect.cpp:10:3: error: use auto when initializing with a cast [modernize-use-auto,-warnings-as-errors]
{SRC}/lib/Dialect/AIE/Transforms/AIEFoo.cpp:5:1: error: variable 'x' is not initialized [cppcoreguidelines-init-variables,-warnings-as-errors]
{SRC}/lib/Dialect/AIE/Transforms/AIEFoo.cpp:9:1: error: redundant cast [readability-redundant-casting,-warnings-as-errors]
1 error generated.
Error while processing {SRC}/lib/Broken.cpp.
{SRC}/lib/Broken.cpp:1:10: error: 'gen.h.inc' file not found [clang-diagnostic-error]
    1 | #include "gen.h.inc"
      |          ^~~~~~~~~~~
/usr/include/c++/13/bits/stl.h:1:1: warning: something [bugprone-x]
Found compiler error(s).
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


def test_a_unit_broken_by_its_header_is_not_ready():
    # clang-tidy's report of a missing tablegen'd .inc: the error is in the
    # header, the translation unit is named by "Error while processing".
    log = f"""1 error generated.
Error while processing {SRC}/lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp.
{SRC}/include/aie/Dialect/AIE/IR/AIEDialect.h:30:10: error: 'aie/Dialect/AIE/IR/AIEAttrs.h.inc' file not found [clang-diagnostic-error]
   30 | #include "aie/Dialect/AIE/IR/AIEAttrs.h.inc"
      |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Found compiler error(s).
"""
    files = [
        "lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp",
        "lib/Clean.cpp",
    ]
    piece = ch.summarize_tidy(log, files, re.compile(r"lib/Enabled\.cpp$"), SRC)
    assert piece["could_not_parse"] == [
        "lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp"
    ]
    assert piece["ready"] == ["lib/Clean.cpp"]


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
    assert cs["new_by_severity"] == {"high": 1}
    # The first publish has nothing to call new.
    assert ch.scanning_status(get, "Xilinx/mlir-aie", None)["new"] == 0


def test_a_new_low_alert_is_kept_ahead_of_known_high_ones():
    known = [alert(n, "high", "c-cpp", "lib/a.cpp") for n in range(ch.KEEP_ALERTS + 49)]
    fresh = alert(ch.KEEP_ALERTS + 49, None, "python", "python/b.py")
    pages = {1: known[:100], 2: known[100:200], 3: [*known[200:], fresh]}
    get = lambda path: pages.get(int(re.search(r"[?&]page=(\d+)", path).group(1)), [])
    previous = {"alert_ids": [a["number"] for a in known]}
    cs = ch.scanning_status(get, "Xilinx/mlir-aie", previous)
    assert cs["new"] == 1 and len(cs["alerts"]) == ch.KEEP_ALERTS
    assert (cs["alerts"][0]["number"], cs["alerts"][0]["new"]) == (
        fresh["number"],
        True,
    )
    assert cs["by_severity"] == {"high": ch.KEEP_ALERTS + 49, "note": 1}


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


# A sanitizer's log_path file holds the report as it would print it.
ASAN_LOG = ASAN_OUT.split("--\n", 1)[1]


@pytest.mark.skipif(
    sys.platform == "win32", reason="the sanitizer nightly runs on Linux"
)
def test_a_report_piped_into_filecheck_is_read_from_its_log(tmp_path):
    suite, logs = tmp_path / "suite", tmp_path / "logs"
    suite.mkdir()
    logs.mkdir()
    (suite / "lit.cfg.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(ROOT / 'python')!r})\n"
        "from aie_lit_utils import SanitizerLogShTest\n"
        'config.name = "AIE"\n'
        f"config.test_format = SanitizerLogShTest({str(logs)!r})\n"
        'config.suffixes = [".test"]\n'
        'config.environment["ASAN_OPTIONS"] = "detect_leaks=0"\n',
        encoding="utf-8",
    )
    show = f'"{sys.executable}" -c "import os; print(os.environ[\'ASAN_OPTIONS\'])"'
    # Split so the outer lit doesn't read these as this file's own RUN lines.
    for name in ("seen.test", "piped.test"):
        (suite / name).write_text(f"# RUN" f": {show}\n", encoding="utf-8")
    report = tmp_path / "lit.json"
    subprocess.run(
        [sys.executable, "-c", "from lit.main import main; main()"]
        + ["-q", "-o", str(report), str(suite)],
        check=True,
        capture_output=True,
    )
    results = json.loads(report.read_text(encoding="utf-8"))
    tests = {t["name"]: t for t in results["tests"]}
    log_path = {
        name: re.search(r"detect_leaks=0:log_path=(\S+)", t["output"])[1]
        for name, t in tests.items()
    }
    assert {Path(p).parent for p in log_path.values()} == {logs}
    assert len(set(log_path.values())) == 2
    # One test printed its report, which lit kept; the other piped the same
    # bug into FileCheck, so only the sanitizer's file has it.
    tests["AIE :: seen.test"]["output"] += ASAN_OUT
    report.write_text(json.dumps(results), encoding="utf-8")
    Path(log_path["AIE :: piped.test"] + ".4242").write_text(ASAN_LOG, encoding="utf-8")
    # A log no test in the report wrote, as the runtime names it.
    (logs / "0123456789abcdef.777").write_text(UBSAN_OUT, encoding="utf-8")
    out = tmp_path / "asan.json"
    argv = ["--source-root", SRC, "sanitizer", "--name", "asan-ubsan"]
    assert ch.main([*argv, "--logs", str(logs), "--out", str(out), str(report)]) == 0
    piece = json.loads(out.read_text(encoding="utf-8"))
    assert piece["finding_count"] == 2
    asan, ubsan = piece["findings"]
    assert asan["sanitizer"] == "asan" and asan["test_count"] == 2
    assert sorted(asan["tests"]) == ["AIE :: piped.test", "AIE :: seen.test"]
    assert ubsan["sanitizer"] == "ubsan"
    assert (ubsan["test_count"], ubsan["tests"]) == (0, [])


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
    wf = (ROOT / ".github/workflows/nightlyCodeHealth.yml").read_text(encoding="utf-8")
    assert "code_health.py publish" in wf and "code-health/" in wf
    assert "group: gh-pages-publish" in wf
    page = (ROOT / "utils/kernel_checks/index.html").read_text(encoding="utf-8")
    assert "'../code-health'" in page
    listed = json.loads(
        (ROOT / "utils/kernel_checks/ci_workflows.json").read_text(encoding="utf-8")
    )
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
    assert piece["finding_count"] == 400 and len(piece["findings"]) == 400
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
    for name in ("asan-ubsan", "tsan"):
        shown = status[name]["findings"]
        assert len(shown) == ch.KEEP_FINDINGS
        assert sum(bool(f["excerpt"]) for f in shown) == ch.EXCERPTS_KEPT
    size = len(json.dumps(status, separators=(",", ":")))
    assert size < ch.STATUS_BUDGET, size
    history = None
    for n in range(800):
        history = ch.update_history(history, {**ch.history_point(status), "run": n})
    assert len(history["points"]) == ch.HISTORY_POINTS
    assert len(json.dumps(history, separators=(",", ":"))) < ch.HISTORY_BUDGET
