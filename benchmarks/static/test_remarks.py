# benchmarks/static/test_remarks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Host-only tests of the remark parser.

The sample below mirrors records captured from a real llvm-aie
22.0.0.2026090201 build of aie_kernels (STATIC_CHECKS.md documents the
probe): the pass is ``pipeliner``, its give-up record is
``Missed/canPipelineLoop`` located only by ``DebugLoc``, and its schedule
diagnostics arrive as ``Analysis/schedule`` with the message split around
typed values.
"""

import textwrap

from .remarks import StaticReport, parse_stderr, parse_yaml

SAMPLE = textwrap.dedent("""\
    --- !Passed
    Pass:            pipeliner
    Name:            schedule
    DebugLoc:        { File: k.cc, Line: 21, Column: 1 }
    Function:        foo
    Args:
      - String:          Schedule found
      - Pipeliner:       postpipeliner
      - II:              '18'
      - NS:              '2'
      - Loop:            bb.1.for.body
      - Prologue:        entry
      - PrologueBundles: '18'
      - Epilogue:        ''
      - EpilogueBundles: '9'
    ...
    --- !Analysis
    Pass:            pipeliner
    Name:            schedule
    DebugLoc:        { File: k.cc, Line: 40, Column: 3 }
    Function:        bar
    Args:
      - String:          'Minimal Initiation Interval too large: '
      - MII:             '34'
      - String:          ' > '
      - SwpMaxMii:       '27'
      - String:          .
      - String:          Refer to -pipeliner-max-mii.
    ...
    --- !Missed
    Pass:            pipeliner
    Name:            canPipelineLoop
    DebugLoc:        { File: k.cc, Line: 40, Column: 3 }
    Function:        bar
    Args:
      - String:          Failed to pipeline loop
    ...
    --- !Analysis
    Pass:            aie-hardware-loops
    Name:            analysis
    Function:        foo
    Args:
      - LoopID:          '0'
      - BasicBlock:      for.body
      - Zero-Overhead-Loop: 'true'
    ...
    --- !Analysis
    Pass:            aie-asm-printer
    Name:            analysis
    Function:        foo
    Args:
      - BasicBlock:      for.body
      - BundleCount:     '18'
      - ByteCount:       '288'
    ...
    --- !Analysis
    Pass:            aie-asm-printer
    Name:            analysis
    Function:        foo
    Args:
      - BasicBlock:      entry
      - BundleCount:     '4'
      - ByteCount:       '64'
    ...
    --- !Missed
    Pass:            aie-multi-slot-pseudo
    Name:            missing-memory-bank
    Function:        bar
    Args:
      - Instruction:     'VLDA.UPS.S32.S16 ...'
    ...
""")


def _report(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(SAMPLE)
    return parse_yaml(p)


def test_pipelined_loop_fields(tmp_path):
    r = _report(tmp_path)
    # The pipeliner said `bb.1.for.body`; hardware-loops and the asm printer
    # said `for.body`. All three must land on one LoopInfo.
    assert ("foo", "bb.1.for.body") not in r.loops
    loop = r.loops[("foo", "for.body")]
    assert (loop.ii, loop.ns, loop.prologue_bundles, loop.epilogue_bundles) == (
        18,
        2,
        18,
        9,
    )
    assert loop.pipelined is True and loop.zol is True
    assert loop.pipeliner == "postpipeliner"
    assert loop.bundle_count == 18 and loop.byte_count == 288


def test_missed_loop_is_keyed_by_source_line(tmp_path):
    # Missed/canPipelineLoop carries no Loop name; only DebugLoc says which.
    r = _report(tmp_path)
    missed = r.loops[("bar", "L40")]
    assert missed.pipelined is False
    assert missed.missed_reason == "Failed to pipeline loop"
    assert r.unpipelined_loops == 1


def test_schedule_analysis_message_is_reassembled(tmp_path):
    r = _report(tmp_path)
    # Typed values are interleaved with the String fragments, as clang prints.
    assert r.schedule_notes == [
        "bar@L40: Minimal Initiation Interval too large: 34 > 27.Refer to -pipeliner-max-mii."
    ]


def test_counts(tmp_path):
    r = _report(tmp_path)
    assert r.non_zol_loops == 0
    assert r.missing_bank_loads == 1
    assert r.pm_bytes_by_function["foo"] == 352


def test_rows_carry_ii_with_context(tmp_path):
    from .remarks import rows

    out = rows(_report(tmp_path), "k/case", "extra")
    ii_rows = [x for x in out if x["name"].endswith("/II")]
    assert [x["name"] for x in ii_rows] == ["k/case/loop/foo/for.body/II"]
    assert ii_rows[0]["value"] == 18
    assert "via=postpipeliner" in ii_rows[0]["range"]
    # The five aggregate rows always come first, in a fixed order.
    assert [x["name"].rsplit("/", 1)[1] for x in out[:5]] == [
        "unpipelined_loops",
        "non_zol_loops",
        "missing_bank_loads",
        "pass_failed_warnings",
        "pm_bytes",
    ]


def test_stderr_channel():
    r = StaticReport()
    parse_stderr(
        "x.cc:3:1: warning: loop not vectorized: ... [-Wpass-failed]\n"
        "warning: No memory bank assigned to load in function 'f', block 'b' at x.cc:9:2: ...\n",
        r,
    )
    assert r.pass_failed_warnings == 1 and r.missing_bank_loads == 1


def test_stderr_does_not_double_count_bank_warning(tmp_path):
    r = _report(tmp_path)
    parse_stderr("warning: No memory bank assigned to load in function 'bar' ...\n", r)
    assert r.missing_bank_loads == 1


def test_loop_source_location_reaches_the_ii_row(tmp_path):
    from .remarks import rows

    r = _report(tmp_path)
    assert (r.loops[("foo", "for.body")].file, r.loops[("foo", "for.body")].line) == (
        "k.cc",
        21,
    )
    assert (r.loops[("bar", "L40")].file, r.loops[("bar", "L40")].line) == ("k.cc", 40)
    ii = [x for x in rows(r, "k", "e") if x["name"].endswith("/II")][0]
    assert ii["range"].endswith("at k.cc:21")


def test_dropped_pragmas_keep_their_text():
    r = StaticReport()
    parse_stderr(
        "/w/aie_kernels/aie2/k.cc:3:1: warning: loop not unrolled: the optimizer "
        "was unable to perform the requested transformation [-Wpass-failed=transform-warning]\n"
        "note: something else\n",
        r,
    )
    assert r.pass_failed_warnings == 1
    assert r.pass_failed[0].startswith("/w/aie_kernels/aie2/k.cc:3:1: warning:")


def test_annotations_name_the_file_and_line_relative_to_the_checkout(tmp_path):
    from .remarks import annotations

    root = tmp_path
    src = root / "aie_kernels" / "aie2" / "k.cc"
    src.parent.mkdir(parents=True)
    src.write_text("")
    r = StaticReport()
    parse_stderr(f"{src}:3:1: warning: loop not unrolled: x [-Wpass-failed]\n", r)
    (line,) = annotations("scale/1024x16/int16", r, "ok", str(root))
    assert line == (
        "::warning file=aie_kernels/aie2/k.cc,line=3,title=scale/1024x16/int16%3A "
        "pragma dropped by the compiler::loop not unrolled: x [-Wpass-failed]"
    )


def test_annotations_drop_a_file_outside_the_checkout_and_escape_the_message(
    tmp_path,
):
    from .remarks import annotations

    r = StaticReport()
    parse_stderr("/site/aie_api/x.hpp:9:2: warning: 100% [-Wpass-failed]\n", r)
    (line,) = annotations("k", r, "ok", str(tmp_path))
    assert line.startswith("::warning title=k%3A pragma dropped by the compiler::")
    assert line.endswith("::100%25 [-Wpass-failed]")


def test_compile_failure_becomes_one_error_annotation():
    from .remarks import annotations

    detail = "compile failed: junk\nk.cc:4:5: error: unknown type name 'v16acc'\n"
    (line,) = annotations("k", None, detail)
    assert line == (
        "::error title=k%3A kernel failed to compile::"
        "k.cc:4:5: error: unknown type name 'v16acc'"
    )
