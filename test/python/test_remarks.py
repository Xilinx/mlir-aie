# test_remarks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""``aie.utils.compile.remarks``: the parser, the annotations and one real build.

The parser tests use a fixed sample of the record shapes the module's
docstring documents. Those shapes are Peano's to change, so the last test
compiles one library kernel with the installed Peano and checks that the
records it emits today still land where the parser expects them; it is
skipped when no Peano is installed.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from aie.iron import kernels
from aie.iron.device import NPU2Col1
from aie.utils import config
from aie.utils.compile import remarks
from aie.utils.compile.remarks import (
    StaticReport,
    compile_command,
    kernel_builds,
    parse_stderr,
    parse_yaml,
    report_rows,
    workflow_annotations,
)
from aie.utils.hostruntime import set_current_device

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


@pytest.fixture
def report(tmp_path) -> StaticReport:
    p = tmp_path / "r.yaml"
    p.write_text(SAMPLE)
    return parse_yaml(p)


@pytest.fixture(autouse=True)
def _aie2p_device():
    set_current_device(NPU2Col1())
    yield
    set_current_device(None)


def _peano_available() -> bool:
    try:
        return os.path.isfile(config.peano_cxx_path())
    except RuntimeError:
        return False


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def test_pipelined_loop_fields(report):
    # The pipeliner said `bb.1.for.body`; hardware-loops and the asm printer
    # said `for.body`. All three must land on one LoopInfo.
    assert ("foo", "bb.1.for.body") not in report.loops
    loop = report.loops[("foo", "for.body")]
    assert (loop.ii, loop.ns, loop.prologue_bundles, loop.epilogue_bundles) == (
        18,
        2,
        18,
        9,
    )
    assert loop.pipelined is True and loop.zol is True
    assert loop.pipeliner == "postpipeliner"
    assert loop.bundle_count == 18 and loop.byte_count == 288
    assert (loop.file, loop.line) == ("k.cc", 21)


def test_missed_loop_is_keyed_by_source_line(report):
    # Missed/canPipelineLoop carries no Loop name; only DebugLoc says which.
    missed = report.loops[("bar", "L40")]
    assert missed.pipelined is False
    assert missed.missed_reason == "Failed to pipeline loop"
    assert (missed.file, missed.line) == ("k.cc", 40)
    assert report.unpipelined_loops == 1


def test_schedule_analysis_message_is_reassembled(report):
    # Typed values are interleaved with the String fragments, as clang prints.
    assert report.schedule_notes == [
        "bar@L40: Minimal Initiation Interval too large: 34 > 27.Refer to -pipeliner-max-mii."
    ]


def test_counts(report):
    assert report.non_zol_loops == 0
    assert report.missing_bank_loads == 1
    assert report.pm_bytes_by_function["foo"] == 352


def test_stderr_channel():
    r = StaticReport()
    parse_stderr(
        "x.cc:3:1: warning: loop not vectorized: ... [-Wpass-failed]\n"
        "warning: No memory bank assigned to load in function 'f', block 'b' at x.cc:9:2: ...\n",
        r,
    )
    assert r.pass_failed_warnings == 1 and r.missing_bank_loads == 1


def test_stderr_does_not_double_count_bank_warning(report):
    parse_stderr(
        "warning: No memory bank assigned to load in function 'bar' ...\n", report
    )
    assert report.missing_bank_loads == 1


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


# --------------------------------------------------------------------------
# rows and annotations
# --------------------------------------------------------------------------


def test_rows_carry_ii_with_context(report):
    out = report_rows(report, "k/case", "extra")
    ii_rows = [x for x in out if x["name"].endswith("/II")]
    assert [x["name"] for x in ii_rows] == ["k/case/loop/foo/for.body/II"]
    assert ii_rows[0]["value"] == 18
    assert "via=postpipeliner" in ii_rows[0]["range"]
    assert ii_rows[0]["range"].endswith("at k.cc:21")
    # The aggregate rows always come first, in a fixed order.
    assert [x["name"].rsplit("/", 1)[1] for x in out[:5]] == [
        "unpipelined_loops",
        "non_zol_loops",
        "missing_bank_loads",
        "pass_failed_warnings",
        "pm_bytes",
    ]


def test_annotations_name_the_file_and_line_relative_to_the_checkout(tmp_path):
    src = tmp_path / "aie_kernels" / "aie2" / "k.cc"
    src.parent.mkdir(parents=True)
    src.write_text("")
    r = StaticReport()
    parse_stderr(f"{src}:3:1: warning: loop not unrolled: x [-Wpass-failed]\n", r)
    (line,) = workflow_annotations("scale (aie2)", r, "ok", str(tmp_path))
    assert line == (
        "::warning file=aie_kernels/aie2/k.cc,line=3,title=scale (aie2)%3A "
        "pragma dropped by the compiler::loop not unrolled: x [-Wpass-failed]"
    )


def test_annotations_drop_a_file_outside_the_checkout_and_escape_the_message(tmp_path):
    r = StaticReport()
    parse_stderr("/site/aie_api/x.hpp:9:2: warning: 100% [-Wpass-failed]\n", r)
    (line,) = workflow_annotations("k", r, "ok", str(tmp_path))
    assert line.startswith("::warning title=k%3A pragma dropped by the compiler::")
    assert line.endswith("::100%25 [-Wpass-failed]")


def test_compile_failure_becomes_one_error_annotation():
    detail = "compile failed: junk\nk.cc:4:5: error: unknown type name 'v16acc'\n"
    (line,) = workflow_annotations("k", None, detail)
    assert line == (
        "::error title=k%3A kernel failed to compile::"
        "k.cc:4:5: error: unknown type name 'v16acc'"
    )


# --------------------------------------------------------------------------
# which kernels, from which sources
# --------------------------------------------------------------------------


def test_kernel_builds_cover_every_factory_once():
    builds = dict(kernel_builds())
    names = list(builds)
    assert len(names) == len(set(names))
    assert {n.split("/")[0] for n in names} >= {"scale", "mm", "gelu", "passthrough"}
    # The default build is listed under the bare name; a dtypes entry that is
    # the default is not listed twice, the others carry their kwargs.
    assert "scale" in builds and "scale/dtype=int16" in builds
    assert "scale/dtype=bf16" not in builds
    assert all(not ef.use_chess for ef in builds.values())


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_compile_command_uses_the_kernels_own_directory(tmp_path):
    ef = kernels.scale()
    cmd, yaml_out = compile_command(ef, "aie2p", tmp_path)
    assert cmd[0].endswith(os.path.basename(config.peano_cxx_path()))
    assert str(Path(ef.source_file)) in cmd
    assert f"-foptimization-record-file={yaml_out}" in cmd
    assert any(flag.startswith("-Rpass=") for flag in cmd)


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_inline_source_kernels_are_written_out_first(tmp_path):
    set_current_device(None)
    from aie.iron.device import NPU1Col1

    set_current_device(NPU1Col1())
    ef = kernels.gelu()  # the aie2 LUT activations include their sources inline
    assert ef.source_file is None
    cmd, _ = compile_command(ef, "aie2", tmp_path)
    written = tmp_path / f"{ef.name}.cc"
    assert str(written) in cmd and "lut_based_ops.cpp" in written.read_text()


def test_kernel_sources_follow_the_environment_override(tmp_path, monkeypatch):
    # A pull request compiles the checkout's kernels against the installed
    # wheel: MLIR_AIE_KERNEL_SOURCES names the checkout.
    src = tmp_path / "aie_kernels" / "aie2p" / "scale.cc"
    src.parent.mkdir(parents=True)
    src.write_text("// stand-in\n")
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(tmp_path))
    assert config.aie_kernels_dir() == str(tmp_path / "aie_kernels")
    assert config.aie_runtime_lib_dir() == str(tmp_path / "aie_runtime_lib")
    from aie.iron.kernels._common import _kernel_source

    assert _kernel_source("aie2p", "aie2p", "scale.cc") == src


def test_chess_kernels_are_refused(tmp_path):
    import types

    ef = types.SimpleNamespace(name="k", use_chess=True)
    with pytest.raises(ValueError, match="xchesscc"):
        compile_command(ef, "aie2p", tmp_path)


# --------------------------------------------------------------------------
# the real compiler
# --------------------------------------------------------------------------


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_real_build_produces_the_documented_record_shapes(tmp_path):
    # scale.cc is one vectorised loop over a tile: Peano must report it as a
    # pipelined (or at least seen) loop, with a bundle count from the asm
    # printer. If a Peano bump changes the record shapes, this is the test
    # that goes red.
    rep, detail = remarks.analyze(kernels.scale(), "aie2p", tmp_path)
    assert rep is not None, detail
    assert rep.loops, "no loop records parsed from a kernel that has a loop"
    assert rep.pm_bytes > 0
    assert any(loop.bundle_count for loop in rep.loops.values())
    assert any(loop.pipelined is not None for loop in rep.loops.values())
    assert rep.pass_failed == []
