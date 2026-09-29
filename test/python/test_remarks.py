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

import json
import os
import shutil
import textwrap
from pathlib import Path

import numpy as np
import pytest
from aie.iron import ExternalFunction, kernels
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
def _aie2p_device(npu2_device):
    yield


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
    assert [x["name"].rsplit("/", 1)[1] for x in out[:6]] == [
        "unpipelined_loops",
        "non_zol_loops",
        "missing_bank_loads",
        "pass_failed_warnings",
        "pm_bytes",
        "libcalls",
    ]


# --------------------------------------------------------------------------
# the object: what the entry reaches
# --------------------------------------------------------------------------


def _section(index, name, alloc=True, info=0):
    flags = [{"Name": "SHF_ALLOC"}, {"Name": "SHF_EXECINSTR"}] if alloc else []
    return {
        "Section": {
            "Index": index,
            "Name": {"Name": name},
            "Flags": {"Flags": flags},
            "Info": info,
        }
    }


def _symbol(name, section, kind="Function"):
    return {
        "Symbol": {
            "Name": {"Name": name},
            "Type": {"Name": kind},
            "Section": {
                "Name": "Undefined" if section == 0 else "x",
                "Value": section,
            },
        }
    }


def _relocs(section, *symbols):
    return {
        "SectionIndex": section,
        "Relocs": [{"Relocation": {"Symbol": {"Value": s}}} for s in symbols],
    }


# llvm-readobj's JSON for: k calls helper and __divsf3; unused, emitted
# standalone, calls __mulsf3; .debug_info refers to all of them.
READOBJ = {
    "Sections": [
        _section(0, "", alloc=False),
        _section(1, ".text.k"),
        _section(2, ".text.helper"),
        _section(3, ".text.unused"),
        _section(4, ".rela.text.k", alloc=False, info=1),
        _section(5, ".rela.text.unused", alloc=False, info=3),
        _section(6, ".debug_info", alloc=False),
        _section(7, ".rela.debug_info", alloc=False, info=6),
    ],
    "Symbols": [
        _symbol("", 0, "None"),
        _symbol("k", 1),
        _symbol("helper", 2),
        _symbol("unused", 3),
        _symbol("__divsf3", 0, "None"),
        _symbol("__mulsf3", 0, "None"),
    ],
    "Relocations": [_relocs(4, 2, 4), _relocs(5, 5), _relocs(7, 1, 2, 3)],
}


def test_the_entry_reaches_its_callees_and_their_runtime_calls():
    reached = remarks.parse_readobj(READOBJ, "k")
    assert reached.functions == {"k", "helper"}
    assert reached.undefined == ["__divsf3"]


def _frames(**sizes):
    return [{"Entry": {"Functions": [n], "Size": b}} for n, b in sizes.items()]


def test_the_stack_is_the_deepest_path_from_the_entry():
    # k (32) calls helper (64); unused's frame is never on the path.
    doc = dict(READOBJ, StackSizes=_frames(k=32, helper=64, unused=512))
    assert remarks.parse_readobj(doc, "k").stack == 96


def test_recursion_leaves_the_stack_unbounded():
    # helper calls back into k.
    doc = dict(
        READOBJ,
        Sections=READOBJ["Sections"] + [_section(8, ".rela.text.helper", False, 2)],
        Relocations=READOBJ["Relocations"] + [_relocs(8, 1)],
        StackSizes=_frames(k=32, helper=64),
    )
    assert remarks.parse_readobj(doc, "k").stack is None


def test_a_stack_row_notes_the_runtime_calls_it_leaves_out(report):
    report.stack_bytes = 96
    report.libcalls = ["__divsf3"]
    (row,) = [r for r in report_rows(report, "k", "") if r["name"] == "k/stack_bytes"]
    assert row["value"] == 96 and "runtime" in row["range"]


def test_only_shipped_functions_count(report):
    # bar's missed loop and bytes belong to a function the entry never calls.
    report.pm_bytes_by_function["bar"] = 1000
    report.shipped = {"foo"}
    assert report.pm_bytes == 352
    assert report.unpipelined_loops == 0
    report.libcalls = ["__divsf3"]
    (row,) = [r for r in report_rows(report, "k", "") if r["name"] == "k/libcalls"]
    assert (row["value"], row["range"]) == (1, "__divsf3")


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
# trace markers, on the -O2 IR shapes Peano emits
# --------------------------------------------------------------------------

E0 = "  tail call void @llvm.aie2p.event(i32 0)"
E1 = "  tail call void @llvm.aie2p.event(i32 1)"


def _fn(name, *blocks):
    """``define`` ``name`` from ``(label, lines)`` blocks; the first is the entry."""
    out = [f"define dso_local void @{name}(ptr noalias %a) local_unnamed_addr #0 {{"]
    for i, (label, lines) in enumerate(blocks):
        if i:
            out += ["", f"{label}:                                 ; preds = %x"]
        out += list(lines)
    return "\n".join(out + ["}", ""])


def _loop(before=(), body=(), after=()):
    return (
        ("entry", [*before, "  br label %for.body"]),
        ("for.body", [*body, "  br i1 %c, label %for.end, label %for.body"]),
        ("for.end", [*after, "  ret void"]),
    )


def test_markers_bracketing_the_whole_call():
    assert remarks.trace_markers(_fn("k", *_loop([E0], [], [E1])), "k") == "whole_call"


def test_no_markers_is_none():
    assert remarks.trace_markers(_fn("k", *_loop()), "k") == "none"


def test_markers_only_in_a_sibling_do_not_count():
    ir = _fn("zero", ("entry", [E0, E1, "  ret void"])) + _fn("k", *_loop())
    assert remarks.trace_markers(ir, "k") == "none"


def test_markers_inside_a_loop_are_named():
    shape = remarks.trace_markers(_fn("k", *_loop(body=[E0, E1])), "k")
    assert "inside a loop (block for.body)" in shape


def test_an_early_return_past_the_markers_is_named():
    ir = _fn(
        "k",
        ("entry", ["  br i1 %masked, label %done, label %work"]),
        ("work", [E0, E1, "  br label %done"]),
        ("done", ["  ret void"]),
    )
    assert "sequences '', '01'" in remarks.trace_markers(ir, "k")


def test_a_helper_called_once_brackets_the_call():
    ir = _fn("helper", ("entry", [E0, E1, "  ret void"])) + _fn(
        "k", ("entry", ["  tail call void @helper(ptr %a)", "  ret void"])
    )
    assert remarks.trace_markers(ir, "k") == "whole_call"


def test_a_helper_called_in_a_loop_is_inside_that_loop():
    # mm_fused_mmul.h's shape: the pair wraps an inner band, not the call.
    ir = _fn("band", ("entry", [E0, E1, "  ret void"])) + _fn(
        "k", *_loop(body=["  call void @band(ptr %a)"])
    )
    assert "inside a loop" in remarks.trace_markers(ir, "k")


def test_two_pairs_per_call_are_not_whole_call():
    ir = _fn("k", ("entry", [E0, E1, E0, E1, "  ret void"]))
    assert "'0101'" in remarks.trace_markers(ir, "k")


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
def test_lut_kernel_command_uses_native_source(tmp_path):
    set_current_device(None)
    from aie.iron.device import NPU1Col1

    set_current_device(NPU1Col1())
    ef = kernels.gelu()
    assert ef.source_string is None
    assert Path(ef.source_file).name == "lut_kernel.cc"
    cmd, _ = compile_command(ef, "aie2", tmp_path)
    assert ef.source_file in cmd
    assert any(arg.startswith("-DAIE_LUT_KERNEL_SOURCE=") for arg in cmd)
    assert not (tmp_path / f"{ef.name}.cc").exists()


def test_kernel_sources_follow_the_environment_override(tmp_path, monkeypatch):
    # A pull request compiles the checkout's kernels against the installed
    # wheel: MLIR_AIE_KERNEL_SOURCES names the checkout.
    src = tmp_path / "aie_kernels" / "eltwise" / "scale.cc"
    src.parent.mkdir(parents=True)
    src.write_text("// stand-in\n")
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(tmp_path))
    assert config.aie_kernels_dir() == str(tmp_path / "aie_kernels")
    assert config.aie_runtime_lib_dir() == str(tmp_path / "aie_runtime_lib")
    from aie.iron.kernels._common import _kernel_source

    assert _kernel_source("eltwise/scale.cc") == src


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
    # scale.cc is one vectorized loop over a tile: Peano must report it as a
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
    assert rep.libcalls == [] and rep.shipped


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_real_object_names_its_runtime_calls_and_drops_what_it_never_calls(
    tmp_path,
):
    # On AIE2P scalar int-to-float is a runtime routine; ``unused`` is emitted
    # standalone but the core link would drop it, with its divide.
    ef = ExternalFunction(
        "k",
        source_string=textwrap.dedent("""\
            extern "C" float unused(float x) { return x / 3.0f; }
            extern "C" void k(float *a, int n) { a[0] = (float)n; }
            """),
        arg_types=[np.ndarray[(16,), np.dtype[np.float32]], np.int32],
    )
    rep, detail = remarks.analyze(ef, "aie2p", tmp_path)
    assert rep is not None, detail
    assert "__floatsisf" in rep.libcalls and "__divsf3" not in rep.libcalls
    assert rep.shipped == {"k"}
    assert 0 < rep.pm_bytes < sum(rep.pm_bytes_by_function.values())


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_real_object_reports_the_stack_its_entry_needs(tmp_path):
    # The volatile buffer must live in helper's frame, below k's.
    ef = ExternalFunction(
        "k",
        source_string=textwrap.dedent(
            """            __attribute__((noinline)) static int helper(int n) {
              volatile int buf[64];
              for (int i = 0; i < 64; i++) buf[i] = n + i;
              return buf[n & 63];
            }
            extern "C" void k(int *a, int n) { a[0] = helper(n); }
            """
        ),
        arg_types=[np.ndarray[(16,), np.dtype[np.int32]], np.int32],
    )
    rep, detail = remarks.analyze(ef, "aie2p", tmp_path)
    assert rep is not None, detail
    assert rep.stack_bytes >= 64 * 4


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_baseline_tree_prints_the_rows_that_differ(tmp_path, capsys):
    # The baseline's relu divides once per call; the current tree's does not.
    base = tmp_path / "base"
    shutil.copytree(config.aie_kernels_dir(), base / "aie_kernels")
    shutil.copytree(config.aie_runtime_lib_dir(), base / "aie_runtime_lib")
    relu = base / "aie_kernels" / "eltwise" / "relu_aie2p.h"
    relu.write_text(
        relu.read_text().replace(
            "  event1();",
            "  c[0] = (bfloat16)((float)a[0] / (float)a[1]);\n  event1();",
        )
    )
    meta = tmp_path / "meta.json"
    code = remarks.main(
        [
            "--target=aie2p",
            "--only=^relu$",
            f"--out={tmp_path / 'rows.json'}",
            f"--meta={meta}",
            f"--baseline-sources={base}",
        ]
    )
    assert code == 0
    changed = json.loads(meta.read_text())["baseline"]["changed"]
    before, after = changed["relu/libcalls"]
    assert before > 0 and after == 0
    assert "relu/libcalls:" in capsys.readouterr().out
