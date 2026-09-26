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

import collections
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
    assert "trips=None" in ii_rows[0]["range"]
    assert not [x for x in out if x["name"].endswith("/II_x_trips")]
    # The aggregate rows always come first, in a fixed order.
    assert [x["name"].rsplit("/", 1)[1] for x in out[:6]] == [
        "unpipelined_loops",
        "non_zol_loops",
        "missing_bank_loads",
        "pass_failed_warnings",
        "pm_bytes",
        "libcalls",
    ]


def test_a_loop_with_known_trips_gets_ii_times_trips(report):
    report.loops["foo", "for.body"].trips = 32
    out = {x["name"]: x for x in report_rows(report, "k/case", "extra")}
    assert out["k/case/loop/foo/for.body/II_x_trips"]["value"] == 18 * 32
    assert "trips=32 " in out["k/case/loop/foo/for.body/II"]["range"]


# What Peano's opt -passes=print<scalar-evolution> prints, abridged.
SCEV = textwrap.dedent("""\
    Classifying expressions for: @transpose_8x8
      %i = phi i32 [ 0, %entry ], [ %inc, %for.body.i ]
    Determining loop execution counts for: @transpose_8x8
    Loop %for.body.i: backedge-taken count is i32 31
    Loop %for.body.i: constant max backedge-taken count is i32 31
    Loop %for.body.i: symbolic max backedge-taken count is i32 31
    Loop %for.body.i: Trip multiple is 32
    Determining loop execution counts for: @"eltwise_add_bf16_vector"
    Loop %for.body: backedge-taken count is (-1 + (%vector_size /u 32))<nsw>
    Loop %for.body: constant max backedge-taken count is i32 134217726
    Loop %"for.body20.i": backedge-taken count is i32 3
    Loop %while.body: Unpredictable backedge-taken count.
    """)


def test_trip_counts_are_the_constant_backedge_counts_plus_one():
    assert remarks.parse_trip_counts(SCEV) == {
        ("transpose_8x8", "for.body.i"): 32,
        ("eltwise_add_bf16_vector", "for.body20.i"): 4,
    }


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


def test_the_stack_row_names_the_kernel_frame_not_the_core(report):
    # dwconv1d_channels_last on AIE2P: 64 bytes here, 256 for its core,
    # whose main holds a frame of its own.
    report.kernel_stack_bytes = 64
    rows = {r["name"]: r for r in report_rows(report, "k", "")}
    assert "k/stack_bytes" not in rows
    row = rows["k/kernel_stack_bytes"]
    assert row["value"] == 64 and "main's" in row["range"]
    assert "runtime" not in row["range"]


def test_a_stack_row_notes_the_runtime_calls_it_leaves_out(report):
    report.kernel_stack_bytes = 96
    report.libcalls = ["__divsf3"]
    (row,) = [
        r for r in report_rows(report, "k", "") if r["name"] == "k/kernel_stack_bytes"
    ]
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


def test_a_compile_failure_keeps_its_first_error_above_the_tail():
    notes = "".join(f"x.h:{i}:1: note: candidate {i}\n" for i in range(200))
    stderr = f"k.cc:4:5: error: no matching function\n{notes}1 error generated.\n"
    detail = remarks.compile_failure(stderr, tail=500)
    assert detail.startswith("k.cc:4:5: error: no matching function\n...\n")
    assert detail.endswith("1 error generated.")
    assert remarks.compile_failure("k.cc:1:1: error: short") == "k.cc:1:1: error: short"


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


KERNEL_CASES = Path(__file__).parent / "npu" / "kernel_cases.py"


def test_case_builds_name_the_build_each_case_runs(tmp_path):
    cases = tmp_path / "some_cases.py"
    cases.write_text(textwrap.dedent("""\
        from dataclasses import dataclass, field
        from aie.iron import kernels

        @dataclass
        class Case:
            name: str
            kwargs: dict = field(default_factory=dict)
            devices: tuple = ()

            def fn(self):
                return kernels.tanh(**self.kwargs)

        CASES = [
            Case("tanh/a"),
            Case("tanh/b"),
            Case("tanh/lut", dict(use_lut=True), devices=("npu2",)),
            Case("tanh/lut-npu1", dict(use_lut=True), devices=("npu1",)),
        ]
        """))
    coverage = collections.Counter()
    builds = dict(remarks.case_builds(str(cases), "npu2", coverage=coverage))
    # a and b build one object; the npu1-only case is not this device's.
    assert list(builds) == ["tanh/a", "tanh/lut"]
    assert builds["tanh/lut"].object_file_name != builds["tanh/a"].object_file_name
    assert remarks.case_coverage(coverage) == (
        "cases: 2 builds for 3 cases (1 share an earlier case's build); "
        "skipped 1 by devices, 0 other-architecture only"
    )
    coverage.clear()
    assert list(dict(remarks.case_builds(str(cases), "npu2", "b$", coverage))) == [
        "tanh/b"
    ]
    assert remarks.case_coverage(coverage).endswith(", 3 not matching --only")


def test_case_builds_reach_the_library_cases_the_default_sweep_misses():
    defaults = {ef.object_file_name for _, ef in kernel_builds()}
    builds = dict(remarks.case_builds(str(KERNEL_CASES), "npu2", "^tanh/"))
    lut = builds["tanh/1024x16/bfloat16/use_lut=True/lut"]
    assert lut.object_file_name not in defaults


def test_a_build_spec_parses_each_value_as_its_parameters_type():
    assert remarks.parse_build(
        "cascade_mm:dim_m=12,input_dtype=int32,output_dtype=bfloat16,use_chess=False"
    ) == (
        "cascade_mm",
        dict(
            dim_m=12,
            input_dtype=np.int32,
            output_dtype=np.dtype("bfloat16").type,
            use_chess=False,
        ),
    )
    factory, kwargs = remarks.parse_build("set_rounding:mode=conv_even")
    [(name, _)] = remarks.spec_builds([(factory, kwargs)])
    assert name == "set_rounding/mode=conv_even"
    assert remarks.parse_build("exp2f_vec:min_x=-3.5")[1] == {"min_x": -3.5}
    assert remarks.parse_build("rms_norm:cols=64")[1] == {"cols": 64}


@pytest.mark.parametrize(
    "spec,error",
    [
        ("nope:x=1", "not a kernel factory"),
        ("cascade_mm:dim_q=1", "dim_m, dim_k, dim_n"),
        ("cascade_mm:dim_m", "not KEY=VALUE"),
        ("cascade_mm:dim_m=x", "invalid literal"),
        ("cascade_mm:use_chess=yes", "True or False"),
        ("cascade_mm:input_dtype=int77", "not understood"),
        ("fused_mm:clamp=1", "not settable"),
    ],
)
def test_a_bad_build_spec_names_what_is_wrong(spec, error):
    with pytest.raises(ValueError, match=error):
        remarks.parse_build(spec)


def test_a_build_its_factory_refuses_exits_with_the_factorys_reason(tmp_path, capsys):
    code = remarks.main(
        [f"--out={tmp_path / 'rows.json'}", "--build=scale:dtype=bfloat16"]
    )
    assert code == 2
    assert "--build scale/dtype=bfloat16: scale() dtype" in capsys.readouterr().err


def test_build_and_cases_are_exclusive(tmp_path):
    with pytest.raises(SystemExit):
        remarks.main(
            [
                f"--out={tmp_path / 'rows.json'}",
                "--build=cascade_mm",
                f"--cases={KERNEL_CASES}",
            ]
        )


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_build_on_the_command_line_reaches_a_shapes_own_code_path(tmp_path):
    # M not a multiple of 8 takes cascade_mm's scalar fallback, which no
    # default build compiles.
    meta = tmp_path / "meta.json"
    code = remarks.main(
        [
            "--target=aie2p",
            f"--out={tmp_path / 'rows.json'}",
            f"--meta={meta}",
            "--build=cascade_mm",
            "--build=cascade_mm:dim_m=12,dim_k=16,dim_n=16",
        ]
    )
    assert code == 0
    written = json.loads(meta.read_text())["kernels"]
    default, fallback = (
        written["cascade_mm"],
        written["cascade_mm/dim_k=16/dim_m=12/dim_n=16"],
    )
    assert len(fallback["loops"]) < len(default["loops"])
    assert fallback["pm_bytes"] != default["pm_bytes"]


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
def test_a_real_build_knows_its_trip_counts(tmp_path):
    # The llama-decode transpose bakes DIM_m=256, DIM_n=32; the AIE2P 8x8
    # kernel does a 32-column block per iteration, (32 / 8) * (256 / 32).
    name = "transpose/8192x16/bfloat16/subtile=8/llama-decode"
    ef = dict(remarks.case_builds(str(KERNEL_CASES), "npu2", f"^{name}$"))[name]
    rep, detail = remarks.analyze(ef, "aie2p", tmp_path)
    assert rep is not None, detail
    loop = rep.loops["transpose_8x8", "for.body.i"]
    assert loop.trips == 32 and loop.ii
    rows = {x["name"]: x["value"] for x in report_rows(rep, "t", "")}
    assert rows["t/loop/transpose_8x8/for.body.i/II_x_trips"] == loop.ii * 32


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
def test_a_real_template_error_is_reported_by_its_error_line(tmp_path):
    # Its candidate notes run past the 2000 characters of tail kept.
    ef = ExternalFunction(
        "k",
        source_string=textwrap.dedent("""\
            #include <aie_api/aie.hpp>
            extern "C" void k(int *a) {
              aie::vector<bfloat16, 32> v;
              auto r = aie::mul(v, aie::vector<int32, 7>());
            }
            """),
        arg_types=[np.ndarray[(16,), np.dtype[np.int32]]],
        include_dirs=[config.cxx_header_path()],
    )
    rep, detail = remarks.analyze(ef, "aie2p", tmp_path)
    assert rep is None
    assert "k.cc:4:12: error: no matching function for call to 'mul'" in detail


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
    assert rep.kernel_stack_bytes >= 64 * 4


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


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_failed_build_keeps_the_rows_of_the_others(tmp_path, monkeypatch, capsys):
    trees = {}
    for tree in ("base", "broken"):
        trees[tree] = tmp_path / tree
        shutil.copytree(config.aie_kernels_dir(), trees[tree] / "aie_kernels")
        shutil.copytree(config.aie_runtime_lib_dir(), trees[tree] / "aie_runtime_lib")
    for tree, source in (("broken", "relu_aie2p.h"), ("base", "scale.cc")):
        path = trees[tree] / "aie_kernels" / "eltwise" / source
        path.write_text("#error broken\n" + path.read_text())
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(trees["broken"]))
    out, meta = tmp_path / "rows.json", tmp_path / "meta.json"
    code = remarks.main(
        [
            "--target=aie2p",
            "--only=^(relu|scale)$",
            f"--out={out}",
            f"--meta={meta}",
            f"--baseline-sources={trees['base']}",
        ]
    )
    # Neither failure is a change: this tree's relu is not compiled again,
    # and the baseline's scale leaves this tree's rows out of the diff.
    assert code == 3
    printed = capsys.readouterr()
    assert "RESULTS INVALID: relu: compile failed" in printed.err
    assert "baseline fails to compile scale, not compared" in printed.out
    names = {r["name"].split("/")[0] for r in json.loads(out.read_text())}
    assert names == {"scale"}
    written = json.loads(meta.read_text())
    assert list(written["failed"]) == ["relu"]
    assert list(written["baseline"]["failed"]) == ["scale"]
    assert written["baseline"]["changed"] == {}


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_sources_names_the_tree_compiled(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    shutil.copytree(config.aie_kernels_dir(), tree / "aie_kernels")
    shutil.copytree(config.aie_runtime_lib_dir(), tree / "aie_runtime_lib")
    monkeypatch.delenv("MLIR_AIE_KERNEL_SOURCES", raising=False)
    meta = tmp_path / "meta.json"
    code = remarks.main(
        [
            "--target=aie2p",
            "--only=^relu$",
            f"--out={tmp_path / 'rows.json'}",
            f"--meta={meta}",
            f"--sources={tree}",
        ]
    )
    assert code == 0
    source = json.loads(meta.read_text())["kernels"]["relu"]["source"]
    assert Path(source).is_relative_to(tree / "aie_kernels")
    assert "MLIR_AIE_KERNEL_SOURCES" not in os.environ


def test_an_unset_kernel_tree_warns_that_it_is_the_installed_copy(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("MLIR_AIE_KERNEL_SOURCES", raising=False)
    current, warning = remarks.current_kernel_sources(str(tmp_path))
    assert current == config.aie_kernels_dir()
    assert "MLIR_AIE_KERNEL_SOURCES is unset" in warning and current in warning


def test_a_kernel_tree_that_is_the_baseline_warns(tmp_path, monkeypatch):
    monkeypatch.setenv("MLIR_AIE_KERNEL_SOURCES", str(tmp_path))
    current, warning = remarks.current_kernel_sources(f"{tmp_path}/")
    assert current == str(tmp_path / "aie_kernels")
    assert "both" in warning
    other = tmp_path / "other"
    assert remarks.current_kernel_sources(str(other)) == (current, None)


def test_a_renamed_loop_with_the_same_rows_is_not_a_change():
    loop = "mm/loop/matmul_bf16"
    base = {
        "mm/pm_bytes": 900,
        f"{loop}/for.body20.i/II": 8,
        f"{loop}/for.body20.i/not_zol": 0,
        f"{loop}/for.body31.i/II": 4,
        f"{loop}/for.body40.i/II": 2,
        f"{loop}/for.body50.i/II": 6,
        "mv/loop/matvec/for.body.i/II": 3,
    }
    current = {
        "mm/pm_bytes": 912,
        # Renamed, rows the same: dropped.
        f"{loop}/for.body23.i/II": 8,
        f"{loop}/for.body23.i/not_zol": 0,
        # body40 took body31's name as body31 moved to body34: both match.
        f"{loop}/for.body31.i/II": 2,
        f"{loop}/for.body34.i/II": 4,
        # Renamed and changed: kept, under both names.
        f"{loop}/for.body52.i/II": 5,
        # Another function's loop with the same II is no match.
        "mv/loop/matvec/for.body.i/II": 3,
        "mv/loop/other/for.body.i/II": 6,
    }
    diff = remarks.diff_rows(base, current)
    assert diff["rows"] == {
        "mm/pm_bytes": [900, 912],
        f"{loop}/for.body50.i/II": [6, None],
        f"{loop}/for.body52.i/II": [None, 5],
        "mv/loop/other/for.body.i/II": [None, 6],
    }
    assert sorted(diff["renamed"]) == [
        [f"{loop}/for.body20.i", f"{loop}/for.body23.i"],
        [f"{loop}/for.body31.i", f"{loop}/for.body34.i"],
        [f"{loop}/for.body40.i", f"{loop}/for.body31.i"],
    ]


@pytest.mark.skipif(not _peano_available(), reason="needs an installed Peano")
def test_a_baseline_tree_compares_the_builds_cases_run(tmp_path, capsys):
    # Only the LUT build compiles the extra loop.
    base = tmp_path / "base"
    shutil.copytree(config.aie_kernels_dir(), base / "aie_kernels")
    shutil.copytree(config.aie_runtime_lib_dir(), base / "aie_runtime_lib")
    tanh = base / "aie_kernels" / "activation" / "tanh.cc"
    marker = "  event0();\n"
    assert tanh.read_text().count(marker) == 1
    tanh.write_text(
        tanh.read_text().replace(
            marker,
            f"{marker}#if !ACTIVATIONS_NATIVE_TANH\n"
            "  for (volatile int k = 0; k < 3; ++k)\n    ;\n#endif\n",
        )
    )
    meta = tmp_path / "meta.json"
    code = remarks.main(
        [
            "--target=aie2p",
            "--only=^tanh/1024x16/",
            f"--cases={KERNEL_CASES}",
            f"--out={tmp_path / 'rows.json'}",
            f"--meta={meta}",
            f"--baseline-sources={base}",
        ]
    )
    assert code == 0
    assert "\ncases: 2 builds for " in capsys.readouterr().out
    written = json.loads(meta.read_text())
    lut = "tanh/1024x16/bfloat16/use_lut=True/lut"
    assert set(written["kernels"]) == {"tanh/1024x16/bfloat16", lut}
    changed = written["baseline"]["changed"]
    assert changed and all(name.startswith(f"{lut}/") for name in changed)
    assert changed[f"{lut}/pm_bytes"][0] > changed[f"{lut}/pm_bytes"][1]
