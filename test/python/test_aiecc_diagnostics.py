# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

"""Check diagnostic forwarding from successful and failed aiecc builds."""

import os
import subprocess
import traceback
import types

import pytest
from aie.helpers import sourceloc
from aie.helpers.errors import IronCompileError
from aie.ir import Context, Module
from aie.utils.compile import utils

THIS_FILE = os.path.abspath(__file__)


def _failing_aiecc(monkeypatch, stderr):
    monkeypatch.setattr(utils.config, "aiecc_path", lambda: "aiecc")
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, stdout="", stderr=stderr
        ),
    )


@pytest.mark.parametrize("severity", ["warning", "error", "note"])
@pytest.mark.parametrize("location", ["", "input.mlir:4:2: "])
def test_successful_diagnostics(monkeypatch, capsys, severity, location):
    diagnostic = f"{location}{severity}: diagnostic message"
    monkeypatch.setattr(utils.config, "aiecc_path", lambda: "aiecc")
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="build output", stderr=f"progress\n{diagnostic}\n"
        ),
    )

    utils._run_aiecc("input.mlir", [])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"[aiecc] {diagnostic}\n"


def test_failed_diagnostics(monkeypatch, capsys):
    _failing_aiecc(monkeypatch, "error: compilation failed\nnote: reason\n")

    with pytest.raises(RuntimeError, match="error: compilation failed\nnote: reason"):
        utils._run_aiecc("input.mlir", [])

    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("legacy_notes", [False, True])
def test_located_failure_reports_against_the_design(monkeypatch, legacy_notes):
    """A located aiecc failure reads as a Python error against the user's line.

    aiecc verifies in-process and reports through MLIR's SourceMgr handler, so
    once the design it compiled carried locations, its stderr names a real
    file -- which is the whole point: the failure should arrive as a traceback
    into that file rather than as a wall of tool output.
    """
    if legacy_notes:
        monkeypatch.setattr(IronCompileError, "add_note", None, raising=False)
    _failing_aiecc(
        monkeypatch,
        f"{THIS_FILE}:1:1: error: 'aie.dma_bd' op exceeds the maximum\n"
        f"{THIS_FILE}:1:1: note: see current operation\n",
    )

    with pytest.raises(IronCompileError) as caught:
        utils._run_aiecc("input.mlir", [])

    assert "exceeds the maximum" in str(caught.value)
    frames = traceback.extract_tb(caught.value.__traceback__)
    assert THIS_FILE in [os.path.abspath(f.filename) for f in frames], frames
    # The note is the explanation, so it rides along rather than being dropped.
    rendered = "".join(traceback.format_exception(caught.value))
    assert rendered.count(f"[aiecc] {THIS_FILE}:1:1: note: see current operation") == 1


def test_glued_progress_output_still_locates(monkeypatch):
    """aiecc writes progress with no trailing newline, gluing it to the path.

    Fixed in aiecc itself, but an older binary on PATH must not silently cost
    the user their location.
    """
    _failing_aiecc(
        monkeypatch, f"(4/28) [0/1] input.mlir{THIS_FILE}:1:1: error: op rejected\n"
    )

    with pytest.raises(IronCompileError) as caught:
        utils._run_aiecc("input.mlir", [])

    frames = traceback.extract_tb(caught.value.__traceback__)
    assert THIS_FILE in [os.path.abspath(f.filename) for f in frames], frames


def test_unreadable_location_falls_back_to_raw_output(monkeypatch):
    """A frame we cannot quote is worse than the tool's own text."""
    _failing_aiecc(monkeypatch, "/nonexistent/design.py:1:1: error: op rejected\n")

    with pytest.raises(RuntimeError, match="op rejected") as caught:
        utils._run_aiecc("input.mlir", [])

    assert not isinstance(caught.value, IronCompileError)


@pytest.mark.parametrize("line", [0, 100_000])
def test_invalid_source_line_falls_back_to_raw_output(monkeypatch, line):
    _failing_aiecc(monkeypatch, f"{THIS_FILE}:{line}:1: error: op rejected\n")

    with pytest.raises(RuntimeError, match="op rejected") as caught:
        utils._run_aiecc("input.mlir", [])

    assert not isinstance(caught.value, IronCompileError)


def test_source_site_without_frame_positions(monkeypatch):
    """Python 3.10 frame records do not expose exact source positions."""
    monkeypatch.setattr(sourceloc, "_is_internal", lambda _: False)
    monkeypatch.setattr(
        sourceloc.inspect,
        "getframeinfo",
        lambda *_: types.SimpleNamespace(filename=THIS_FILE, lineno=1),
    )

    site = sourceloc.capture_source_site()

    assert site is not None
    assert (site.filename, site.line, site.col) == (THIS_FILE, 1, 0)


def test_module_text_keeps_locations():
    """str() prints no locations, so aiecc would lose them at the handoff."""
    with Context():
        module = Module.parse('func.func @f() { return loc("design.py":42:7) }')

    assert "design.py" not in str(module)
    assert "design.py" in utils._module_text(module)
    assert utils._module_text("already text") == "already text"
