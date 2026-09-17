# test_kernel_source_materialization_windows.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Windows-only unit tests for cached kernel source materialization."""

import os

import aie.utils.compile.utils as compile_utils
import pytest
from aie.utils.compile.utils import _copy_source

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows-only test")

SOURCE = "// kernel\nvoid k() {}\n" * 64


def test_windows_replace_race_reuses_identical_source(tmp_path, monkeypatch):
    """A Windows sharing violation is harmless when the destination matches."""
    src = tmp_path / "shared.cc"
    src.write_text(SOURCE)
    dest = tmp_path / "kernel.cc"
    dest.write_text(SOURCE)
    real_replace = os.replace

    def fake_replace(tmp, final):
        if final == str(dest):
            raise PermissionError("sharing violation")
        real_replace(tmp, final)

    monkeypatch.setattr(compile_utils.os, "replace", fake_replace)

    _copy_source(str(dest), str(src))

    assert dest.read_text() == SOURCE
    assert sorted(path.name for path in tmp_path.iterdir()) == ["kernel.cc", "shared.cc"]


def test_windows_replace_race_still_raises_on_mismatch(tmp_path, monkeypatch):
    """Only identical staged content may bypass a Windows replace failure."""
    src = tmp_path / "shared.cc"
    src.write_text(SOURCE)
    dest = tmp_path / "kernel.cc"
    dest.write_text("// different\n")
    real_replace = os.replace

    def fake_replace(tmp, final):
        if final == str(dest):
            raise PermissionError("sharing violation")
        real_replace(tmp, final)

    monkeypatch.setattr(compile_utils.os, "replace", fake_replace)

    with pytest.raises(PermissionError):
        _copy_source(str(dest), str(src))

    assert dest.read_text() == "// different\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["kernel.cc", "shared.cc"]
