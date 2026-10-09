# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

"""Check diagnostic forwarding from successful and failed aiecc builds."""

import sys

import pytest
from aie.utils.compile import utils


@pytest.fixture
def aiecc(tmp_path, monkeypatch):
    """An aiecc that writes `AIECC_STDOUT` and `AIECC_STDERR`, then exits with
    `AIECC_EXIT`."""
    stand_in = tmp_path / "aiecc"
    stand_in.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "sys.stdout.write(os.environ['AIECC_STDOUT'])\n"
        "sys.stderr.write(os.environ['AIECC_STDERR'])\n"
        "sys.exit(int(os.environ['AIECC_EXIT']))\n"
    )
    stand_in.chmod(0o755)
    monkeypatch.setenv("AIECC_PATH", str(stand_in))


@pytest.mark.parametrize("severity", ["warning", "error", "note"])
@pytest.mark.parametrize("location", ["", "input.mlir:4:2: "])
def test_successful_diagnostics(aiecc, monkeypatch, capsys, severity, location):
    diagnostic = f"{location}{severity}: diagnostic message"
    monkeypatch.setenv("AIECC_EXIT", "0")
    monkeypatch.setenv("AIECC_STDOUT", "build output")
    monkeypatch.setenv("AIECC_STDERR", f"progress\n{diagnostic}\n")

    utils._run_aiecc("input.mlir", [])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"[aiecc] {diagnostic}\n"


def test_failed_diagnostics(aiecc, monkeypatch, capsys):
    monkeypatch.setenv("AIECC_EXIT", "1")
    monkeypatch.setenv("AIECC_STDOUT", "")
    monkeypatch.setenv("AIECC_STDERR", "error: compilation failed\nnote: reason\n")

    with pytest.raises(RuntimeError, match="error: compilation failed\nnote: reason"):
        utils._run_aiecc("input.mlir", [])

    assert capsys.readouterr().err == ""
