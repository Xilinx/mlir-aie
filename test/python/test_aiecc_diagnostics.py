# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

"""Check diagnostic forwarding from successful and failed aiecc builds."""

import subprocess

import pytest
from aie.utils.compile import utils


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
    monkeypatch.setattr(utils.config, "aiecc_path", lambda: "aiecc")
    monkeypatch.setattr(
        utils.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args, 1, stdout="", stderr="error: compilation failed\nnote: reason\n"
        ),
    )

    with pytest.raises(RuntimeError, match="error: compilation failed\nnote: reason"):
        utils._run_aiecc("input.mlir", [])

    assert capsys.readouterr().err == ""
