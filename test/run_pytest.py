# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "utils" / "run_pytest.py"
_SPEC = importlib.util.spec_from_file_location("run_pytest", _MODULE_PATH)
run_pytest = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(run_pytest)


def test_host_pytest_arguments(monkeypatch):
    monkeypatch.delenv("MLIR_AIE_NPU_TEST", raising=False)
    monkeypatch.delenv("NPU_RUNTIME", raising=False)
    assert run_pytest.pytest_arguments(["test.py"]) == ["test.py"]


@pytest.mark.parametrize(
    "environment",
    [
        {"MLIR_AIE_NPU_TEST": "1"},
        {"NPU_RUNTIME": "hrx"},
        {"NPU_RUNTIME": "hsa"},
    ],
)
def test_npu_pytest_arguments(monkeypatch, environment):
    monkeypatch.delenv("MLIR_AIE_NPU_TEST", raising=False)
    monkeypatch.delenv("NPU_RUNTIME", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    arguments = run_pytest.pytest_arguments(["test.py"])
    assert arguments[:7] == [
        "-n1",
        "--reruns",
        "1",
        "--reruns-delay",
        "3",
        "--rerun-show-tracebacks",
        "test.py",
    ]


def test_crashed_item_is_rerun(tmp_path):
    attempts = tmp_path / "attempts"
    test_file = tmp_path / "test_crash.py"
    test_file.write_text(
        """
import os
from pathlib import Path


def test_crash_once():
    attempts = Path(os.environ[\"ATTEMPTS_FILE\"])
    count = int(attempts.read_text()) if attempts.exists() else 0
    attempts.write_text(str(count + 1))
    if count == 0:
        os._exit(3)
"""
    )
    environment = os.environ.copy()
    environment["ATTEMPTS_FILE"] = str(attempts)
    environment["MLIR_AIE_NPU_TEST"] = "1"
    result = subprocess.run(
        [sys.executable, str(_MODULE_PATH), "-q", str(test_file)],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert attempts.read_text() == "2"