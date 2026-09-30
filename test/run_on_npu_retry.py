# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parent.parent / "utils" / "run_on_npu.py"
_SPEC = importlib.util.spec_from_file_location("run_on_npu", _MODULE_PATH)
run_on_npu = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(run_on_npu)


@pytest.mark.parametrize(
    "output",
    [
        "No such device with index '0'",
        "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-22): Invalid argument",
        "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed (err=-110): Connection timed out",
        "DRM_IOCTL_AMDXDNA_EXEC_CMD IOCTL failed (err=-5): Input/output error",
        "DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2): No such file or directory",
        "DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-22): Invalid argument",
        "idx 7: 42 != 42",
    ],
)
def test_transient_failure_detection(output):
    assert run_on_npu.is_transient_failure(output)


def test_deterministic_failure_is_not_transient():
    assert not run_on_npu.is_transient_failure("idx 7: 42 != 43")


def test_pytest_command_uses_item_retry_only():
    command = [
        "python",
        "utils/run_with_test_cache.py",
        "test-key",
        "utils/run_pytest.py",
        "test_file.py",
    ]
    assert run_on_npu.command_runs_pytest(command)
    assert not run_on_npu.command_runs_pytest(["./test.exe"])


def test_hrx_command_does_not_source_xrt(monkeypatch):
    monkeypatch.setenv("NPU_RUNTIME", "hrx")
    command = ["./test.exe"]
    assert run_on_npu.wrapped_command("/opt/xilinx/xrt", command) == command