# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from aie.iron.hostruntime.xrtruntime.hostruntime import (
    XRTHostRuntime,
    XRTKernelHandle,
    IronRuntimeError,
)


@pytest.fixture
def mock_xrt_runtime():
    # Reset singleton
    if XRTHostRuntime._instance:
        XRTHostRuntime._instance.reset()

    with patch("aie.iron.hostruntime.xrtruntime.hostruntime.pyxrt") as mock_pyxrt:
        # Setup mock device
        mock_device = MagicMock()
        mock_device.get_info.return_value = "NPU Phoenix"
        mock_pyxrt.device.return_value = mock_device

        # Setup mock xclbin and context
        mock_xclbin = MagicMock()
        mock_xclbin.get_uuid.return_value = "uuid"

        # Mock get_kernels to return k1, k2, k3
        k1 = MagicMock()
        k1.get_name.return_value = "k1"
        k2 = MagicMock()
        k2.get_name.return_value = "k2"
        k3 = MagicMock()
        k3.get_name.return_value = "k3"
        mock_xclbin.get_kernels.return_value = [k1, k2, k3]

        mock_pyxrt.xclbin.return_value = mock_xclbin

        mock_context = MagicMock()
        mock_pyxrt.hw_context.return_value = mock_context

        # Setup mock kernel
        mock_kernel = MagicMock()
        mock_pyxrt.kernel.return_value = mock_kernel

        # Setup mock bo
        mock_bo = MagicMock()
        mock_pyxrt.bo.return_value = mock_bo

        # Mock ert_cmd_state
        mock_pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED = 1

        # Setup mock kernel execution result
        mock_run_handle = MagicMock()
        mock_run_handle.wait.return_value = 1
        mock_kernel.return_value = mock_run_handle

        runtime = XRTHostRuntime()
        # Mock read_insts to return dummy data
        runtime.read_insts = MagicMock(
            return_value=MagicMock(nbytes=10, view=lambda x: x)
        )

        # Mock Path checks
        with patch("pathlib.Path.exists", return_value=True), patch(
            "pathlib.Path.is_file", return_value=True
        ):
            yield runtime

    # Cleanup
    if XRTHostRuntime._instance:
        XRTHostRuntime._instance.reset()


def test_kernel_handle_equality():
    h1 = XRTKernelHandle(Path("a.xclbin"), "kernel", Path("insts.txt"))
    h2 = XRTKernelHandle(Path("a.xclbin"), "kernel", Path("insts.txt"))

    assert h1 == h2
    assert hash(h1) == hash(h2)

    d = {}
    d[h1] = 1
    assert h2 in d


def test_lru_cache(mock_xrt_runtime):
    # Set cache size to 2
    original_max = XRTHostRuntime.MAX_LOADED_KERNELS
    XRTHostRuntime.MAX_LOADED_KERNELS = 2

    try:
        h1 = mock_xrt_runtime.load(Path("a.xclbin"), Path("i1.txt"), "k1")
        h2 = mock_xrt_runtime.load(Path("a.xclbin"), Path("i2.txt"), "k2")

        assert len(mock_xrt_runtime._kernels) == 2
        assert h1 in mock_xrt_runtime._kernels
        assert h2 in mock_xrt_runtime._kernels

        # Access h1 to make it MRU
        mock_xrt_runtime.load(Path("a.xclbin"), Path("i1.txt"), "k1")

        # Load h3, should evict h2 (LRU)
        h3 = mock_xrt_runtime.load(Path("a.xclbin"), Path("i3.txt"), "k3")

        assert len(mock_xrt_runtime._kernels) == 2
        assert h1 in mock_xrt_runtime._kernels
        assert h3 in mock_xrt_runtime._kernels
        assert h2 not in mock_xrt_runtime._kernels

    finally:
        XRTHostRuntime.MAX_LOADED_KERNELS = original_max


def test_only_if_loaded(mock_xrt_runtime):
    h1 = XRTKernelHandle(Path("a.xclbin"), "k1", Path("i1.txt"))

    # Should fail because not loaded
    with pytest.raises(IronRuntimeError, match="is not loaded"):
        mock_xrt_runtime.run(h1, [], only_if_loaded=True)

    # Load it
    mock_xrt_runtime.load(Path("a.xclbin"), Path("i1.txt"), "k1")

    # Should succeed (mock run logic)
    mock_xrt_runtime.run(h1, [], only_if_loaded=True)


def test_fail_if_full(mock_xrt_runtime):
    original_max = XRTHostRuntime.MAX_LOADED_KERNELS
    XRTHostRuntime.MAX_LOADED_KERNELS = 2

    try:
        mock_xrt_runtime.load(Path("a.xclbin"), Path("i1.txt"), "k1")
        mock_xrt_runtime.load(Path("a.xclbin"), Path("i2.txt"), "k2")

        # Cache is full

        # Should fail
        with pytest.raises(IronRuntimeError, match="Cache is full"):
            mock_xrt_runtime.load(
                Path("a.xclbin"), Path("i3.txt"), "k3", fail_if_full=True
            )

        # Should succeed if fail_if_full=False (evicts)
        mock_xrt_runtime.load(
            Path("a.xclbin"), Path("i3.txt"), "k3", fail_if_full=False
        )

    finally:
        XRTHostRuntime.MAX_LOADED_KERNELS = original_max
