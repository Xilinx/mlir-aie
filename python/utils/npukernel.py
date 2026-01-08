# npukernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
from pathlib import Path
from .trace import TraceConfig


class NPUKernel:
    """
    Represents a compiled NPU kernel.
    """

    def __init__(
        self,
        xclbin_path,
        insts_path,
        device_index=0,
        kernel_name="MLIR_AIE",
        trace_config: TraceConfig | None = None,
    ):
        """
        Initialize the NPUKernel.

        Args:
            xclbin_path (str | Path): Path to the xclbin file.
            insts_path (str | Path): Path to the instructions file.
            device_index (int, optional): Device index. Defaults to 0.
            kernel_name (str, optional): Name of the kernel. Defaults to "MLIR_AIE".
            trace_config (TraceConfig | None, optional): Trace configuration. Defaults to None.
        """
        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._kernel_name = kernel_name
        self._trace_config = trace_config

    @property
    def trace_config(self) -> TraceConfig | None:
        """
        Get the trace configuration.

        Returns:
            TraceConfig | None: The trace configuration.
        """
        return self._trace_config

    @property
    def xclbin_path(self):
        """
        Get the path to the xclbin file.

        Returns:
            str | Path: The xclbin path.
        """
        return self._xclbin_path

    @property
    def insts_path(self):
        """
        Get the path to the instructions file.

        Returns:
            str | Path: The instructions path.
        """
        return self._insts_path

    @property
    def kernel_name(self):
        """
        Get the kernel name.

        Returns:
            str: The kernel name.
        """
        return self._kernel_name

    # Blocking call.
    def __call__(self, *args, **kwargs):
        """
        Run the kernel with the given arguments.
        This is a blocking call.

        Args:
            *args: Arguments passed to the kernel.
            **kwargs: Additional arguments passed to the runtime load_and_run method.

        Returns:
            KernelResult: The result of the kernel execution.
        """
        from . import DefaultNPURuntime

        return DefaultNPURuntime.load_and_run(
            self,
            list(args),
            **kwargs,
        )
