# npukernel.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
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
        num_host_bos: int | None = None,
    ):
        """
        Initialize the NPUKernel.

        Args:
            xclbin_path (str | Path): Path to the xclbin file.
            insts_path (str | Path): Path to the instructions file.
            device_index (int, optional): Device index. Defaults to 0.
            kernel_name (str, optional): Name of the kernel. Defaults to "MLIR_AIE".
            trace_config (TraceConfig | None, optional): Trace configuration. Defaults to None.
            num_host_bos (int | None, optional): The compiled design's true
                host-buffer count -- the number of ``aie.runtime_sequence``
                operands, including any trace/ctrl-packet buffer the lowering
                appended. This is floor-independent (unlike the kernels.json
                ``boN`` slot count, which aiecc floors to the firmware
                command-chain minimum), so it is the correct value to validate
                host buffer counts against. ``None`` when it could not be
                determined (validation is then skipped).
        """
        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._kernel_name = kernel_name
        self._trace_config = trace_config
        self._device_index = device_index
        self._num_host_bos = num_host_bos

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

    @property
    def num_host_bos(self) -> int | None:
        """
        Get the compiled design's true host-buffer count.

        Returns:
            int | None: The number of ``aie.runtime_sequence`` operands the
            design was compiled with (including any appended trace buffer), or
            ``None`` if it could not be determined.
        """
        return self._num_host_bos

    # Blocking call.
    def __call__(self, *args, **kwargs):
        """
        Run the kernel with the given arguments.
        This is a blocking call.

        Args:
            *args: Arguments passed to the kernel.
            **kwargs: Additional arguments passed to the runtime load_and_run method.

        Returns:
            The result returned by the runtime ``load_and_run`` call.
        """
        from . import DefaultNPURuntime

        if DefaultNPURuntime is None:
            raise Exception("Cannot run kernel; DefaultNPURuntime not set.")
        return DefaultNPURuntime.load_and_run(
            self,
            list(args),
            **kwargs,
        )
