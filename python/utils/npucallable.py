# npucallable.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .trace import TraceConfig

if TYPE_CHECKING:
    from .compilable import Compilable


class NPUCallable:
    """
    Represents a compiled NPU kernel or a lazy callable that compiles on first use.
    """

    def __init__(
        self,
        xclbin_path: str | Path | None = None,
        insts_path: str | Path | None = None,
        kernel_name: str = "MLIR_AIE",
        trace_config: TraceConfig | None = None,
        compilable: "Compilable | None" = None,
    ):
        """
        Initialize the NPUCallable.

        Args:
            xclbin_path (str | Path | None, optional): Path to the xclbin file. Defaults to None.
            insts_path (str | Path | None, optional): Path to the instructions file. Defaults to None.
            kernel_name (str, optional): Name of the kernel. Defaults to "MLIR_AIE".
            trace_config (TraceConfig | None, optional): Trace configuration. Defaults to None.
            compilable (Compilable | None, optional): The Compilable object for lazy compilation. Defaults to None.
        """
        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._kernel_name = kernel_name
        self._trace_config = trace_config
        self._compilable = compilable

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
    def compilable(self) -> "Compilable | None":
        """
        Get the Compilable object.

        Returns:
            Compilable | None: The Compilable object if available.
        """
        return self._compilable

    # Blocking call.
    def __call__(self, *args, **kwargs) -> Any:
        """
        Run the kernel with the given arguments.
        This is a blocking call.
        If the kernel is not yet compiled, it will be compiled using the provided arguments.

        Args:
            *args: Arguments passed to the kernel.
            **kwargs: Additional arguments passed to the runtime load_and_run method.

        Returns:
            KernelResult: The result of the kernel execution.
        """
        if self._compilable:
            # Compile using the arguments
            # Note: We assume the arguments passed here are suitable for compilation (generation)
            # if the user used @jit. If they used @compileconfig, they should have called .compile() explicitly
            # unless they want this lazy behavior.
            compiled_callable = self._compilable.compile(*args, **kwargs)
            self._xclbin_path = compiled_callable.xclbin_path
            self._insts_path = compiled_callable.insts_path
            # We don't overwrite other attributes as they should be consistent or set by init
            return compiled_callable(*args, **kwargs)

        from . import DefaultNPURuntime

        return DefaultNPURuntime.load_and_run(
            self,
            list(args),
            **kwargs,
        )
