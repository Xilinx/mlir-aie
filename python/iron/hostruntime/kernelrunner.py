# kernelrunner.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import numpy as np
from .hostruntime import DEFAULT_IRON_RUNTIME


class NPUKernel:
    """
    NPUKernel class wrapper for NPU kernels.
    """

    def __init__(self, xclbin_path, insts_path, kernel_name: str | None = None):
        """
        Initialize the NPUKernel object.
        Parameters:
            xclbin_path (str): Path to the XCLBIN file containing the kernel.
            insts_path (str): Path to the instruction binary file for the kernel.
        """
        import pyxrt as xrt
        from .xrtruntime.xrt import read_insts_binary

        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._kernel_name = kernel_name

    # Blocking call.
    def __call__(self, *args):
        """
        Allows the kernel to be called as a function with the provided arguments.

        Parameters:
            args (IRON Tensors): Arguments to pass to the kernel.
        """
        # Skip callable arguments since these are inlined in the kernel
        tensors = [t for t in args if not callable(t)]
        handle = DEFAULT_IRON_RUNTIME.load(
            self._xclbin_path, self._insts_path, kernel_name=self._kernel_name
        )
        DEFAULT_IRON_RUNTIME.run(handle, *tensors)
