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

    def __init__(self, xclbin_path, insts_path):
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

    def load(self):
        DEFAULT_IRON_RUNTIME.load(self._xclbin_path, self._insts_path)

    # Blocking call.
    def __call__(self, *args):
        """
        Allows the kernel to be called as a function with the provided arguments.

        Parameters:
            args (IRON Tensors): Arguments to pass to the kernel.
        """
        import pyxrt as xrt

        opcode = 3
        kernel_args = []

        for tensor in args:
            # Skip callable arguments since these are inlined in the kernel
            if callable(tensor):
                continue
            if not hasattr(tensor, "buffer_object"):
                raise TypeError(
                    f"Expected Tensor with .buffer_object(), got {type(tensor)}"
                )
            kernel_args.append(tensor.buffer_object())

        h = self.__kernel(opcode, self.__insts_buffer_bo, self.__n_insts, *kernel_args)
        r = h.wait()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise NPUKernel_Error(f"Kernel returned {r}")

    def __del__(self):
        """
        Destructor to clean up resources and delete the kernel and device objects.
        """
        if hasattr(self, "_NPUKernel__insts_buffer_bo"):
            del self.__insts_buffer_bo
            self.__insts_buffer_bo = None
        if hasattr(self, "_NPUKernel__kernel"):
            del self.__kernel
            self.__kernel = None
        if hasattr(self, "_NPUKernel__context"):
            del self.__context
            self.__context = None
        if hasattr(self, "_NPUKernel__xclbin"):
            del self.__xclbin
            self.__xclbin = None
        if hasattr(self, "_NPUKernel__device"):
            del self.__device
            self.__device = None


class NPUKernel_Error(Exception):
    """
    Error raised when a NPU kernel encounters an error during execution.
    """

    pass
