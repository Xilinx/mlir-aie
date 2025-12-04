# kernelrunner.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
# kernelrunner.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import numpy as np
from pathlib import Path
from . import DEFAULT_IRON_RUNTIME


class NPUKernel:
    """
    NPUKernel class wrapper for NPU kernels.
    """

    def __init__(
        self, xclbin_path, insts_path, device_index=0, kernel_name="PP_FD_PRE"
    ):
        """
        Initialize the NPUKernel object.
        Parameters:
            xclbin_path (str): Path to the XCLBIN file containing the kernel.
            insts_path (str): Path to the instruction binary file for the kernel.
            device_index (int, optional): Index of the device. Defaults to 0.
            kernel_name (str, optional): Name of the kernel. Defaults to "PP_FD_PRE".
        """
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
        return DEFAULT_IRON_RUNTIME.load_and_run(
            [Path(self._xclbin_path), Path(self._insts_path), self._kernel_name],
            list(args),
        )
