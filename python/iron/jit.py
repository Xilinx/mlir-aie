# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
import tempfile
import hashlib
import numpy as np

try:
    import pyxrt as xrt
except ImportError:
    print("pyxrt not available; skipping module.")
    import sys

    sys.exit(0)


from aie.extras.context import mlir_mod_ctx
from ..utils.compile import compile_mlir_to_binary
from ..utils.xrt import read_insts_binary


IRON_CACHE_DIR = os.path.expanduser("~/.iron/cache")


class NPUKernel:
    def __init__(
        self, xclbin_path, insts_path, device_index=0, kernel_name="PP_FD_PRE"
    ):
        self.__device = xrt.device(device_index)

        # Find kernel by name in the xclbin
        self.__xclbin = xrt.xclbin(xclbin_path)
        kernels = self.__xclbin.get_kernels()

        try:
            xkernel = [k for k in kernels if kernel_name == k.get_name()][0]
        except KeyError:
            raise NPUKernel_Error("No such kernel: " + kernel_name)

        self.__device.register_xclbin(self.__xclbin)
        self.__context = xrt.hw_context(self.__device, self.__xclbin.get_uuid())
        self.__kernel = xrt.kernel(self.__context, xkernel.get_name())

        # Set up instruction stream
        insts = read_insts_binary(insts_path)
        self.__n_insts = len(insts)
        insts_buffers_bytes = self.__n_insts * np.dtype(insts.dtype).itemsize

        # Magic number for RyzenAI group id that will be fixed in the future. See same code at XRT:
        # https://github.com/Xilinx/XRT/blob/56222ed5cfd119dff0d5bd920735b87024e8c829/src/runtime_src/core/common/api/xrt_module.cpp#L1621
        group_id = 1

        self.__insts_buffer_bo = xrt.bo(
            self.__device,
            insts_buffers_bytes,
            xrt.bo.cacheable,
            group_id,
        )

        # Copy into a temporary numpy buffer
        insts_buffer_bo_np = np.frombuffer(
            self.__insts_buffer_bo.map(), dtype=insts.dtype
        ).reshape(insts.shape)
        insts_buffer_bo_np[:] = insts

        # Always sync to the device in the constructor
        self.__insts_buffer_bo.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Blocking call.
    def __call__(self, *args):

        opcode = 3
        kernel_args = []

        for tensor in args:
            if not hasattr(tensor, "buffer_object"):
                raise TypeError(
                    f"Expected Tensor with .buffer_object(), got {type(tensor)}"
                )
            kernel_args.append(tensor.buffer_object())
        h = self.__kernel(opcode, self.__insts_buffer_bo, self.__n_insts, *kernel_args)
        r = h.wait()
        if r != xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise Exception(f"Kernel returned {r}")

    def __del__(self):
        del self.__kernel
        del self.__device


class NPUKernel_Error(Exception):
    pass


def jit(*, debug=False, verify=False, use_cache=True):
    def decorator(fn):
        with mlir_mod_ctx() as ctx:
            fn()
            if verify:
                assert (
                    ctx.module.operation.verify()
                ), f"Verification failed for '{fn.__name__}'"
            mlir_module = ctx.module

        # Hash of the IR string
        module_hash = hash_module(mlir_module)
        kernel_dir = os.path.join(IRON_CACHE_DIR, f"{module_hash}")
        mlir_path = os.path.join(kernel_dir, "aie.mlir")

        # Ensure cache directory exists
        os.makedirs(kernel_dir, exist_ok=True)

        # Write MLIR to file if not already cached
        inst_filename = "insts.bin"
        xclbin_filename = "final.xclbin"
        xclbin_path = os.path.join(kernel_dir, xclbin_filename)
        inst_path = os.path.join(kernel_dir, inst_filename)

        xclbin_exists = os.path.exists(xclbin_path)
        inst_exists = os.path.exists(inst_path)

        if not use_cache or not xclbin_exists or not inst_exists:
            with open(mlir_path, "w", encoding="utf-8") as f:
                print(mlir_module, file=f)
            compile_mlir_to_binary(
                mlir_path=mlir_path,
                inst_filename=inst_filename,
                xclbin_filename=xclbin_filename,
            )

        kernel = load_kernel(xclbin_path, inst_path)
        return kernel

    return decorator


def hash_module(module):
    mlir_str = str(module)
    return hashlib.sha256(mlir_str.encode("utf-8")).hexdigest()[:16]


def load_kernel(binary_path, inst_path):
    kernel_name = "MLIR_AIE"
    return NPUKernel(binary_path, inst_path, kernel_name=kernel_name)
