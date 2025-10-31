# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import functools
import numpy as np
import pyxrt as xrt

from ..utils.xrt import read_insts_binary
from .compileconfig import Compilable, PreCompiled
from aie.iron.tensor import Tensor
from pathlib import Path
from aie.iron.kernel import ExternalFunction


class CircularCache:
    def __init__(self, max_size):
        self.max_size = max_size
        self.cache = [None] * max_size
        self.keys = [None] * max_size
        self.index = 0

    def __contains__(self, key):
        return key in self.keys

    def __getitem__(self, key):
        idx = self.keys.index(key)
        return self.cache[idx]

    def __setitem__(self, key, value):
        self.cache[self.index] = value
        self.keys[self.index] = key
        self.index = (self.index + 1) % self.max_size

    def __len__(self):
        return sum(1 for k in self.keys if k is not None)

    def clear(self):
        self.cache = [None] * self.max_size
        self.keys = [None] * self.max_size
        self.index = 0


# Global cache for compiled kernels at the function level
# Key: (function_name, args_signature) -> NPUKernel instance
# There is a limit on the number of kernels we have in cache
_compiled_kernels = CircularCache(max_size=1)


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
        """
        Allows the kernel to be called as a function with the provided arguments.

        Parameters:
            args (IRON Tensors): Arguments to pass to the kernel.
        """

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


class Callable:
    def __init__(self, function, **kwargs):
        if isinstance(function, (Compilable, PreCompiled)):
            self.compilable = function
        else:
            self.compilable = Compilable(function, **kwargs)
        functools.update_wrapper(self, function)

    def to_json(self):
        return self.compilable.to_json()

    @classmethod
    def from_json(cls, json_str, func):
        import json

        data = json.loads(json_str)
        func_name = data.pop("function")
        compilable = Compilable.from_json(json_str, func)

        def new_func(*args, **kwargs):
            return compilable.function(*args, **kwargs)

        new_func.__name__ = func_name
        return cls(new_func, **data)

    def __call__(self, *args, **kwargs):
        if isinstance(self.compilable, PreCompiled):
            xclbin_path, inst_path = self.compilable.get_artifacts()
            cache_key = (str(xclbin_path), str(inst_path))
        else:
            cache_key = _create_function_cache_key(
                self.compilable.function, args, kwargs
            )
        if cache_key in _compiled_kernels:
            cached_kernel = _compiled_kernels[cache_key]
            return cached_kernel(*args, **kwargs)

        if not isinstance(self.compilable, PreCompiled):
            xclbin_path, inst_path = self.compilable.compile(*args, **kwargs)
        else:
            xclbin_path, inst_path = self.compilable.get_artifacts()

        kernel_name = "MLIR_AIE"
        try:
            kernel = NPUKernel(xclbin_path, inst_path, kernel_name=kernel_name)
            _compiled_kernels[cache_key] = kernel
            result = kernel(*args, **kwargs)
            return result
        except Exception as e:
            raise


def jit(function=None, **kwargs):
    """
    Decorator to JIT compile and run an IRON kernel on the NPU.
    """

    if function is None:
        return functools.partial(jit, **kwargs)

    return Callable(function, **kwargs)


def _create_function_cache_key(function, args, kwargs):
    """
    Create a cache key for a function call based on function name and argument types/shapes.
    This allows us to cache compiled kernels at the function level.
    Note that it is not necessary that we cache the tensor shapes since the kernel may be agonstic
    to the shape changes but we are doing here for safety.
    """
    # Get function name
    func_name = function.__name__

    # Create signature from argument types and shapes
    signature_parts = []

    for arg in args:
        if isinstance(arg, Tensor):
            # Tensor argument - include shape and dtype
            signature_parts.append(f"tensor_{arg.shape}_{arg.dtype}")
        elif isinstance(arg, ExternalFunction):
            # ExternalFunction argument - use its custom hash method
            func_hash = hash(arg)
            signature_parts.append(f"externalfunction_{func_hash}")
        elif callable(arg):
            # Function argument - use hash of function address for uniqueness
            func_hash = hash(arg)
            signature_parts.append(f"function_{func_hash}")
        else:
            # Unsupported type - use type name
            signature_parts.append(f"{type(arg).__name__}")

    for key, value in sorted(kwargs.items()):
        if isinstance(value, Tensor):
            # Tensor argument - include shape and dtype
            signature_parts.append(f"{key}_tensor_{value.shape}_{value.dtype}")
        elif isinstance(value, ExternalFunction):
            # ExternalFunction argument - use its custom hash method
            func_hash = hash(value)
            signature_parts.append(f"{key}_externalfunction_{func_hash}")
        elif callable(value):
            # Function argument - use hash of function address for uniqueness
            func_hash = hash(value)
            signature_parts.append(f"{key}_function_{func_hash}")
        else:
            # Unsupported type - use type name
            signature_parts.append(f"{key}_{type(value).__name__}")

    signature = "_".join(signature_parts)
    return (func_name, signature)
