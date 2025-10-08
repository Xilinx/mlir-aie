# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import os
import functools
import hashlib
import numpy as np
import pyxrt as xrt
import shutil
import fcntl
import contextlib
import time

from aie.extras.context import mlir_mod_ctx
from ..utils.xrt import read_insts_binary
from .device import NPU1, NPU2, NPU1Col1, NPU2Col1
from .compile import compile_mlir_module
from .config import get_current_device
from aie.dialects.aie import AIEDevice


# The `iron.jit` decorator below caches compiled kenrels inside the `IRON_CACHE_HOME` directory.
# Kernels are cached based on their hash value of the MLIR module string. If during compilation,
# we hit in the cache, the `iron.jit` will load the xclbin and instruction binary files from the cache.
IRON_CACHE_HOME = os.environ.get("IRON_CACHE_HOME", os.path.expanduser("~/.iron/cache"))


def _cleanup_failed_compilation(cache_dir):
    """Clean up cache directory after failed compilation, preserving the lock file."""
    if not os.path.exists(cache_dir):
        return

    for item in os.listdir(cache_dir):
        if item == ".lock":
            continue
        item_path = os.path.join(cache_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)


@contextlib.contextmanager
def file_lock(lock_file_path, timeout_seconds=60):
    """
    Context manager for file locking using flock to prevent race conditions.

    Args:
        lock_file_path (str): Path to the lock file
        timeout_seconds (int): Maximum time to wait for lock acquisition in seconds
    """
    lock_file = None
    try:
        # Create lock file if it doesn't exist
        os.makedirs(os.path.dirname(lock_file_path), exist_ok=True)
        try:
            f = os.open(lock_file_path, os.O_CREAT | os.O_EXCL)
            os.close(f)
        except FileExistsError:
            pass  # File already exists
        lock_file = open(lock_file_path, "a")

        # Try to acquire exclusive lock with timeout
        start_time = time.time()
        while True:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                # Lock is held by another process
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(
                        f"Could not acquire lock on {lock_file_path} within {timeout_seconds} seconds"
                    )
                time.sleep(0.1)

        yield lock_file

    finally:
        if lock_file is not None:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass  # Ignore errors when releasing lock
            lock_file.close()


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


def jit(function=None, is_placed=True, use_cache=True):
    """
    Decorator to compile an IRON kernel into a binary to run on the NPU.

    Parameters:
    - is_placed (bool): Whether the kernel is using explicit or implicit placement Defaults to True.
    - use_cache (bool): Use cached MLIR module if available. Defaults to True.
    """

    if function is None:
        return functools.partial(jit, is_placed=is_placed, use_cache=use_cache)

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        from .kernel import ExternalFunction

        # Check if we already have a compiled kernel for this function signature
        cache_key = _create_function_cache_key(function, args, kwargs)
        if cache_key in _compiled_kernels:
            cached_kernel = _compiled_kernels[cache_key]
            return cached_kernel(*args, **kwargs)

        # Clear any instances from previous runs to make sure if the user provided any broken code we don't try to recompile it
        ExternalFunction._instances.clear()

        # Find ExternalFunction instances in arguments and kwargs
        external_kernels = []
        for arg in args:
            if isinstance(arg, ExternalFunction):
                external_kernels.append(arg)
        for value in kwargs.values():
            if isinstance(value, ExternalFunction):
                external_kernels.append(value)

        # Execute the function to generate MLIR
        try:
            if is_placed:
                with mlir_mod_ctx() as ctx:
                    function(*args, **kwargs)
                    assert (
                        ctx.module.operation.verify()
                    ), f"Verification failed for '{function.__name__}'"
                    mlir_module = ctx.module
            else:
                mlir_module = function(*args, **kwargs)
        except Exception as e:
            raise

        # Compile all ExternalFunction instances that were created during this JIT compilation
        for func in ExternalFunction._instances:
            if (
                not hasattr(func, "_compiled") or not func._compiled
            ):  # Don't compile if already compiled
                external_kernels.append(func)

        # Determine target architecture based on device type
        try:
            current_device = get_current_device()

            # Determine target architecture based on device type
            if isinstance(current_device, (NPU2, NPU2Col1)):
                target_arch = "aie2p"
            elif isinstance(current_device, (NPU1, NPU1Col1)):
                target_arch = "aie2"
            elif current_device in (AIEDevice.npu2, AIEDevice.npu2_1col):
                target_arch = "aie2p"
            elif current_device in (AIEDevice.npu1, AIEDevice.npu1_1col):
                target_arch = "aie2"
            else:
                raise RuntimeError(f"Unsupported device type: {type(current_device)}")
        except Exception as e:
            raise

        # Hash of the IR string, ExternalFunction compiler options, and target architecture
        module_hash = hash_module(mlir_module, external_kernels, target_arch)
        kernel_dir = os.path.join(IRON_CACHE_HOME, f"{module_hash}")
        lock_file_path = os.path.join(kernel_dir, ".lock")
        mlir_path = os.path.join(kernel_dir, "aie.mlir")

        # Use file locking to prevent race conditions when accessing cache directory
        with file_lock(lock_file_path):
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
                try:
                    with open(mlir_path, "w", encoding="utf-8") as f:
                        print(mlir_module, file=f)

                    # Compile ExternalFunctions from inside the JIT compilation directory
                    for func in external_kernels:
                        compile_external_kernel(func, kernel_dir, target_arch)

                    # Compile the MLIR module
                    compile_mlir_module(
                        mlir_module=mlir_module,
                        insts_path=inst_path,
                        xclbin_path=xclbin_path,
                        work_dir=kernel_dir,
                    )
                except Exception as e:
                    # Clean up cache directory on any compilation failure to avoid any corrupted objects in the cache
                    _cleanup_failed_compilation(kernel_dir)
                    raise e

        kernel_name = "MLIR_AIE"
        try:
            kernel = NPUKernel(xclbin_path, inst_path, kernel_name=kernel_name)

            # Cache the kernel for this function signature
            _compiled_kernels[cache_key] = kernel

            result = kernel(*args, **kwargs)
            return result
        except Exception as e:
            raise

    return decorator


def compile_external_kernel(func, kernel_dir, target_arch):
    """
    Compile an ExternalFunction to an object file in the kernel directory.

    Args:
        func: ExternalFunction instance to compile
        kernel_dir: Directory to place the compiled object file
        target_arch: Target architecture (e.g., "aie2" or "aie2p")
    """
    # Skip if already compiled
    if hasattr(func, "_compiled") and func._compiled:
        return

    # Check if object file already exists in kernel directory
    output_file = os.path.join(kernel_dir, func._object_file_name)
    if os.path.exists(output_file):
        return

    # Create source file in kernel directory
    source_file = os.path.join(kernel_dir, f"{func._name}.cc")

    # Handle both source_string and source_file cases
    if func._source_string is not None:
        # Use source_string (write to file)
        try:
            with open(source_file, "w") as f:
                f.write(func._source_string)
        except Exception as e:
            raise
    elif func._source_file is not None:
        # Use source_file (copy existing file)
        # Check if source file exists before copying
        if os.path.exists(func._source_file):
            try:
                shutil.copy2(func._source_file, source_file)
            except Exception as e:
                raise
        else:
            return
    else:
        raise ValueError("Neither source_string nor source_file is provided")

    from .compile.compile import compile_cxx_core_function

    try:
        compile_cxx_core_function(
            source_path=source_file,
            target_arch=target_arch,
            output_path=output_file,
            include_dirs=func._include_dirs,
            compile_args=func._compile_flags,
            cwd=kernel_dir,
            verbose=False,
        )
    except Exception as e:
        raise

    # Mark the function as compiled
    func._compiled = True


def hash_module(module, external_kernels=None, target_arch=None):
    """
    Hash the MLIR module and ExternalFunction compiler options to create a unique identifier.
    """
    mlir_str = str(module)

    # Include ExternalFunction compiler options and source code in the hash
    if external_kernels:
        running_hash = ""
        source_contents = []
        for func in external_kernels:
            running_hash += str(hash(func))

        combined_str = mlir_str + "|" + "|".join(running_hash)
    else:
        combined_str = mlir_str

    # Include target architecture in the hash
    if target_arch:
        combined_str += f"|target_arch={target_arch}"

    hash_result = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16]
    return hash_result


def _hash_argument(arg, prefix=""):
    """
    Helper function to hash supported argument types (tensors and callables).
    Returns a string representation for cache key generation.
    """
    from aie.iron.tensor import Tensor
    from aie.iron.kernel import ExternalFunction

    if isinstance(arg, Tensor):
        # Tensor argument - include shape and dtype
        return f"{prefix}tensor_{arg.shape}_{arg.dtype}"
    elif isinstance(arg, ExternalFunction):
        # ExternalFunction argument - use its custom hash method
        func_hash = hash(arg)
        return f"{prefix}externalfunction_{func_hash}"
    elif callable(arg):
        # Function argument - use hash of function address for uniqueness
        func_hash = hash(arg)
        return f"{prefix}function_{func_hash}"
    else:
        # Unsupported type - use type name
        return f"{prefix}{type(arg).__name__}"


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
        result = _hash_argument(arg)
        signature_parts.append(result)

    for key, value in sorted(kwargs.items()):
        result = _hash_argument(value, f"{key}_")
        signature_parts.append(result)

    signature = "_".join(signature_parts)
    return (func_name, signature)
