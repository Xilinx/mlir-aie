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

from aie.extras.context import mlir_mod_ctx
from ..device import NPU1, NPU2, NPU1Col1, NPU2Col1
from ..compile import compile_mlir_module, compile_external_kernel
from ..kernel import ExternalFunction
from .npukernel import NPUKernel
from aie.dialects.aie import AIEDevice
from ..compile.cache.circular_cache import CircularCache
from ..compile.cache.utils import _create_function_cache_key, file_lock
from ..compile import IRON_CACHE_HOME
from ..compile.utils import _cleanup_failed_compilation
from . import DEFAULT_IRON_RUNTIME


# Global cache for compiled kernels at the function level
# Key: (function_name, args_signature) -> NPUKernel instance
# There is a limit on the number of kernels we have in cache
_compiled_kernels = CircularCache(max_size=1)


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
        if DEFAULT_IRON_RUNTIME is None:
            raise Exception("Cannot use JIT; DEFAULT_IRON_RUNTIME not set.")

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
        if is_placed:
            with mlir_mod_ctx() as ctx:
                function(*args, **kwargs)
                assert (
                    ctx.module.operation.verify()
                ), f"Verification failed for '{function.__name__}'"
                mlir_module = ctx.module
        else:
            mlir_module = function(*args, **kwargs)

        # Compile all ExternalFunction instances that were created during this JIT compilation
        for func in ExternalFunction._instances:
            if (
                not hasattr(func, "_compiled") or not func._compiled
            ):  # Don't compile if already compiled
                external_kernels.append(func)

        # Determine target architecture based on device type
        current_device = DEFAULT_IRON_RUNTIME.device()

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

        # Hash of the IR string, ExternalFunction compiler options, and target architecture
        module_hash = hash_module(mlir_module, external_kernels, target_arch)
        kernel_dir = IRON_CACHE_HOME / f"{module_hash}"
        lock_file_path = kernel_dir / ".lock"
        mlir_path = kernel_dir / "aie.mlir"

        # Use file locking to prevent race conditions when accessing cache directory
        with file_lock(lock_file_path):
            # Ensure cache directory exists
            os.makedirs(kernel_dir, exist_ok=True)

            # Write MLIR to file if not already cached
            inst_filename = "insts.bin"
            xclbin_filename = "final.xclbin"
            xclbin_path = kernel_dir / xclbin_filename
            inst_path = kernel_dir / inst_filename

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

        _compiled_kernels[cache_key] = NPUKernel(
            xclbin_path, inst_path, kernel_name="MLIR_AIE"
        )
        _compiled_kernels[cache_key](*args)

    return decorator


def hash_module(module, external_kernels=None, target_arch=None):
    """
    Hash the MLIR module and ExternalFunction compiler options to create a unique identifier.
    """
    mlir_str = str(module)

    # Include ExternalFunction compiler options and source code in the hash
    if external_kernels:
        running_hash = ""
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
