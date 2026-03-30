# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
"""JIT decorator for compiling and running IRON-decorated functions on the NPU."""

import os
import functools
import hashlib
import numpy as np

from aie.extras.context import mlir_mod_ctx
from .compile import compile_mlir_module, compile_external_kernel
from .npukernel import NPUKernel
from aie.dialects.aie import AIEDevice
from .compile.cache.circular_cache import CircularCache
from .compile.cache.utils import _create_function_cache_key, file_lock
from .compile import NPU_CACHE_HOME
from .compile.utils import _cleanup_failed_compilation

# Global cache for compiled kernels at the function level
# Key: (function_name, args_signature) -> NPUKernel instance
# There is a limit on the number of kernels we have in cache
_compiled_kernels = CircularCache(max_size=1)


def jit(function=None, use_cache=True):
    """
    Decorator to compile an NPU kernel into a binary to run on the NPU.

    The decorated function may either return an MLIR module directly (unplaced
    style, using the IRON API) or return None and populate the module implicitly
    through the active ``mlir_mod_ctx`` context (placed style, using low-level
    dialects).  The mode is detected automatically from the return value.

    Args:
        function (callable, optional): The function to compile.
        use_cache (bool, optional): Use cached MLIR module if available. Defaults to True.

    Returns:
        callable: The decorated function.
    """
    if function is None:
        return functools.partial(jit, use_cache=use_cache)

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        from aie.iron.device import NPU1, NPU2, NPU1Col1, NPU2Col1
        from aie.iron.kernel import ExternalFunction
        from . import DefaultNPURuntime

        if DefaultNPURuntime is None:
            raise Exception("Cannot use JIT; DefaultNPURuntime not set.")

        trace_config = kwargs.get("trace_config")

        # Strip compile-time-only kwargs that must not be forwarded to the NPU
        # kernel at runtime (e.g. trace_config is consumed by NPUKernel.__init__).
        runtime_kwargs = {k: v for k, v in kwargs.items() if k != "trace_config"}

        effective_use_cache = use_cache

        # Check if we already have a compiled kernel for this function signature
        cache_key = _create_function_cache_key(function, args, kwargs)
        if effective_use_cache and cache_key in _compiled_kernels:
            cached_kernel = _compiled_kernels[cache_key]
            if cached_kernel is None:
                raise RuntimeError(
                    f"Cached kernel for '{function.__name__}' is None; this is a bug."
                )
            # Filter out non-tensor arguments (ExternalFunction, scalars)
            # Only tensor args should be passed to the kernel
            tensor_args = _filter_tensor_args(args)
            return cached_kernel(*tensor_args, **runtime_kwargs)

        # Collect ExternalFunction instances that need JIT compilation.
        # Note: bare Kernel instances (pre-compiled .o) are intentionally
        # excluded here — they require no compilation step. Both Kernel and
        # ExternalFunction are stripped from the tensor args passed to the NPU
        # kernel (see _filter_tensor_args).
        # ExternalFunction.__init__ registers to _instances at construction time
        # (before this JIT call), so they must be captured before the clear below.
        external_kernels = [
            arg for arg in args if isinstance(arg, ExternalFunction)
        ] + [v for v in kwargs.values() if isinstance(v, ExternalFunction)]
        seen = set(id(k) for k in external_kernels)

        # Clear stale instances from previous (possibly failed) runs so that a
        # broken kernel doesn't prevent a corrected one from being recompiled.
        ExternalFunction._instances.clear()

        # Execute the function to generate MLIR.
        # Always wrap in mlir_mod_ctx so that placed-style functions (which
        # populate the module implicitly) work correctly.  If the function
        # returns a module directly (unplaced style) we use that instead.
        with mlir_mod_ctx() as ctx:
            result = function(*args, **kwargs)

        if result is None:
            # Placed style: module was built implicitly via the context.
            assert (
                ctx.module.operation.verify()
            ), f"Verification failed for '{function.__name__}'"
            mlir_module = ctx.module
        else:
            # Unplaced style: function returned the module directly.
            mlir_module = result

        # Also collect ExternalFunction instances created during function()
        # execution (e.g. inside algorithm helpers that construct them internally).
        for func in ExternalFunction._instances:
            if not func._compiled and id(func) not in seen:
                external_kernels.append(func)
                seen.add(id(func))

        current_device = DefaultNPURuntime.device()

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
        kernel_dir = NPU_CACHE_HOME / f"{module_hash}"
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

            if not effective_use_cache or not xclbin_exists or not inst_exists:
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
                except Exception:
                    # Clean up cache directory on any compilation failure to avoid any corrupted objects in the cache
                    _cleanup_failed_compilation(kernel_dir)
                    raise

        kernel = NPUKernel(
            xclbin_path,
            inst_path,
            kernel_name="MLIR_AIE",
            trace_config=trace_config,
        )
        if effective_use_cache:
            _compiled_kernels[cache_key] = kernel

        # Filter out non-tensor arguments (ExternalFunction, scalars) before calling kernel
        # Only tensor args should be passed to the kernel
        tensor_args = _filter_tensor_args(args)
        kernel(*tensor_args, **runtime_kwargs)

    return decorator


def _filter_tensor_args(args):
    """
    Filter out non-tensor arguments from args.

    Algorithm functions may include Kernel/ExternalFunction instances and scalar
    compile-time constants in their Python signature that must not be forwarded
    to the NPU kernel as runtime buffer arguments.

    Removes:
    - Kernel and ExternalFunction instances (resolved at compile time via link_with)
    - Scalar values (int, float, np.integer, np.floating) used as MLIR constants
    - Callables (e.g. lambda configuration helpers)
    """
    from aie.iron.kernel import ExternalFunction, Kernel

    tensor_args = []
    for arg in args:
        # Skip any kernel handle (Kernel, ExternalFunction, or subclasses)
        if isinstance(arg, Kernel):
            continue
        # Skip scalar types (MLIR constants)
        if isinstance(arg, (int, float, np.integer, np.floating)):
            continue
        # Skip callables (lambda functions)
        if callable(arg):
            continue
        tensor_args.append(arg)

    return tensor_args


def hash_module(module, external_kernels=None, target_arch=None):
    """
    Hash the MLIR module and ExternalFunction compiler options to create a unique identifier.

    Args:
        module: The MLIR module.
        external_kernels (list, optional): List of external kernels. Defaults to None.
        target_arch (str, optional): Target architecture. Defaults to None.

    Returns:
        str: The hash string.
    """
    mlir_str = str(module)

    # Include ExternalFunction compiler options and source code in the hash
    if external_kernels:
        combined_str = (
            mlir_str + "|" + "|".join(sorted(str(hash(f)) for f in external_kernels))
        )
    else:
        combined_str = mlir_str

    # Include target architecture in the hash
    if target_arch:
        combined_str += f"|target_arch={target_arch}"

    hash_result = hashlib.sha256(combined_str.encode("utf-8")).hexdigest()[:16]
    return hash_result
