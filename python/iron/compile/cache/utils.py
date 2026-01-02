# filelock.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import contextlib
import fcntl
import os
import time

from ...hostruntime.tensor_class import Tensor
from ...kernel import ExternalFunction


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
            if hasattr(arg, "__code__"):
                # Use bytecode and constants hash for Python functions/lambdas
                code = arg.__code__
                func_hash = hash((code.co_code, code.co_consts, code.co_names))
                signature_parts.append(f"function_{func_hash}")
            else:
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
            if hasattr(value, "__code__"):
                # Use bytecode and constants hash for Python functions/lambdas
                code = value.__code__
                func_hash = hash((code.co_code, code.co_consts, code.co_names))
                signature_parts.append(f"{key}_function_{func_hash}")
            else:
                # Function argument - use hash of function address for uniqueness
                func_hash = hash(value)
                signature_parts.append(f"{key}_function_{func_hash}")
        else:
            # Unsupported type - use type name
            signature_parts.append(f"{key}_{type(value).__name__}")

    signature = "_".join(signature_parts)
    return (func_name, signature)


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
