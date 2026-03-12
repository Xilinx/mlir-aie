# utils.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
import contextlib
import fcntl
import hashlib
import os
import pickle
import time

from aie.utils.hostruntime.tensor_class import Tensor


def _cell_val_to_key(val):
    """Convert a single closure cell value to a stable, hashable key component.

    Priority:
    1. value-hash   — cheap O(1) path for types with a proper value-based
                      __hash__ (int, str, float, tuple, frozenset, and any
                      custom immutable type).
    2. pickle       — serialises full object state (__dict__ / __getstate__)
                      for mutable objects whose identity-based __hash__ would
                      miss mutations.
    3. pickle_dict  — for objects whose class is not picklable by name (e.g.
                      locally-defined classes), pickle __dict__ directly; it
                      is always a plain dict and therefore always picklable.
    4. repr         — last resort for non-picklable, identity-hashable objects.
    """
    # 1. Value-based hash: object.__hash__ is identity-based (id >> 4), so any
    #    override is contractually value-based and cheap.
    h = type(val).__hash__
    if h is not None and h is not object.__hash__:
        try:
            return ("hash", hash(val))
        except Exception:
            pass

    # 2. pickle: captures __dict__ and custom __getstate__ regardless of whether
    #    __eq__ / __hash__ / __repr__ are defined.  If the object itself is not
    #    picklable (e.g. locally-defined class), fall back to pickling __dict__,
    #    which is always a plain dict and therefore always picklable.
    try:
        return ("pickle", hashlib.sha256(pickle.dumps(val)).hexdigest())
    except Exception:
        pass
    if hasattr(val, "__dict__"):
        try:
            return (
                "pickle_dict",
                hashlib.sha256(pickle.dumps(val.__dict__)).hexdigest(),
            )
        except Exception:
            pass

    # 3. repr fallback
    return ("repr", repr(val))


def _closure_key(fn):
    """Return a hashable representation of a callable's closure cell contents.

    Uses _cell_val_to_key so that mutable objects (including those with no
    custom __eq__ / __hash__ / __repr__) produce distinct keys when their
    state changes.
    """
    if not fn.__closure__:
        return ()
    names = fn.__code__.co_freevars
    parts = []
    for name, cell in zip(names, fn.__closure__):
        try:
            val = cell.cell_contents
        except ValueError:
            parts.append((name, "<empty>"))
            continue
        parts.append((name, _cell_val_to_key(val)))
    return tuple(parts)


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
        elif callable(arg):
            if hasattr(arg, "__code__"):
                # Use bytecode and constants hash for Python functions/lambdas
                code = arg.__code__
                defaults = arg.__defaults__ if hasattr(arg, "__defaults__") else None
                closure_vals = _closure_key(arg)
                func_hash = hash(
                    (
                        code.co_code,
                        code.co_consts,
                        code.co_names,
                        defaults,
                        closure_vals,
                    )
                )
                signature_parts.append(f"function_{func_hash}")
            else:
                # Function argument - use hash of function address for uniqueness
                func_hash = hash(arg)
                signature_parts.append(f"function_{func_hash}")
        else:
            # Other type - use type name
            arg_hash = hash(arg)
            signature_parts.append(f"{type(arg).__name__}_{arg_hash}")

    for key, value in sorted(kwargs.items()):
        if isinstance(value, Tensor):
            # Tensor argument - include shape and dtype
            signature_parts.append(f"{key}_tensor_{value.shape}_{value.dtype}")
        elif callable(value):
            if hasattr(value, "__code__"):
                # Use bytecode and constants hash for Python functions/lambdas
                code = value.__code__
                defaults = (
                    value.__defaults__ if hasattr(value, "__defaults__") else None
                )
                closure_vals = _closure_key(value)
                func_hash = hash(
                    (
                        code.co_code,
                        code.co_consts,
                        code.co_names,
                        defaults,
                        closure_vals,
                    )
                )
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
