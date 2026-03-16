# utils.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc.
"""Cache key utilities and file locking for the JIT compilation cache."""

import contextlib
import hashlib
import os
import pickle
import time

# Cross-platform file locking:
# - POSIX: fcntl.flock
# - Windows: msvcrt.locking
if os.name == "nt":
    import msvcrt
else:
    import fcntl

from aie.utils.hostruntime.tensor_class import Tensor


def _cell_val_to_key(val):
    """Convert a closure cell value to a stable, hashable key component.

    Tries value-based hash first (for types with a proper __hash__), then
    pickle (to capture full object state for mutable objects), then pickle
    of __dict__ (for locally-defined classes), and finally repr as a fallback.
    """
    # Value-based hash: any __hash__ override is contractually value-based.
    h = type(val).__hash__
    if h is not None and h is not object.__hash__:
        try:
            return ("hash", hash(val))
        except Exception:
            pass

    # pickle captures __dict__ and __getstate__ for mutable objects.
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


def _try_acquire_lock(lock_file) -> bool:
    if os.name == "nt":
        # msvcrt.locking locks a byte-range starting at the current file position.
        # Try to use the first byte to provide a process-wide lock.
        # Hacky and brittle, but works well enough to prevent a race.
        try:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    else:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            return False


def _release_lock(lock_file) -> None:
    if os.name == "nt":
        try:
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
    else:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass


def _create_function_cache_key(function, args, kwargs):
    """Create a cache key for a function call based on function name and argument types/shapes."""
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
            # Include the value in the key so that different scalar/immutable
            # values (e.g. tile_size=4 vs tile_size=8) produce different keys.
            try:
                val_hash = hash(value)
            except TypeError:
                val_hash = hashlib.sha256(pickle.dumps(value)).hexdigest()
            signature_parts.append(f"{key}_{type(value).__name__}_{val_hash}")

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
            f = os.open(lock_file_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(f)
        except FileExistsError:
            pass  # File already exists
        lock_file = open(lock_file_path, "a+")

        # Try to acquire exclusive lock with timeout
        start_time = time.time()
        while True:
            try:
                if _try_acquire_lock(lock_file):
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
                _release_lock(lock_file)
            except OSError:
                pass  # Ignore errors when releasing lock
            lock_file.close()
