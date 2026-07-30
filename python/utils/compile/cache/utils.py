# utils.py -*- Python -*-
#
# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Cache key utilities and file locking for the JIT compilation cache."""

import contextlib
import hashlib
import os
import pickle
import queue
import threading
import time

# Cross-platform file locking:
# - POSIX: fcntl.flock
# - Windows: msvcrt.locking
if os.name == "nt":
    import msvcrt
else:
    import fcntl

from aie.utils.hostruntime.tensor_class import Tensor

# Windows has no indefinite blocking lock -- msvcrt.locking's LK_LOCK gives up
# after ~10s -- so that platform waits by retrying.  See _wait_for_lock.
_LOCK_POLL_SECONDS = 0.005


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
        # Lock the first byte to provide a process-wide lock.
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


def _create_function_cache_key(function, args, kwargs, *, extra_key=()):
    """Create a cache key for a function call based on call inputs.

    ``extra_key`` is an optional immutable discriminator owned by the caller.
    It is kept outside ``kwargs`` so internal cache dimensions cannot collide
    with user-facing compile-time parameter names.
    """
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
                        code.co_filename,
                        (
                            code.co_qualname  # pyright: ignore[reportAttributeAccessIssue]
                            if hasattr(code, "co_qualname")
                            else code.co_name
                        ),
                        defaults,
                        closure_vals,
                    )
                )
                signature_parts.append(f"function_{func_hash}")
            else:
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
                        code.co_filename,
                        (
                            code.co_qualname  # pyright: ignore[reportAttributeAccessIssue]
                            if hasattr(code, "co_qualname")
                            else code.co_name
                        ),
                        defaults,
                        closure_vals,
                    )
                )
                signature_parts.append(f"{key}_function_{func_hash}")
            else:
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
    key = (func_name, signature)
    return (*key, extra_key) if extra_key else key


def _blocking_acquire(lock_file_path):
    """Block until the lock is ours, returning the holding file object.

    POSIX only: `flock` without LOCK_NB parks the caller in the kernel until the
    holder releases, so there is nothing to poll and no handoff delay.
    """
    lock_file = open(lock_file_path, "a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
    except OSError:
        lock_file.close()
        raise
    return lock_file


def _wait_for_lock(lock_file_path, timeout_seconds):
    """Wait out a held lock, honouring `timeout_seconds`, and return the lock.

    On POSIX the wait happens inside a blocking `flock` on a helper thread, so
    the kernel wakes us the instant the holder releases rather than us guessing
    when to look. The timeout then applies to the handoff instead of to a poll
    interval. Windows has no indefinite blocking lock, so it retries.
    """
    if os.name == "nt":
        deadline = time.time() + timeout_seconds
        lock_file = open(lock_file_path, "a+")
        while not _try_acquire_lock(lock_file):
            if time.time() > deadline:
                lock_file.close()
                raise TimeoutError(
                    f"Could not acquire lock on {lock_file_path} within "
                    f"{timeout_seconds} seconds"
                )
            time.sleep(_LOCK_POLL_SECONDS)
        return lock_file

    handoff = queue.Queue()
    cancelled = threading.Event()
    # The waiter's cancelled-check and its handoff must be atomic against the
    # timeout below.  Otherwise the thread can win the lock in the instant the
    # caller gives up, hand it to nobody, and hold it until the process exits.
    state = threading.Lock()

    def waiter():
        try:
            acquired = _blocking_acquire(lock_file_path)
        except OSError:
            return
        with state:
            if not cancelled.is_set():
                handoff.put(acquired)
                return
        _release_lock(acquired)
        acquired.close()

    threading.Thread(target=waiter, name="aie-jit-cache-lock", daemon=True).start()
    try:
        return handoff.get(timeout=timeout_seconds)
    except queue.Empty:
        with state:
            cancelled.set()
            while not handoff.empty():  # it may have landed as we gave up
                late = handoff.get_nowait()
                _release_lock(late)
                late.close()
        raise TimeoutError(
            f"Could not acquire lock on {lock_file_path} within "
            f"{timeout_seconds} seconds"
        )


@contextlib.contextmanager
def file_lock(lock_file_path, timeout_seconds=60):
    """Context manager for file locking using flock to prevent race conditions.

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

        if not _try_acquire_lock(lock_file):
            # Drop our handle before waiting, and clear it first: the waiter can
            # raise, and the cleanup below must not touch a closed file.
            closed, lock_file = lock_file, None
            closed.close()
            lock_file = _wait_for_lock(lock_file_path, timeout_seconds)

        yield lock_file

    finally:
        if lock_file is not None:
            try:
                _release_lock(lock_file)
            except OSError:
                pass  # Ignore errors when releasing lock
            lock_file.close()
