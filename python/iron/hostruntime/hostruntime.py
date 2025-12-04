# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from ..compile.cache.circular_cache import CircularCache
from ..compile.cache.utils import file_lock
from ..device import Device

IRON_MAX_INSTR_CACHE_ENTRIES = 256


class KernelHandle:
    pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime"""

    # An in-memory instruction cache to avoid reparsing instruction files
    _instruction_cache = CircularCache(IRON_MAX_INSTR_CACHE_ENTRIES)

    @abstractmethod
    def load(self, *args, **kwargs) -> KernelHandle:
        pass

    @abstractmethod
    def run(self, kernel_handle: KernelHandle, *args):
        pass

    def load_and_run(self, load_args: list, run_args: list):
        handle = self.load(*load_args)
        self.run(handle, list(run_args))

    @abstractmethod
    def device(self) -> Device:
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up all resources"""
        pass

    @abstractmethod
    def reset(self):
        """Reset the runtime"""
        pass

    def read_insts_sequence(cls, insts_path: Path):
        """Reads instructions from a text file (hex numbers, one per line)."""
        lock_path = insts_path.with_suffix(insts_path.suffix + ".lock")
        with file_lock(str(lock_path)):
            mod_time = insts_path.stat().st_mtime
            with open(insts_path, "r") as f:
                insts_text = f.readlines()
            insts_text = [l for l in insts_text if l != ""]
            insts_v = np.array([int(c, 16) for c in insts_text], dtype=np.uint32)
            cls._instruction_cache[insts_path] = (mod_time, insts_v)
        return insts_v

    # Read instruction stream from bin file and reformat it to be passed into the
    # instruction buffer for the xrt.kernel call
    def read_insts_binary(cls, insts_path: Path):
        """Reads instructions from a binary file."""
        lock_path = insts_path.with_suffix(insts_path.suffix + ".lock")
        with file_lock(str(lock_path)):
            mod_time = insts_path.stat().st_mtime
            with open(insts_path, "rb") as f:
                data = f.read()
            # Interpret the binary data as an array of uint32 values.
            insts_v = np.frombuffer(data, dtype=np.uint32)
            cls._instruction_cache[insts_path] = (mod_time, insts_v)
        return insts_v

    def read_insts(cls, insts_path: Path):
        """
        Reads instructions from the given file.
        If the file extension is .bin, uses binary read.
        If the file extension is .txt, uses sequence (text) read.
        """
        ext = insts_path.suffix.lower()
        if insts_path in cls._instruction_cache:
            # Speed up things if we re-configure the array a lot: Don't re-parse
            # the insts.bin each time
            mtime, insts_v = cls._instruction_cache[insts_path]
            if mtime == insts_path.stat().st_mtime:
                return insts_v
        if ext == ".bin":
            return cls.read_insts_binary(insts_path)
        elif ext == ".txt":
            return cls.read_insts_sequence(insts_path)
        else:
            raise IronRuntimeError(
                "Unsupported file extension for instruction file: expected .bin or .txt"
            )

    def clear_instr_cache(cls):
        cls._instruction_cache.clear()


class IronRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass


# Set default tensor class
try:
    from .xrtruntime.hostruntime import XRTHostRuntime

    DEFAULT_IRON_RUNTIME = XRTHostRuntime()
except ImportError:
    DEFAULT_IRON_RUNTIME = None
