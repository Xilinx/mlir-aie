# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from ..device import Device


class KernelHandle(ABC):
    """KernelHandles may be used a cache keys, and so should implement these methods."""

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime"""

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
        with open(insts_path, "r") as f:
            insts_text = f.readlines()
        insts_text = [l for l in insts_text if l != ""]

    # Read instruction stream from bin file and reformat it to be passed into the
    # instruction buffer for the xrt.kernel call
    def read_insts_binary(cls, insts_path: Path):
        """Reads instructions from a binary file."""
        with open(insts_path, "rb") as f:
            data = f.read()
        # Interpret the binary data as an array of uint32 values.
        return np.frombuffer(data, dtype=np.uint32)

    def read_insts(cls, insts_path: Path):
        """
        Reads instructions from the given file.
        If the file extension is .bin, uses binary read.
        If the file extension is .txt, uses sequence (text) read.
        """
        ext = insts_path.suffix.lower()
        if ext == ".bin":
            return cls.read_insts_binary(insts_path)
        elif ext == ".txt":
            return cls.read_insts_sequence(insts_path)
        else:
            raise IronRuntimeError(
                "Unsupported file extension for instruction file: expected .bin or .txt"
            )


class IronRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass
