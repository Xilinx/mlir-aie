# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from ..device import Device
from .tensor import Tensor


class KernelHandle(ABC):
    """KernelHandles may be used a cache keys, and so should implement these methods."""

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class KernelResult(ABC):
    """A wrapper around data produced as the result of running a kernel"""

    def __init__(self, npu_time: int, trace_data: Tensor | None = None):
        self._npu_time = npu_time
        self._trace_data = trace_data

    @property
    def npu_execution_time(self) -> int:
        return self._npu_time

    @property
    def trace_data(self) -> Tensor | None:
        return self._trace_data

    @property
    def trace_size(self) -> int:
        if self._trace_data:
            return self._trace_data.size()
        else:
            return 0

    def has_trace(self) -> bool:
        return not (self._trace_data is None)

    @abstractmethod
    def is_success(self) -> bool:
        pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime"""

    @abstractmethod
    def load(self, *args, **kwargs) -> KernelHandle:
        pass

    @abstractmethod
    def run(
        self, kernel_handle: KernelHandle, *args, only_if_loaded=False
    ) -> KernelResult:
        pass

    def load_and_run(
        self, load_args: list, run_args: list
    ) -> tuple[KernelHandle, KernelResult]:
        handle = self.load(*load_args)
        return handle, self.run(handle, list(run_args))

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
        If the file extension is .bin or .elf, uses binary read.
        If the file extension is .txt, uses sequence (text) read.
        """
        ext = insts_path.suffix.lower()
        if ext == ".bin" or ext == ".elf":
            return cls.read_insts_binary(insts_path)
        elif ext == ".txt":
            return cls.read_insts_sequence(insts_path)
        else:
            raise HostRuntimeError(
                "Unsupported file extension for instruction file: expected .bin, .elf, or .txt"
            )


class HostRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass
