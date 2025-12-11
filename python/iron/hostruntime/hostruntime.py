# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path

from ..device import Device
from .tensor_class import Tensor


class HostRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass


class TraceConfig:
    DEFAULT_TRACE_BUFFER_INDEX = 7

    def __init__(
        self,
        trace_size: int,
        trace_after_last_tensor: bool = False,
        enable_ctrl_pkts: bool = False,
        last_tensor_shape=None,
        last_tensor_dtype=None,
    ):
        if trace_size <= 0:
            raise HostRuntimeError(f"Invalid trace size: {trace_size}")
        self.trace_size = trace_size
        self.trace_after_last_tensor = trace_after_last_tensor
        self.enable_ctrl_pkts = enable_ctrl_pkts
        self.last_tensor_shape = last_tensor_shape
        self.last_tensor_dtype = last_tensor_dtype


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

    def __init__(
        self,
        npu_time: int,
        trace_config: TraceConfig | None = None,
        tensors: list[Tensor] | None = None,
    ):
        self._npu_time = npu_time
        self._trace_config = trace_config
        self._tensors = tensors

    @property
    def npu_time(self) -> int:
        return self._npu_time

    @property
    def trace_config(self) -> TraceConfig | None:
        return self._trace_config

    def has_trace(self) -> bool:
        return not (self._trace_config is None)

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
        self,
        kernel_handle: KernelHandle,
        *args,
        trace_config: TraceConfig | None = None,
        only_if_loaded=False,
    ) -> KernelResult:
        pass

    def load_and_run(
        self, load_args: list, run_args: list, trace_config: TraceConfig | None = None
    ) -> tuple[KernelHandle, KernelResult]:
        handle = self.load(*load_args)
        return handle, self.run(handle, list(run_args), trace_config)

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

    @classmethod
    def read_insts_sequence(cls, insts_path: Path):
        """Reads instructions from a text file (hex numbers, one per line)."""
        with open(insts_path, "r") as f:
            insts_text = f.readlines()
        insts_text = [l for l in insts_text if l != ""]

    # Read instruction stream from bin file and reformat it to be passed into the
    # instruction buffer for the xrt.kernel call
    @classmethod
    def read_insts_binary(cls, insts_path: Path):
        """Reads instructions from a binary file."""
        with open(insts_path, "rb") as f:
            data = f.read()
        # Interpret the binary data as an array of uint32 values.
        return np.frombuffer(data, dtype=np.uint32)

    @classmethod
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

    @classmethod
    def prepare_args_for_trace(
        cls, args: list[Tensor], trace_config: TraceConfig
    ) -> list[Tensor]:
        from aie.iron.hostruntime import tensor

        if trace_config.trace_after_last_tensor:
            # Create a new, extended out tensor.
            out_size = trace_config.trace_size
            if len(args) > 0:
                out_size += args[-1].nbytes
                # TODO(erika): should really copy previous contents of output into this buffer...? What if it's in/out?
                args[-1] = tensor(out_size, dtype=np.uint8)
            else:
                out = tensor(out_size, dtype=np.uint8)
                args.append(out)
        else:
            while len(args) < trace_config.DEFAULT_TRACE_BUFFER_INDEX:
                # TODO out always needed so register buf 7 succeeds (not needed in C/C++ host code)
                filler = tensor((1,), dtype=np.uint32)
                args.append(filler)

                trace_buff = tensor((trace_config.trace_size,), dtype=np.uint8)
                args.append(trace_buff)

    @classmethod
    def extract_trace_from_args(
        cls, args: list[Tensor], trace_config: TraceConfig
    ) -> tuple[Tensor, Tensor | None]:
        trace_buff = None
        ctrl_buff = None

        if trace_config.trace_after_last_tensor:
            args[-1], trace_buff = cls._extract_prefix(
                args[-1], trace_config.last_tensor_shape, trace_config.last_tensor_dtype
            )
        else:
            # The trace position is always last.
            trace_buff = args.pop(-1)

        if trace_config.enable_ctrl_pkts:
            trace_buff, ctrl_buff = cls._extract_prefix(
                trace_buff, trace_config.trace_size, np.uint8
            )
        trace_buff = trace_buff.view(np.uint32).reshape(
            trace_config.trace_size // np.uint32.itemsize
        )
        return trace_buff, ctrl_buff

    @classmethod
    def _extract_prefix(cls, tensor: Tensor, prefix_shape, prefix_dtype):
        # Wrapper function to separate output data and trace data from a single output buffer stream
        flat_tensor = tensor.reshape((-1,)).view(np.uint8)
        prefix_bytes = np.prod(prefix_shape) * prefix_dtype.itemsize
        output_prefix = (
            flat_tensor[:prefix_bytes].view(prefix_dtype).reshape(prefix_shape)
        )
        output_suffix = flat_tensor[-prefix_bytes:]
        return output_prefix, output_suffix
