# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from . import tensor

if TYPE_CHECKING:
    from aie.iron.device import Device
from .tensor_class import Tensor
from .trace import TraceConfig
from .npukernel import NPUKernel


class HostRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass


class KernelHandle(ABC):
    """KernelHandles may be used a cache keys, and so should implement these methods."""

    ...


class KernelResult(ABC):
    """A wrapper around data produced as the result of running a kernel"""

    def __init__(
        self,
        npu_time: int,
        trace_config: TraceConfig | None = None,
    ):
        self._npu_time = npu_time
        self._trace_config = trace_config

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
    def load(self, npu_kernel: NPUKernel) -> KernelHandle:
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
        self,
        npu_kernel: NPUKernel,
        run_args: list,
        trace_config: TraceConfig | None = None,
    ) -> tuple[KernelHandle, KernelResult]:
        handle = self.load(npu_kernel)
        return handle, self.run(handle, list(run_args), trace_config)

    @abstractmethod
    def device(self) -> "Device":
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
        If the file extension is .bin or uses binary read.
        If the file extension is .txt, uses sequence (text) read.
        """
        ext = insts_path.suffix.lower()
        if ext == ".bin":
            return cls.read_insts_binary(insts_path)
        elif ext == ".txt":
            return cls.read_insts_sequence(insts_path)
        else:
            raise HostRuntimeError(
                "Unsupported file extension for instruction file: expected .bin or .txt"
            )

    @classmethod
    def prepare_args_for_trace(
        cls, args: list[Tensor], trace_config: TraceConfig
    ) -> list[Tensor]:
        if trace_config.trace_after_last_tensor:
            # Create a new, extended out tensor.
            out_size = trace_config.trace_size
            if len(args) > 0:
                out_size += args[-1].nbytes
                # TODO(erika): should really copy previous contents of output into this buffer...? What if it's in/out?
                args[-1] = tensor((out_size,), dtype=np.uint8)
            else:
                out = tensor((out_size,), dtype=np.uint8)
                args.append(out)
        else:
            pad_until = trace_config.DEFAULT_TRACE_BUFFER_INDEX
            if trace_config.enable_ctrl_pkts:
                pad_until -= 1
            while len(args) < pad_until:
                # TODO out always needed so register buf 7 succeeds (not needed in C/C++ host code)
                filler = tensor((1,), dtype=np.uint32)
                args.append(filler)

            if trace_config.enable_ctrl_pkts:
                # write ctrl packets
                ctrl_pkts = [
                    TraceConfig.create_ctrl_pkt(1, 0, 0x32004),  # core status
                    TraceConfig.create_ctrl_pkt(1, 0, 0x340D8),  # trace status
                ]
                # Pad to 8 words
                ctrl_pkts += [0] * (8 - len(ctrl_pkts))

                header = tensor(np.array(ctrl_pkts, dtype=np.uint32))
                args.append(header)

            # Allocate extra space for control packets if enabled
            alloc_size = trace_config.trace_size
            if trace_config.enable_ctrl_pkts:
                alloc_size = trace_config.trace_size * 4

            trace_buff = tensor((alloc_size,), dtype=np.uint8)
            args.append(trace_buff)
        return args

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
            trace_buff = args[-1].numpy()

        if trace_config.enable_ctrl_pkts:
            trace_buff, ctrl_buff = cls._extract_prefix(
                trace_buff, trace_config.trace_size, np.dtype(np.uint8)
            )
        trace_buff = trace_buff.view(np.uint32).reshape(
            trace_config.trace_size // np.dtype(np.uint32).itemsize
        )
        return trace_buff, ctrl_buff

    @classmethod
    def _extract_prefix(cls, tensor, prefix_shape, prefix_dtype):
        # Wrapper function to separate output data and trace data from a single output buffer stream
        if not isinstance(tensor, np.ndarray):
            tensor = tensor.numpy()
        flat_tensor = tensor.reshape((-1,)).view(np.uint8)
        prefix_bytes = np.prod(prefix_shape) * prefix_dtype.itemsize
        output_prefix = (
            flat_tensor[:prefix_bytes].view(prefix_dtype).reshape(prefix_shape).copy()
        )
        output_suffix = flat_tensor[prefix_bytes:].copy()
        return output_prefix, output_suffix

    @classmethod
    def process_trace(cls, trace_buffer, ctrl_buffer, trace_config, verbosity=0):
        if verbosity >= 1:
            print("trace_buffer shape: ", trace_buffer.shape)
            print("trace_buffer dtype: ", trace_buffer.dtype)
        trace_config.write_trace(trace_buffer)

        if trace_config.enable_ctrl_pkts:
            if verbosity >= 1:
                print("ctrl_buffer shape: ", ctrl_buffer.shape)
                print("ctrl_buffer dtype: ", ctrl_buffer.dtype)
                print("ctrl buffer: ", [hex(d) for d in ctrl_buffer])
            for i in range(ctrl_buffer.size // 2):
                col, row, pkt_type, pkt_id = TraceConfig.extract_tile(
                    ctrl_buffer[i * 2]
                )
                overflow = True if (ctrl_buffer[i * 2 + 1] >> 8) == 3 else False
                if overflow:
                    print(
                        f"WARNING: Trace overflow detected in tile({row},{col}). Trace results may be invalid."
                    )

    @classmethod
    def verify_results(cls, io_args, ref, verbosity=0):
        errors = 0
        if verbosity >= 1:
            print("Verifying results ...")

        # Handle ref being list or single
        if not isinstance(ref, list):
            ref = [ref]

        for item in ref:
            if isinstance(item, tuple) and len(item) == 2:
                idx, r = item
                if idx >= len(io_args):
                    print(
                        f"Error: Reference index {idx} out of bounds for {len(io_args)} IO buffers"
                    )
                    return 1
                io_args[idx].to("cpu")
                o = io_args[idx].numpy()
                e = np.equal(r, o)
                errors += np.size(e) - np.count_nonzero(e)
            else:
                print("Error: Reference data must be a list of (index, data) tuples")
                return 1
        return errors

    def run_test(
        self,
        io_args,
        ref,
        npu_kernel,
        verify: bool = True,
        verbosity: int = 0,
    ) -> int:
        kernel_handle = self.load(npu_kernel)
        trace_config = npu_kernel.trace_config

        # Ensure io_args is a list
        if not isinstance(io_args, list):
            io_args = [io_args] if io_args else []

        buffers = io_args
        last_out = buffers[-1] if buffers else None

        if trace_config:
            trace_config.last_tensor_shape = last_out.shape if last_out else None
            trace_config.last_tensor_dtype = last_out.dtype if last_out else None
            self.prepare_args_for_trace(buffers, trace_config)

        ret = self.run(kernel_handle, buffers)

        if verbosity >= 1:
            print("npu_time: ", ret.npu_time / 1000.0, " us")

        if trace_config:
            trace_buffer, ctrl_buffer = self.extract_trace_from_args(
                buffers, trace_config
            )
            self.process_trace(trace_buffer, ctrl_buffer, trace_config, verbosity)

        errors = 0
        if verify:
            errors = self.verify_results(io_args, ref, verbosity)

        if not errors:
            return 0
        else:
            if verbosity >= 1:
                print("\nError count: ", errors)
                print("\nFailed.\n")
            return 1
