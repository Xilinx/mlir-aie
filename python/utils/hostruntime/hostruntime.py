# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING

from .. import tensor

if TYPE_CHECKING:
    from aie.iron.device import Device
from .tensor_class import Tensor
from ..trace import TraceConfig
from ..trace.utils import create_ctrl_pkt, extract_tile
from ..npukernel import NPUKernel
from . import bfloat16_safe_allclose


class HostRuntimeError(Exception):
    """
    Error raised when a NPU kernel encounters an error during runtime operations.
    """

    pass


class KernelHandle(ABC):
    """
    Abstract representation that represents a kernel already registered/loaded with a runtime.
    """

    ...


class KernelResult(ABC):
    """A wrapper around data produced as the result of running a kernel"""

    def __init__(
        self,
        npu_time: int,
        trace_config: TraceConfig | None = None,
    ):
        """
        Initialize the KernelResult.

        Args:
            npu_time (int): The execution time on the NPU in nanoseconds.
            trace_config (TraceConfig | None, optional): Configuration for tracing. Defaults to None.
        """
        self._npu_time = npu_time
        self._trace_config = trace_config

    @property
    def npu_time(self) -> int:
        """
        Get the NPU execution time.

        Returns:
            int: The execution time in nanoseconds.
        """
        return self._npu_time

    @property
    def trace_config(self) -> TraceConfig | None:
        """
        Get the trace configuration.

        Returns:
            TraceConfig | None: The trace configuration if available, else None.
        """
        return self._trace_config

    def has_trace(self) -> bool:
        """
        Check if trace data is available.

        Returns:
            bool: True if trace configuration is present, False otherwise.
        """
        return not (self._trace_config is None)

    @abstractmethod
    def is_success(self) -> bool:
        """
        Check if the kernel execution was successful.

        Returns:
            bool: True if successful, False otherwise.
        """
        pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime"""

    @abstractmethod
    def load(self, npu_kernel: NPUKernel, **kwargs) -> KernelHandle:
        """
        Load an NPU kernel into the runtime.

        Args:
            npu_kernel (NPUKernel): The NPU kernel to load.
            **kwargs: Additional arguments for loading.

        Returns:
            KernelHandle: A handle to the loaded kernel.
        """
        pass

    @abstractmethod
    def run(
        self,
        kernel_handle: KernelHandle,
        *args,
        trace_config: TraceConfig | None = None,
        only_if_loaded=False,
    ) -> KernelResult:
        """
        Run a loaded kernel.

        Args:
            kernel_handle (KernelHandle): The handle to the loaded kernel.
            *args: Arguments to pass to the kernel.
            trace_config (TraceConfig | None, optional): Configuration for tracing. Defaults to None.
            only_if_loaded (bool, optional): If True, only run if already loaded. Defaults to False.

        Returns:
            KernelResult: The result of the kernel execution.
        """
        pass

    def load_and_run(
        self,
        npu_kernel: NPUKernel,
        run_args: list,
        **kwargs,
    ) -> tuple[KernelHandle, KernelResult]:
        """
        Load and run an NPU kernel.

        Args:
            npu_kernel (NPUKernel): The NPU kernel to load and run.
            run_args (list): Arguments to pass to the kernel.
            **kwargs: Additional arguments passed to load.

        Returns:
            tuple[KernelHandle, KernelResult]: A tuple containing the kernel handle and the execution result.
        """
        trace_config = npu_kernel.trace_config
        handle = self.load(npu_kernel, **kwargs)
        if trace_config:
            if trace_config.trace_after_last_tensor and len(run_args) > 0:
                trace_config.last_tensor_shape = run_args[-1].shape
                trace_config.last_tensor_dtype = np.dtype(run_args[-1].dtype)
            self.prepare_args_for_trace(run_args, trace_config)

        ret = self.run(handle, list(run_args), trace_config=trace_config)

        if trace_config:
            trace_buffer, ctrl_buffer = self.extract_trace_from_args(
                run_args, trace_config
            )
            self.process_trace(trace_buffer, ctrl_buffer, trace_config)

        return handle, ret

    @abstractmethod
    def device(self) -> "Device":
        """
        Get the device associated with this runtime.

        Returns:
            Device: The device object.
        """
        pass

    # Read instruction stream from bin file and reformat it to be passed into the
    # instruction buffer for the xrt.kernel call
    @classmethod
    def read_insts_binary(cls, insts_path: Path):
        """
        Reads instructions from a binary file.

        Args:
            insts_path (Path): Path to the binary instruction file.

        Returns:
            np.ndarray: Array of uint32 instructions.
        """
        with open(insts_path, "rb") as f:
            data = f.read()
        # Interpret the binary data as an array of uint32 values.
        return np.frombuffer(data, dtype=np.uint32)

    @classmethod
    def read_insts(cls, insts_path: Path):
        """
        Reads instructions from the given file.

        If the file extension is .bin, uses binary read.
        If the file extension is .txt, uses sequence (text) read.

        Args:
            insts_path (Path): Path to the instruction file.

        Returns:
            np.ndarray: Array of instructions.

        Raises:
            HostRuntimeError: If the file extension is not supported.
        """
        ext = insts_path.suffix.lower()
        if ext == ".bin":
            return cls.read_insts_binary(insts_path)
        else:
            raise HostRuntimeError(
                "Unsupported file extension for instruction file: expected .bin"
            )

    @classmethod
    def prepare_args_for_trace(
        cls, args: list[Tensor], trace_config: TraceConfig
    ) -> list[Tensor]:
        """
        Prepare arguments for tracing by appending necessary buffers.

        Args:
            args (list[Tensor]): List of input/output tensors.
            trace_config (TraceConfig): Trace configuration.

        Returns:
            list[Tensor]: The updated list of tensors with trace buffers appended.
        """
        if trace_config.trace_after_last_tensor:
            # Create a new, extended out tensor.
            out_size = trace_config.trace_size
            if len(args) > 0:
                out_size += args[-1].nbytes
                # TODO: should really copy previous contents of output into this buffer...? What if it's in/out?
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
                    create_ctrl_pkt(1, 0, 0x32004),  # core status
                    create_ctrl_pkt(1, 0, 0x340D8),  # trace status
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
        """
        Extract trace and control buffers from the arguments.

        Args:
            args (list[Tensor]): List of tensors used in execution.
            trace_config (TraceConfig): Trace configuration.

        Returns:
            tuple[Tensor, Tensor | None]: A tuple containing the trace buffer and optionally the control buffer.
        """
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
        """
        Separate output data and trace data from a single output buffer stream.

        Args:
            tensor (Tensor | np.ndarray): The combined tensor.
            prefix_shape (tuple): Shape of the prefix (output data).
            prefix_dtype (np.dtype): Data type of the prefix.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the output prefix and the suffix (trace data).
        """
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
        """
        Process the trace buffer and control buffer.

        Args:
            trace_buffer (np.ndarray): The trace data buffer.
            ctrl_buffer (np.ndarray): The control packet buffer.
            trace_config (TraceConfig): Trace configuration.
            verbosity (int, optional): Verbosity level. Defaults to 0.
        """
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
                col, row, pkt_type, pkt_id = extract_tile(ctrl_buffer[i * 2])
                overflow = True if (ctrl_buffer[i * 2 + 1] >> 8) == 3 else False
                if overflow:
                    print(
                        f"WARNING: Trace overflow detected in tile({row},{col}). Trace results may be invalid."
                    )

    @classmethod
    def verify_results(cls, io_args, refs={}, verbosity=0):
        """
        Verify the results of the kernel execution against reference data.

        Args:
            io_args (list[Tensor]): List of input/output tensors.
            refs (dict, optional): Dictionary mapping index to reference numpy array. Defaults to {}.
            verbosity (int, optional): Verbosity level. Defaults to 0.

        Returns:
            int: Number of errors found.

        Raises:
            HostRuntimeError: If a reference index is out of bounds.
        """
        errors = 0
        if verbosity >= 1:
            print("Verifying results ...")

        for idx, ref in refs.items():
            if idx >= len(io_args):
                raise HostRuntimeError(
                    f"Error: Reference index {idx} out of bounds for {len(io_args)} IO buffers"
                )
            io_args[idx].to("cpu")
            o = io_args[idx].numpy()
            e = bfloat16_safe_allclose(ref.dtype, ref, o)
            errors += np.size(e) - np.count_nonzero(e)
        return errors

    def run_test(
        self,
        npu_kernel,
        io_args,
        ref,
        verify: bool = True,
        verbosity: int = 0,
    ) -> int:
        """
        Run a test for the given NPU kernel.

        Args:
            npu_kernel (NPUKernel): The NPU kernel to test.
            io_args (list[Tensor]): List of input/output tensors.
            ref (dict): Reference data for verification.
            verify (bool, optional): Whether to verify results. Defaults to True.
            verbosity (int, optional): Verbosity level. Defaults to 0.

        Returns:
            int: 0 if successful, 1 otherwise.
        """
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
