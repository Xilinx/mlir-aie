# Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..tensor_factory import tensor

if TYPE_CHECKING:
    from aie.iron.device import Device

    # Annotation-only: NPUKernel imports the dispatch bridge, which needs
    # HostRuntimeError from this module. Importing it for real would close
    # that loop.
    from ..npukernel import NPUKernel
from ..trace import TraceConfig
from ..trace.utils import create_ctrl_pkt, extract_tile
from . import bfloat16_safe_allclose
from .tensor_class import NpuTensor

logger = logging.getLogger(__name__)


class HostRuntimeError(Exception):
    """Error raised when a NPU kernel encounters an error during runtime operations."""

    pass


class KernelHandle(ABC):
    """Abstract representation that represents a kernel already registered/loaded with a runtime."""

    @property
    def needs_dispatch_insts(self) -> bool:
        """Whether every run must be handed freshly synthesized instructions.

        True for a ``DispatchTime[T]`` design, which holds no static
        instruction stream: each call rebuilds one through the dispatch
        bridge. Backends override this to name the artifact they lack, so the
        "no static stream" test lives in one place per backend instead of
        being re-derived at every use.
        """
        return False


class KernelResult(ABC):
    """A wrapper around data produced as the result of running a kernel."""

    def __init__(
        self,
        npu_time: int,
        trace_config: TraceConfig | None = None,
    ):
        """Initialize the KernelResult.

        Args:
            npu_time (int): The execution time on the NPU in nanoseconds.
            trace_config (TraceConfig | None, optional): Configuration for tracing. Defaults to None.
        """
        self._npu_time = npu_time
        self._trace_config = trace_config

    @property
    def npu_time(self) -> int:
        """Get the NPU execution time.

        Returns:
            int: The execution time in nanoseconds.
        """
        return self._npu_time

    @property
    def trace_config(self) -> TraceConfig | None:
        """Get the trace configuration.

        Returns:
            TraceConfig | None: The trace configuration if available, else None.
        """
        return self._trace_config

    def has_trace(self) -> bool:
        """Check if trace data is available.

        Returns:
            bool: True if trace configuration is present, False otherwise.
        """
        return self._trace_config is not None

    @abstractmethod
    def is_success(self) -> bool:
        """Check if the kernel execution was successful.

        Returns:
            bool: True if successful, False otherwise.
        """
        pass


class HostRuntime(ABC):
    """An abstract class for a generic host runtime."""

    def check_device_consistency(self):
        """Check if the overridden device is loadable on the runtime device.

        A 1- or N-column variant of a generation (e.g. NPU1Col1) is loadable
        on a wider device of the same generation (e.g. a 4-column NPU1), so we
        accept any override whose arch matches and whose column count is <=
        the runtime device's column count.
        """
        assert __package__ is not None
        mod = sys.modules[__package__]
        override = getattr(mod, "_CURRENT_DEVICE", None)
        if override is None:
            return
        runtime_device = self.device()
        try:
            same_arch = override.arch == runtime_device.arch
            fits = override.cols <= runtime_device.cols
        except AttributeError:
            same_arch = fits = False
        if not (same_arch and fits):
            raise RuntimeError(
                f"Overridden device {override} is not loadable on runtime "
                f"device {runtime_device}"
            )

    def cleanup(self) -> None:
        """Release any cached device/runtime resources held by this runtime.

        Base implementation is a no-op: a plain runtime holds nothing to
        release. Caching runtimes override this to free hardware contexts,
        loaded executables, instruction buffers, etc. Safe to call even if the
        runtime never ran anything.
        """
        return

    def evict_context(self, xclbin_path: Path) -> None:
        """Drop any cached device context associated with ``xclbin_path``.

        Recovery hook invoked after the driver rejects a submit against a stale
        context (e.g. an XRT IOCTL EINVAL) so the next ``load`` rebuilds a fresh
        context. Base implementation is a no-op for runtimes that keep no
        evictable context cache.
        """
        return

    @abstractmethod
    def load(self, npu_kernel: NPUKernel, **kwargs) -> KernelHandle:
        """Load an NPU kernel into the runtime.

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
        args,
        trace_config: TraceConfig | None = None,
        fail_on_error: bool = True,
        only_if_loaded=False,
        dispatch_insts: np.ndarray | None = None,
        **kwargs,
    ) -> KernelResult:
        """Run a loaded kernel.

        Args:
            kernel_handle (KernelHandle): The handle to the loaded kernel.
            args: Arguments to pass to the kernel.
            trace_config (TraceConfig | None, optional): Configuration for tracing. Defaults to None.
            fail_on_error (bool, optional): Whether to raise an exception on kernel failure. Defaults to True.
            only_if_loaded (bool, optional): If True, only run if already loaded. Defaults to False.
            dispatch_insts (np.ndarray | None, optional): Freshly-generated
                instruction words for a DispatchTime[T] design (see
                ``_maybe_generate_dispatch_insts``), to be submitted
                *instead of* any static instruction stream the kernel handle
                may hold. ``None`` for a design with no DispatchTime[T]
                parameters -- the ordinary static/cached path.
            **kwargs: Additional arguments.

        Returns:
            KernelResult: The result of the kernel execution.
        """
        pass

    @staticmethod
    def _resolve_insts_path(npu_kernel) -> Path | None:
        """Resolve and validate a kernel's static insts.bin.

        ``None`` when the design has no static stream at all -- a full ELF
        carries its control code inside the ELF, and a DispatchTime[T] design
        synthesizes a fresh stream per call.
        """
        if not npu_kernel.insts_path:
            return None
        insts_path = Path(npu_kernel.insts_path).resolve()
        if not insts_path.is_file():
            raise HostRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )
        return insts_path

    @staticmethod
    def _require_dispatch_insts(kernel_handle: KernelHandle, dispatch_insts) -> None:
        """Reject a run of a dispatch design that was given no instructions.

        ``load_and_run`` synthesizes them via ``_maybe_generate_dispatch_insts``;
        anything reaching ``run()`` by another route (``run_test``,
        ``run_chain``, a direct ``load()`` + ``run()``) would otherwise submit a
        null instruction stream, which each backend fails differently and
        unrecognizably.
        """
        if kernel_handle.needs_dispatch_insts and dispatch_insts is None:
            raise HostRuntimeError(
                "this kernel declares DispatchTime[T] parameter(s), so it has "
                "no static instruction stream and run() cannot submit it "
                "directly. Call the kernel (or load_and_run) with the "
                "DispatchTime[T] value(s) so the stream is built for this call."
            )

    def _maybe_generate_dispatch_insts(
        self, npu_kernel: NPUKernel, dispatch_scalars: dict | None
    ) -> np.ndarray | None:
        """Return fresh instruction words for a DispatchTime[T] design, or ``None``.

        Fully backend-agnostic: validates *dispatch_scalars* against the
        compiled design's declared ``DispatchTime[T]`` parameters and, if any
        are declared, calls the design's ``DispatchBridge`` to
        synthesize this call's instruction stream. Every backend calls this
        the same way; only what a backend does with the returned words
        (rebuild a BO, memmove into a device buffer, rebuild an executable...)
        is backend-specific -- see each concrete ``run()``.
        """
        dispatch_params = npu_kernel.dispatch_params
        dispatch_scalars = dispatch_scalars or {}
        if not dispatch_params:
            if dispatch_scalars:
                raise HostRuntimeError(
                    f"got dispatch scalar(s) {list(dispatch_scalars)} but this "
                    "compiled design declares no DispatchTime[T] parameters"
                )
            return None
        missing = set(dispatch_params) - set(dispatch_scalars)
        extra = set(dispatch_scalars) - set(dispatch_params)
        if missing or extra:
            raise HostRuntimeError(
                f"dispatch scalar mismatch: missing={missing or None} "
                f"extra={extra or None}; design expects exactly {dispatch_params}"
            )
        return npu_kernel._get_dispatch_bridge().generate(dispatch_scalars)

    def load_and_run(
        self,
        npu_kernel: NPUKernel,
        run_args: list,
        dispatch_scalars: dict | None = None,
        **kwargs,
    ) -> tuple[KernelHandle, KernelResult]:
        """Load and run an NPU kernel.

        Args:
            npu_kernel (NPUKernel): The NPU kernel to load and run.
            run_args (list): Arguments to pass to the kernel.
            dispatch_scalars (dict | None, optional): DispatchTime[T] scalar
                values for this call, keyed by parameter name. Popped here
                (never forwarded to ``load()``) so a design with
                DispatchTime[T] parameters actually reaches ``run()``,
                which is where the fresh instruction stream is needed.
            **kwargs: Additional arguments passed to load.

        Returns:
            tuple[KernelHandle, KernelResult]: A tuple containing the kernel handle and the execution result.
        """
        trace_config = npu_kernel.trace_config
        handle = self.load(npu_kernel, **kwargs)
        dispatch_insts = self._maybe_generate_dispatch_insts(
            npu_kernel, dispatch_scalars
        )
        if trace_config:
            if trace_config.reuse_output_buffer and len(run_args) > 0:
                trace_config.last_tensor_shape = run_args[-1].shape
                trace_config.last_tensor_dtype = np.dtype(run_args[-1].dtype)
            self.prepare_args_for_trace(run_args, trace_config)

            # Passing a trace_config to a design that never called enable_trace
            # means the lowering appended no trace operand, yet the host just
            # appended a trace buffer above. The extra buffer has no matching
            # runtime_sequence operand and would run with an empty trace (or,
            # before the firmware-ABI floor over-declared kernels.json, segfault
            # in XRT argument setup). Compare against the design's true operand
            # count -- floor-independent, unlike the kernels.json boN slot count.
            num_host_bos = npu_kernel.num_host_bos
            if num_host_bos is not None and len(run_args) > num_host_bos:
                raise HostRuntimeError(
                    f"A trace_config was supplied but the compiled design has "
                    f"{num_host_bos} host buffer argument(s), while running with "
                    f"a trace buffer requires {len(run_args)}. The design must "
                    f"call enable_trace(...) so trace lowering appends a trace "
                    f"buffer operand; otherwise the trace buffer has nowhere to "
                    f"land."
                )

        ret = self.run(
            handle,
            list(run_args),
            trace_config=trace_config,
            dispatch_insts=dispatch_insts,
        )

        if trace_config:
            trace_buffer, ctrl_buffer = self.extract_trace_from_args(
                run_args, trace_config
            )
            self.process_trace(trace_buffer, ctrl_buffer, trace_config)

        return handle, ret

    @abstractmethod
    def device(self) -> "Device":
        """Get the device associated with this runtime.

        Returns:
            Device: The device object.
        """
        pass

    # Read instruction stream from bin file and reformat it to be passed into the
    # instruction buffer for the xrt.kernel call
    @classmethod
    def read_insts_binary(cls, insts_path: Path):
        """Read instructions from a binary file.

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
        """Read instructions from the given file.

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
        cls, args: list[NpuTensor], trace_config: TraceConfig
    ) -> list[NpuTensor]:
        """Prepare arguments for tracing by appending necessary buffers.

        Args:
            args (list[NpuTensor]): List of input/output tensors.
            trace_config (TraceConfig): Trace configuration.

        Returns:
            list[NpuTensor]: The updated list of tensors with trace buffers appended.
        """
        if trace_config.reuse_output_buffer:
            # Trace data is written into the tail of the last output buffer.
            # Extend that buffer by the trace size; no new host buffer is added.
            out_size = trace_config.trace_size
            if len(args) > 0:
                out_size += args[-1].nbytes
                # TODO: should really copy previous contents of output into this buffer...? What if it's in/out?
                args[-1] = tensor((out_size,), dtype=np.uint8)
            else:
                out = tensor((out_size,), dtype=np.uint8)
                args.append(out)
        else:
            # Dedicated trace buffer: trace lowering appended one trailing
            # argument to the runtime_sequence, so the host appends exactly one
            # trailing buffer here. The trace buffer lands at index len(args),
            # which matches the appended argument's index by construction -- no
            # positional padding is needed.
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
        cls, args: list[NpuTensor], trace_config: TraceConfig
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Extract trace and control buffers from the arguments.

        Args:
            args (list[NpuTensor]): List of tensors used in execution.
            trace_config (TraceConfig): Trace configuration.

        Returns:
            tuple[np.ndarray, np.ndarray | None]: A tuple containing the trace buffer and optionally the control buffer.
        """
        trace_buff = None
        ctrl_buff = None

        if trace_config.reuse_output_buffer:
            prefix, trace_buff = cls._extract_prefix(
                args[-1], trace_config.last_tensor_shape, trace_config.last_tensor_dtype
            )
            args[-1] = prefix  # pyright: ignore[reportCallIssue, reportArgumentType]
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
        """Separate output data and trace data from a single output buffer stream.

        Args:
            tensor (NpuTensor | np.ndarray): The combined tensor.
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
        """Process the trace buffer and control buffer.

        Args:
            trace_buffer (np.ndarray): The trace data buffer.
            ctrl_buffer (np.ndarray): The control packet buffer.
            trace_config (TraceConfig): Trace configuration.
            verbosity (int, optional): Verbosity level. Defaults to 0.
        """
        logger.debug("trace_buffer shape: %s", trace_buffer.shape)
        logger.debug("trace_buffer dtype: %s", trace_buffer.dtype)
        trace_config.write_trace(trace_buffer)

        if trace_config.enable_ctrl_pkts:
            logger.debug("ctrl_buffer shape: %s", ctrl_buffer.shape)
            logger.debug("ctrl_buffer dtype: %s", ctrl_buffer.dtype)
            logger.debug("ctrl buffer: %s", [hex(d) for d in ctrl_buffer])
            for i in range(ctrl_buffer.size // 2):
                col, row, pkt_type, pkt_id = extract_tile(ctrl_buffer[i * 2])
                overflow = True if (ctrl_buffer[i * 2 + 1] >> 8) == 3 else False
                if overflow:
                    logger.warning(
                        "Trace overflow detected in tile(%d,%d). Trace results may be invalid.",
                        row,
                        col,
                    )

    @classmethod
    def verify_results(cls, io_args, refs=None, verbosity=0):
        """Verify the results of the kernel execution against reference data.

        Args:
            io_args (list[NpuTensor]): List of input/output tensors.
            refs (dict | None, optional): Dictionary mapping index to reference numpy array. Defaults to None (empty dict).
            verbosity (int, optional): Verbosity level. Defaults to 0.

        Returns:
            int: Number of errors found.

        Raises:
            HostRuntimeError: If a reference index is out of bounds.
        """
        if refs is None:
            refs = {}
        errors = 0
        if verbosity >= 1:
            logger.info("Verifying results ...")

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
        """Run a test for the given NPU kernel.

        Args:
            npu_kernel (NPUKernel): The NPU kernel to test.
            io_args (list[NpuTensor]): List of input/output tensors.
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
            logger.info("npu_time: %s us", ret.npu_time / 1000.0)

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
            logger.error("Error count: %d", errors)
            logger.error("Failed.")
            return 1
