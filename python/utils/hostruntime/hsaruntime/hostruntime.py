# hostruntime.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA/ROCR implementation of the HostRuntime.

Consumes the aiecc artifacts ``insts.bin`` + ``main.pdi`` (the xclbin is
ignored on this path) and dispatches them as AIE AQL packets:

    insts.bin + main.pdi -> HSA device heap (hsa_amd_memory_pool_allocate)
    I/O tensors          -> kernarg buffer of 2*N uint64 (VAs then sizes)
    fill AQL packet(s), ring doorbell, wait on completion signal

A single ``run`` issues one packet; ``run_chain`` issues N packets that share
one completion signal on the in-order AIE queue (producer -> consumer ordering
via the queue order plus the packets' system-scope fences).
"""

import atexit
import ctypes
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from .tensor import HSATensor
from .context import HSAContext
from ._bindings import HSATimeoutError

if TYPE_CHECKING:
    from aie.iron.device import Device

_TRACE_UNSUPPORTED_MSG = (
    "Trace capture is not supported on the HSA backend. Re-run without a "
    "trace_config, or use the XRT backend (NPU_RUNTIME=xrt) for trace-enabled "
    "designs."
)


class HSAKernelHandle(KernelHandle):
    """Handle for a loaded HSA kernel (PDI + insts in region memory)."""

    def __init__(self, pdi_ptr, insts_ptr, insts_size, kernel_name):
        self.pdi_ptr = pdi_ptr
        self.insts_ptr = insts_ptr
        self.insts_size = insts_size
        self.kernel_name = kernel_name


class HSAKernelResult(KernelResult):
    """Result wrapper for an HSA dispatch (raises on failure, so success here)."""

    def __init__(self, npu_time, success=True, trace_config=None):
        super().__init__(npu_time, trace_config)
        self._success = success

    def is_success(self) -> bool:
        return self._success


class HSAHostRuntime(HostRuntime):
    """Uncached HostRuntime that dispatches IRON designs through HSA/ROCR.

    Every :meth:`load` copies the design's insts + PDI into fresh HSA region
    allocations and never reuses them across calls -- the analogue of
    :class:`XRTHostRuntime` / :class:`HRXHostRuntime`. Allocations are tracked
    so :meth:`cleanup` frees them; :class:`CachedHSAHostRuntime` layers an LRU
    cache on top for the common single-process case.
    """

    _tensor_class = HSATensor

    def __init__(self):
        self._ctx = HSAContext.get()
        # Handles created by load(), retained so cleanup() frees their region
        # allocations (this uncached runtime never reuses one across loads).
        self._handles = []

    def _find_pdi(self, xclbin_path: Path) -> Path:
        kernel_dir = xclbin_path.parent
        main_pdi = kernel_dir / "main.pdi"
        if main_pdi.is_file():
            return main_pdi
        pdis = sorted(kernel_dir.glob("*.pdi"))
        if pdis:
            return pdis[0]
        raise HostRuntimeError(
            f"No PDI (main.pdi) found in {kernel_dir}. The HSA backend needs a "
            f"PDI; ensure aiecc emitted one alongside the xclbin."
        )

    def _resolve_kernel(self, npu_kernel):
        """Resolve + validate an npu_kernel to (insts_path, pdi_path, name)."""
        self.check_device_consistency()
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = Path(npu_kernel.insts_path).resolve()
        kernel_name = npu_kernel.kernel_name or "MLIR_AIE"
        if not insts_path.exists() or not insts_path.is_file():
            raise HostRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )
        pdi_path = self._find_pdi(xclbin_path)
        return insts_path, pdi_path, kernel_name

    def _build_handle(self, insts_path, pdi_path, kernel_name) -> HSAKernelHandle:
        """Copy insts + PDI into fresh device-heap allocations and wrap in a handle."""
        insts_bytes = insts_path.read_bytes()
        if len(insts_bytes) % 4 != 0:
            raise HostRuntimeError("insts.bin length is not a multiple of 4 bytes")
        pdi_bytes = pdi_path.read_bytes()

        insts_ptr = self._ctx.alloc_dev(len(insts_bytes))
        ctypes.memmove(insts_ptr, insts_bytes, len(insts_bytes))
        try:
            pdi_ptr = self._ctx.alloc_dev(len(pdi_bytes))
            ctypes.memmove(pdi_ptr, pdi_bytes, len(pdi_bytes))
        except BaseException:
            self._ctx.free_dev(insts_ptr)
            raise
        return HSAKernelHandle(pdi_ptr, insts_ptr, len(insts_bytes), kernel_name)

    def _free_handle(self, handle) -> None:
        self._ctx.free_dev(handle.pdi_ptr)
        self._ctx.free_dev(handle.insts_ptr)

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        insts_path, pdi_path, kernel_name = self._resolve_kernel(npu_kernel)
        handle = self._build_handle(insts_path, pdi_path, kernel_name)
        self._handles.append(handle)
        return handle

    @staticmethod
    def _arg_pairs(kept):
        """(device_va, logical byte size) per tensor, in dispatch order.

        The logical ``nbytes`` (not the granule-rounded allocation size) is what
        the kernarg block must carry, matching ROCR's dispatch.cc."""
        return [(t.buffer_object(), t.nbytes) for t in kept]

    def _validate_args(self, args):
        kept = [a for a in args if not callable(a)]
        if not all(isinstance(a, self._tensor_class) for a in kept):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take "
                f"{self._tensor_class.__name__} as arguments, but got: {kept}"
            )
        return kept

    def run(
        self,
        kernel_handle,
        args,
        trace_config=None,
        fail_on_error=True,
        only_if_loaded=False,
        **kwargs,
    ) -> HSAKernelResult:
        """``fail_on_error`` is accepted for API compatibility but not honored:
        HSA always raises on failure via the context's ``_check`` (see the
        HSATimeoutError leak-on-timeout note below for the one path where
        cleanup is intentionally skipped rather than run unconditionally)."""
        assert isinstance(kernel_handle, HSAKernelHandle)
        if trace_config is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        self.check_device_consistency()

        kept = self._validate_args(args)
        timed_out = False
        overflows = []
        signal = self._ctx.arm_signal(1)
        try:
            start = time.time_ns()
            overflows = self._ctx.dispatch(
                kernel_handle.pdi_ptr,
                kernel_handle.insts_ptr,
                kernel_handle.insts_size,
                self._arg_pairs(kept),
                signal,
            )
            self._ctx.wait(signal)
            stop = time.time_ns()
        except HSATimeoutError:
            timed_out = True
            raise
        finally:
            # Kernargs normally live in the context's fixed slot pool and the
            # completion signal is reused across dispatches, so the success path
            # frees nothing; only an over-capacity argument list allocates. On a
            # timeout the dispatch is wedged on the device with the packet
            # in-flight, so the device still owns the shared signal (it may
            # decrement it later) and any overflow buffer: replace the signal
            # rather than reuse it, and leak the buffer rather than free it (see
            # HSATimeoutError docstring).
            if timed_out:
                self._ctx.discard_signal()
            else:
                for overflow in overflows:
                    self._ctx.vmem_free(*overflow)

        return HSAKernelResult(stop - start, success=True)

    def run_chain(self, runs, fail_on_error: bool = True) -> HSAKernelResult:
        """Execute a chain of dispatches sharing one completion signal.

        ``runs`` is a sequence of ``(kernel_handle, args)`` entries recorded, in
        order, onto the single in-order AIE queue. One completion signal is
        initialized to ``len(runs)``; each completed packet decrements it, so a
        single wait covers the whole chain. Ordering (producer -> consumer) is
        guaranteed by the in-order queue plus the system-scope acquire/release
        fences in every packet header. Chains longer than the queue capacity
        auto-batch (wrap-around).

        Kernargs are written into the context's fixed slot pool as each packet's
        ring slot is reserved -- never all up front, since a chain longer than
        the queue reuses slots.

        ``fail_on_error`` is accepted for API compatibility but not honored:
        HSA always raises on failure via the context's ``_check``.
        """
        self.check_device_consistency()
        runs = list(runs)
        if not runs:
            return HSAKernelResult(0, success=True)

        items = []  # (pdi_ptr, insts_ptr, insts_size, arg_pairs)
        timed_out = False
        overflows = []
        signal = self._ctx.arm_signal(len(runs))
        try:
            for kernel_handle, args in runs:
                assert isinstance(kernel_handle, HSAKernelHandle)
                kept = self._validate_args(args)
                items.append(
                    (
                        kernel_handle.pdi_ptr,
                        kernel_handle.insts_ptr,
                        kernel_handle.insts_size,
                        self._arg_pairs(kept),
                    )
                )

            start = time.time_ns()
            overflows = self._ctx.dispatch_chain(items, signal)
            self._ctx.wait(signal)
            stop = time.time_ns()
        except HSATimeoutError:
            timed_out = True
            raise
        finally:
            # Kernargs normally come from the context's fixed slot pool, written
            # per packet as its ring slot is reserved, and the completion signal is
            # reused across dispatches; only over-capacity argument lists allocate.
            # On a timeout the chain is wedged on the device with packets
            # in-flight -- and a partly-enqueued chain leaves the shared signal
            # stuck above 0 -- so replace the signal rather than reuse it, and leak
            # any overflow buffers (see HSATimeoutError docstring).
            if timed_out:
                self._ctx.discard_signal()
            else:
                for overflow in overflows:
                    self._ctx.vmem_free(*overflow)

        return HSAKernelResult(stop - start, success=True)

    def load_and_run(self, npu_kernel, run_args, **kwargs):
        """Reject trace up front, then defer to the base load/run pipeline.

        The base ``load_and_run`` mutates ``run_args`` (appends a trace buffer
        via ``prepare_args_for_trace``) *before* calling ``run``. HSA cannot
        honor trace, so fail here -- before touching the args -- keeping the
        caller's ``run_args`` untouched on the error path (mirrors HRX).
        """
        if getattr(npu_kernel, "trace_config", None) is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        return super().load_and_run(npu_kernel, run_args, **kwargs)

    def device(self) -> "Device":
        from aie.iron.device import from_name

        return from_name(self._ctx.device_gen, n_cols=None)

    def cleanup(self) -> None:
        """Free the region allocations this runtime created."""
        handles = getattr(self, "_handles", None)
        if not handles:
            return
        while handles:
            self._free_handle(handles.pop())


class CachedHSAHostRuntime(HSAHostRuntime):
    """HSA runtime that caches loaded kernels (analogue of CachedXRTRuntime).

    Reuses a handle's region allocations across :meth:`load` calls for the same
    artifacts, evicting the least-recently-used entry once ``HSA_EXE_CACHE_SIZE``
    (default 32) is exceeded. Registers an ``atexit`` cleanup so cached
    allocations are freed at interpreter shutdown.
    """

    def __init__(self):
        super().__init__()
        self._exe_cache = OrderedDict()
        self._cache_size = int(os.environ.get("HSA_EXE_CACHE_SIZE", "32"))
        atexit.register(self.cleanup)

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        insts_path, pdi_path, kernel_name = self._resolve_kernel(npu_kernel)
        key = (
            str(insts_path),
            insts_path.stat().st_mtime,
            str(pdi_path),
            pdi_path.stat().st_mtime,
            kernel_name,
        )
        if key in self._exe_cache:
            self._exe_cache.move_to_end(key)
            return self._exe_cache[key]

        handle = self._build_handle(insts_path, pdi_path, kernel_name)
        if self._cache_size > 0 and len(self._exe_cache) >= self._cache_size:
            _, old = self._exe_cache.popitem(last=False)
            self._free_handle(old)
        self._exe_cache[key] = handle
        return handle

    def cleanup(self) -> None:
        """Free cached handles, then any tracked by the base runtime."""
        cache = getattr(self, "_exe_cache", None)
        if cache:
            while cache:
                _, handle = cache.popitem(last=False)
                self._free_handle(handle)
        super().cleanup()
