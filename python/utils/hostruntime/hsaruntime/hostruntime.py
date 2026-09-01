# hostruntime.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA/ROCR implementation of the HostRuntime.

Consumes the aiecc artifacts ``insts.bin`` + ``main.pdi`` (the xclbin is
ignored on this path) and dispatches them as AIE AQL packets:

    insts.bin + main.pdi -> HSA device heap (hsa_amd_memory_pool_allocate)
    I/O tensors          -> pooled kernarg slot of 2*N uint64 (VAs then sizes)
    fill AQL packet(s), ring doorbell, wait on completion signal

A single ``run`` issues one packet; ``run_chain`` issues N packets that share
one completion signal on the in-order AIE queue (producer -> consumer ordering
via the queue order plus the packets' system-scope fences).
"""

import atexit
import ctypes
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from .context import HSAContext
from .tensor import HSATensor

if TYPE_CHECKING:
    from aie.iron.device import Device

_logger = logging.getLogger(__name__)

_TRACE_UNSUPPORTED_MSG = (
    "Trace capture is not supported on the HSA backend. Re-run without a "
    "trace_config, or use the XRT backend (NPU_RUNTIME=xrt) for trace-enabled "
    "designs."
)

_DEFAULT_EXE_CACHE_SIZE = 32


def _exe_cache_size() -> int:
    """Read the optional HSA_EXE_CACHE_SIZE (LRU cap on loaded designs).

    A malformed value warns and falls back to the default rather than raising,
    mirroring ``_hsa_sync_timeout_s``. Raising here would surface as an opaque
    failure from ``aie.utils.__getattr__`` during runtime construction, far from
    the variable that caused it.
    """
    raw = os.environ.get("HSA_EXE_CACHE_SIZE")
    if raw is None:
        return _DEFAULT_EXE_CACHE_SIZE
    try:
        return int(raw)
    except ValueError:
        _logger.warning(
            "Ignoring invalid HSA_EXE_CACHE_SIZE=%r (want an integer); using %d.",
            raw,
            _DEFAULT_EXE_CACHE_SIZE,
        )
        return _DEFAULT_EXE_CACHE_SIZE


class HSAKernelHandle(KernelHandle):
    """Handle for a loaded HSA kernel (PDI + insts in region memory)."""

    def __init__(self, pdi_ptr, insts_ptr, insts_size):
        self.pdi_ptr = pdi_ptr
        self.insts_ptr = insts_ptr
        self.insts_size = insts_size

    @property
    def needs_dispatch_insts(self) -> bool:
        """No device-side instruction buffer -- so a dispatch design."""
        return self.insts_ptr is None


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
        """Resolve + validate an npu_kernel to (insts_path, pdi_path, name).

        ``insts_path`` is ``None`` for a DispatchTime[T] design -- its
        instruction stream is synthesized fresh per call (see ``run``'s
        ``dispatch_insts`` handling) instead of read from a static insts.bin.
        """
        self.check_device_consistency()
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = (
            Path(npu_kernel.insts_path).resolve() if npu_kernel.insts_path else None
        )
        kernel_name = npu_kernel.kernel_name or "MLIR_AIE"
        if insts_path is not None and (
            not insts_path.exists() or not insts_path.is_file()
        ):
            raise HostRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )
        pdi_path = self._find_pdi(xclbin_path)
        return insts_path, pdi_path, kernel_name

    def _build_handle(self, insts_path, pdi_path) -> HSAKernelHandle:
        """Copy insts (if any) + PDI into fresh device-heap allocations.

        ``insts_path=None`` (a DispatchTime[T] design) allocates only the PDI
        -- there is no static instruction stream to load; ``run()`` builds
        (and frees) a fresh device buffer from ``dispatch_insts`` every call.
        """
        pdi_bytes = pdi_path.read_bytes()
        if insts_path is None:
            pdi_ptr = self._ctx.alloc_dev(len(pdi_bytes))
            ctypes.memmove(pdi_ptr, pdi_bytes, len(pdi_bytes))
            return HSAKernelHandle(pdi_ptr, None, 0)

        insts_bytes = insts_path.read_bytes()
        if len(insts_bytes) % 4 != 0:
            raise HostRuntimeError("insts.bin length is not a multiple of 4 bytes")

        insts_ptr = self._ctx.alloc_dev(len(insts_bytes))
        ctypes.memmove(insts_ptr, insts_bytes, len(insts_bytes))
        try:
            pdi_ptr = self._ctx.alloc_dev(len(pdi_bytes))
            ctypes.memmove(pdi_ptr, pdi_bytes, len(pdi_bytes))
        except BaseException:
            self._ctx.free_dev(insts_ptr)
            raise
        return HSAKernelHandle(pdi_ptr, insts_ptr, len(insts_bytes))

    def _free_handle(self, handle) -> None:
        self._ctx.free_dev(handle.pdi_ptr)
        if handle.insts_ptr is not None:
            self._ctx.free_dev(handle.insts_ptr)

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        insts_path, pdi_path, _ = self._resolve_kernel(npu_kernel)
        handle = self._build_handle(insts_path, pdi_path)
        self._handles.append(handle)
        return handle

    @staticmethod
    def _arg_pairs(kept):
        """(device_va, logical byte size) per tensor, in dispatch order.

        The logical ``nbytes`` (not the granule-rounded allocation size) is what
        the kernarg block must carry, matching ROCR's dispatch.cc.
        """
        return [(t.buffer_object(), t.nbytes) for t in kept]

    def _release_dispatch(self, failed, overflows):
        """Release what a completed dispatch owns; leak what a failed one may still.

        The steady-state path frees nothing: kernargs come from the context's
        fixed slot pool and the completion signal is reused. Only an
        over-capacity argument list allocates.

        Any failure once packets have been rung compromises the shared signal,
        not just a timeout: `dispatch_chain` rings the packets it already wrote
        before propagating a non-timeout error, and those will decrement the
        signal whenever they complete. Reusing it would let the next dispatch's
        wait see somebody else's decrements and return early. So once the device
        holds the signal it is replaced, and the overflow buffers are leaked
        rather than freed, since the device may still read them.

        A failure *before* any doorbell -- a rejected argument, a conversion
        error, a failed kernarg allocation -- never reached the device. Both the
        signal and the buffers are still ours, so both are kept: discarding there
        would leak one signal (a kernel event) per failure, which a caller
        retrying bad arguments in a loop turns into signal exhaustion.
        """
        if failed and self._ctx.signal_in_flight():
            self._ctx.discard_signal()
            return
        for overflow in overflows:
            self._ctx.vmem_free(*overflow)

    def _validate_args(self, args):
        kept = [a for a in args if not callable(a)]
        if not all(isinstance(a, self._tensor_class) for a in kept):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take "
                f"{self._tensor_class.__name__} as arguments, but got: {kept}"
            )
        return kept

    @staticmethod
    def _mark_device_resident(tensors):
        """Record that a completed dispatch wrote these tensors on-device.

        The vmem mapping is CPU+AIE coherent, so unlike XRT and HRX there is no
        stale host copy to invalidate and the sync hooks stay no-ops. The
        residency marker still has to move: ``.device`` is public API, and
        ``NpuTensor``'s ``out=`` check rejects a tensor whose residency does not
        match the one requested. Leaving it at ``cpu`` made the same call
        sequence succeed on XRT/HRX and fail here.
        """
        for t in tensors:
            t.device = "npu"

    def run(
        self,
        kernel_handle,
        args,
        trace_config=None,
        fail_on_error=True,
        only_if_loaded=False,
        dispatch_insts=None,
        **kwargs,
    ) -> HSAKernelResult:
        """Dispatch one packet for ``kernel_handle`` and wait for it to complete.

        ``fail_on_error`` is accepted for API compatibility but not honored:
        HSA always raises on failure via the context's ``_check`` (see the
        _release_dispatch note below for the one path where cleanup is
        intentionally skipped rather than run unconditionally).

        ``dispatch_insts`` (np.ndarray | None): freshly-generated instruction
        words for a DispatchTime[T] design. Its exact size is only known per
        call (see DispatchBridge), so -- unlike XRT's cacheable BO -- this
        allocates a fresh device buffer, copies the words in, dispatches, and
        frees it every call; ``kernel_handle.insts_ptr`` is ``None`` for a
        dispatch design (see ``_build_handle``) and is never touched here.
        """
        assert isinstance(kernel_handle, HSAKernelHandle)
        if trace_config is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        self._require_dispatch_insts(kernel_handle, dispatch_insts)
        self.check_device_consistency()

        kept = self._validate_args(args)
        failed = False
        overflows = []
        dispatch_ptr = None
        signal = self._ctx.arm_signal(1)
        try:
            if dispatch_insts is not None:
                # dispatch_insts is already a fresh contiguous copy owned by
                # this call; memmove reads it directly rather than paying for
                # an intermediate bytes object the size of the stream.
                nbytes = dispatch_insts.nbytes
                dispatch_ptr = self._ctx.alloc_dev(nbytes)
                ctypes.memmove(dispatch_ptr, dispatch_insts.ctypes.data, nbytes)
                insts_ptr, insts_size = dispatch_ptr, nbytes
            else:
                insts_ptr = kernel_handle.insts_ptr
                insts_size = kernel_handle.insts_size

            start = time.perf_counter_ns()
            overflows = self._ctx.dispatch(
                kernel_handle.pdi_ptr,
                insts_ptr,
                insts_size,
                self._arg_pairs(kept),
                signal,
            )
            self._ctx.wait(signal)
            stop = time.perf_counter_ns()
        except BaseException:
            failed = True
            raise
        finally:
            self._release_dispatch(failed, overflows)
            # Same rule _release_dispatch applies to the overflow buffers: a
            # failed dispatch may have reached the device, which can still be
            # reading these words, so leak rather than hand the pages back.
            if dispatch_ptr is not None and not failed:
                self._ctx.free_dev(dispatch_ptr)

        self._mark_device_resident(kept)
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

        # Built before the signal is armed, not inside the try: validation and
        # argument conversion are host-side and reject bad input without ever
        # reaching the device. Arming first would send every such rejection
        # through the failure path, which discards (and leaks) the signal.
        items = []
        tensors = []
        for kernel_handle, args in runs:
            assert isinstance(kernel_handle, HSAKernelHandle)
            kept = self._validate_args(args)
            tensors.extend(kept)
            items.append(
                (
                    kernel_handle.pdi_ptr,
                    kernel_handle.insts_ptr,
                    kernel_handle.insts_size,
                    self._arg_pairs(kept),
                )
            )

        failed = False
        overflows = []
        signal = self._ctx.arm_signal(len(runs))
        try:
            start = time.perf_counter_ns()
            overflows = self._ctx.dispatch_chain(items, signal)
            self._ctx.wait(signal)
            stop = time.perf_counter_ns()
        except BaseException:
            failed = True
            raise
        finally:
            self._release_dispatch(failed, overflows)

        self._mark_device_resident(tensors)
        return HSAKernelResult(stop - start, success=True)

    def load_and_run(self, npu_kernel, run_args, dispatch_scalars=None, **kwargs):
        """Reject trace up front, then defer to the base load/run pipeline.

        The base ``load_and_run`` mutates ``run_args`` (appends a trace buffer
        via ``prepare_args_for_trace``) *before* calling ``run``. HSA cannot
        honor trace, so fail here -- before touching the args -- keeping the
        caller's ``run_args`` untouched on the error path (mirrors HRX).
        """
        if getattr(npu_kernel, "trace_config", None) is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        return super().load_and_run(
            npu_kernel, run_args, dispatch_scalars=dispatch_scalars, **kwargs
        )

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
        self._cache_size = _exe_cache_size()
        atexit.register(self.cleanup)

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        insts_path, pdi_path, kernel_name = self._resolve_kernel(npu_kernel)
        if insts_path is None:
            # DispatchTime[T] design: no static insts.bin to key on, so key on
            # the PDI alone and repeated calls reuse one PDI allocation. run()
            # allocates the instruction stream fresh regardless of this cache.
            key = (str(pdi_path), pdi_path.stat().st_mtime, kernel_name, "dispatch")
        else:
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

        handle = self._build_handle(insts_path, pdi_path)
        if self._cache_size <= 0:
            # Caching disabled. Track the handle so cleanup still frees it,
            # rather than never evicting (which is what a bare `>= size` test
            # would do here, and what a size of 0 previously meant).
            self._handles.append(handle)
            return handle
        while self._exe_cache and len(self._exe_cache) >= self._cache_size:
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
