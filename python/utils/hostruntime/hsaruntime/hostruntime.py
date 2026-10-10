# hostruntime.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA/ROCR implementation of the HostRuntime.

Packs the aiecc artifacts into an hsaco (HSA code object), loads it through
ROCR's code-object loader, and dispatches its kernel object as AIE AQL packets:

    xclbin + insts.bin, or full ELF -> hsaco -> HSA executable -> kernel object
    I/O tensors -> pooled kernarg slot of 2*N uint64 (VAs then sizes)
    fill AQL packet(s), ring doorbell, wait on completion signal

An xclbin contributes its PDI and partition column count, and the kernel is a
PDI plus instruction sequence. A full ELF (aie2p only) carries its own PDIs and
control code. A DispatchTime[T] design has no static instruction sequence: each
distinct per-call sequence is packed and loaded as its own executable.

A single ``run`` issues one packet; ``run_chain`` issues N packets that share
one completion signal on the in-order AIE queue (producer -> consumer ordering
via the queue order plus the packets' system-scope fences).
"""

import atexit
import logging
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from aie.compiler.hsaco import pack

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from ._bindings import HSA_SIGNAL_CONDITION_EQ, HSA_WAIT_STATE_BLOCKED, lib
from .context import HSAContext
from .tensor import HSATensor

if TYPE_CHECKING:
    from aie.iron.device import Device

_logger = logging.getLogger(__name__)

_DEFAULT_EXE_CACHE_SIZE = 32

# Executables a DispatchTime[T] handle keeps loaded, one per distinct per-call
# instruction sequence, so that alternating between a few scalar values does not
# repack and reload on every call.
_MAX_CACHED_STREAMS = 16


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


def _pack_hsaco(arch, kernels):
    """Return the bytes of an hsaco holding ``kernels`` in its ``arch`` section."""
    section = pack.build_section(arch, kernels)
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "kernel.hsaco")
        pack.ensure_hsaco(path)
        pack.inject(path, arch, section)
        return Path(path).read_bytes()


class HSAKernelHandle(KernelHandle):
    """Handle for a loaded HSA kernel.

    ``executable`` is the loaded `HSAExecutable` dispatched by ``run``. A
    DispatchTime[T] design has none: ``stream_kernel`` instead holds what its
    per-call instruction sequences are packed with (name, PDI, column count),
    and ``streams`` the executables already loaded for them, keyed by the
    sequence's bytes, least recently used first.
    """

    def __init__(self, executable, stream_kernel=None):
        super().__init__(needs_dispatch_insts=executable is None)
        self.executable = executable
        self.stream_kernel = stream_kernel
        self.streams = OrderedDict()


class HSAKernelResult(KernelResult):
    """Result wrapper for an HSA dispatch (raises on failure, so success here)."""

    def __init__(self, npu_time, success=True, trace_config=None):
        super().__init__(npu_time, trace_config)
        self._success = success

    def is_success(self) -> bool:
        return self._success


class HSAHostRuntime(HostRuntime):
    """Uncached HostRuntime that dispatches IRON designs through HSA/ROCR.

    Every `load` packs the design into an hsaco and loads it as a fresh HSA
    executable, never reusing one across calls -- the analogue of
    `XRTHostRuntime` / `HRXHostRuntime`. Executables are tracked so
    `cleanup` destroys them; `CachedHSAHostRuntime` layers an LRU cache on
    top for the common single-process case.
    """

    _tensor_class = HSATensor

    def __init__(self):
        self._ctx = HSAContext.get()
        # Handles created by load(), retained so cleanup() destroys their
        # executables (this uncached runtime never reuses one across loads).
        self._handles = []
        # (executable, signal) for every executable a failed dispatch may still
        # be running, and the executables their owner gave up meanwhile; see
        # _release and _reclaim_pending.
        self._in_flight = []
        self._released = []
        self._pending_cleanup_registered = False

    def _resolve_kernel(self, npu_kernel):
        """Resolve + validate an npu_kernel to (source_path, insts_path, name, full_elf).

        ``source_path`` is the full ELF when ``full_elf``, and the xclbin
        otherwise. ``insts_path`` is None for a full ELF, and for a
        DispatchTime[T] design, which has no static instruction sequence.
        """
        self.check_device_consistency()
        kernel_name = npu_kernel.kernel_name or "MLIR_AIE"
        if npu_kernel.elf_path is not None:
            return Path(npu_kernel.elf_path).resolve(), None, kernel_name, True
        xclbin_path = Path(npu_kernel.xclbin_path).resolve()
        insts_path = self._resolve_insts_path(npu_kernel)
        return xclbin_path, insts_path, kernel_name, False

    def _load_hsaco(self, kernel):
        """Pack ``kernel`` into an hsaco of its own and load it, returning the executable."""
        try:
            hsaco = _pack_hsaco(self._ctx.arch, [kernel])
        except (OSError, ValueError, RuntimeError) as e:
            raise HostRuntimeError(
                f"could not pack kernel {kernel['name']!r} into an hsaco: {e}"
            ) from e
        return self._ctx.load_executable(hsaco, kernel["name"])

    def _build_handle(
        self, source_path, insts_path, kernel_name, full_elf
    ) -> HSAKernelHandle:
        """Pack the kernel at ``source_path`` and load it (see `_resolve_kernel`)."""
        if full_elf:
            try:
                (kernel,) = pack.kernels_from_full_elf(
                    str(source_path), names=[kernel_name]
                )
            except (OSError, ValueError) as e:
                raise HostRuntimeError(
                    f"could not read full ELF {source_path}: {e}"
                ) from e
            return HSAKernelHandle(self._load_hsaco(kernel))

        try:
            pdi, num_cols = pack.partition_from_xclbin(str(source_path))
        except (ValueError, RuntimeError) as e:
            raise HostRuntimeError(
                f"could not read the PDI out of {source_path}: {e}"
            ) from e
        if num_cols is None:
            raise HostRuntimeError(
                f"{source_path} does not record its partition's column count"
            )
        kernel = {"name": kernel_name, "pdi": pdi, "num_cols": num_cols}
        if insts_path is None:
            return HSAKernelHandle(None, stream_kernel=kernel)

        insts_bytes = insts_path.read_bytes()
        if len(insts_bytes) % 4 != 0:
            raise HostRuntimeError("insts.bin length is not a multiple of 4 bytes")
        return HSAKernelHandle(self._load_hsaco({**kernel, "insts": insts_bytes}))

    def _stream_executable(self, handle, dispatch_insts):
        """Return the executable for one call of a DispatchTime[T] design.

        Loaded on first use of an instruction sequence and kept in
        ``handle.streams``, evicting the least recently used one.
        """
        key = dispatch_insts.tobytes()
        executable = handle.streams.get(key)
        if executable is not None:
            handle.streams.move_to_end(key)
            return executable
        executable = self._load_hsaco({**handle.stream_kernel, "insts": key})
        while len(handle.streams) >= _MAX_CACHED_STREAMS:
            _, old = handle.streams.popitem(last=False)
            self._release(old)
        handle.streams[key] = executable
        return executable

    def _free_handle(self, handle) -> None:
        if handle.executable is not None:
            self._release(handle.executable)
        while handle.streams:
            self._release(handle.streams.popitem()[1])

    def _track_in_flight(self, executables, signal):
        """Record that a failed dispatch on ``signal`` may still run ``executables``."""
        self._in_flight.extend((executable, signal) for executable in executables)
        if not self._pending_cleanup_registered:
            atexit.register(self._reclaim_at_exit)
            self._pending_cleanup_registered = True

    def _release(self, executable):
        """Destroy an executable its owner no longer needs.

        Every owner -- a handle, its stream cache, the design cache -- gives up
        executables through here, so none is destroyed while a failed dispatch
        may still be running it: such an executable waits in ``_released``
        until `_reclaim_pending` sees that dispatch complete.
        """
        if any(e is executable for e, _ in self._in_flight):
            self._released.append(executable)
        else:
            executable.destroy()

    def _reclaim_pending(self):
        """Forget failed dispatches once they have completed, destroying what they held.

        A drained queue (or destroying/inactivating it) is not proof that a
        dispatched kernel has stopped using its operands. Each retained signal
        belongs to exactly one packet and is never rearmed after publication.
        An acquire observation of zero therefore establishes completion even if
        the original wait raised. A zero timeout is only a hint: always inspect
        the returned value, including on a spurious wakeup.

        Discarded signals are deliberately leaked by the context, so they remain
        valid to inspect here. Do not destroy or reuse them.
        """
        self._in_flight = [
            (executable, signal)
            for executable, signal in self._in_flight
            if lib.hsa_signal_wait_scacquire(
                signal, HSA_SIGNAL_CONDITION_EQ, 0, 0, HSA_WAIT_STATE_BLOCKED
            )
            != 0
        ]
        released, self._released = self._released, []
        for executable in released:
            self._release(executable)
        if not self._in_flight and self._pending_cleanup_registered:
            atexit.unregister(self._reclaim_at_exit)
            self._pending_cleanup_registered = False

    def _reclaim_at_exit(self):
        # Retain this runtime if its caller drops it while a failed dispatch is
        # still running. Unfinished work at process exit remains intentionally
        # allocated; neither interpreter shutdown nor queue destruction proves
        # device completion.
        self._pending_cleanup_registered = False
        self.cleanup()

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        handle = self._build_handle(*self._resolve_kernel(npu_kernel))
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

        ``dispatch_insts`` (np.ndarray | None): Per-call instruction words. Each
        distinct sequence is packed and loaded as its own executable, cached on
        the handle.

        A failed submission the device may still be running keeps its executable
        alive until completion is observed, on a later dispatch or during
        cleanup, even if its owner is freed or evicted meanwhile.
        """
        assert isinstance(kernel_handle, HSAKernelHandle)
        self._require_dispatch_insts(kernel_handle, dispatch_insts)
        self.check_device_consistency()
        self._reclaim_pending()

        kept = self._validate_args(args)
        if dispatch_insts is not None:
            executable = self._stream_executable(kernel_handle, dispatch_insts)
        else:
            executable = kernel_handle.executable
        failed = False
        overflows = []
        signal = self._ctx.arm_signal(1)
        try:
            start = time.perf_counter_ns()
            overflows = self._ctx.dispatch(
                executable.kernel_object, self._arg_pairs(kept), signal
            )
            self._ctx.wait(signal)
            stop = time.perf_counter_ns()
        except BaseException:
            failed = True
            raise
        finally:
            # Checked before _release_dispatch replaces the signal and clears
            # its publication flag. Unpublished failures never reached the
            # device; published ones may still be running this executable.
            if failed and self._ctx.signal_in_flight():
                self._track_in_flight([executable], signal)
            self._release_dispatch(failed, overflows)

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
        auto-batch (wrap-around). ROCR rejects a batch that mixes PDI-plus-
        instruction kernels with full-ELF ones.

        Kernargs are written into the context's fixed slot pool as each packet's
        ring slot is reserved -- never all up front, since a chain longer than
        the queue reuses slots.

        ``fail_on_error`` is accepted for API compatibility but not honored:
        HSA always raises on failure via the context's ``_check``.
        """
        self.check_device_consistency()
        self._reclaim_pending()
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
            self._require_dispatch_insts(kernel_handle, None)
            kept = self._validate_args(args)
            tensors.extend(kept)
            items.append(
                (kernel_handle.executable.kernel_object, self._arg_pairs(kept))
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
            if failed and self._ctx.signal_in_flight():
                self._track_in_flight([h.executable for h, _ in runs], signal)
            self._release_dispatch(failed, overflows)

        self._mark_device_resident(tensors)
        return HSAKernelResult(stop - start, success=True)

    def device(self) -> "Device":
        from aie.iron.device import from_name

        return from_name(self._ctx.device_gen, n_cols=None)

    def cleanup(self) -> None:
        """Destroy the executables this runtime loaded."""
        self._reclaim_pending()
        handles = getattr(self, "_handles", None)
        if not handles:
            return
        while handles:
            self._free_handle(handles.pop())


class CachedHSAHostRuntime(HSAHostRuntime):
    """HSA runtime that caches loaded kernels (analogue of CachedXRTRuntime).

    Reuses a handle's executable across `load` calls for the same artifacts,
    evicting the least-recently-used entry once ``HSA_EXE_CACHE_SIZE``
    (default 32) is exceeded. Registers an ``atexit`` cleanup so cached
    executables are destroyed at interpreter shutdown.
    """

    def __init__(self):
        super().__init__()
        self._exe_cache = OrderedDict()
        self._cache_size = _exe_cache_size()
        atexit.register(self.cleanup)

    def load(self, npu_kernel, **kwargs) -> HSAKernelHandle:
        source_path, insts_path, kernel_name, full_elf = self._resolve_kernel(
            npu_kernel
        )
        # With no insts.bin, the key rests on the xclbin or full ELF alone, so
        # repeated calls share one handle -- and a DispatchTime[T] design its
        # cached per-call executables.
        key = (
            str(source_path),
            source_path.stat().st_mtime,
            str(insts_path) if insts_path else None,
            insts_path.stat().st_mtime if insts_path else None,
            kernel_name,
        )
        if key in self._exe_cache:
            self._exe_cache.move_to_end(key)
            return self._exe_cache[key]

        handle = self._build_handle(source_path, insts_path, kernel_name, full_elf)
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
        self._reclaim_pending()
        cache = getattr(self, "_exe_cache", None)
        if cache:
            while cache:
                _, handle = cache.popitem(last=False)
                self._free_handle(handle)
        super().cleanup()
