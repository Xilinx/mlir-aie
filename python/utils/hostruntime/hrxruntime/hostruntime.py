# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""HRX-based implementation of the HostRuntime.

It consumes the ``aiecc`` artifacts (``final.xclbin`` + ``insts.bin``) and
dispatches them through ``libhrx``:

    insts.bin words -> transaction        (one executable, cached by content)
    I/O tensors     -> bindings           (binding order = DDR-patch arg index)
    hrx_stream_dispatch(...) + hrx_stream_synchronize(...)

libhrx patches the buffer addresses into the control code from binding order +
the TXN's own DDR-patch ops (npu4 COMMAND_CHAIN path).

Producer-independent DDR-patch ABI: the ``insts.bin`` consumed here is compiled
*unfolded* (aiecc ``--fold-ddr-addr-offset=false``; the JIT picks this whenever
NPU_RUNTIME=hrx), so each DDR patch's ``arg_plus`` carries only the raw
intra-buffer offset. libhrx adds the AIE DDR aperture offset (0x80000000) for
every argument itself, exactly once -- it does not depend on the firmware's
first-5-args translation cutoff. This is what lets designs with more than five
host buffers dispatch correctly (the XRT/instruction-buffer path instead folds
the aperture offset into ``arg_plus`` for args >= 5 to match its firmware).
"""

import atexit
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from ..hostruntime import HostRuntime, HostRuntimeError, KernelHandle, KernelResult
from . import HRXContext, HRXError, control_code_from_elf
from .tensor import HRXTensor

if TYPE_CHECKING:
    from aie.iron.device import Device

logger = logging.getLogger(__name__)

# amdxdna NPU PCI device IDs (vendor 0x1022 = AMD), used to infer the device
# generation without XRT. Extend as new silicon ships.
_AMD_PCI_VENDOR = "0x1022"
_PHOENIX_PCI_IDS = {"0x1502"}  # Phoenix -> npu1
_STRIX_PCI_IDS = {"0x17f0", "0x17f1", "0x1640", "0x1641"}  # Strix/Krackan -> npu2


def _detect_hrx_device_gen() -> str:
    """Best-effort amdxdna device-generation detection (npu1/npu2).

    Order:
      1. ``IRON_HRX_DEVICE`` env override (always wins);
      2. sysfs PCI device-id probe (amdxdna-bound device, else any AMD NPU id);
      3. fall back to ``npu2`` (the common XDNA2 case) with a debug log.

    Fully offline and XRT-free (no dlopen, no device init), so it is safe to call
    from the runtime constructor.
    """
    env = os.environ.get("IRON_HRX_DEVICE")
    if env:
        return env

    def _pci_ids():
        ids = []
        drv = Path("/sys/bus/pci/drivers/amdxdna")
        try:
            if drv.is_dir():
                for entry in drv.iterdir():
                    dev_file = entry / "device"
                    if dev_file.is_file():
                        ids.append(dev_file.read_text().strip().lower())
        except OSError:
            pass
        if ids:
            return ids
        # Driver not bound / not found: scan AMD PCI devices for a known NPU id.
        try:
            for dev_file in Path("/sys/bus/pci/devices").glob("*/device"):
                vendor_file = dev_file.with_name("vendor")
                try:
                    vendor = vendor_file.read_text().strip().lower()
                except OSError:
                    continue
                if vendor == _AMD_PCI_VENDOR:
                    ids.append(dev_file.read_text().strip().lower())
        except OSError:
            pass
        return ids

    try:
        ids = _pci_ids()
        if any(i in _PHOENIX_PCI_IDS for i in ids):
            return "npu1"
        if any(i in _STRIX_PCI_IDS for i in ids):
            return "npu2"
    except Exception as e:  # detection must never break runtime construction
        logger.debug("HRX device auto-detect failed: %s", e)

    logger.debug("HRX device generation not detected; defaulting to npu2")
    return "npu2"


class HRXKernelHandle(KernelHandle):
    """Handle for a loaded HRX executable (one XADX export)."""

    def __init__(
        self, executable, export_ordinal, kernel_name, xclbin_path, insts_path, ctx=None
    ):
        self.executable = executable
        self.export_ordinal = export_ordinal
        self.kernel_name = kernel_name
        self.xclbin_path = xclbin_path
        self.insts_path = insts_path
        self._xclbin_bytes: bytes | None = None
        # Own an independent libhrx reference to the executable. The executable
        # cache holds only a single reference and drops it on LRU eviction; a
        # live handle (e.g. every step of a batched run_chain, kept in the
        # sequence callable for the whole dispatch) must not be left dangling
        # when an unrelated load evicts its cache entry. Balanced in __del__.
        self._ctx = ctx
        if ctx is not None and executable:
            ctx.retain_executable(executable)

    def __del__(self):
        ctx = getattr(self, "_ctx", None)
        exe = getattr(self, "executable", None)
        if ctx is not None and exe:
            try:
                ctx.release_executable(exe)
            except Exception:
                pass
            self.executable = None

    @property
    def needs_dispatch_insts(self) -> bool:
        """No prebuilt executable -- so a dispatch design (run() builds one)."""
        return self.executable is None

    def xclbin_image(self) -> bytes:
        """Return the xclbin bytes, read once and kept for later dispatches.

        A dispatch design rebuilds its executable every call (libhrx has no
        way to swap an existing one's transaction bytes), but the xclbin
        behind it never changes -- so only the executable needs rebuilding,
        not the file read.
        """
        if self._xclbin_bytes is None:
            self._xclbin_bytes = Path(self.xclbin_path).read_bytes()
        return self._xclbin_bytes


class HRXKernelResult(KernelResult):
    """Result wrapper for an HRX dispatch.

    HRX raises (via ``_check``) on a non-OK dispatch/sync, so reaching
    construction means the run completed.
    """

    def __init__(self, npu_time, success=True, trace_config=None):
        super().__init__(npu_time, trace_config)
        self._success = success

    def is_success(self) -> bool:
        return self._success


_TRACE_UNSUPPORTED_MSG = (
    "Trace capture is not supported on the HRX backend. Re-run without a "
    "trace_config, or use the XRT backend (NPU_RUNTIME=xrt) for trace-enabled "
    "designs."
)


class HRXHostRuntime(HostRuntime):
    """Uncached HostRuntime that dispatches IRON designs through HRX.

    Every :meth:`load` builds a fresh amdxdna executable and never reuses one
    across calls -- the analogue of :class:`XRTHostRuntime`. On shared systems
    where holding onto device executables is undesirable this is the runtime to
    pick; :class:`CachedHRXRuntime` layers an LRU executable cache on top for
    the common single-process case. Created executables are tracked so
    :meth:`cleanup` can release them.
    """

    _tensor_class = HRXTensor

    def __init__(self):
        self._ctx = HRXContext.get()
        # Executables created by load(), retained only so cleanup() can release
        # them (this uncached runtime never reuses one across load() calls).
        self._executables = []
        # Device generation (npu1/npu2). Detected from the amdxdna device when
        # possible so a Phoenix box is not silently mislabeled as Strix; the
        # IRON_HRX_DEVICE env var always overrides.
        self._device_gen = _detect_hrx_device_gen()

    def _resolve_kernel(self, npu_kernel):
        """Resolve + validate an npu_kernel to (xclbin_path, insts_path, name).

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

        if not xclbin_path.exists() or not xclbin_path.is_file():
            raise HostRuntimeError(
                f"xclbin {xclbin_path} does not exist or is not a file."
            )
        if insts_path is not None and (
            not insts_path.exists() or not insts_path.is_file()
        ):
            raise HostRuntimeError(
                f"insts {insts_path} does not exist or is not a file."
            )
        return xclbin_path, insts_path, kernel_name

    def _create_executable_from_bytes(self, xclbin_bytes, insts_data, kernel_name):
        """Create + look up a fresh amdxdna executable from raw artifact bytes.

        Shared by the static path (``_build_executable``, file-backed) and
        the DispatchTime[T] path (``run()``, backed by this call's
        freshly-generated instruction words) -- both need the same
        create+lookup sequence, only the byte source differs.
        """
        # libhrx builds the amdxdna XADX package and derives the patch table
        # from the XAie transaction internally, so we just hand it the raw
        # artifacts. The transaction is the raw insts.bin TXN words; for an ELF
        # input (aiecc --aie-generate-elf) we extract .ctrltext (the TXN verbatim)
        # so libhrx still sees the BLOCKWRITE/DDR_PATCH ops it patches from.
        if insts_data[:4] == b"\x7fELF":
            insts_bytes = control_code_from_elf(insts_data).tobytes()
        else:
            insts_bytes = insts_data
        try:
            exe = self._ctx.create_executable(xclbin_bytes, insts_bytes, kernel_name)
            ordv = self._ctx.lookup_export(exe, kernel_name)
        except HRXError as e:
            raise HostRuntimeError(f"HRX failed to load kernel: {e}") from e
        return exe, ordv

    def _build_executable(self, xclbin_path, insts_path, kernel_name):
        """Create + look up a fresh amdxdna executable from the raw artifacts (file-backed)."""
        xclbin_bytes = xclbin_path.read_bytes()
        insts_data = insts_path.read_bytes()
        return self._create_executable_from_bytes(xclbin_bytes, insts_data, kernel_name)

    def load(self, npu_kernel, **kwargs) -> HRXKernelHandle:
        """Build a fresh amdxdna executable for ``npu_kernel``.

        Args:
            npu_kernel (NPUKernel): The kernel to load; its ``xclbin_path`` /
                ``insts_path`` / ``kernel_name`` are read and validated.
            **kwargs: Accepted for API compatibility; ignored by HRX.

        Returns:
            HRXKernelHandle: A handle wrapping the loaded executable and its
            resolved export ordinal. For a DispatchTime[T] design (no static
            insts.bin), ``executable``/``export_ordinal`` are left ``None`` --
            ``run()`` builds (and releases) a fresh executable from each
            call's freshly-generated instruction words instead. libhrx bakes
            the transaction into an immutable executable object at creation
            time, with no lower-level "patch the bytes on an existing
            executable" primitive, so this is a real per-call cost on this
            backend, unlike XRT (rebuilds a BO) or HSA (memmoves into a fresh
            device buffer) -- both far cheaper than recreating a whole
            executable.

        Raises:
            HostRuntimeError: If the artifacts are missing or libhrx fails to
                create/resolve the executable.
        """
        xclbin_path, insts_path, kernel_name = self._resolve_kernel(npu_kernel)
        if insts_path is None:
            return HRXKernelHandle(
                None, None, kernel_name, xclbin_path, None, ctx=self._ctx
            )
        exe, ordv = self._build_executable(xclbin_path, insts_path, kernel_name)
        self._executables.append(exe)
        return HRXKernelHandle(
            exe, ordv, kernel_name, xclbin_path, insts_path, ctx=self._ctx
        )

    def _prepare_bindings(self, args):
        """Validate/sync a run's args and return its HRX dispatch bindings.

        Drops callables (the ``@iron.jit`` trailing kernel ref), checks every
        remaining arg is an ``HRXTensor``, pushes host-side inputs to the device
        (a cheap ``flush_range`` clflush on the persistent mapping — no copy),
        and returns both the kept tensors and the ``(buffer, size)`` bindings.

        Sync cost note (review r3623783388): this layer does not know an arg's
        direction (input / output / in-out), so it is deliberately conservative
        both ways. It flushes *every* binding host->device here, and :meth:`run`
        marks *every* binding device-resident afterwards so the next host read
        invalidates it. For a pure input, the post-run mark forces one extra
        device->host *invalidate* on next access. Both directions are cheap cache
        maintenance (clflush / cache-line invalidate) over a single persistent
        host-coherent mapping — never a host<->device copy — so the redundant op
        on a pure input is a bounded, no-copy cost. Tracking direction would
        require plumbing arg intents from the design down to dispatch, which is
        the same conservative trade-off the XRT runtime makes today.
        """
        kept = [a for a in args if not callable(a)]
        if not all(isinstance(a, self._tensor_class) for a in kept):
            raise HostRuntimeError(
                f"The {self.__class__.__name__} can only take "
                f"{self._tensor_class.__name__} as arguments, but got: {kept}"
            )
        for a in kept:
            a.to("npu")
            a._sync_to_device()
        bindings = [(a.buffer_object(), a.nbytes_alloc()) for a in kept]
        return kept, bindings

    def run(
        self,
        kernel_handle: KernelHandle,
        args,
        trace_config=None,
        fail_on_error: bool = True,
        only_if_loaded: bool = False,
        dispatch_insts=None,
        **kwargs,
    ) -> HRXKernelResult:
        """Dispatch a single loaded kernel and wait for it to finish.

        Host-side inputs are flushed to the device, the executable is dispatched
        and synchronized, and every argument is left marked device-resident so
        the next host read invalidates and observes the results.

        Args:
            kernel_handle (HRXKernelHandle): Handle from :meth:`load`.
            args: The kernel arguments (``HRXTensor`` instances; a trailing
                callable, as ``@iron.jit`` appends, is ignored).
            trace_config (optional): Must be ``None`` -- HRX has no trace capture.
            fail_on_error (bool, optional): Raise on a failed dispatch instead of
                returning an unsuccessful result. Defaults to True.
            only_if_loaded (bool, optional): Accepted for API compatibility.
            dispatch_insts (np.ndarray | None, optional): Freshly-generated
                instruction words for a DispatchTime[T] design. When set,
                builds a fresh amdxdna executable from these words (since
                ``kernel_handle.executable`` is ``None`` for a dispatch
                design -- see ``load``), dispatches it, and releases it
                before returning. This re-does load()-equivalent work on
                *every* dynamic call -- a real, structural per-call cost
                specific to this backend (see ``load``'s docstring).
            **kwargs: Accepted for API compatibility; ignored by HRX.

        Returns:
            HRXKernelResult: Wraps the elapsed dispatch time and success flag.

        Raises:
            HostRuntimeError: If ``trace_config`` is set, or the dispatch fails
                and ``fail_on_error`` is True.
        """
        assert isinstance(kernel_handle, HRXKernelHandle)
        # HRX does not implement trace capture; fail loudly rather than silently
        # ignoring the request (which would return a misleading success with no
        # trace). Matches the C++ wrapper's reject_unsupported_features.
        if trace_config is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        # After the trace check: an unsupported feature is the more useful
        # complaint when a caller asks for both at once.
        self._require_dispatch_insts(kernel_handle, dispatch_insts)
        self.check_device_consistency()

        args, bindings = self._prepare_bindings(args)

        exe, ordv = kernel_handle.executable, kernel_handle.export_ordinal
        dispatch_exe = None
        if dispatch_insts is not None:
            dispatch_exe, ordv = self._create_executable_from_bytes(
                kernel_handle.xclbin_image(),
                dispatch_insts.tobytes(),
                kernel_handle.kernel_name,
            )
            exe = dispatch_exe

        try:
            start = time.perf_counter_ns()
            try:
                self._ctx.dispatch(exe, ordv, bindings)
                self._ctx.synchronize()
            except HRXError as e:
                if fail_on_error:
                    raise HostRuntimeError(f"HRX dispatch failed: {e}") from e
                stop = time.perf_counter_ns()
                return HRXKernelResult(stop - start, success=False)
            stop = time.perf_counter_ns()

            # Outputs were written on-device, so the host mapping is stale.
            # Leave the tensors marked device="npu" so the next host read
            # invalidates the cache via _sync_from_device.
            for a in args:
                a.device = "npu"

            return HRXKernelResult(stop - start, success=True)
        finally:
            if dispatch_exe is not None:
                self._release_executable(dispatch_exe)

    def run_chain(self, runs, fail_on_error: bool = True) -> HRXKernelResult:
        """Execute a chain (runlist) of dispatches as a single batched submit.

        ``runs`` is a sequence of ``(kernel_handle, args)`` entries that are
        recorded, in order, into one HRX command buffer with an execution +
        memory barrier between them, then
        submitted with a single ``synchronize``. Because of the barrier, a later
        run observes an earlier run's device writes, so producer -> consumer
        chains work (e.g. ``run0`` writes ``out0`` and ``run1`` reads ``out0``).
        The amdxdna HAL lowers the multi-dispatch command buffer into one
        ``ERT_CMD_CHAIN`` issued/waited once.

        All entries may share one ``kernel_handle`` (re-dispatching the same
        executable with different bindings) or use different handles (a true
        multi-kernel pipeline).

        Args:
            runs: A sequence of ``(kernel_handle, args)`` entries, recorded in
                order. Each ``kernel_handle`` is an :class:`HRXKernelHandle` and
                ``args`` are ``HRXTensor`` instances (a trailing callable is
                ignored, as in :meth:`run`).
            fail_on_error (bool, optional): Raise on a failed chain dispatch
                instead of returning an unsuccessful result. Defaults to True.

        Returns:
            HRXKernelResult: One result covering the whole chain (elapsed time +
            success flag). An empty ``runs`` returns a successful zero-time
            result.

        Raises:
            HostRuntimeError: If the chain dispatch fails and ``fail_on_error``
                is True.
        """
        self.check_device_consistency()
        runs = list(runs)
        if not runs:
            return HRXKernelResult(0, success=True)

        # Record everything first: all host->device flushes happen here, before
        # any device execution, so flushing an intermediate buffer that an
        # earlier run overwrites on-device is harmless.
        items = []
        touched = []
        for kernel_handle, args in runs:
            assert isinstance(kernel_handle, HRXKernelHandle)
            kept, bindings = self._prepare_bindings(args)
            items.append(
                (kernel_handle.executable, kernel_handle.export_ordinal, bindings)
            )
            touched.extend(kept)

        start = time.perf_counter_ns()
        try:
            self._ctx.dispatch_chain(items)
            self._ctx.synchronize()
        except HRXError as e:
            if fail_on_error:
                raise HostRuntimeError(f"HRX chain dispatch failed: {e}") from e
            stop = time.perf_counter_ns()
            return HRXKernelResult(stop - start, success=False)
        stop = time.perf_counter_ns()

        # Mark every touched tensor device-resident so the next host read
        # invalidates and observes the on-device results.
        for a in touched:
            a.device = "npu"

        return HRXKernelResult(stop - start, success=True)

    def load_and_run(self, npu_kernel, run_args, dispatch_scalars=None, **kwargs):
        """Reject trace up front, then defer to the base load/run pipeline.

        The base ``load_and_run`` mutates ``run_args`` (appends a trace buffer
        via ``prepare_args_for_trace``) *before* calling ``run``. HRX cannot
        honor trace, so we fail here -- before touching the args -- instead of
        after, keeping the caller's ``run_args`` untouched on the error path.
        """
        if getattr(npu_kernel, "trace_config", None) is not None:
            raise HostRuntimeError(_TRACE_UNSUPPORTED_MSG)
        return super().load_and_run(
            npu_kernel, run_args, dispatch_scalars=dispatch_scalars, **kwargs
        )

    def device(self) -> "Device":
        from aie.iron.device import from_name

        return from_name(self._device_gen, n_cols=None)

    def _release_executable(self, exe) -> None:
        """Release one executable back to HRX, swallowing release errors."""
        try:
            self._ctx.release_executable(exe)
        except HRXError as e:
            logger.debug("HRX executable release failed during cleanup: %s", e)

    def cleanup(self) -> None:
        """Release the executables this runtime created.

        Invoked by the shared ``aie.utils.cleanup_npu_runtime`` entry point.
        Each executable is released back to HRX; the process-wide device/stream
        owned by :class:`HRXContext` is intentionally left intact (it is a
        shared singleton that other runtimes/tensors may still use and is torn
        down by libhrx at process exit).
        """
        executables = getattr(self, "_executables", None)
        if not executables:
            return
        while executables:
            self._release_executable(executables.pop())


class CachedHRXRuntime(HRXHostRuntime):
    """HRX runtime that caches loaded executables (analogue of CachedXRTRuntime).

    Unlike the uncached :class:`HRXHostRuntime`, this reuses an amdxdna
    executable across :meth:`load` calls for the same artifacts, evicting the
    least-recently-used entry once ``HRX_EXE_CACHE_SIZE`` (default 32) is
    exceeded. It also registers an ``atexit`` cleanup (as ``CachedXRTRuntime``
    does) so cached executables are released on interpreter shutdown.
    """

    def __init__(self):
        super().__init__()
        # Executable cache keyed by (xclbin_path, xclbin_mtime, insts_path,
        # insts_mtime, kernel_name).
        self._exe_cache = OrderedDict()
        self._cache_size = int(os.environ.get("HRX_EXE_CACHE_SIZE", "32"))
        atexit.register(self.cleanup)

    def load(self, npu_kernel, **kwargs) -> HRXKernelHandle:
        xclbin_path, insts_path, kernel_name = self._resolve_kernel(npu_kernel)

        if insts_path is None:
            # DispatchTime[T] design: nothing stable to cache here -- run()
            # builds and releases a fresh executable from each call's
            # instruction words (see HRXHostRuntime.load/run).
            return HRXKernelHandle(
                None, None, kernel_name, xclbin_path, None, ctx=self._ctx
            )

        key = (
            str(xclbin_path),
            xclbin_path.stat().st_mtime,
            str(insts_path),
            insts_path.stat().st_mtime,
            kernel_name,
        )
        if key in self._exe_cache:
            self._exe_cache.move_to_end(key)
            exe, ordv = self._exe_cache[key]
            return HRXKernelHandle(
                exe, ordv, kernel_name, xclbin_path, insts_path, ctx=self._ctx
            )

        exe, ordv = self._build_executable(xclbin_path, insts_path, kernel_name)

        if len(self._exe_cache) >= self._cache_size:
            _, (old_exe, _) = self._exe_cache.popitem(last=False)
            self._release_executable(old_exe)
        self._exe_cache[key] = (exe, ordv)

        return HRXKernelHandle(
            exe, ordv, kernel_name, xclbin_path, insts_path, ctx=self._ctx
        )

    def cleanup(self) -> None:
        """Release cached executables, then any tracked by the base runtime."""
        cache = getattr(self, "_exe_cache", None)
        if cache:
            while cache:
                _, (exe, _) = cache.popitem(last=False)
                self._release_executable(exe)
        super().cleanup()
