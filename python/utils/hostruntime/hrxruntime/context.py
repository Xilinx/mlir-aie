# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Process-wide HRX device/stream context and dispatch orchestration.

This is the mid-level layer between the raw C ABI (:mod:`._bindings`) and the
IRON ``HostRuntime`` (:mod:`.hostruntime`): :class:`HRXContext` owns the single
amdxdna device + dispatch stream, allocates/maps persistent buffers, creates
amdxdna executables, and records/submits (chained) dispatches.
"""

import ctypes
import logging
import threading

from ._bindings import (
    HRX_AMDXDNA_CONTEXT_MODE_CREATE,
    HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0,
    HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0,
    HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0,
    HRX_BUFFER_USAGE_DEFAULT,
    HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
    HRX_DISPATCH_FLAG_NONE,
    HRX_MAP_READ,
    HRX_MAP_WRITE,
    HRX_MAPPING_MODE_PERSISTENT,
    HRX_MEMORY_TYPE_DEVICE_VISIBLE,
    HRX_MEMORY_TYPE_HOST_LOCAL,
    HrxAmdxdnaExecutableCreateParams,
    HrxAmdxdnaExecutableEntryPoint,
    HrxAmdxdnaExecutableRun,
    HrxBufferRef,
    HrxConstByteSpan,
    HrxDispatchConfig,
    HrxStringView,
    HRXError,
    _check,
    _handle,
    _hrx_sync_timeout_s,
    lib,
)

logger = logging.getLogger(__name__)


class HRXContext:
    """Process-wide singleton owning the HRX device + dispatch stream.

    The amdxdna NPU is a single shared device, so a single device/stream pair is
    shared across all tensors and kernels in the process.

    Concurrency / multi-tenancy model:

    * **Multiple processes / users** -- fully isolated. Each process builds its
      own :class:`HRXContext` (its own ``hrx_gpu_initialize`` /
      ``hrx_gpu_device_get`` / ``hrx_stream_create``) and allocates its own
      buffers, so handles are never shared across processes; the amdxdna driver
      isolates each process's hardware context and device memory. The only
      shared resource is the finite, system-wide pool of amdxdna hardware
      contexts: under heavy parallelism, context/stream creation can fail with
      exhaustion (the same capacity constraint the XRT runtime documents via
      ``CachedXRTRuntime.NPU_CONTEXT_CACHE_SIZE``) -- a capacity limit, not a
      data-safety issue.
    * **One context per process** -- by design there is exactly one device +
      stream per process (the NPU is a single shared device); we never create
      several.
    * **Multiple threads in one process** -- singleton creation and libhrx
      binding are thread-safe (see :meth:`get` / :meth:`~._bindings._HrxLib.ensure`),
      but the shared ``stream`` is *not* built for concurrent dispatch: recording
      ``dispatch``/``dispatch_chain`` and ``synchronize`` from several threads at
      once would interleave into one pending command buffer. Callers must
      serialize dispatch on a single context (the XRT Python runtime has the same
      expectation).
    """

    _instance = None
    # Guards lazy singleton creation for concurrent first-touch from threads.
    _lock = threading.Lock()

    def __init__(self):
        # Load libhrx on first context creation (the import of this package
        # itself stays dlopen-free for the cheap has_hrx probe).
        lib.ensure()
        self.device = _handle()
        self.stream = _handle()
        _check(lib.hrx_gpu_initialize(0), "hrx_gpu_initialize")
        _check(
            lib.hrx_gpu_device_get(0, ctypes.byref(self.device)), "hrx_gpu_device_get"
        )
        _check(
            lib.hrx_stream_create(self.device, 0, ctypes.byref(self.stream)),
            "hrx_stream_create",
        )

    @classmethod
    def get(cls) -> "HRXContext":
        # Double-checked locking: lock-free once the singleton exists.
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = HRXContext()
        return cls._instance

    # -- buffers -----------------------------------------------------------
    def allocate_persistent(self, size: int):
        """Allocate a device-visible, host-coherent BO and map it persistently.

        Args:
            size (int): Number of bytes to allocate.

        Returns:
            tuple: ``(buffer_handle, host_ptr)`` -- the opaque ``hrx_buffer_t``
            and the address of its persistent host mapping. Coherence is
            maintained explicitly via :meth:`flush_range` / :meth:`invalidate_range`.
        """
        buf = _handle()
        _check(
            lib.hrx_buffer_allocate(
                self.stream,
                ctypes.c_size_t(size),
                HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
                HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
                ctypes.byref(buf),
            ),
            "hrx_buffer_allocate",
        )
        ptr = ctypes.c_void_p()
        _check(
            lib.hrx_buffer_map_with_mode(
                buf,
                HRX_MAPPING_MODE_PERSISTENT,
                HRX_MAP_READ | HRX_MAP_WRITE,
                ctypes.c_size_t(0),
                ctypes.c_size_t(size),
                ctypes.byref(ptr),
            ),
            "hrx_buffer_map_with_mode",
        )
        return buf, ptr.value

    def flush_range(self, buf, offset: int, size: int):
        """Flush host writes in ``[offset, offset+size)`` out to the device.

        Args:
            buf: The ``hrx_buffer_t`` handle to flush.
            offset (int): Byte offset into the buffer.
            size (int): Number of bytes to flush.
        """
        _check(
            lib.hrx_buffer_flush_range(
                buf, ctypes.c_size_t(offset), ctypes.c_size_t(size)
            ),
            "hrx_buffer_flush_range",
        )

    def invalidate_range(self, buf, offset: int, size: int):
        """Invalidate the host cache for ``[offset, offset+size)``.

        Ensures subsequent host reads observe device writes.

        Args:
            buf: The ``hrx_buffer_t`` handle to invalidate.
            offset (int): Byte offset into the buffer.
            size (int): Number of bytes to invalidate.
        """
        _check(
            lib.hrx_buffer_invalidate_range(
                buf, ctypes.c_size_t(offset), ctypes.c_size_t(size)
            ),
            "hrx_buffer_invalidate_range",
        )

    def release_buffer(self, buf):
        if buf:
            lib.hrx_buffer_release(buf)

    # -- executables -------------------------------------------------------
    def create_executable(
        self, xclbin_bytes: bytes, insts_bytes: bytes, entry_name: str
    ):
        """Create + load an amdxdna executable from the raw artifacts.

        Hands libhrx the xclbin bytes and the XAie transaction (the raw
        ``insts.bin`` words) via ``hrx_amdxdna_executable_create``.

        Args:
            xclbin_bytes (bytes): The packaged ``final.xclbin`` contents.
            insts_bytes (bytes): The raw XAie transaction (TXN) stream, i.e. the
                ``insts.bin`` uint32 words. Its length must be a multiple of 4
                (libhrx rejects otherwise).
            entry_name (str): The kernel/export name to build an entry point for
                (e.g. ``"MLIR_AIE"``).

        Returns:
            The loaded ``hrx_executable_t`` handle.

        Raises:
            HRXError: If ``xclbin_bytes`` is empty, or ``insts_bytes`` is empty
                or not a multiple of 4 bytes, or libhrx fails to create the
                executable.
        """
        if not xclbin_bytes:
            raise HRXError("xclbin bytes are empty")
        if not insts_bytes or len(insts_bytes) % 4 != 0:
            raise HRXError(
                "insts (XAie transaction) bytes are empty or not a multiple of 4"
            )

        # Keep every backing buffer alive for the duration of the call: libhrx
        # borrows all input storage and only reads it before returning.
        name_bytes = entry_name.encode("utf-8")

        xclbin_span = HrxConstByteSpan(
            data=ctypes.cast(ctypes.c_char_p(xclbin_bytes), ctypes.c_void_p),
            data_length=len(xclbin_bytes),
        )
        run = HrxAmdxdnaExecutableRun(
            record_length=ctypes.sizeof(HrxAmdxdnaExecutableRun),
            abi_version=HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0,
            transaction=HrxConstByteSpan(
                data=ctypes.cast(ctypes.c_char_p(insts_bytes), ctypes.c_void_p),
                data_length=len(insts_bytes),
            ),
            data_payload=HrxConstByteSpan(data=None, data_length=0),
        )
        entry = HrxAmdxdnaExecutableEntryPoint(
            record_length=ctypes.sizeof(HrxAmdxdnaExecutableEntryPoint),
            abi_version=HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0,
            name=HrxStringView(data=name_bytes, size=len(name_bytes)),
            context_mode=HRX_AMDXDNA_CONTEXT_MODE_CREATE,
            xclbin_ordinal=0,
            pdi_ordinal=0,
            source_line=0,
            source_file=HrxStringView(data=None, size=0),
            runs=ctypes.pointer(run),
            run_count=1,
        )
        params = HrxAmdxdnaExecutableCreateParams(
            record_length=ctypes.sizeof(HrxAmdxdnaExecutableCreateParams),
            abi_version=HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0,
            flags=0,
            reserved=0,
            xclbins=ctypes.pointer(xclbin_span),
            xclbin_count=1,
            entry_points=ctypes.pointer(entry),
            entry_point_count=1,
        )

        exe = _handle()
        _check(
            lib.hrx_amdxdna_executable_create(
                self.device, ctypes.byref(params), ctypes.byref(exe)
            ),
            "hrx_amdxdna_executable_create",
        )
        return exe

    def lookup_export(self, exe, name: str) -> int:
        """Resolve an export name to its ordinal within an executable.

        Args:
            exe: The ``hrx_executable_t`` handle returned by
                :meth:`create_executable`.
            name (str): The export/kernel name to look up.

        Returns:
            int: The export ordinal, passed to :meth:`dispatch` as
            ``export_ordinal``.

        Raises:
            HRXError: If ``name`` is not an export of ``exe``.
        """
        ordv = ctypes.c_uint32()
        _check(
            lib.hrx_executable_lookup_export_by_name(
                exe, name.encode("utf-8"), ctypes.byref(ordv)
            ),
            "hrx_executable_lookup_export_by_name",
        )
        return ordv.value

    def release_executable(self, exe):
        if exe:
            lib.hrx_executable_release(exe)

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, exe, export_ordinal: int, bindings):
        """Record a single dispatch of ``exe`` into the stream's command buffer.

        The dispatch config is the unit config the amdxdna path expects
        (``{1,1,1}`` workgroup count/size); the I/O addresses are bound by
        binding order plus the TXN's DDR-patch ops. This records only -- call
        :meth:`synchronize` to submit and wait.

        Args:
            exe: The ``hrx_executable_t`` handle to dispatch.
            export_ordinal (int): The export ordinal from :meth:`lookup_export`.
            bindings (list): Ordered ``(buffer_handle, size)`` tuples; the list
                index is the DDR-patch argument index.

        Raises:
            HRXError: If libhrx rejects the dispatch.
        """
        n = len(bindings)
        arr = (HrxBufferRef * n)()
        for i, (buf, size) in enumerate(bindings):
            arr[i].buffer = buf
            arr[i].offset = 0
            arr[i].length = size
        cfg = HrxDispatchConfig()
        cfg.workgroup_count[0] = cfg.workgroup_count[1] = cfg.workgroup_count[2] = 1
        cfg.workgroup_size[0] = cfg.workgroup_size[1] = cfg.workgroup_size[2] = 1
        cfg.subgroup_size = 0
        _check(
            lib.hrx_stream_dispatch(
                self.stream,
                exe,
                ctypes.c_uint32(export_ordinal),
                ctypes.byref(cfg),
                None,
                0,
                arr,
                ctypes.c_size_t(n),
                ctypes.c_uint32(HRX_DISPATCH_FLAG_NONE),
            ),
            "hrx_stream_dispatch",
        )

    def dispatch_chain(self, items):
        """Record a sequence of dispatches into one command buffer (no submit).

        Each entry is recorded via :meth:`dispatch`, and HRX inserts an execution
        + memory barrier after every dispatch, so a later dispatch observes an
        earlier one's device writes (producer -> consumer chains are correct).
        The whole batch stays pending until :meth:`synchronize`, which submits it
        as a single execution -- the amdxdna HAL lowers a multi-dispatch command
        buffer into one ``ERT_CMD_CHAIN`` issued/waited once.

        Records only; call :meth:`synchronize` to submit and wait.

        Args:
            items: An iterable of ``(executable, export_ordinal, bindings)``,
                where ``bindings`` is a list of ``(buffer_handle, size)`` tuples
                (the same shape :meth:`dispatch` takes).

        Raises:
            HRXError: If libhrx rejects any recorded dispatch.
        """
        for exe, export_ordinal, bindings in items:
            self.dispatch(exe, export_ordinal, bindings)

    def synchronize(self):
        """Submit the pending command buffer and block until it completes.

        With ``IRON_HRX_TIMEOUT`` unset (or ``<= 0``) this blocks until libhrx
        returns. Otherwise a best-effort watchdog bounds the *wait*: the blocking
        sync runs on a daemon thread and, on expiry, a diagnosable error is
        raised (the underlying sync cannot be cancelled, so the device work keeps
        running -- this avoids hanging forever, it does not abort the dispatch).

        Raises:
            HRXError: If libhrx reports a synchronization failure, or the wait
                exceeds ``IRON_HRX_TIMEOUT`` seconds.
        """
        timeout = _hrx_sync_timeout_s()
        if timeout <= 0:
            # Default: block until libhrx returns (unchanged behavior).
            _check(lib.hrx_stream_synchronize(self.stream), "hrx_stream_synchronize")
            return

        # Best-effort watchdog: run the blocking sync on a helper thread and
        # bound the *wait* with a timeout. NOTE: libhrx's hrx_stream_synchronize
        # cannot be cancelled, so on expiry the underlying call keeps running in
        # the daemon thread -- this raises a diagnosable error instead of
        # hanging forever, it does not abort the device work.
        result = {}

        def _worker():
            try:
                _check(
                    lib.hrx_stream_synchronize(self.stream), "hrx_stream_synchronize"
                )
                result["ok"] = True
            except BaseException as e:  # forwarded to the caller below
                result["err"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise HRXError(
                f"hrx_stream_synchronize did not complete within "
                f"IRON_HRX_TIMEOUT={timeout:g}s. The dispatch may be wedged; the "
                f"underlying sync cannot be cancelled and is still pending. "
                f"Recover the device (e.g. reload the amdxdna driver) if this "
                f"persists."
            )
        if "err" in result:
            raise result["err"]
