# context.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Process-wide HSA device/queue context and dispatch orchestration.

This is the mid-level layer between the raw C ABI (:mod:`._bindings`) and the
IRON ``HostRuntime`` (:mod:`.hostruntime`): :class:`HSAContext` owns the single
AIE + CPU agents, the data and device-heap memory pools, and a dispatch queue, and
issues/waits on AIE kernel-dispatch packets.
"""

import ctypes
import logging
import os
import threading
import time

from ._bindings import (
    _DISPATCH_HEADER,
    _HSA_WAIT_FOREVER,
    HSA_ACCESS_PERMISSION_NONE,
    HSA_ACCESS_PERMISSION_RW,
    HSA_AGENT_INFO_DEVICE,
    HSA_AGENT_INFO_NAME,
    HSA_AGENT_INFO_QUEUE_MIN_SIZE,
    HSA_AMD_AIE_PACKET_OPCODE_KMQ,
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
    HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
    HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
    HSA_AMD_SEGMENT_GLOBAL,
    HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
    HSA_DEVICE_TYPE_AIE,
    HSA_DEVICE_TYPE_CPU,
    HSA_QUEUE_TYPE_SINGLE,
    HSA_SIGNAL_CONDITION_EQ,
    HSA_STATUS_INFO_BREAK,
    HSA_STATUS_SUCCESS,
    HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT,
    HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY,
    HSA_WAIT_STATE_BLOCKED,
    MEMORY_TYPE_PINNED,
    HsaAieKernelDispatchPacket,
    HsaAmdMemoryAccessDesc,
    HSAError,
    HsaQueue,
    HSATimeoutError,
    _check,
    _hsa_sync_timeout_s,
    hsa_agent_t,
    hsa_amd_memory_pool_t,
    hsa_amd_vmem_alloc_handle_t,
    hsa_signal_t,
    lib,
)

_logger = logging.getLogger(__name__)

# Tensor arguments a pooled kernarg slot holds (as 2*N uint64: N addresses then
# N sizes). Sized to cover real designs with headroom -- IRON designs in-tree go
# up to 9 tensor arguments -- while keeping the whole pool small: the backing
# allocation is this * 16B * queue_size, i.e. 16KB for a 64-slot queue. Argument
# lists longer than this still work, via a per-dispatch fallback allocation.
_MAX_POOLED_KERNARGS = 16
_KERNARG_SLOT_SIZE = _MAX_POOLED_KERNARGS * 2 * 8

# Packets to enqueue before ringing the doorbell. ROCR submits every packet
# pending at the ring as ONE ERT command chain, so batching the doorbell is what
# turns a chain into an actual hardware chain -- measured 134us -> 63us per
# dispatch. Returns are flat beyond ~16, so this sits deliberately below the two
# ceilings rather than at them: the firmware's maximum chain length (measured at
# 40 on NPU firmware 1.5.5.391 -- 41 aborts inside ROCR on an assert, which is
# not a catchable error), and the queue capacity, which enqueue cannot exceed
# without a ring to drain it. The effective batch is clamped to the queue size.
_MAX_DOORBELL_BATCH = 32


class HSAContext:
    """Process-wide singleton owning the HSA AIE device, memory, and queue.

    The single in-order AIE queue + doorbell is NOT safe for concurrent
    dispatch from multiple threads; callers must serialize dispatches (same
    constraint HRX documents).
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        lib._ensure()
        _check(lib.hsa_init(), "hsa_init")

        # Tick conversion for the bounded wait in wait(); fixed for the
        # process, so query once rather than per wait.
        self.timestamp_freq = self._system_info_u64(
            HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, "TIMESTAMP_FREQUENCY"
        )
        self.signal_max_wait = self._system_info_u64(
            HSA_SYSTEM_INFO_SIGNAL_MAX_WAIT, "SIGNAL_MAX_WAIT"
        )

        self.aie_agent = self._find_agent(HSA_DEVICE_TYPE_AIE)
        if self.aie_agent == 0:
            raise HSAError("No HSA AIE agent found")
        self.cpu_agent = self._find_agent(HSA_DEVICE_TYPE_CPU)
        if self.cpu_agent == 0:
            raise HSAError("No HSA CPU agent found")

        self.pool = self._find_pool(self.aie_agent, dev_heap=False)
        self.dev_pool = self._find_pool(self.aie_agent, dev_heap=True)
        # Fixed for the life of the singleton; query once instead of per vmem_alloc.
        self.pool_granule = self._pool_granule()
        self.device_gen = self._detect_device_gen()

        min_size = ctypes.c_uint32()
        _check(
            lib.hsa_agent_get_info(
                self.aie_agent,
                HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                ctypes.byref(min_size),
            ),
            "hsa_agent_get_info(QUEUE_MIN_SIZE)",
        )
        qptr = ctypes.POINTER(HsaQueue)()
        _check(
            lib.hsa_queue_create(
                self.aie_agent,
                min_size.value,
                HSA_QUEUE_TYPE_SINGLE,
                None,
                None,
                0,
                0,
                ctypes.byref(qptr),
            ),
            "hsa_queue_create",
        )
        self.queue = qptr
        # Fixed for the life of the queue; cache instead of dereferencing
        # q.contents.* (and re-casting base_address) on every dispatch.
        self.queue_size = qptr.contents.size
        self.queue_doorbell = qptr.contents.doorbell_signal
        self.queue_packets = ctypes.cast(
            qptr.contents.base_address,
            ctypes.POINTER(HsaAieKernelDispatchPacket),
        )

        # Fixed-slot kernarg pool: one backing allocation carved into a slot per
        # queue ring slot, so the dispatch path writes kernargs with plain stores
        # and makes no HSA call. Slot i belongs to ring slot i, which makes reuse
        # safe for free: `enqueue` only writes ring slot i once the device has
        # consumed the previous packet there, and that is exactly when the device
        # is done reading slot i's kernargs.
        # Never freed: the context is a process-global singleton with no teardown.
        _, self._kernarg_va, _ = self.vmem_alloc(_KERNARG_SLOT_SIZE * self.queue_size)

        # Fixed for the life of the queue, like the caches above.
        self.doorbell_batch = min(_MAX_DOORBELL_BATCH, self.queue_size)

        # One completion signal reused by every dispatch, armed per dispatch
        # rather than created and destroyed each time. Safe for the same reason
        # the single queue is: callers must serialize dispatches.
        self._signal = self.create_signal(0)

    @classmethod
    def get(cls) -> "HSAContext":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = HSAContext()
        return cls._instance

    # -- discovery ---------------------------------------------------------
    def _find_agent(self, device_type):
        found = ctypes.c_uint64(0)

        @ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p)
        def cb(agent, _data):
            dt = ctypes.c_int()
            s = lib.hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, ctypes.byref(dt))
            if s != HSA_STATUS_SUCCESS:
                return s
            if dt.value == device_type:
                found.value = agent
                return HSA_STATUS_INFO_BREAK
            return HSA_STATUS_SUCCESS

        status = lib.hsa_iterate_agents(cb, None)
        if status not in (HSA_STATUS_SUCCESS, HSA_STATUS_INFO_BREAK):
            raise HSAError(f"hsa_iterate_agents failed (hsa status {status})")
        return found.value

    def _find_pool(self, agent, dev_heap):
        """Find the AIE agent's device heap (``dev_heap``) or data pool.

        ROCR distinguishes the two by the recommended allocation granule: the
        device heap is reported with a REC_GRANULE of 0, every ordinary pool
        with a nonzero one (see ``AieAgent::InitRegionList``). Allocations out
        of the device heap become ``AMDXDNA_BO_DEV`` buffer objects, which is
        the only BO type the driver's ``aie2_config_cu`` accepts for a PDI;
        anything else is an ``AMDXDNA_BO_SHARE`` and the submit ioctl fails
        with EIO.
        """
        found = ctypes.c_uint64(0)

        @ctypes.CFUNCTYPE(ctypes.c_int, hsa_amd_memory_pool_t, ctypes.c_void_p)
        def cb(pool, _data):
            seg = ctypes.c_int()
            if (
                lib.hsa_amd_memory_pool_get_info(
                    pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(seg)
                )
                != HSA_STATUS_SUCCESS
            ):
                return HSA_STATUS_SUCCESS
            if seg.value != HSA_AMD_SEGMENT_GLOBAL:
                return HSA_STATUS_SUCCESS
            flags = ctypes.c_uint32()
            if (
                lib.hsa_amd_memory_pool_get_info(
                    pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(flags)
                )
                != HSA_STATUS_SUCCESS
            ):
                return HSA_STATUS_SUCCESS
            if (flags.value & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0:
                return HSA_STATUS_SUCCESS
            rec = ctypes.c_size_t()
            if (
                lib.hsa_amd_memory_pool_get_info(
                    pool,
                    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
                    ctypes.byref(rec),
                )
                != HSA_STATUS_SUCCESS
            ):
                return HSA_STATUS_SUCCESS
            if (rec.value == 0) != dev_heap:
                return HSA_STATUS_SUCCESS
            found.value = pool
            return HSA_STATUS_INFO_BREAK

        status = lib.hsa_amd_agent_iterate_memory_pools(agent, cb, None)
        if status not in (HSA_STATUS_SUCCESS, HSA_STATUS_INFO_BREAK):
            raise HSAError(
                f"hsa_amd_agent_iterate_memory_pools failed (hsa status {status})"
            )
        if found.value == 0:
            kind = "device heap" if dev_heap else "data"
            raise HSAError(f"No coarse-grained {kind} pool found on AIE agent")
        return found.value

    def _detect_device_gen(self):
        """Map the HSA AIE agent to an IRON device generation.

        ROCR names the agent after its ISA rather than its marketing name:
        ``aie2`` on Phoenix (npu1) and ``aie2p`` on Strix (npu2), see
        ``XdnaDriver`` ``XDNADeviceType::Phx``/``Stx``. Test the ``aie2p``
        prefix first, since ``aie2p``/``aie2ps`` also start with ``aie2``.

        Guessing wrong here is expensive: the design compiles for the wrong
        architecture and the dispatch wedges the NPU until the driver's
        timeout detection fires, so an unrecognized agent raises instead of
        falling back to a default.
        """
        env = os.environ.get("IRON_HSA_DEVICE")
        if env:
            return env
        name = (ctypes.c_char * 64)()
        status = lib.hsa_agent_get_info(self.aie_agent, HSA_AGENT_INFO_NAME, name)
        if status != HSA_STATUS_SUCCESS:
            raise HSAError(f"hsa_agent_get_info(NAME) failed (hsa status {status})")
        text = name.value.decode("utf-8", "replace").lower()
        if "aie2p" in text or "strix" in text or "krackan" in text or "npu2" in text:
            return "npu2"
        if "aie2" in text or "phoenix" in text or "npu1" in text:
            return "npu1"
        raise HSAError(
            f"Cannot map HSA AIE agent name {text!r} to a device generation; "
            f"set IRON_HSA_DEVICE=npu1|npu2 to override."
        )

    @staticmethod
    def _system_info_u64(attribute, what):
        value = ctypes.c_uint64()
        _check(
            lib.hsa_system_get_info(attribute, ctypes.byref(value)),
            f"hsa_system_get_info({what})",
        )
        return value.value

    def _pool_granule(self):
        gran = ctypes.c_size_t()
        _check(
            lib.hsa_amd_memory_pool_get_info(
                self.pool,
                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                ctypes.byref(gran),
            ),
            "hsa_amd_memory_pool_get_info(GRANULE)",
        )
        return gran.value

    # -- device heap memory (PDI/insts) -----------------------------------
    def alloc_dev(self, size):
        """Allocate from the device heap, where PDI and insts must live."""
        ptr = ctypes.c_void_p()
        _check(
            lib.hsa_amd_memory_pool_allocate(self.dev_pool, size, 0, ctypes.byref(ptr)),
            "hsa_amd_memory_pool_allocate",
        )
        if not ptr.value:
            raise HSAError("hsa_amd_memory_pool_allocate returned a null pointer")
        return ptr.value

    def free_dev(self, ptr):
        if ptr:
            lib.hsa_amd_memory_pool_free(ctypes.c_void_p(ptr))

    # -- vmem memory (I/O + kernargs) -------------------------------------
    def vmem_alloc(self, size):
        granule = self.pool_granule
        size = ((size + granule - 1) // granule) * granule
        handle = hsa_amd_vmem_alloc_handle_t()
        _check(
            lib.hsa_amd_vmem_handle_create(
                self.pool,
                size,
                MEMORY_TYPE_PINNED,
                0,
                ctypes.byref(handle),
            ),
            "hsa_amd_vmem_handle_create",
        )
        va = ctypes.c_void_p()
        _check(
            lib.hsa_amd_vmem_address_reserve_align(
                ctypes.byref(va),
                size,
                0,
                0,
                HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
            ),
            "hsa_amd_vmem_address_reserve_align",
        )
        _check(
            lib.hsa_amd_vmem_map(va, size, 0, handle, 0),
            "hsa_amd_vmem_map",
        )
        self._set_vmem_access(va, size, HSA_ACCESS_PERMISSION_RW)
        return handle.value, va.value, size

    def _set_vmem_access(self, va, size, permission):
        """Grant or revoke CPU+AIE access to a mapped vmem range."""
        descs = (HsaAmdMemoryAccessDesc * 2)(
            HsaAmdMemoryAccessDesc(permission, self.cpu_agent),
            HsaAmdMemoryAccessDesc(permission, self.aie_agent),
        )
        _check(
            lib.hsa_amd_vmem_set_access(va, size, descs, 2),
            "hsa_amd_vmem_set_access",
        )

    def vmem_free(self, handle, va, size):
        """Tear down a vmem allocation made by :meth:`vmem_alloc`.

        Access must be revoked *before* unmapping. With an agent grant still in
        place ROCR refuses the unmap (HSA_STATUS_ERROR) and then the address
        free (HSA_STATUS_ERROR_RESOURCE_FREE), leaving the range mapped -- after
        which the next :meth:`vmem_alloc` reserving that VA fails in
        ``hsa_amd_vmem_map``. Statuses are logged rather than raised: this runs
        from ``__del__`` and from ``finally`` cleanup, where raising would either
        be swallowed or mask the original error.
        """
        if va:
            va_p = ctypes.c_void_p(va)
            try:
                self._set_vmem_access(va_p, size, HSA_ACCESS_PERMISSION_NONE)
            except HSAError as e:
                _logger.warning("vmem_free: revoking access failed: %s", e)
            self._log_if_error(
                lib.hsa_amd_vmem_unmap(va_p, size), "hsa_amd_vmem_unmap"
            )
            self._log_if_error(
                lib.hsa_amd_vmem_address_free(va_p, size),
                "hsa_amd_vmem_address_free",
            )
        if handle:
            self._log_if_error(
                lib.hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t(handle)),
                "hsa_amd_vmem_handle_release",
            )

    @staticmethod
    def _log_if_error(status, what):
        """Report a failed teardown call without raising.

        Ignoring these silently is what let a failed unmap corrupt the next
        allocation instead of surfacing at the point of failure.
        """
        if status != HSA_STATUS_SUCCESS:
            _logger.warning("%s failed (hsa status %s)", what, status)

    # -- signals -----------------------------------------------------------
    def create_signal(self, initial):
        sig = hsa_signal_t()
        _check(
            lib.hsa_signal_create(initial, 0, None, ctypes.byref(sig)),
            "hsa_signal_create",
        )
        return sig.value

    def arm_signal(self, value):
        """Arm the shared completion signal to ``value`` and return it.

        ``value`` is the number of packets that will decrement it (1 for a single
        dispatch, len(chain) for a chain), so a single wait covers the batch."""
        lib.hsa_signal_store_screlease(self._signal, value)
        return self._signal

    def discard_signal(self):
        """Abandon the shared signal and install a fresh one.

        Called when a dispatch is left in flight (a timeout): the device may still
        decrement the old signal at any point, which would corrupt the count of
        whatever dispatch armed it next, so it must never be reused. The old
        signal is deliberately leaked rather than destroyed -- the device still
        owns it (see HSATimeoutError)."""
        self._signal = self.create_signal(0)

    # -- dispatch ----------------------------------------------------------
    def _fill_packet(
        self, pdi_ptr, insts_ptr, insts_size, kernarg_ptr, num_kernargs, signal
    ):
        pkt = HsaAieKernelDispatchPacket()
        pkt.header = _DISPATCH_HEADER
        pkt.opcode = HSA_AMD_AIE_PACKET_OPCODE_KMQ
        pkt.count = 24
        pkt.completion_signal = signal
        pkt.insts_addr_low = insts_ptr & 0xFFFFFFFF
        pkt.insts_addr_high = insts_ptr >> 32
        pkt.num_kernargs = num_kernargs
        pkt.kernarg_address = kernarg_ptr
        pkt.insts_size = insts_size
        pkt.pdi_addr = pdi_ptr
        return pkt

    @staticmethod
    def _write_kernargs(va, addrs, sizes):
        """Write the 2*N uint64 kernarg block: N addresses, then N byte sizes.

        Takes pre-converted ints so that this cannot raise -- it runs after the
        queue write index has been reserved, where a failure would be unrecoverable
        (see :meth:`enqueue`)."""
        n = len(addrs)
        ka = (ctypes.c_uint64 * (2 * n)).from_address(va)
        for i in range(n):
            ka[i] = addrs[i]
            ka[n + i] = sizes[i]

    def enqueue(self, pdi_ptr, insts_ptr, insts_size, args, signal):
        """Write one packet at the next queue slot (no doorbell).

        ``args`` is a sequence of ``(device_va, nbytes)`` pairs, one per tensor
        argument. Returns ``(wr_idx, overflow)``, where ``overflow`` is ``None``
        for the common pooled case, or the ``(handle, va, size)`` of a one-off
        kernarg allocation the caller must free once the dispatch has completed.

        Kernargs are written into this ring slot's preallocated pool slot, so the
        hot path performs no HSA allocation. Argument lists longer than
        ``_MAX_POOLED_KERNARGS`` do not fit a slot and fall back to a per-dispatch
        allocation.

        This is all-or-nothing with respect to the queue: everything that can fail
        (argument conversion, the overflow allocation, waiting for a free slot)
        happens *before* the write index is reserved. Reserving an index and then
        failing to store a packet at it would leave the queue permanently
        inconsistent -- read index behind write index with an unwritten slot
        between them -- and the next doorbell would submit that garbage slot,
        killing the context for the rest of the process.

        Spins while the queue is full so an in-flight batch drains (wrap-around),
        yielding each iteration so it doesn't peg a core. When ``IRON_HSA_TIMEOUT``
        is set, the spin is bounded by that timeout and raises
        :class:`HSATimeoutError` (mirroring :meth:`wait`)."""
        # -- fallible section: nothing here has touched the queue yet ----------
        n = len(args)
        addrs = [int(va) for va, _ in args]
        sizes = [int(nbytes) for _, nbytes in args]
        pdi_ptr = int(pdi_ptr)
        insts_ptr = int(insts_ptr)
        insts_size = int(insts_size)
        signal = int(signal)

        q = self.queue
        qsize = self.queue_size
        # Single-producer by contract (callers serialize dispatches), so peeking
        # the write index and reserving it after the wait is race-free -- and it
        # keeps a timed-out wait from consuming an index it will never fill.
        wr_idx = lib.hsa_queue_load_write_index_relaxed(q)
        # The queue is normally not full, so the loop body never runs; read the
        # timeout / compute the deadline lazily on first spin to keep the common
        # path free of a per-dispatch os.environ read.
        deadline = None
        timeout = 0.0  # only meaningful once the spin below reads it
        while wr_idx - lib.hsa_queue_load_read_index_scacquire(q) >= qsize:
            if deadline is None:
                timeout = _hsa_sync_timeout_s()
                deadline = time.monotonic() + timeout if timeout > 0 else 0.0
            elif deadline and time.monotonic() >= deadline:
                raise HSATimeoutError(
                    f"queue did not drain within IRON_HSA_TIMEOUT={timeout:g}s "
                    f"while enqueuing a dispatch. The device may be wedged; "
                    f"recover it (e.g. reload the amdxdna driver) if this persists."
                )
            time.sleep(0)  # yield so a full-queue spin doesn't peg a core

        overflow = self.vmem_alloc(2 * n * 8) if n > _MAX_POOLED_KERNARGS else None

        # -- committed section: reserve the index, then only plain stores ------
        wr_idx = lib.hsa_queue_add_write_index_relaxed(q, 1)
        # Only now that the ring slot is ours may its kernarg slot be reused: the
        # device has finished reading whatever the previous occupant wrote.
        slot = wr_idx % qsize
        ka_va = (
            overflow[1]
            if overflow is not None
            else self._kernarg_va + slot * _KERNARG_SLOT_SIZE
        )
        self._write_kernargs(ka_va, addrs, sizes)
        self.queue_packets[slot] = self._fill_packet(
            pdi_ptr, insts_ptr, insts_size, ka_va if n else None, n, signal
        )
        return wr_idx, overflow

    def ring(self, wr_idx):
        lib.hsa_signal_store_screlease(self.queue_doorbell, wr_idx)

    def dispatch(self, pdi_ptr, insts_ptr, insts_size, args, signal):
        """Single dispatch: enqueue one packet and ring the doorbell.

        Returns the list of one-off kernarg allocations to free after the wait
        (empty in the common pooled case)."""
        wr_idx, overflow = self.enqueue(pdi_ptr, insts_ptr, insts_size, args, signal)
        self.ring(wr_idx)
        return [overflow] if overflow is not None else []

    def dispatch_chain(self, items, signal):
        """Enqueue a sequence of dispatches sharing one completion signal.

        ``items`` is a sequence of ``(pdi_ptr, insts_ptr, insts_size, args)``
        tuples, where ``args`` is that dispatch's list of ``(device_va, nbytes)``
        pairs. All packets carry the same ``signal`` (initialized by the caller to
        ``len(items)``); each completed packet decrements it, so a single
        ``wait(signal)`` covers the whole chain. Ordering is guaranteed by the
        single in-order AIE queue plus the system-scope acquire/release fences in
        every packet header, so a later dispatch observes an earlier one's device
        writes (producer -> consumer). Chains longer than the queue capacity
        auto-batch: the doorbell is rung every ``_MAX_DOORBELL_BATCH`` packets (and
        once more for the remainder), which both submits that group as a single
        hardware command chain and drains ring slots so the next group can wrap
        around instead of deadlocking on a full queue.

        Kernargs are written per packet as its ring slot is reserved, never all up
        front -- a chain longer than the queue reuses slots, so an up-front fill
        would be overwritten before the device read it.

        Returns the list of one-off kernarg allocations to free after the wait.
        """
        overflows = []
        pending = 0
        wr_idx = None
        try:
            for pdi_ptr, insts_ptr, insts_size, args in items:
                wr_idx, overflow = self.enqueue(
                    pdi_ptr, insts_ptr, insts_size, args, signal
                )
                if overflow is not None:
                    overflows.append(overflow)
                pending += 1
                # Ring every `batch` packets rather than every packet: ROCR chains
                # the whole pending group into one submission, and ringing also
                # drains slots so a chain longer than the queue wraps around
                # instead of deadlocking on a full queue.
                if pending == self.doorbell_batch:
                    self.ring(wr_idx)
                    pending = 0
        except HSATimeoutError:
            # The device is not draining the queue; ringing would block in the
            # synchronous submit, so leave the pending packets for whatever
            # recovers the device.
            raise
        except BaseException:
            # A packet failed to build, but the ones already written are valid and
            # the device is healthy. Ring them so the queue is not left carrying
            # un-rung packets that a later, unrelated dispatch would submit.
            if pending:
                self.ring(wr_idx)
            raise
        if pending:
            self.ring(wr_idx)
        return overflows

    def wait(self, signal):
        """Block until ``signal`` reaches 0, optionally bounded by IRON_HSA_TIMEOUT.

        The bounded form uses ``hsa_signal_wait``'s own tick-unit timeout rather
        than a watchdog thread, so arming the timeout costs nothing per dispatch.
        The timeout is only a *hint* and the wait may resume spuriously with the
        condition unmet, so the native wait is retried until either the signal is
        observed at 0 or the wall-clock deadline passes. As before, an expired
        wait cannot cancel the dispatch -- the device work keeps running and this
        raises a diagnosable error instead of hanging forever.
        """
        timeout = _hsa_sync_timeout_s()
        if timeout <= 0:
            # Default: block until the signal reaches 0 (unchanged behavior).
            lib.hsa_signal_wait_scacquire(
                signal,
                HSA_SIGNAL_CONDITION_EQ,
                0,
                _HSA_WAIT_FOREVER,
                HSA_WAIT_STATE_BLOCKED,
            )
            return

        # Clamp the per-attempt hint to the longest wait the system supports; the
        # wall-clock deadline below, not the hint, is what bounds the total.
        ticks = int(timeout * self.timestamp_freq)
        if self.signal_max_wait and ticks > self.signal_max_wait:
            ticks = self.signal_max_wait
        deadline = time.monotonic() + timeout
        while True:
            if (
                lib.hsa_signal_wait_scacquire(
                    signal,
                    HSA_SIGNAL_CONDITION_EQ,
                    0,
                    ticks,
                    HSA_WAIT_STATE_BLOCKED,
                )
                == 0
            ):
                return
            if time.monotonic() >= deadline:
                raise HSATimeoutError(
                    f"hsa_signal_wait did not complete within IRON_HSA_TIMEOUT="
                    f"{timeout:g}s. The dispatch may be wedged; the underlying wait "
                    f"cannot be cancelled and is still pending. Recover the device "
                    f"(e.g. reload the amdxdna driver) if this persists."
                )
