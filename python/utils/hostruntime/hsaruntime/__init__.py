# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""HSA/ROCR host-runtime backend for IRON.

Wraps the ``hsa_*`` entry points bound by :mod:`._bindings` in a process-wide
``HSAContext`` singleton owning the AIE device, memory region/pool, and a
dispatch queue. Bindings load lazily so importing this package for the
``hsa_available`` probe performs no dlopen.
"""

import ctypes
import os

from ._bindings import (
    HSA_ACCESS_PERMISSION_RW,
    HSA_AGENT_INFO_DEVICE,
    HSA_AGENT_INFO_NAME,
    HSA_AGENT_INFO_QUEUE_MIN_SIZE,
    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED,
    HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE,
    HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
    HSA_AMD_SEGMENT_GLOBAL,
    HSA_AMD_AIE_PACKET_OPCODE_KMQ,
    HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
    HSA_DEVICE_TYPE_AIE,
    HSA_DEVICE_TYPE_CPU,
    HSA_QUEUE_TYPE_SINGLE,
    HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED,
    HSA_REGION_INFO_SEGMENT,
    HSA_REGION_SEGMENT_GLOBAL,
    HSA_SIGNAL_CONDITION_EQ,
    HSA_STATUS_INFO_BREAK,
    HSA_STATUS_SUCCESS,
    HSA_WAIT_STATE_BLOCKED,
    MEMORY_TYPE_PINNED,
    HSAError,
    HsaAieKernelDispatchPacket,
    HsaAmdMemoryAccessDesc,
    HsaQueue,
    _check,
    _DISPATCH_HEADER,
    _HSA_WAIT_FOREVER,
    hsa_agent_t,
    hsa_amd_memory_pool_t,
    hsa_amd_vmem_alloc_handle_t,
    hsa_region_t,
    hsa_signal_t,
    lib,
)


class HSAContext:
    """Process-wide singleton owning the HSA AIE device, memory, and queue."""

    _instance = None

    def __init__(self):
        lib._ensure()
        _check(lib.hsa_init(), "hsa_init")

        self.aie_agent = self._find_agent(HSA_DEVICE_TYPE_AIE)
        if self.aie_agent == 0:
            raise HSAError("No HSA AIE agent found")
        self.cpu_agent = self._find_agent(HSA_DEVICE_TYPE_CPU)
        if self.cpu_agent == 0:
            raise HSAError("No HSA CPU agent found")

        self.region = self._find_region(self.aie_agent)
        self.pool = self._find_pool(self.aie_agent)
        # Fixed for the life of the singleton; query once instead of per vmem_alloc.
        self.pool_granule = self._pool_granule()
        self.device_gen = self._detect_device_gen()

        min_size = ctypes.c_uint32()
        _check(
            lib.hsa_agent_get_info(
                self.aie_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                ctypes.byref(min_size),
            ),
            "hsa_agent_get_info(QUEUE_MIN_SIZE)",
        )
        qptr = ctypes.POINTER(HsaQueue)()
        _check(
            lib.hsa_queue_create(
                self.aie_agent, min_size.value, HSA_QUEUE_TYPE_SINGLE, None, None,
                0, 0, ctypes.byref(qptr),
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

    @classmethod
    def get(cls):
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

    def _find_region(self, agent):
        found = ctypes.c_uint64(0)

        @ctypes.CFUNCTYPE(ctypes.c_int, hsa_region_t, ctypes.c_void_p)
        def cb(region, _data):
            seg = ctypes.c_int()
            if lib.hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, ctypes.byref(seg)) != HSA_STATUS_SUCCESS:
                return HSA_STATUS_SUCCESS
            if seg.value != HSA_REGION_SEGMENT_GLOBAL:
                return HSA_STATUS_SUCCESS
            allowed = ctypes.c_bool()
            if lib.hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, ctypes.byref(allowed)) != HSA_STATUS_SUCCESS:
                return HSA_STATUS_SUCCESS
            if not allowed.value:
                return HSA_STATUS_SUCCESS
            found.value = region
            return HSA_STATUS_INFO_BREAK

        status = lib.hsa_agent_iterate_regions(agent, cb, None)
        if status not in (HSA_STATUS_SUCCESS, HSA_STATUS_INFO_BREAK):
            raise HSAError(f"hsa_agent_iterate_regions failed (hsa status {status})")
        if found.value == 0:
            raise HSAError("No allocatable global HSA region found on AIE agent")
        return found.value

    def _find_pool(self, agent):
        found = ctypes.c_uint64(0)

        @ctypes.CFUNCTYPE(ctypes.c_int, hsa_amd_memory_pool_t, ctypes.c_void_p)
        def cb(pool, _data):
            seg = ctypes.c_int()
            if lib.hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(seg)) != HSA_STATUS_SUCCESS:
                return HSA_STATUS_SUCCESS
            if seg.value != HSA_AMD_SEGMENT_GLOBAL:
                return HSA_STATUS_SUCCESS
            flags = ctypes.c_uint32()
            if lib.hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(flags)) != HSA_STATUS_SUCCESS:
                return HSA_STATUS_SUCCESS
            if (flags.value & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0:
                return HSA_STATUS_SUCCESS
            rec = ctypes.c_size_t()
            if lib.hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, ctypes.byref(rec)) != HSA_STATUS_SUCCESS:
                return HSA_STATUS_SUCCESS
            if rec.value == 0:  # allocatable pools have a nonzero rec granule
                return HSA_STATUS_SUCCESS
            found.value = pool
            return HSA_STATUS_INFO_BREAK

        status = lib.hsa_amd_agent_iterate_memory_pools(agent, cb, None)
        if status not in (HSA_STATUS_SUCCESS, HSA_STATUS_INFO_BREAK):
            raise HSAError(f"hsa_amd_agent_iterate_memory_pools failed (hsa status {status})")
        if found.value == 0:
            raise HSAError("No allocatable coarse-grained pool found on AIE agent")
        return found.value

    def _detect_device_gen(self):
        env = os.environ.get("IRON_HSA_DEVICE")
        if env:
            return env
        name = (ctypes.c_char * 64)()
        if lib.hsa_agent_get_info(self.aie_agent, HSA_AGENT_INFO_NAME, name) == HSA_STATUS_SUCCESS:
            text = name.value.decode("utf-8", "replace").lower()
            if "phoenix" in text or "npu1" in text:
                return "npu1"
        return "npu2"

    def _pool_granule(self):
        gran = ctypes.c_size_t()
        _check(
            lib.hsa_amd_memory_pool_get_info(
                self.pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE,
                ctypes.byref(gran),
            ),
            "hsa_amd_memory_pool_get_info(GRANULE)",
        )
        return gran.value

    # -- region memory (PDI/insts) ----------------------------------------
    def alloc_region(self, size):
        ptr = ctypes.c_void_p()
        _check(
            lib.hsa_memory_allocate(self.region, size, ctypes.byref(ptr)),
            "hsa_memory_allocate",
        )
        return ptr.value

    def free_region(self, ptr):
        if ptr:
            lib.hsa_memory_free(ctypes.c_void_p(ptr))

    # -- vmem memory (I/O + kernargs) -------------------------------------
    def vmem_alloc(self, size):
        granule = self.pool_granule
        size = ((size + granule - 1) // granule) * granule
        handle = hsa_amd_vmem_alloc_handle_t()
        _check(
            lib.hsa_amd_vmem_handle_create(
                self.pool, size, MEMORY_TYPE_PINNED, 0, ctypes.byref(handle),
            ),
            "hsa_amd_vmem_handle_create",
        )
        va = ctypes.c_void_p()
        _check(
            lib.hsa_amd_vmem_address_reserve_align(
                ctypes.byref(va), size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
            ),
            "hsa_amd_vmem_address_reserve_align",
        )
        _check(
            lib.hsa_amd_vmem_map(va, size, 0, handle, 0),
            "hsa_amd_vmem_map",
        )
        descs = (HsaAmdMemoryAccessDesc * 2)(
            HsaAmdMemoryAccessDesc(HSA_ACCESS_PERMISSION_RW, self.cpu_agent),
            HsaAmdMemoryAccessDesc(HSA_ACCESS_PERMISSION_RW, self.aie_agent),
        )
        _check(
            lib.hsa_amd_vmem_set_access(va, size, descs, 2),
            "hsa_amd_vmem_set_access",
        )
        return handle.value, va.value, size

    def vmem_free(self, handle, va, size):
        if va:
            lib.hsa_amd_vmem_unmap(ctypes.c_void_p(va), size)
            lib.hsa_amd_vmem_address_free(ctypes.c_void_p(va), size)
        if handle:
            lib.hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t(handle))

    # -- signals -----------------------------------------------------------
    def create_signal(self, initial):
        sig = hsa_signal_t()
        _check(
            lib.hsa_signal_create(initial, 0, None, ctypes.byref(sig)),
            "hsa_signal_create",
        )
        return sig.value

    def destroy_signal(self, sig):
        if sig:
            lib.hsa_signal_destroy(hsa_signal_t(sig))

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, pdi_ptr, insts_ptr, insts_size, kernarg_ptr, num_kernargs, signal):
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

        q = self.queue
        qsize = self.queue_size
        wr_idx = lib.hsa_queue_add_write_index_relaxed(q, 1)
        while wr_idx - lib.hsa_queue_load_read_index_scacquire(q) >= qsize:
            pass
        self.queue_packets[wr_idx % qsize] = pkt
        lib.hsa_signal_store_screlease(self.queue_doorbell, wr_idx)

    def wait(self, signal):
        lib.hsa_signal_wait_scacquire(
            signal, HSA_SIGNAL_CONDITION_EQ, 0, _HSA_WAIT_FOREVER,
            HSA_WAIT_STATE_BLOCKED,
        )
