# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""ctypes bindings for the HSA/ROCR C ABI (``libhsa-runtime64.so``).

Binds the handful of ``hsa_*`` entry points the AIE dispatch path needs and
wraps them in a process-wide ``HSAContext`` singleton owning the AIE device,
memory region/pool, and a dispatch queue. Bindings load lazily so importing
this package for the ``hsa_available`` probe performs no dlopen.
"""

import ctypes
import os

from .discovery import find_libhsa

# --- hsa.h constants -------------------------------------------------------
HSA_STATUS_SUCCESS = 0
HSA_STATUS_INFO_BREAK = 0x1

HSA_DEVICE_TYPE_CPU = 0
HSA_DEVICE_TYPE_GPU = 1
HSA_DEVICE_TYPE_AIE = 3  # from hsa_device_type_t (CPU=0, GPU=1, DSP=2, AIE=3)

HSA_AGENT_INFO_DEVICE = 17
HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13
HSA_AGENT_INFO_NAME = 0

HSA_REGION_INFO_SEGMENT = 0
HSA_REGION_INFO_GLOBAL_FLAGS = 1
HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED = 5
HSA_REGION_SEGMENT_GLOBAL = 0

HSA_AMD_SEGMENT_GLOBAL = 0
HSA_AMD_MEMORY_POOL_INFO_SEGMENT = 0
HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS = 1
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE = 6
HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED = 4
HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE = 18

MEMORY_TYPE_PINNED = 1  # hsa_amd_memory_type_t
HSA_AMD_VMEM_ADDRESS_NO_REGISTER = 1
HSA_ACCESS_PERMISSION_RW = 3

HSA_QUEUE_TYPE_SINGLE = 1

# hsa_packet_header_t bit offsets
HSA_PACKET_HEADER_TYPE = 0
HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE = 9
HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE = 11
HSA_FENCE_SCOPE_SYSTEM = 2

HSA_AMD_AIE_PACKET_TYPE_READY = 0
HSA_AMD_AIE_PACKET_OPCODE_KMQ = 0

HSA_SIGNAL_CONDITION_EQ = 0
HSA_WAIT_STATE_BLOCKED = 0

# Opaque handle-carrying structs are all {uint64_t handle}.
hsa_agent_t = ctypes.c_uint64
hsa_region_t = ctypes.c_uint64
hsa_amd_memory_pool_t = ctypes.c_uint64
hsa_signal_t = ctypes.c_int64  # signal handle is a 64-bit value
hsa_amd_vmem_alloc_handle_t = ctypes.c_uint64


class HSAError(RuntimeError):
    """Raised when an hsa_* call returns a non-success status."""


class HsaAmdMemoryAccessDesc(ctypes.Structure):
    _fields_ = [
        ("permissions", ctypes.c_int),      # hsa_access_permission_t
        ("agent", hsa_agent_t),
    ]


class HsaAieKernelDispatchPacket(ctypes.Structure):
    """Mirror of hsa_amd_aie_kernel_dispatch_packet_t (64-byte AQL packet)."""
    _fields_ = [
        ("header", ctypes.c_uint16),
        ("opcode", ctypes.c_uint16),
        ("count", ctypes.c_uint16),
        ("reserved0", ctypes.c_uint8),
        ("reserved1", ctypes.c_uint8),
        ("completion_signal", hsa_signal_t),
        ("reserved2", ctypes.c_uint32),
        ("insts_addr_low", ctypes.c_uint32),
        ("insts_addr_high", ctypes.c_uint32),
        ("num_kernargs", ctypes.c_uint16),
        ("reserved3", ctypes.c_uint16),
        ("kernarg_address", ctypes.c_void_p),
        ("insts_size", ctypes.c_uint64),
        ("pdi_addr", ctypes.c_void_p),
        ("reserved4", ctypes.c_uint64),
    ]


class HsaQueue(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("base_address", ctypes.c_void_p),
        ("doorbell_signal", hsa_signal_t),
        ("size", ctypes.c_uint32),
        ("reserved1", ctypes.c_uint32),
        ("id", ctypes.c_uint64),
    ]


_lib = None
_bindings_ready = False


def _load_libhsa() -> ctypes.CDLL:
    tried = []
    last_err = None
    for c in [find_libhsa(), "libhsa-runtime64.so"]:
        if not c:
            continue
        tried.append(c)
        try:
            return ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last_err = e
    raise HSAError(
        f"Could not load libhsa-runtime64.so (tried: {tried}). Set "
        f"HSA_RUNTIME_LIB/ROCM_PATH or add it to LD_LIBRARY_PATH. "
        f"Last error: {last_err}"
    )


def _decl(name, restype, argtypes):
    f = getattr(_lib, name)
    f.restype = restype
    f.argtypes = argtypes
    return f


def _ensure_bindings():
    global _lib, _bindings_ready
    if _bindings_ready:
        return
    _lib = _load_libhsa()

    g = globals()
    g["_hsa_init"] = _decl("hsa_init", ctypes.c_int, [])
    g["_hsa_shut_down"] = _decl("hsa_shut_down", ctypes.c_int, [])
    g["_hsa_iterate_agents"] = _decl(
        "hsa_iterate_agents", ctypes.c_int,
        [ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p), ctypes.c_void_p],
    )
    g["_hsa_agent_get_info"] = _decl(
        "hsa_agent_get_info", ctypes.c_int,
        [hsa_agent_t, ctypes.c_int, ctypes.c_void_p],
    )
    g["_hsa_agent_iterate_regions"] = _decl(
        "hsa_agent_iterate_regions", ctypes.c_int,
        [hsa_agent_t, ctypes.CFUNCTYPE(ctypes.c_int, hsa_region_t, ctypes.c_void_p),
         ctypes.c_void_p],
    )
    g["_hsa_region_get_info"] = _decl(
        "hsa_region_get_info", ctypes.c_int,
        [hsa_region_t, ctypes.c_int, ctypes.c_void_p],
    )
    g["_hsa_memory_allocate"] = _decl(
        "hsa_memory_allocate", ctypes.c_int,
        [hsa_region_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
    )
    g["_hsa_memory_free"] = _decl("hsa_memory_free", ctypes.c_int, [ctypes.c_void_p])

    g["_hsa_amd_agent_iterate_memory_pools"] = _decl(
        "hsa_amd_agent_iterate_memory_pools", ctypes.c_int,
        [hsa_agent_t,
         ctypes.CFUNCTYPE(ctypes.c_int, hsa_amd_memory_pool_t, ctypes.c_void_p),
         ctypes.c_void_p],
    )
    g["_hsa_amd_memory_pool_get_info"] = _decl(
        "hsa_amd_memory_pool_get_info", ctypes.c_int,
        [hsa_amd_memory_pool_t, ctypes.c_int, ctypes.c_void_p],
    )
    g["_hsa_amd_vmem_handle_create"] = _decl(
        "hsa_amd_vmem_handle_create", ctypes.c_int,
        [hsa_amd_memory_pool_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint64,
         ctypes.POINTER(hsa_amd_vmem_alloc_handle_t)],
    )
    g["_hsa_amd_vmem_handle_release"] = _decl(
        "hsa_amd_vmem_handle_release", ctypes.c_int, [hsa_amd_vmem_alloc_handle_t]
    )
    g["_hsa_amd_vmem_address_reserve_align"] = _decl(
        "hsa_amd_vmem_address_reserve_align", ctypes.c_int,
        [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint64,
         ctypes.c_uint64, ctypes.c_uint64],
    )
    g["_hsa_amd_vmem_address_free"] = _decl(
        "hsa_amd_vmem_address_free", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t],
    )
    g["_hsa_amd_vmem_map"] = _decl(
        "hsa_amd_vmem_map", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
         hsa_amd_vmem_alloc_handle_t, ctypes.c_uint64],
    )
    g["_hsa_amd_vmem_unmap"] = _decl(
        "hsa_amd_vmem_unmap", ctypes.c_int, [ctypes.c_void_p, ctypes.c_size_t]
    )
    g["_hsa_amd_vmem_set_access"] = _decl(
        "hsa_amd_vmem_set_access", ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(HsaAmdMemoryAccessDesc),
         ctypes.c_size_t],
    )

    g["_hsa_queue_create"] = _decl(
        "hsa_queue_create", ctypes.c_int,
        [hsa_agent_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
         ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
         ctypes.POINTER(ctypes.POINTER(HsaQueue))],
    )
    g["_hsa_queue_destroy"] = _decl(
        "hsa_queue_destroy", ctypes.c_int, [ctypes.POINTER(HsaQueue)]
    )
    g["_hsa_queue_add_write_index_relaxed"] = _decl(
        "hsa_queue_add_write_index_relaxed", ctypes.c_uint64,
        [ctypes.POINTER(HsaQueue), ctypes.c_uint64],
    )
    g["_hsa_queue_load_read_index_scacquire"] = _decl(
        "hsa_queue_load_read_index_scacquire", ctypes.c_uint64,
        [ctypes.POINTER(HsaQueue)],
    )
    g["_hsa_signal_create"] = _decl(
        "hsa_signal_create", ctypes.c_int,
        [ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p,
         ctypes.POINTER(hsa_signal_t)],
    )
    g["_hsa_signal_destroy"] = _decl("hsa_signal_destroy", ctypes.c_int, [hsa_signal_t])
    g["_hsa_signal_store_screlease"] = _decl(
        "hsa_signal_store_screlease", None, [hsa_signal_t, ctypes.c_int64]
    )
    g["_hsa_signal_wait_scacquire"] = _decl(
        "hsa_signal_wait_scacquire", ctypes.c_int64,
        [hsa_signal_t, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64, ctypes.c_int],
    )

    _bindings_ready = True


def _check(status, what):
    if status != HSA_STATUS_SUCCESS:
        raise HSAError(f"{what} failed (hsa status {status})")


class HSAContext:
    """Process-wide singleton owning the HSA AIE device, memory, and queue."""

    _instance = None

    def __init__(self):
        _ensure_bindings()
        _check(_hsa_init(), "hsa_init")  # noqa: F821

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
            _hsa_agent_get_info(  # noqa: F821
                self.aie_agent, HSA_AGENT_INFO_QUEUE_MIN_SIZE,
                ctypes.byref(min_size),
            ),
            "hsa_agent_get_info(QUEUE_MIN_SIZE)",
        )
        qptr = ctypes.POINTER(HsaQueue)()
        _check(
            _hsa_queue_create(  # noqa: F821
                self.aie_agent, min_size.value, HSA_QUEUE_TYPE_SINGLE, None, None,
                0, 0, ctypes.byref(qptr),
            ),
            "hsa_queue_create",
        )
        self.queue = qptr

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
            s = _hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, ctypes.byref(dt))  # noqa: F821
            if s != HSA_STATUS_SUCCESS:
                return s
            if dt.value == device_type:
                found.value = agent
                return HSA_STATUS_INFO_BREAK
            return HSA_STATUS_SUCCESS

        status = _hsa_iterate_agents(cb, None)  # noqa: F821
        if status not in (HSA_STATUS_SUCCESS, HSA_STATUS_INFO_BREAK):
            raise HSAError(f"hsa_iterate_agents failed (hsa status {status})")
        return found.value

    def _find_region(self, agent):
        found = ctypes.c_uint64(0)

        @ctypes.CFUNCTYPE(ctypes.c_int, hsa_region_t, ctypes.c_void_p)
        def cb(region, _data):
            seg = ctypes.c_int()
            if _hsa_region_get_info(region, HSA_REGION_INFO_SEGMENT, ctypes.byref(seg)) != HSA_STATUS_SUCCESS:  # noqa: F821
                return HSA_STATUS_SUCCESS
            if seg.value != HSA_REGION_SEGMENT_GLOBAL:
                return HSA_STATUS_SUCCESS
            allowed = ctypes.c_bool()
            if _hsa_region_get_info(region, HSA_REGION_INFO_RUNTIME_ALLOC_ALLOWED, ctypes.byref(allowed)) != HSA_STATUS_SUCCESS:  # noqa: F821
                return HSA_STATUS_SUCCESS
            if not allowed.value:
                return HSA_STATUS_SUCCESS
            found.value = region
            return HSA_STATUS_INFO_BREAK

        status = _hsa_agent_iterate_regions(agent, cb, None)  # noqa: F821
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
            if _hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, ctypes.byref(seg)) != HSA_STATUS_SUCCESS:  # noqa: F821
                return HSA_STATUS_SUCCESS
            if seg.value != HSA_AMD_SEGMENT_GLOBAL:
                return HSA_STATUS_SUCCESS
            flags = ctypes.c_uint32()
            if _hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, ctypes.byref(flags)) != HSA_STATUS_SUCCESS:  # noqa: F821
                return HSA_STATUS_SUCCESS
            if (flags.value & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) == 0:
                return HSA_STATUS_SUCCESS
            rec = ctypes.c_size_t()
            if _hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, ctypes.byref(rec)) != HSA_STATUS_SUCCESS:  # noqa: F821
                return HSA_STATUS_SUCCESS
            if rec.value == 0:  # allocatable pools have a nonzero rec granule
                return HSA_STATUS_SUCCESS
            found.value = pool
            return HSA_STATUS_INFO_BREAK

        status = _hsa_amd_agent_iterate_memory_pools(agent, cb, None)  # noqa: F821
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
        if _hsa_agent_get_info(self.aie_agent, HSA_AGENT_INFO_NAME, name) == HSA_STATUS_SUCCESS:  # noqa: F821
            text = name.value.decode("utf-8", "replace").lower()
            if "phoenix" in text or "npu1" in text:
                return "npu1"
        return "npu2"

    def _pool_granule(self):
        gran = ctypes.c_size_t()
        _check(
            _hsa_amd_memory_pool_get_info(  # noqa: F821
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
            _hsa_memory_allocate(self.region, size, ctypes.byref(ptr)),  # noqa: F821
            "hsa_memory_allocate",
        )
        return ptr.value

    def free_region(self, ptr):
        if ptr:
            _hsa_memory_free(ctypes.c_void_p(ptr))  # noqa: F821

    # -- vmem memory (I/O + kernargs) -------------------------------------
    def vmem_alloc(self, size):
        granule = self.pool_granule
        size = ((size + granule - 1) // granule) * granule
        handle = hsa_amd_vmem_alloc_handle_t()
        _check(
            _hsa_amd_vmem_handle_create(  # noqa: F821
                self.pool, size, MEMORY_TYPE_PINNED, 0, ctypes.byref(handle),
            ),
            "hsa_amd_vmem_handle_create",
        )
        va = ctypes.c_void_p()
        _check(
            _hsa_amd_vmem_address_reserve_align(  # noqa: F821
                ctypes.byref(va), size, 0, 0, HSA_AMD_VMEM_ADDRESS_NO_REGISTER,
            ),
            "hsa_amd_vmem_address_reserve_align",
        )
        _check(
            _hsa_amd_vmem_map(va, size, 0, handle, 0),  # noqa: F821
            "hsa_amd_vmem_map",
        )
        descs = (HsaAmdMemoryAccessDesc * 2)(
            HsaAmdMemoryAccessDesc(HSA_ACCESS_PERMISSION_RW, self.cpu_agent),
            HsaAmdMemoryAccessDesc(HSA_ACCESS_PERMISSION_RW, self.aie_agent),
        )
        _check(
            _hsa_amd_vmem_set_access(va, size, descs, 2),  # noqa: F821
            "hsa_amd_vmem_set_access",
        )
        return handle.value, va.value, size

    def vmem_free(self, handle, va, size):
        if va:
            _hsa_amd_vmem_unmap(ctypes.c_void_p(va), size)  # noqa: F821
            _hsa_amd_vmem_address_free(ctypes.c_void_p(va), size)  # noqa: F821
        if handle:
            _hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t(handle))  # noqa: F821

    # -- signals -----------------------------------------------------------
    def create_signal(self, initial):
        sig = hsa_signal_t()
        _check(
            _hsa_signal_create(initial, 0, None, ctypes.byref(sig)),  # noqa: F821
            "hsa_signal_create",
        )
        return sig.value

    def destroy_signal(self, sig):
        if sig:
            _hsa_signal_destroy(hsa_signal_t(sig))  # noqa: F821

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, pdi_ptr, insts_ptr, insts_size, kernarg_ptr, num_kernargs, signal):
        pkt = HsaAieKernelDispatchPacket()
        pkt.header = (
            (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE)
            | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
            | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)
        )
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
        wr_idx = _hsa_queue_add_write_index_relaxed(q, 1)  # noqa: F821
        qsize = q.contents.size
        while wr_idx - _hsa_queue_load_read_index_scacquire(q) >= qsize:  # noqa: F821
            pass
        base = ctypes.cast(
            q.contents.base_address,
            ctypes.POINTER(HsaAieKernelDispatchPacket),
        )
        base[wr_idx % qsize] = pkt
        _hsa_signal_store_screlease(q.contents.doorbell_signal, wr_idx)  # noqa: F821

    def wait(self, signal):
        _hsa_signal_wait_scacquire(  # noqa: F821
            signal, HSA_SIGNAL_CONDITION_EQ, 0, ctypes.c_uint64(-1).value,
            HSA_WAIT_STATE_BLOCKED,
        )
