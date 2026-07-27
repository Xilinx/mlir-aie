# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Low-level ctypes bindings for the HSA/ROCR C ABI (``libhsa-runtime64.so``).

This module owns everything at the C-ABI boundary: the enum/flag constants and
``ctypes`` struct mirrors from ``hsa.h`` / ``hsa_ext_amd.h``, library discovery
+ ``dlopen``, and the bound ``hsa_*`` entry points. The higher-level device /
memory / queue orchestration lives alongside :class:`HSAContext` in the package
``__init__``.

Importing this module is side-effect-free: it performs no ``dlopen`` and no
device init. Binding is deferred to :meth:`_HsaLib._ensure`, which the first
``HSAContext`` triggers. That is what lets the cheap ``hsa_available`` probe
in ``aie.utils`` (which only imports the sibling :mod:`.discovery` module)
stay as cheap and safe as a plain import.
"""

import ctypes
import logging
import os
import threading

from .discovery import find_libhsa

logger = logging.getLogger(__name__)

# --- hsa.h constants -------------------------------------------------------
HSA_STATUS_SUCCESS = 0
HSA_STATUS_INFO_BREAK = 0x1

HSA_DEVICE_TYPE_CPU = 0
HSA_DEVICE_TYPE_AIE = 3  # from hsa_device_type_t (CPU=0, GPU=1, DSP=2, AIE=3)

HSA_AGENT_INFO_DEVICE = 17
HSA_AGENT_INFO_QUEUE_MIN_SIZE = 13
HSA_AGENT_INFO_NAME = 0

HSA_REGION_INFO_SEGMENT = 0
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

# AQL packet header: READY type with system-scope acquire/release fences. All
# operands are constants, so build it once at import instead of per dispatch.
_DISPATCH_HEADER = (
    (HSA_AMD_AIE_PACKET_TYPE_READY << HSA_PACKET_HEADER_TYPE)
    | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
    | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)
)
# Block indefinitely (UINT64_MAX timeout) in hsa_signal_wait_scacquire.
_HSA_WAIT_FOREVER = 0xFFFFFFFFFFFFFFFF

# Opaque handle-carrying structs are all {uint64_t handle}.
hsa_agent_t = ctypes.c_uint64
hsa_region_t = ctypes.c_uint64
hsa_amd_memory_pool_t = ctypes.c_uint64
hsa_signal_t = ctypes.c_int64  # signal handle is a 64-bit value
hsa_amd_vmem_alloc_handle_t = ctypes.c_uint64


class HSAError(RuntimeError):
    """Raised when an hsa_* call returns a non-success status."""


class HSATimeoutError(HSAError):
    """Raised when a signal wait exceeds IRON_HSA_TIMEOUT.

    The underlying hsa_signal_wait cannot be cancelled, so the dispatch is still
    pending on the device when this is raised. The device retains ownership of
    the completion signal and any in-flight buffers, so callers must NOT free
    those on this path (doing so is a use-after-free / destroys a signal another
    thread still waits on) -- they are intentionally leaked until the device is
    recovered (e.g. driver reload / process teardown).
    """


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


class _HsaLib:
    """Lazily-bound handle to libhsa-runtime64.so.

    Importing this module performs no dlopen; the first attribute access that
    needs a bound function triggers _bind(), which dlopens libhsa and declares
    the hsa_* entry points as attributes. This keeps the cheap has_hsa probe
    (which only imports .discovery) free of any device/library side effects.
    """

    def __init__(self):
        self._cdll = None
        self._ready = False
        self._lock = threading.Lock()

    def _ensure(self):
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._bind()
            self._ready = True

    def __getattr__(self, name):
        # Only reached for names not yet set as instance attributes.
        if name.startswith("hsa_"):
            self._ensure()
            return object.__getattribute__(self, name)
        raise AttributeError(name)

    def _bind(self):
        cdll = _load_libhsa()
        self._cdll = cdll

        def decl(fn, restype, argtypes):
            f = getattr(cdll, fn)
            f.restype = restype
            f.argtypes = argtypes
            setattr(self, fn, f)

        decl("hsa_init", ctypes.c_int, [])
        decl("hsa_shut_down", ctypes.c_int, [])
        decl(
            "hsa_iterate_agents", ctypes.c_int,
            [ctypes.CFUNCTYPE(ctypes.c_int, hsa_agent_t, ctypes.c_void_p), ctypes.c_void_p],
        )
        decl(
            "hsa_agent_get_info", ctypes.c_int,
            [hsa_agent_t, ctypes.c_int, ctypes.c_void_p],
        )
        decl(
            "hsa_agent_iterate_regions", ctypes.c_int,
            [hsa_agent_t, ctypes.CFUNCTYPE(ctypes.c_int, hsa_region_t, ctypes.c_void_p),
             ctypes.c_void_p],
        )
        decl(
            "hsa_region_get_info", ctypes.c_int,
            [hsa_region_t, ctypes.c_int, ctypes.c_void_p],
        )
        decl(
            "hsa_memory_allocate", ctypes.c_int,
            [hsa_region_t, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)],
        )
        decl("hsa_memory_free", ctypes.c_int, [ctypes.c_void_p])

        decl(
            "hsa_amd_agent_iterate_memory_pools", ctypes.c_int,
            [hsa_agent_t,
             ctypes.CFUNCTYPE(ctypes.c_int, hsa_amd_memory_pool_t, ctypes.c_void_p),
             ctypes.c_void_p],
        )
        decl(
            "hsa_amd_memory_pool_get_info", ctypes.c_int,
            [hsa_amd_memory_pool_t, ctypes.c_int, ctypes.c_void_p],
        )
        decl(
            "hsa_amd_vmem_handle_create", ctypes.c_int,
            [hsa_amd_memory_pool_t, ctypes.c_size_t, ctypes.c_int, ctypes.c_uint64,
             ctypes.POINTER(hsa_amd_vmem_alloc_handle_t)],
        )
        decl(
            "hsa_amd_vmem_handle_release", ctypes.c_int, [hsa_amd_vmem_alloc_handle_t]
        )
        decl(
            "hsa_amd_vmem_address_reserve_align", ctypes.c_int,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint64,
             ctypes.c_uint64, ctypes.c_uint64],
        )
        decl(
            "hsa_amd_vmem_address_free", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_size_t],
        )
        decl(
            "hsa_amd_vmem_map", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
             hsa_amd_vmem_alloc_handle_t, ctypes.c_uint64],
        )
        decl(
            "hsa_amd_vmem_unmap", ctypes.c_int, [ctypes.c_void_p, ctypes.c_size_t]
        )
        decl(
            "hsa_amd_vmem_set_access", ctypes.c_int,
            [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(HsaAmdMemoryAccessDesc),
             ctypes.c_size_t],
        )

        decl(
            "hsa_queue_create", ctypes.c_int,
            [hsa_agent_t, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
             ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32,
             ctypes.POINTER(ctypes.POINTER(HsaQueue))],
        )
        decl(
            "hsa_queue_destroy", ctypes.c_int, [ctypes.POINTER(HsaQueue)]
        )
        decl(
            "hsa_queue_add_write_index_relaxed", ctypes.c_uint64,
            [ctypes.POINTER(HsaQueue), ctypes.c_uint64],
        )
        decl(
            "hsa_queue_load_read_index_scacquire", ctypes.c_uint64,
            [ctypes.POINTER(HsaQueue)],
        )
        decl(
            "hsa_signal_create", ctypes.c_int,
            [ctypes.c_int64, ctypes.c_uint32, ctypes.c_void_p,
             ctypes.POINTER(hsa_signal_t)],
        )
        decl("hsa_signal_destroy", ctypes.c_int, [hsa_signal_t])
        decl(
            "hsa_signal_store_screlease", None, [hsa_signal_t, ctypes.c_int64]
        )
        decl(
            "hsa_signal_wait_scacquire", ctypes.c_int64,
            [hsa_signal_t, ctypes.c_int, ctypes.c_int64, ctypes.c_uint64, ctypes.c_int],
        )


lib = _HsaLib()


def _check(status, what):
    if status != HSA_STATUS_SUCCESS:
        raise HSAError(f"{what} failed (hsa status {status})")


def _hsa_sync_timeout_s() -> float:
    """Read the optional IRON_HSA_TIMEOUT (seconds). 0/unset/invalid => disabled."""
    raw = os.environ.get("IRON_HSA_TIMEOUT")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid IRON_HSA_TIMEOUT=%r (want seconds)", raw)
        return 0.0
