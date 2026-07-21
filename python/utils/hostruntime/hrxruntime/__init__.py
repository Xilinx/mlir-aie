# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ctypes bindings for the HRX C ABI (``libhrx.so``).

For the IRON host stack: it binds only the handful of ``hrx_*`` entry points
the dispatch path needs, and wraps them in a tiny ``HRXContext`` singleton that
owns the device + stream and creates/loads amdxdna executables.

Library discovery order for ``libhrx.so``:
  1. ``$HRX_LIBHRX``                       (explicit full path to the .so)
  2. ``$LIBHRX_DIR/libhrx.so``             (set by activate_env.sh)
  3. ``$HRX_DIR/lib/libhrx.so``            (HRX install prefix)
  4. plain ``libhrx.so`` via the loader (LD_LIBRARY_PATH)
"""

import ctypes
import logging
import os
from pathlib import Path

from .discovery import find_libhrx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enum / flag constants (mirror hrx_runtime.h; values match IREE HAL).
# ---------------------------------------------------------------------------
HRX_MEMORY_TYPE_HOST_LOCAL = 0x00000046
HRX_MEMORY_TYPE_DEVICE_VISIBLE = 0x00000010

HRX_BUFFER_USAGE_DEFAULT = 0x00000C03
HRX_BUFFER_USAGE_MAPPING_PERSISTENT = 0x02000000
HRX_BUFFER_USAGE_MAPPING_SCOPED = 0x01000000

HRX_MAP_READ = 0x01
HRX_MAP_WRITE = 0x02

# hrx_mapping_mode_t (hrx_buffer_map_with_mode). PERSISTENT keeps host_ptr valid
# across dispatches (requires HRX_BUFFER_USAGE_MAPPING_PERSISTENT).
HRX_MAPPING_MODE_SCOPED = 0x00000001
HRX_MAPPING_MODE_PERSISTENT = 0x00000002

HRX_DISPATCH_FLAG_NONE = 0

# hrx_amdxdna_executable_create ABI (hrx_amdxdna.h). Every record starts with
# (record_length, abi_version) so libhrx can stride/validate it; abi_version 0 is
# the only version defined today. libhrx exposes a distinct v0 macro per record
# type (run / entry-point / create-params); all three are 0, so a single value
# covers them, but we mirror the header names for clarity.
HRX_AMDXDNA_EXECUTABLE_RUN_ABI_VERSION_0 = 0
HRX_AMDXDNA_EXECUTABLE_ENTRY_POINT_ABI_VERSION_0 = 0
HRX_AMDXDNA_EXECUTABLE_CREATE_PARAMS_ABI_VERSION_0 = 0
HRX_AMDXDNA_CONTEXT_MODE_CREATE = 0
HRX_AMDXDNA_CONTEXT_MODE_REUSE = 1


class HRXError(RuntimeError):
    """Raised when an hrx_* call returns a non-OK status."""


# ---------------------------------------------------------------------------
# ctypes struct mirrors
# ---------------------------------------------------------------------------
class HrxDispatchConfig(ctypes.Structure):
    _fields_ = [
        ("workgroup_count", ctypes.c_uint32 * 3),
        ("workgroup_size", ctypes.c_uint32 * 3),
        ("subgroup_size", ctypes.c_uint32),
    ]


class HrxBufferRef(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("offset", ctypes.c_size_t),
        ("length", ctypes.c_size_t),
    ]


# --- hrx_amdxdna_executable_create parameter structs (hrx_amdxdna.h) ---------
# Borrowed byte storage: {const void* data; size_t data_length;}.
class HrxConstByteSpan(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("data_length", ctypes.c_size_t),
    ]


# Borrowed string storage: {const char* data; size_t size;}.
class HrxStringView(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_char_p),
        ("size", ctypes.c_size_t),
    ]


# One amdxdna executable run. HRX derives the patch table from |transaction|.
class HrxAmdxdnaExecutableRun(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("transaction", HrxConstByteSpan),
        ("data_payload", HrxConstByteSpan),
    ]


# One dispatchable entry point (CREATE selects a PDI from an xclbin).
class HrxAmdxdnaExecutableEntryPoint(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("name", HrxStringView),
        ("context_mode", ctypes.c_uint32),
        ("xclbin_ordinal", ctypes.c_uint32),
        ("pdi_ordinal", ctypes.c_uint32),
        ("source_line", ctypes.c_uint32),
        ("source_file", HrxStringView),
        ("runs", ctypes.POINTER(HrxAmdxdnaExecutableRun)),
        ("run_count", ctypes.c_size_t),
    ]


# Top-level create params: a set of xclbins + a set of entry points.
class HrxAmdxdnaExecutableCreateParams(ctypes.Structure):
    _fields_ = [
        ("record_length", ctypes.c_uint32),
        ("abi_version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
        ("xclbins", ctypes.POINTER(HrxConstByteSpan)),
        ("xclbin_count", ctypes.c_size_t),
        ("entry_points", ctypes.POINTER(HrxAmdxdnaExecutableEntryPoint)),
        ("entry_point_count", ctypes.c_size_t),
    ]


def _load_libhrx() -> ctypes.CDLL:
    last_err = None
    tried = []
    # Auto-detected path first (env hints + standard locations), then a bare
    # name so the dynamic loader's LD_LIBRARY_PATH search still works.
    for c in [find_libhrx(), "libhrx.so"]:
        if not c:
            continue
        tried.append(c)
        try:
            return ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            last_err = e
    raise HRXError(
        f"Could not load libhrx.so (tried: {tried}). "
        f"Install HRX to a standard location, set HRX_DIR/LIBHRX_DIR, or add "
        f"libhrx to LD_LIBRARY_PATH. Last error: {last_err}"
    )


# Status is an opaque pointer; NULL == OK.
_status_t = ctypes.c_void_p
_handle = ctypes.c_void_p

# ctypes handle to libhrx plus the bound C ABI entry points. These are populated
# lazily by _ensure_bindings() so that *importing* this package performs no
# dlopen and no device init. The cheap ``has_hrx`` capability probe in
# ``aie.utils`` imports the sibling ``discovery`` module, which drags in this
# package initializer; keeping it side-effect-free is what makes that probe as
# cheap and safe as a plain module import.
_lib = None
_bindings_ready = False


def _decl(fn, restype, argtypes):
    f = getattr(_lib, fn)
    f.restype = restype
    f.argtypes = argtypes
    return f


def _ensure_bindings():
    """Load libhrx and bind the C ABI (idempotent).

    Deferred until first real use (i.e. when an :class:`HRXContext` is created)
    so that merely importing this package -- as the ``discovery``-based
    ``has_hrx`` probe does transitively -- never dlopen()s a library or touches
    the device.
    """
    global _lib, _bindings_ready
    if _bindings_ready:
        return
    _lib = _load_libhrx()

    global _hrx_status_code, _hrx_status_to_string, _hrx_status_free_message
    global _hrx_status_ignore
    global _hrx_gpu_initialize, _hrx_gpu_device_get, _hrx_stream_create
    global _hrx_buffer_allocate, _hrx_buffer_map_with_mode
    global _hrx_buffer_flush_range, _hrx_buffer_invalidate_range
    global _hrx_buffer_release, _hrx_stream_copy_h2d
    global _hrx_amdxdna_executable_create, _hrx_executable_lookup_export_by_name
    global _hrx_executable_release, _hrx_stream_dispatch, _hrx_stream_synchronize

    # Status helpers
    _hrx_status_code = _decl("hrx_status_code", ctypes.c_int, [_status_t])
    _hrx_status_to_string = _decl(
        "hrx_status_to_string",
        _status_t,
        [_status_t, ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_size_t)],
    )
    _hrx_status_free_message = _decl("hrx_status_free_message", None, [ctypes.c_char_p])
    _hrx_status_ignore = _decl("hrx_status_ignore", None, [_status_t])

    # Lifecycle
    _hrx_gpu_initialize = _decl("hrx_gpu_initialize", _status_t, [ctypes.c_uint32])
    _hrx_gpu_device_get = _decl(
        "hrx_gpu_device_get", _status_t, [ctypes.c_int, ctypes.POINTER(_handle)]
    )
    _hrx_stream_create = _decl(
        "hrx_stream_create",
        _status_t,
        [_handle, ctypes.c_uint32, ctypes.POINTER(_handle)],
    )

    # Buffers
    _hrx_buffer_allocate = _decl(
        "hrx_buffer_allocate",
        _status_t,
        [
            _handle,
            ctypes.c_size_t,
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.POINTER(_handle),
        ],
    )
    # hrx_buffer_map_with_mode(buffer, hrx_mapping_mode_t (u32),
    #   hrx_map_flags_t (u16), size_t offset, size_t size, void** mapped_ptr).
    _hrx_buffer_map_with_mode = _decl(
        "hrx_buffer_map_with_mode",
        _status_t,
        [
            _handle,
            ctypes.c_uint32,
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_void_p),
        ],
    )
    _hrx_buffer_flush_range = _decl(
        "hrx_buffer_flush_range",
        _status_t,
        [_handle, ctypes.c_size_t, ctypes.c_size_t],
    )
    _hrx_buffer_invalidate_range = _decl(
        "hrx_buffer_invalidate_range",
        _status_t,
        [_handle, ctypes.c_size_t, ctypes.c_size_t],
    )
    _hrx_buffer_release = _decl("hrx_buffer_release", None, [_handle])
    _hrx_stream_copy_h2d = _decl(
        "hrx_stream_copy_h2d",
        _status_t,
        [_handle, ctypes.c_void_p, _handle, ctypes.c_size_t, ctypes.c_size_t],
    )

    # Executables: libhrx builds the amdxdna XADX package (and derives the patch
    # table from the transaction) internally, so we bind the high-level
    # hrx_amdxdna_executable_create instead of the raw hrx_executable_load_data.
    _hrx_amdxdna_executable_create = _decl(
        "hrx_amdxdna_executable_create",
        _status_t,
        [
            _handle,
            ctypes.POINTER(HrxAmdxdnaExecutableCreateParams),
            ctypes.POINTER(_handle),
        ],
    )
    _hrx_executable_lookup_export_by_name = _decl(
        "hrx_executable_lookup_export_by_name",
        _status_t,
        [_handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)],
    )
    _hrx_executable_release = _decl("hrx_executable_release", None, [_handle])

    # Dispatch / sync
    _hrx_stream_dispatch = _decl(
        "hrx_stream_dispatch",
        _status_t,
        [
            _handle,
            _handle,
            ctypes.c_uint32,
            ctypes.POINTER(HrxDispatchConfig),
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(HrxBufferRef),
            ctypes.c_size_t,
            ctypes.c_uint32,
        ],
    )
    _hrx_stream_synchronize = _decl("hrx_stream_synchronize", _status_t, [_handle])

    _bindings_ready = True


def _hrx_sync_timeout_s() -> float:
    """Read the optional IRON_HRX_TIMEOUT (seconds). 0/unset/invalid => disabled."""
    raw = os.environ.get("IRON_HRX_TIMEOUT")
    if not raw:
        return 0.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning("Ignoring invalid IRON_HRX_TIMEOUT=%r (want seconds)", raw)
        return 0.0


def _check(status, what: str):
    """Raise HRXError if status is non-OK (non-NULL)."""
    if not status:  # NULL == OK
        return
    msg_ptr = ctypes.c_char_p()
    msg_len = ctypes.c_size_t()
    s2 = _hrx_status_to_string(status, ctypes.byref(msg_ptr), ctypes.byref(msg_len))
    code = _hrx_status_code(status)
    text = msg_ptr.value.decode("utf-8", "replace") if msg_ptr.value else "?"
    # to_string may allocate the message; free it.
    if msg_ptr.value:
        _hrx_status_free_message(msg_ptr)
    _hrx_status_ignore(s2)
    _hrx_status_ignore(status)
    raise HRXError(f"{what} failed (hrx status code {code}): {text}")


# ---------------------------------------------------------------------------
# Control ELF -> transaction words
# ---------------------------------------------------------------------------
def control_code_from_elf(elf_bytes: bytes):
    """Extract the XAie transaction words (``.ctrltext``) from a control ELF.

    Only needed for an ELF input (``aiecc --aie-generate-elf``): ``.ctrltext``
    is the TXN stream verbatim, which is what libhrx patches from. A raw
    ``insts.bin`` is already the transaction and needs no extraction. libhrx
    derives the buffer patch table from the transaction itself inside
    ``hrx_amdxdna_executable_create``, so no host-side patch table is produced.

    Returns the control code as a ``numpy.uint32`` array.
    """
    import struct

    import numpy as np

    d = elf_bytes
    if len(d) < 52 or d[0] != 0x7F or d[1:4] != b"ELF":
        raise HRXError("control ELF is not a valid ELF32 file")

    def rd(fmt, off):
        return struct.unpack_from(fmt, d, off)[0]

    # ELF32 header fields (little-endian).
    e_shoff = rd("<I", 0x20)
    e_shentsize = rd("<H", 0x2E)
    e_shnum = rd("<H", 0x30)
    e_shstrndx = rd("<H", 0x32)

    def sh(i, f):
        return e_shoff + i * e_shentsize + f

    shstr_off = rd("<I", sh(e_shstrndx, 0x10))

    def sname(i):
        nm = rd("<I", sh(i, 0))
        end = d.index(b"\x00", shstr_off + nm)
        return d[shstr_off + nm : end].decode("ascii", "replace")

    for i in range(e_shnum):
        if sname(i) == ".ctrltext":
            coff = rd("<I", sh(i, 0x10))
            csize = rd("<I", sh(i, 0x14))
            return np.frombuffer(d[coff : coff + csize], dtype=np.uint32).copy()
    raise HRXError("control ELF has no .ctrltext section")


class HRXContext:
    """Process-wide singleton owning the HRX device + dispatch stream.

    The amdxdna NPU is a single shared device, so a single device/stream pair is
    shared across all tensors and kernels in the process.
    """

    _instance = None

    def __init__(self):
        # Load libhrx on first context creation (the import of this package
        # itself stays dlopen-free for the cheap has_hrx probe).
        _ensure_bindings()
        self.device = _handle()
        self.stream = _handle()
        _check(_hrx_gpu_initialize(0), "hrx_gpu_initialize")
        _check(_hrx_gpu_device_get(0, ctypes.byref(self.device)), "hrx_gpu_device_get")
        _check(
            _hrx_stream_create(self.device, 0, ctypes.byref(self.stream)),
            "hrx_stream_create",
        )

    @classmethod
    def get(cls) -> "HRXContext":
        if cls._instance is None:
            cls._instance = HRXContext()
        return cls._instance

    # -- buffers -----------------------------------------------------------
    def allocate_persistent(self, size: int):
        """Allocate a device-visible, host-coherent BO and map it persistently.

        Returns (buffer_handle, host_ptr). Coherence is maintained explicitly
        via flush_range / invalidate_range.
        """
        buf = _handle()
        _check(
            _hrx_buffer_allocate(
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
            _hrx_buffer_map_with_mode(
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
        _check(
            _hrx_buffer_flush_range(
                buf, ctypes.c_size_t(offset), ctypes.c_size_t(size)
            ),
            "hrx_buffer_flush_range",
        )

    def invalidate_range(self, buf, offset: int, size: int):
        _check(
            _hrx_buffer_invalidate_range(
                buf, ctypes.c_size_t(offset), ctypes.c_size_t(size)
            ),
            "hrx_buffer_invalidate_range",
        )

    def release_buffer(self, buf):
        if buf:
            _hrx_buffer_release(buf)

    # -- executables -------------------------------------------------------
    def create_executable(
        self, xclbin_bytes: bytes, insts_bytes: bytes, entry_name: str
    ):
        """Create + load an amdxdna executable from the raw artifacts.

        Hands libhrx the xclbin bytes and the XAie transaction (the raw
        ``insts.bin`` words) via ``hrx_amdxdna_executable_create``.

        ``insts_bytes`` must be the raw TXN stream (uint32 words); its length has
        to be a multiple of 4 (libhrx rejects otherwise). Returns the loaded
        ``hrx_executable_t`` handle.
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
            _hrx_amdxdna_executable_create(
                self.device, ctypes.byref(params), ctypes.byref(exe)
            ),
            "hrx_amdxdna_executable_create",
        )
        return exe

    def lookup_export(self, exe, name: str) -> int:
        ordv = ctypes.c_uint32()
        _check(
            _hrx_executable_lookup_export_by_name(
                exe, name.encode("utf-8"), ctypes.byref(ordv)
            ),
            "hrx_executable_lookup_export_by_name",
        )
        return ordv.value

    def release_executable(self, exe):
        if exe:
            _hrx_executable_release(exe)

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, exe, export_ordinal: int, bindings):
        """Dispatch `exe` with `bindings` (list of (buffer_handle, size)).

        cfg is the unit config the amdxdna path expects ({1,1,1}/{1,1,1}); the
        I/O addresses are bound by binding order + the TXN DDR-patch ops.
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
            _hrx_stream_dispatch(
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

        ``items`` is an iterable of ``(executable, export_ordinal, bindings)``,
        where ``bindings`` is a list of ``(buffer_handle, size)`` (same shape
        :meth:`dispatch` takes). Each :meth:`dispatch` records into the stream's
        pending command buffer, and HRX inserts an execution + memory barrier
        after every dispatch, so a later dispatch observes an earlier one's
        device writes (producer -> consumer chains are correct). The whole batch
        stays pending until
        :meth:`synchronize`, which submits it as a single execution — the
        amdxdna HAL lowers a multi-dispatch command buffer into one
        ``ERT_CMD_CHAIN`` issued/waited once.

        Records only; call :meth:`synchronize` to submit and wait.
        """
        for exe, export_ordinal, bindings in items:
            self.dispatch(exe, export_ordinal, bindings)

    def synchronize(self):
        timeout = _hrx_sync_timeout_s()
        if timeout <= 0:
            # Default: block until libhrx returns (unchanged behavior).
            _check(_hrx_stream_synchronize(self.stream), "hrx_stream_synchronize")
            return

        # Best-effort watchdog: run the blocking sync on a helper thread and
        # bound the *wait* with a timeout. NOTE: libhrx's hrx_stream_synchronize
        # cannot be cancelled, so on expiry the underlying call keeps running in
        # the daemon thread -- this raises a diagnosable error instead of
        # hanging forever, it does not abort the device work.
        import threading

        result = {}

        def _worker():
            try:
                _check(_hrx_stream_synchronize(self.stream), "hrx_stream_synchronize")
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
