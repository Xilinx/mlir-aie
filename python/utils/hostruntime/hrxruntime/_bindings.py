# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Low-level ctypes bindings for the HRX C ABI (``libhrx.so``).

This module owns everything at the C-ABI boundary: the enum/flag constants and
``ctypes`` struct mirrors from ``hrx_runtime.h`` / ``hrx_amdxdna.h``, library
discovery + ``dlopen``, and the bound ``hrx_*`` entry points. The higher-level
device/stream/buffer/dispatch orchestration lives in :mod:`.context`
(:class:`~.context.HRXContext`); the package ``__init__`` re-exports both.

Library discovery order for ``libhrx.so``:
  1. ``$HRX_LIBHRX``                       (explicit full path to the .so)
  2. ``$LIBHRX_DIR/libhrx.so``             (set by activate_env.sh)
  3. ``$HRX_DIR/lib/libhrx.so``            (HRX install prefix)
  4. plain ``libhrx.so`` via the loader (LD_LIBRARY_PATH)

Importing this module is side-effect-free: it performs no ``dlopen`` and no
device init. Binding is deferred to :meth:`_HrxLib.ensure`, which the first
:class:`~.context.HRXContext` triggers. That is what lets the cheap ``has_hrx``
capability probe in ``aie.utils`` (which only imports the sibling
:mod:`.discovery` module) stay as cheap and safe as a plain import.
"""

import ctypes
import logging
import os
import platform
import struct
import threading

import numpy as np

from .discovery import find_libhrx

_IS_WINDOWS = platform.system() == "Windows"

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


# Status is an opaque pointer; NULL == OK.
_status_t = ctypes.c_void_p
_handle = ctypes.c_void_p


def _load_libhrx() -> ctypes.CDLL:
    last_err = None
    tried = []

    if _IS_WINDOWS:
        # Windows has no RTLD_GLOBAL, and DLL dependency resolution is driven by
        # the loader's search path -- not PATH alone -- so register the library's
        # own directory (and any LIBHRX_DIR hint) with os.add_dll_directory so
        # hrx.dll's sibling dependencies resolve, then load it by full path.
        libhrx_dir = os.environ.get("LIBHRX_DIR")
        if libhrx_dir and os.path.isdir(libhrx_dir):
            try:
                # os.add_dll_directory is Windows-only (absent from the Linux
                # typeshed stub pyright checks against), hence the ignore.
                os.add_dll_directory(  # pyright: ignore[reportAttributeAccessIssue]
                    libhrx_dir
                )
            except OSError:
                pass
        found = find_libhrx()
        # Auto-detected full path first, then bare names via the default search.
        for c in [found, "hrx.dll", "libhrx.dll"]:
            if not c:
                continue
            tried.append(c)
            dll_dir = os.path.dirname(c)
            if dll_dir and os.path.isdir(dll_dir):
                try:
                    os.add_dll_directory(  # pyright: ignore[reportAttributeAccessIssue]
                        dll_dir
                    )
                except OSError:
                    pass
            try:
                return ctypes.CDLL(c)
            except OSError as e:
                last_err = e
        raise HRXError(
            f"Could not load hrx.dll (tried: {tried}). Install HRX or set "
            f"HRX_DIR/LIBHRX_DIR/HRX_LIBHRX to the release that contains "
            f"bin\\hrx.dll. Last error: {last_err}"
        )

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


class _HrxLib:
    """The dlopen'd ``libhrx`` handle plus its bound C ABI entry points.

    A single process-wide instance (:data:`lib`) is bound lazily by
    :meth:`ensure`; until then no library is opened. Every bound entry point is
    exposed as an attribute (e.g. ``lib.hrx_stream_dispatch``) so callers in
    :mod:`.context` reference one populated object rather than a set of
    rebindable module globals.
    """

    def __init__(self):
        self._cdll = None
        self._ready = False
        # Guards the one-time bind so concurrent first-touch from multiple
        # threads cannot dlopen / bind twice or observe a half-bound library.
        self._lock = threading.Lock()

    def ensure(self) -> None:
        """Load libhrx and bind the C ABI (idempotent, thread-safe).

        Deferred until first real use (i.e. when an :class:`~.context.HRXContext`
        is created) so that merely importing this package -- as the
        ``discovery``-based ``has_hrx`` probe does transitively -- never
        ``dlopen()``s a library or touches the device. Double-checked locking
        keeps the fast path lock-free once bound.
        """
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._bind()

    def _bind(self) -> None:
        lib = _load_libhrx()
        self._cdll = lib

        def decl(fn, restype, argtypes):
            f = getattr(lib, fn)
            f.restype = restype
            f.argtypes = argtypes
            return f

        # Status helpers
        self.hrx_status_code = decl("hrx_status_code", ctypes.c_int, [_status_t])
        self.hrx_status_to_string = decl(
            "hrx_status_to_string",
            _status_t,
            [
                _status_t,
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.POINTER(ctypes.c_size_t),
            ],
        )
        self.hrx_status_free_message = decl(
            "hrx_status_free_message", None, [ctypes.c_char_p]
        )
        self.hrx_status_ignore = decl("hrx_status_ignore", None, [_status_t])

        # Lifecycle
        self.hrx_gpu_initialize = decl(
            "hrx_gpu_initialize", _status_t, [ctypes.c_uint32]
        )
        self.hrx_gpu_device_get = decl(
            "hrx_gpu_device_get", _status_t, [ctypes.c_int, ctypes.POINTER(_handle)]
        )
        self.hrx_stream_create = decl(
            "hrx_stream_create",
            _status_t,
            [_handle, ctypes.c_uint32, ctypes.POINTER(_handle)],
        )

        # Buffers
        self.hrx_buffer_allocate = decl(
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
        self.hrx_buffer_map_with_mode = decl(
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
        self.hrx_buffer_flush_range = decl(
            "hrx_buffer_flush_range",
            _status_t,
            [_handle, ctypes.c_size_t, ctypes.c_size_t],
        )
        self.hrx_buffer_invalidate_range = decl(
            "hrx_buffer_invalidate_range",
            _status_t,
            [_handle, ctypes.c_size_t, ctypes.c_size_t],
        )
        self.hrx_buffer_release = decl("hrx_buffer_release", None, [_handle])
        self.hrx_stream_copy_h2d = decl(
            "hrx_stream_copy_h2d",
            _status_t,
            [_handle, ctypes.c_void_p, _handle, ctypes.c_size_t, ctypes.c_size_t],
        )

        # Executables: libhrx builds the amdxdna XADX package (and derives the
        # patch table from the transaction) internally, so we bind the
        # high-level hrx_amdxdna_executable_create instead of the raw
        # hrx_executable_load_data.
        self.hrx_amdxdna_executable_create = decl(
            "hrx_amdxdna_executable_create",
            _status_t,
            [
                _handle,
                ctypes.POINTER(HrxAmdxdnaExecutableCreateParams),
                ctypes.POINTER(_handle),
            ],
        )
        self.hrx_executable_lookup_export_by_name = decl(
            "hrx_executable_lookup_export_by_name",
            _status_t,
            [_handle, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)],
        )
        self.hrx_executable_release = decl("hrx_executable_release", None, [_handle])

        # Dispatch / sync
        self.hrx_stream_dispatch = decl(
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
        self.hrx_stream_synchronize = decl(
            "hrx_stream_synchronize", _status_t, [_handle]
        )

        self._ready = True


# Process-wide libhrx handle, bound lazily by lib.ensure().
lib = _HrxLib()


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
    s2 = lib.hrx_status_to_string(status, ctypes.byref(msg_ptr), ctypes.byref(msg_len))
    code = lib.hrx_status_code(status)
    text = msg_ptr.value.decode("utf-8", "replace") if msg_ptr.value else "?"
    # to_string may allocate the message; free it.
    if msg_ptr.value:
        lib.hrx_status_free_message(msg_ptr)
    lib.hrx_status_ignore(s2)
    lib.hrx_status_ignore(status)
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
