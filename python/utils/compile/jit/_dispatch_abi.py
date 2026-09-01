# _dispatch_abi.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The call ABI of a compiled dispatch bridge.

``ConvertAIEXToEmitC`` emits both the ``dispatch_generate`` entry point and a
``dispatch_abi()`` naming the C type of each of its scalar parameters -- it
builds them from the ``aie.runtime_sequence`` argument types, so MLIR is the
one authority on what the compiled ``.so`` expects and nothing on this side
parses generated C++.

This module reads that string back. ``compile_dispatch_bridge()``
(_dispatch_compile.py) uses it to check the declared ``DispatchTime[T]`` types
against what was actually built; ``DispatchBridge`` (_dispatch_bridge.py) uses
it to build the ``ctypes`` call signature. Those need one shared C-type table,
so it lives here.

Deliberately imports nothing from ``aie.utils.hostruntime`` or
``aie.utils.npukernel``: _dispatch_bridge reaches into hostruntime for
``HostRuntimeError``, and hostruntime imports ``NPUKernel``, so a shared module
that pulled in either would close an import cycle.
"""

from __future__ import annotations

import ctypes

import numpy as np

# The C symbol the generated code exports to describe itself, and the
# aie-translate flag that emits it. Kept beside each other because a .so built
# without the flag has no ABI to read.
DISPATCH_ABI_SYMBOL = "dispatch_abi"
EMIT_DISPATCH_SHIM_FLAG = "--aie-npu-emit-dispatch-shim"

# C type spelling (as aie-translate emits it) -> (ctypes type, the
# ``DispatchTime[T]`` wrapped type that maps to it, or None if none does).
# Small and fixed: the generated builder only ever takes scalar runtime-sequence
# arguments (i32/index), never anything more exotic. ``size_t`` is what
# ``index`` becomes and has no DispatchTime[T] spelling of its own.
_SCALAR_C_TYPES: dict[str, tuple[type, type | None]] = {
    "int8_t": (ctypes.c_int8, np.int8),
    "uint8_t": (ctypes.c_uint8, np.uint8),
    "int16_t": (ctypes.c_int16, np.int16),
    "uint16_t": (ctypes.c_uint16, np.uint16),
    "int32_t": (ctypes.c_int32, np.int32),
    "uint32_t": (ctypes.c_uint32, np.uint32),
    "int64_t": (ctypes.c_int64, np.int64),
    "uint64_t": (ctypes.c_uint64, np.uint64),
    "size_t": (ctypes.c_size_t, None),
}

C_TYPE_BY_NP_TYPE = {
    np_type: c_name
    for c_name, (_ctype, np_type) in _SCALAR_C_TYPES.items()
    if np_type is not None
}

KNOWN_C_TYPES = sorted(_SCALAR_C_TYPES)


def ctypes_for(c_type_name: str) -> type | None:
    """Return the ``ctypes`` type for a generated C type, or ``None`` if unknown."""
    entry = _SCALAR_C_TYPES.get(c_type_name)
    return entry[0] if entry else None


def value_range(ctype: type) -> tuple[int, int]:
    """Inclusive ``(low, high)`` a *ctype* can hold, derived from the type itself."""
    bits = ctypes.sizeof(ctype) * 8
    if ctype(-1).value < 0:  # type: ignore[call-arg]
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def read_dispatch_abi(lib: ctypes.CDLL, so_path) -> list[str]:
    """C type names an already-loaded dispatch ``.so`` reports for its params.

    Raises ``ValueError`` if the ``.so`` exports no ``dispatch_abi()`` (built
    without ``EMIT_DISPATCH_SHIM_FLAG``, so it has no entry point either) or
    reports something unparseable.
    """
    try:
        abi_fn = getattr(lib, DISPATCH_ABI_SYMBOL)
    except AttributeError:
        raise ValueError(
            f"{so_path} exports no {DISPATCH_ABI_SYMBOL}(); it was not built "
            f"with {EMIT_DISPATCH_SHIM_FLAG} -- clear the compiled kernel "
            "directory and rebuild."
        ) from None
    abi_fn.restype = ctypes.c_char_p
    text = abi_fn().decode()
    if not text.strip():
        return []
    c_types = [field.strip() for field in text.split(",")]
    if not all(c_types):
        raise ValueError(f"{so_path} reports a malformed dispatch ABI {text!r}")
    return c_types
