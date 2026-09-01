# _dispatch_bridge.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``DispatchBridge``: the Python/ctypes side of a compiled dispatch bridge.

Loads the ``dispatch.so`` that ``compile_dispatch_bridge()`` (_dispatch_compile.py)
produces and calls its ``dispatch_generate`` entry point to synthesize a fresh
instruction word stream from ``DispatchTime[T]`` scalar values, once per NPU
dispatch. ``dispatch_generate`` already knows its exact output size (it built
a complete ``std::vector`` before returning) and hands back a pointer +
count into its own thread-local storage -- no buffer-capacity guessing on
this side. See ``_dispatch_compile.py`` for the ``-2`` guard-failed sentinel
this wraps.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
from aie.utils.hostruntime.hostruntime import HostRuntimeError

# C type spelling (as aie-translate emits it) -> ctypes type. Small, fixed
# table because the generated function only ever takes scalar runtime-
# sequence arguments (i32/index), never anything more exotic.
_CTYPES_BY_NAME = {
    "int8_t": ctypes.c_int8,
    "uint8_t": ctypes.c_uint8,
    "int16_t": ctypes.c_int16,
    "uint16_t": ctypes.c_uint16,
    "int32_t": ctypes.c_int32,
    "uint32_t": ctypes.c_uint32,
    "int64_t": ctypes.c_int64,
    "uint64_t": ctypes.c_uint64,
    "size_t": ctypes.c_size_t,
}


def _ctypes_for(c_type_name: str) -> type:
    try:
        return _CTYPES_BY_NAME[c_type_name]
    except KeyError:
        raise HostRuntimeError(
            f"dispatch bridge: unrecognized generated C type {c_type_name!r}; "
            f"known types are {sorted(_CTYPES_BY_NAME)}."
        ) from None


class DispatchBridge:
    """Loads a compiled dispatch bridge once; regenerates instruction words per call.

    Args:
        so_path: Path to the compiled ``dispatch.so``.
        dispatch_params: Declared ``DispatchTime[T]`` parameter names, in the
            same order as the generated function's parameters (validated by
            ``compile_dispatch_bridge()`` (_dispatch_compile.py) at compile time).
        param_ctypes: The C type spelling (e.g. ``"int32_t"``) for each entry
            in ``dispatch_params``, in order -- trusts what the generated C++
            actually expects rather than re-deriving from the Python
            ``DispatchTime[T]`` wrapped type.
    """

    def __init__(
        self,
        so_path: Path,
        dispatch_params: list[str],
        param_ctypes: list[str],
    ):
        if len(dispatch_params) != len(param_ctypes):
            raise ValueError(
                f"dispatch_params ({len(dispatch_params)}) and param_ctypes "
                f"({len(param_ctypes)}) must have the same length"
            )
        self._lib = ctypes.CDLL(str(so_path))
        self._dispatch_params = list(dispatch_params)
        self._arg_ctypes = [_ctypes_for(c) for c in param_ctypes]
        self._lib.dispatch_generate.restype = ctypes.c_int64
        self._lib.dispatch_generate.argtypes = [
            *self._arg_ctypes,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
        ]

    def generate(self, values: dict[str, Any]) -> np.ndarray:
        """Return a fresh ``uint32`` instruction word array for *values*.

        Raises ``HostRuntimeError`` if a value overflowed a hardware BD
        field -- the generator itself returned no stream.
        """
        ordered = [
            ctype(values[name])
            for name, ctype in zip(self._dispatch_params, self._arg_ctypes)
        ]
        out_ptr = ctypes.POINTER(ctypes.c_uint32)()
        n = self._lib.dispatch_generate(*ordered, ctypes.byref(out_ptr))
        if n == -2:
            raise HostRuntimeError(
                f"DispatchTime[T] scalar value(s) {values!r} overflowed a "
                "hardware BD field for this compiled design; try a "
                "smaller value, or a design compiled with a larger "
                "CompileTime[T] bound."
            )
        # Past this point n indexes raw memory, so anything the contract does
        # not allow has to become an exception rather than a bad read.
        if n < 0:
            raise HostRuntimeError(
                f"dispatch bridge returned unexpected status {n} for {values!r}; "
                "the compiled dispatch.so does not match this design."
            )
        if n == 0:
            return np.empty(0, dtype=np.uint32)
        if not out_ptr:
            raise HostRuntimeError(
                f"dispatch bridge reported {n} instruction word(s) for {values!r} "
                "but returned a null pointer."
            )
        # out_ptr points into the wrapper's thread-local storage, valid only
        # until the next call on this thread -- copy out immediately.
        return np.ctypeslib.as_array(out_ptr, shape=(n,)).copy()
