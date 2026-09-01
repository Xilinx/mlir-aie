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
count into its own thread-local storage, so this side never has to guess a
buffer capacity. The call signature comes from the ``.so``'s own
``dispatch_abi()`` export, so the ``.so`` is the only artifact a cached kernel
directory needs. See ``_dispatch_compile.py`` for the ``-2`` guard-failed
sentinel this wraps.
"""

from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Any

import numpy as np
from aie.utils.hostruntime.hostruntime import HostRuntimeError

from ._dispatch_abi import (
    KNOWN_C_TYPES,
    ctypes_for,
    read_dispatch_abi,
    value_range,
)


def _ctypes_for(c_type_name: str) -> type:
    ctype = ctypes_for(c_type_name)
    if ctype is None:
        raise HostRuntimeError(
            f"dispatch bridge: unrecognized generated C type {c_type_name!r}; "
            f"known types are {KNOWN_C_TYPES}."
        )
    return ctype


class DispatchBridge:
    """Loads a compiled dispatch bridge once; regenerates instruction words per call.

    Args:
        so_path: Path to the compiled ``dispatch.so``.
        dispatch_params: Declared ``DispatchTime[T]`` parameter names, in the
            same order as the generated function's parameters (validated by
            ``compile_dispatch_bridge()`` (_dispatch_compile.py) at compile time).
            The C types come from the ``.so``'s own ``dispatch_abi()``, so only
            the names are needed here.
    """

    def __init__(self, so_path: Path, dispatch_params: list[str]):
        try:
            self._lib = ctypes.CDLL(str(so_path))
        except OSError as e:
            raise HostRuntimeError(
                f"dispatch bridge at {so_path} could not be loaded ({e}); the "
                "compiled kernel directory is unusable -- clear it and rebuild."
            ) from None
        try:
            param_ctypes = read_dispatch_abi(self._lib, so_path)
        except ValueError as e:
            raise HostRuntimeError(f"dispatch bridge: {e}") from None
        # A count mismatch means this .so was built for a different signature:
        # rebuild it rather than guess which parameters line up.
        if len(param_ctypes) != len(dispatch_params):
            raise HostRuntimeError(
                f"dispatch bridge at {so_path} takes {len(param_ctypes)} scalar "
                f"parameter(s) but this design declares {len(dispatch_params)} "
                f"({list(dispatch_params)!r}); the cached artifact is stale."
            )
        # Ranges are fixed once the ABI is known; deriving them per dispatch
        # would redo this work on every call.
        self._arg_spec = [
            (name, ctype, *value_range(ctype))
            for name, ctype in zip(
                dispatch_params, (_ctypes_for(c) for c in param_ctypes)
            )
        ]
        self._lib.dispatch_generate.restype = ctypes.c_int64
        self._lib.dispatch_generate.argtypes = [
            *(ctype for _n, ctype, _lo, _hi in self._arg_spec),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint32)),
        ]

    def generate(self, values: dict[str, Any]) -> np.ndarray:
        """Return a fresh ``uint32`` instruction word array for *values*.

        Raises ``HostRuntimeError`` if a value does not fit its generated C
        parameter, or if it overflowed a hardware BD field -- in which case
        the generator itself returned no stream.
        """
        ordered = []
        for name, ctype, low, high in self._arg_spec:
            value = values[name]
            # ctypes truncates silently: c_int32(2**31) is -2147483648 and
            # c_int32(2**70) is 0, either of which would dispatch a
            # plausible-looking stream built from a value the caller never
            # asked for.
            if not low <= int(value) <= high:
                raise HostRuntimeError(
                    f"DispatchTime[T] value {name}={value!r} does not fit its "
                    f"generated C parameter type (valid range {low}..{high}); "
                    "passing it would silently wrap to a different value."
                )
            ordered.append(ctype(int(value)))
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
