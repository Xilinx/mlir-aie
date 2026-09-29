# _dispatch_bridge.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""``DispatchBridge``: the Python/ctypes side of a compiled dispatch bridge.

Loads the immutable ``dispatch-<digest>.so`` (``.dll`` on Windows) generation
that ``compile_dispatch_bridge()`` produces and calls its ``dispatch_generate`` entry point
once per NPU dispatch to synthesize a fresh instruction word stream from
``DispatchTime[T]`` scalar values.

``ConvertAIEXToEmitC`` emits a ``dispatch_abi()`` naming the C type of each
scalar parameter, so the library describes its own call signature.
"""

from __future__ import annotations

import ctypes
import operator
from pathlib import Path
from typing import Any

import numpy as np
from aie.utils.hostruntime.hostruntime import HostRuntimeError

# The C symbol the generated code exports to describe itself, and the
# aiecc flag that emits it. A .so built without the flag has no ABI to
# read and no entry point either.
DISPATCH_ABI_SYMBOL = "dispatch_abi"
EMIT_DISPATCH_SHIM_FLAG = "--npu-cpp-emit-dispatch-shim"

# Covers everything AIEXToEmitC.cpp's cTypeName can emit.
_SCALAR_C_TYPES: dict[str, type] = {
    "bool": ctypes.c_bool,
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


def _value_range(ctype: type) -> tuple[int, int]:
    """Inclusive ``(low, high)`` a *ctype* can hold, derived from the type itself."""
    if ctype is ctypes.c_bool:
        return 0, 1
    bits = ctypes.sizeof(ctype) * 8
    if ctype(-1).value < 0:  # type: ignore[call-arg]
        return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return 0, 2**bits - 1


def read_dispatch_abi(lib: ctypes.CDLL, so_path) -> list[str]:
    """C type names an already-loaded dispatch library reports for its params.

    Raises ``ValueError`` if the library exports no ``dispatch_abi()`` (built
    without ``EMIT_DISPATCH_SHIM_FLAG``) or reports something unparseable.
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
    abi_fn.argtypes = []
    abi = abi_fn()
    if abi is None:
        raise ValueError(f"{so_path} reports a null dispatch ABI")
    try:
        text = abi.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError(f"{so_path} reports a non-ASCII dispatch ABI") from None
    if not text.strip():
        return []
    c_types = [field.strip() for field in text.split(",")]
    if not all(c_types):
        raise ValueError(f"{so_path} reports a malformed dispatch ABI {text!r}")
    return c_types


class DispatchBridge:
    """Loads a compiled dispatch bridge once; regenerates instruction words per call.

    Args:
        so_path: Path to the immutable compiled dispatch generation.
        dispatch_params: Declared ``DispatchTime[T]`` parameter names, in the
            same order as the generated function's parameters (validated by
            ``compile_dispatch_bridge()`` at compile time). The C types come
            from the library's own ``dispatch_abi()``, so only names are needed.
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
        self._arg_spec = []
        for name, c_type_name in zip(dispatch_params, param_ctypes):
            ctype = _SCALAR_C_TYPES.get(c_type_name)
            if ctype is None:
                raise HostRuntimeError(
                    f"dispatch bridge: unrecognized generated C type "
                    f"{c_type_name!r}; known types are {sorted(_SCALAR_C_TYPES)}."
                )
            self._arg_spec.append((name, ctype, *_value_range(ctype)))
        try:
            self._lib.dispatch_generate
        except AttributeError:
            raise HostRuntimeError(
                f"dispatch bridge at {so_path} exports no dispatch_generate(); "
                "clear the compiled kernel directory and rebuild."
            ) from None
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
            try:
                integer = operator.index(value)
            except TypeError:
                raise HostRuntimeError(
                    f"DispatchTime[T] value {name}={value!r} must be an integer."
                ) from None
            if not low <= integer <= high:
                raise HostRuntimeError(
                    f"DispatchTime[T] value {name}={value!r} does not fit its "
                    f"generated C parameter type (valid range {low}..{high}); "
                    "passing it would silently wrap to a different value."
                )
            ordered.append(ctype(integer))
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
                "the compiled dispatch library does not match this design."
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
