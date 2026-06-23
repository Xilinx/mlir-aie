# _serialization.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""JSON-friendly encoding for :class:`CompilableDesign.to_json` / ``from_json``,
plus the runtime-tensor placeholder used during MLIR generation.

Carved out of ``compilabledesign.py`` to keep the main file focused on the
``CompilableDesign`` class itself.
"""

from __future__ import annotations

from typing import Any

_PRIMITIVE_TYPES = (int, float, str, bool, bytes)

_KWARG_TYPE_MAP = {"bool": bool, "int": int, "float": float, "str": str}


def _encode_kwarg(value: Any) -> Any:
    """Encode a compile_kwarg value as ``[typename, value]`` for JSON storage."""
    if isinstance(value, bool):  # must check bool before int
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value]
    if isinstance(value, str):
        return ["str", value]
    return ["str", str(value)]


def _decode_kwarg(encoded: Any) -> Any:
    """Decode a compile_kwarg value from JSON storage."""
    if not isinstance(encoded, list) or len(encoded) != 2:
        return encoded
    t, v = encoded
    converter = _KWARG_TYPE_MAP.get(t, str)
    return converter(v)


class _TensorPlaceholder:
    """Sentinel for ``In``/``Out``/``InOut`` params during MLIR generation.

    Any attribute access raises so generator bodies can't read tensor
    shape/dtype at compile time (use ``CompileTime[T]`` for that instead).
    """

    def __init__(self, param_name: str) -> None:
        object.__setattr__(self, "_param_name", param_name)

    def _raise(self, op: str = "") -> None:
        name = object.__getattribute__(self, "_param_name")
        suffix = f": {op}" if op else ""
        raise RuntimeError(
            f"Generator parameter {name!r} is a runtime tensor (In/Out/InOut) "
            f"and is not available at compile time{suffix}. "
            f"Use CompileTime[T] parameters for shape/dtype information instead."
        )

    def __getattr__(self, name: str):
        self._raise(f".{name}")

    def __setattr__(self, name: str, value) -> None:
        self._raise(f".{name} = ...")

    def __getitem__(self, key):
        self._raise(f"[{key!r}]")

    def __repr__(self) -> str:
        name = object.__getattribute__(self, "_param_name")
        return f"<_TensorPlaceholder for {name!r}>"
