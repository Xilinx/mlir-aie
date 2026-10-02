# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Integer helpers that behave identically on Python ints and staged MLIR values.

The access-pattern algebra in `tap.py` is pure integer
arithmetic plus a validity check. On Python ints these helpers are the
obvious builtins. On a staged value (an `aie.ir.Value` carrying a runtime
scalar inside a runtime-sequence body) they emit the equivalent `arith` and
`cf` ops instead, so the same pattern code serves the static instruction path
and the dynamic C++ transaction builder.
"""

from __future__ import annotations

from typing import Any, Iterable, Union

import numpy as np

from ...dialects import (  # pyright: ignore[reportMissingImports]
    arith,  # pyright: ignore[reportAttributeAccessIssue]
    cf,  # pyright: ignore[reportAttributeAccessIssue]
    emitc,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...extras import types as T  # pyright: ignore[reportMissingImports]
from ...extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    index_cast,
)
from ...ir import (  # pyright: ignore[reportMissingImports]
    IndexType,
    IntegerAttr,
    IntegerType,
    OpView,
    Value,
)

__all__ = [
    "IntLike",
    "is_sym",
    "show",
    "sym_any",
    "sint",
    "sprod",
    "require",
]

IntLike = Union[int, np.integer, Value]
"""An integer: a Python or NumPy int, or a staged `aie.ir.Value`."""


def is_sym(value: Any) -> bool:
    """Return whether `value` is a staged value rather than a Python integer.

    Args:
        value: Any value.

    Returns:
        bool: Whether `value` is an `aie.ir.Value`.
    """
    return isinstance(value, Value)


def show(value: Any) -> str:
    """Render a value for a message: staged values become `<runtime>`.

    A guard message travels into the generated C++ as a comment, so it must
    not quote a staged value's IR.

    Args:
        value: An integer, a staged value, or a list or tuple of those.

    Returns:
        str: The rendering; a list or tuple renders as `[a, b, ...]`.
    """
    if is_sym(value):
        return "<runtime>"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(show(v) for v in value) + "]"
    return str(value)


def sym_any(values: Iterable[Any]) -> bool:
    """Return whether any entry of `values` is a staged value.

    Args:
        values (Iterable[Any]): The values to check.

    Returns:
        bool: Whether any entry is an `aie.ir.Value`.
    """
    return any(is_sym(v) for v in values)


def sint(value: Any) -> Any:
    """Normalise an integer-like: NumPy integers become `int`, staged values `i64`.

    Every staged value is widened to `i64`, so values of any width meet in
    the algebra and its products cannot wrap: a signed one is
    sign-extended, an `index` one (a `range_` induction variable) is cast,
    and an unsigned one (a `DispatchTime[np.uint*]` scalar) is zero-extended,
    or reinterpreted at 64 bits, where a value past the signed range fails
    the algebra's non-negativity guards. A constant unsigned one folds to its
    `int`.

    Args:
        value: A Python or NumPy integer, or a staged value.

    Returns:
        IntLike: An `int`, or a signless integer staged value.

    Raises:
        TypeError: If `value` is neither an integer nor a staged value.
    """
    if isinstance(value, Value):
        if isinstance(value.type, IndexType):
            return index_cast(value, to=T.i64())
        if isinstance(value.type, IntegerType) and value.type.is_unsigned:
            bits = value.type.width
            owner = value.owner
            if isinstance(owner, OpView) and owner.operation.name == "emitc.constant":
                return IntegerAttr(owner.attributes["value"]).value & ((1 << bits) - 1)
            return emitc.CastOp(T.i64(), value).result
        if isinstance(value.type, IntegerType) and value.type.width < 64:
            return arith.extsi(T.i64(), value)
        return value
    if isinstance(value, bool):
        raise TypeError("expected an integer, got a bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(
        f"expected an integer or a staged value, got {type(value).__name__}"
    )


def sprod(values: Iterable[Any]) -> Any:
    """Return the product of `values`.

    Args:
        values (Iterable[Any]): Integers and staged values.

    Returns:
        IntLike: The product as an `int`, or a staged `muli` chain.
    """
    result: Any = 1
    for v in values:
        if not is_sym(v) and v == 1:
            continue
        if is_sym(result) or is_sym(v):
            result = v if not is_sym(result) and result == 1 else result * v
        else:
            result = int(result) * int(v)
    return result


def require(cond: Any, message: str) -> None:
    """Assert a shape constraint at generation time or at dispatch time.

    On a Python bool this is `raise ValueError(message)`. On a staged `i1`
    it emits a `cf.assert`: the generated C++ transaction builder
    refuses a dispatch that fails it, and the host raises a
    `HostRuntimeError` carrying `message`.

    Args:
        cond: A Python bool, or a staged `i1`.
        message (str): The error message; keep it constant, since it travels
            into the generated C++.

    Raises:
        ValueError: If a concrete condition is false.
    """
    if not is_sym(cond):
        if not cond:
            raise ValueError(message)
        return
    cf.assert_(cond, message)
