# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Integer helpers that behave identically on Python ints and staged MLIR values.

The access-pattern algebra in :mod:`.tap` and :mod:`.tas` is pure integer
arithmetic plus a handful of decisions that inspect a value: a minimum, a ceiling division, a conditional
choice, a product, and a validity check. On Python ints these helpers are the
obvious builtins and produce exactly the numbers ``taplib`` produced before the
algebra existed. On a staged value (an ``aie.ir.Value`` carrying a runtime
scalar inside a runtime-sequence body) they emit the equivalent ``arith`` ops
instead, so the same pattern code serves the static instruction path and the
dynamic C++ transaction builder.

Only ``addi/subi/muli/divsi/remsi/cmpi/select`` are ever emitted: upstream
``ArithToEmitC`` has no patterns for ``minsi``, ``ceildivsi`` or ``floordivsi``.
Every operand here is a shape, tile count or offset, hence non-negative, so
``divsi``/``remsi`` coincide with Python's floor semantics.

Nothing in this module imports the MLIR bindings unless a staged value is
actually seen, which keeps ``taplib`` importable without a built ``aie``.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

__all__ = [
    "SymbolicError",
    "is_sym",
    "sym_any",
    "sint",
    "smin",
    "smax",
    "sceildiv",
    "sselect",
    "sprod",
    "require",
]


class SymbolicError(TypeError):
    """A generation-time decision was attempted on a staged (runtime) value."""


def is_sym(value: Any) -> bool:
    """Whether ``value`` is a staged value rather than a Python integer.

    A staged value is an ``aie.ir.Value`` (a runtime scalar inside a runtime
    sequence body) or any object whose type sets ``__aie_symbolic__ = True``.
    The latter is the protocol the tests use to drive every staged branch with
    an expression-tree stand-in and no MLIR bindings: such a type overloads the
    integer operators and comparisons, and may provide ``_select(a, b)`` and
    ``_require(message)`` for the two helpers that otherwise emit dialect ops.
    """
    if isinstance(value, (int, np.integer, bool)):
        return False
    if getattr(type(value), "__aie_symbolic__", False):
        return True
    try:
        from aie.ir import Value  # pyright: ignore[reportMissingImports]
    except ImportError:
        return False
    return isinstance(value, Value)


def show(value: Any) -> str:
    """Render a value for a message: staged values become ``<runtime>``.

    A guard message travels into the generated C++ as a comment, so it must
    not quote a staged value's IR.
    """
    if is_sym(value):
        return "<runtime>"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(show(v) for v in value) + "]"
    return str(value)


def sym_any(values: Iterable[Any]) -> bool:
    """Whether any entry of ``values`` is a staged value."""
    return any(is_sym(v) for v in values)


def sint(value: Any) -> Any:
    """Normalise an integer-like: NumPy integers become ``int``; staged values pass through.

    An ``index``-typed staged value (a ``range_`` induction variable) is cast
    to ``i32``, the width dispatch-time scalars carry, so a loop counter can
    index a tiler directly.

    Raises:
        TypeError: If ``value`` is neither an integer nor a staged value.
    """
    if is_sym(value):
        return _from_index(value)
    if isinstance(value, bool):
        raise TypeError("expected an integer, got a bool")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(
        f"expected an integer or a staged value, got {type(value).__name__}"
    )


def _from_index(value: Any) -> Any:
    """Cast an ``index``-typed MLIR value to ``i32``; anything else passes through."""
    if getattr(value, "type", None) is None or str(value.type) != "index":
        return value
    from aie.extras import types as T  # pyright: ignore[reportMissingImports]
    from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
        index_cast,
    )

    return index_cast(value, to=T.i32())


def _arith():
    from aie.dialects import (  # pyright: ignore[reportMissingImports]
        arith,  # pyright: ignore[reportAttributeAccessIssue]
    )

    return arith


def _cmp_lt(a: Any, b: Any) -> Any:
    """Return ``a < b`` as an ``i1`` value when either side is staged."""
    # ArithValue overloads ``<`` to emit ``arith.cmpi slt``; an int operand is
    # promoted to a constant of the other side's type.
    if is_sym(a):
        return a < b
    return b > a


def _as_staged(value: Any, like: Any) -> Any:
    """Return ``value`` as an MLIR value of ``like``'s type (ints become constants)."""
    if is_sym(value):
        return value
    from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
        constant,
    )

    return constant(int(value), like.type)


def sselect(cond: Any, if_true: Any, if_false: Any) -> Any:
    """``if_true if cond else if_false``; ``arith.select`` when ``cond`` is staged."""
    if is_sym(cond):
        hook = getattr(cond, "_select", None)
        if hook is not None:
            return hook(if_true, if_false)
        from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
            ScalarValue,
        )

        if not is_sym(if_true) and not is_sym(if_false):
            if if_true == if_false:
                return if_true
            # Two plain ints: the result takes the type the condition compared.
            try:
                ref = cond.owner.operands[0]
            except (AttributeError, IndexError):
                from aie.extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
                    constant,
                )

                ref = constant(0, index=False)
        else:
            ref = if_true if is_sym(if_true) else if_false
        result = _arith().select(
            cond, _as_staged(if_true, ref), _as_staged(if_false, ref)
        )
        return ScalarValue(result, dtype=ref.type)
    return if_true if cond else if_false


def smin(a: Any, b: Any) -> Any:
    """``min(a, b)`` that stays branch-free on staged values."""
    if not (is_sym(a) or is_sym(b)):
        return min(a, b)
    return sselect(_cmp_lt(a, b), a, b)


def smax(a: Any, b: Any) -> Any:
    """``max(a, b)`` that stays branch-free on staged values."""
    if not (is_sym(a) or is_sym(b)):
        return max(a, b)
    return sselect(_cmp_lt(a, b), b, a)


def sceildiv(a: Any, b: Any) -> Any:
    """Ceiling division for non-negative operands.

    Staged as ``a // b + (a % b > 0)`` rather than ``-(a // -b)``, because the
    staged ``//`` lowers to ``divsi``, which truncates toward zero, and rather
    than ``(a + b - 1) // b``, whose addition can overflow a runtime ``i32``
    even when the quotient fits.
    """
    if not (is_sym(a) or is_sym(b)):
        return -(-a // b)
    return a // b + sselect(_cmp_lt(0, a % b), 1, 0)


def sprod(values: Iterable[Any]) -> Any:
    """Product of ``values`` as an int, or a staged ``muli`` chain."""
    result: Any = 1
    for v in values:
        if is_sym(result) or is_sym(v):
            result = result * v
        else:
            result = int(result) * int(v)
    return result


def require(cond: Any, message: str) -> None:
    """Assert a shape constraint at generation time or at dispatch time.

    On a Python bool this is ``raise ValueError(message)``. On a staged ``i1``
    it records a runtime guard: the generated C++ transaction builder returns
    ``std::nullopt`` (surfaced as a ``HostRuntimeError``) when the condition
    fails at dispatch, mirroring the existing BD-field overflow guards.

    Raises:
        ValueError: If a concrete condition is false.
        SymbolicError: If ``cond`` is staged and the runtime guard op is not
            available in this build.
    """
    if not is_sym(cond):
        if not cond:
            raise ValueError(message)
        return
    hook = getattr(cond, "_require", None)
    if hook is not None:
        hook(message)
        return
    try:
        from aie.dialects import aiex  # pyright: ignore[reportMissingImports]

        emit = aiex.npu_require
    except (ImportError, AttributeError):
        raise SymbolicError(
            f"a shape constraint depends on a runtime value ({message!r}) but "
            "this build has no aiex.npu.require op to guard it at dispatch."
        ) from None
    emit(cond, message)
