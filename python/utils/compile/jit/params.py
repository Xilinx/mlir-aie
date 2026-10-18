# params.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The words an operator's class body declares its values with.

``M: int = param()`` is a compile-time parameter, ``tile: int = tunable()`` one
a tuner may choose, and ``n = per_call(np.int32)`` a value that may change
every call. Each stays the class attribute it was written as, so ``In(M, K)``
captures ``M`` itself and ``GEMM.M`` names the parameter. On an instance,
``op.M`` is the integer, held in the instance's own ``__dict__``.
"""

from __future__ import annotations

from dataclasses import MISSING, dataclass
from enum import Enum
from types import FunctionType
from typing import Any, Callable

import numpy as np


class Param:
    """What ``param()`` and ``tunable()`` return: a parameter of its class.

    Attributes:
        default: The value when the caller gives none, a function of the
            operator computing it, or ``MISSING`` for a required argument.
            A tunable's default is its starting value, None for one
            ``resolve`` fills.
        derive: A function of the operator a given value must equal, and
            the value when none is given; or None.
        array: The cores read it, though no operand's shape names it.
        repr: It is in the operator's repr and device symbols.
        init: It is a constructor argument.
        tunable: A tuner may choose it.
        domain: A tunable's search space (see `values`), or None.
    """

    def __init__(self, *, default, derive, array, repr, init, tunable, domain):
        self.default = default
        self.derive = derive
        self.array = array
        self.repr = repr
        self.init = init
        self.tunable = tunable
        self.domain = domain
        self.name = ""
        self.owner: type | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.owner = owner

    def __get__(self, op, owner=None):
        # An instance holds every parameter in its __dict__, which a non-data
        # descriptor gives way to; only the layout copy, which holds just the
        # key, falls through to here.
        if op is None:
            return self
        raise AttributeError(
            f"{owner.__name__}.build() reads {self.name}, which is not in the "
            f"core-layout key; declare it param(array=True)"
        )

    def initial(self, op) -> Any:
        """Return the value on ``op`` when the caller gives none."""
        if self.derive is not None:
            return self.derive(op)
        if isinstance(self.default, FunctionType):
            return self.default(op)
        return self.default

    def values(self, op, dev, start) -> tuple:
        """Return the values a tuner tries for this tunable.

        ``op`` is resolved for ``dev`` with ``start``. They are none without
        a domain; a tuple, with ``start`` first if it is not among them; an
        ``Enum``'s members, ``start`` first; or what a function ``(op, dev,
        start)`` returns.
        """
        domain = self.domain
        if domain is None:
            return ()
        if isinstance(domain, type):
            return (start, *(m for m in domain if m != start))
        if isinstance(domain, tuple):
            return domain if start in domain else (start, *domain)
        return domain(op, dev, start)

    def __repr__(self) -> str:
        return f"<{'tunable' if self.tunable else 'param'} {self.name}>"


def param(
    *,
    default: Any = MISSING,
    derive: Callable[[Any], Any] | None = None,
    array: bool = False,
    repr: bool = True,
    init: bool = True,
) -> Any:
    """Declare a compile-time parameter.

    The caller gives it, or the operands' shapes infer it.

    Args:
        default: The value when neither the caller nor an operand gives one.
            A ``lambda`` or ``def`` is a function of the operator computing
            it; anything else, a type included, is the value.
        derive: A function of the operator: the value when none is given,
            and what a given one must equal.
        array: The cores read it (a kernel scalar, a flag the build branches
            on) though no operand's shape, dtype or presence names it.
        repr: It is in the operator's repr and device symbols.
        init: It is a constructor argument.

    Raises:
        TypeError: Both `default` and `derive` are given.
    """
    if default is not MISSING and derive is not None:
        raise TypeError("param(default=, derive=): give one, not both")
    return Param(
        default=default,
        derive=derive,
        array=array,
        repr=repr,
        init=init,
        tunable=False,
        domain=None,
    )


def tunable(
    *,
    default: Any = None,
    domain: Any = None,
    repr: bool = True,
    init: bool = True,
) -> Any:
    """Declare a compile-time parameter a tuner may search.

    ``resolve`` fills it for the device when the caller does not. The cores
    always read it.

    With ``init=False`` it is no constructor argument: a value ``resolve``
    computes, or a subclass's fixed value of an inherited tunable.

    Args:
        default: The starting value; None means ``resolve`` fills it.
        domain: The values a tuner searches: a tuple, an ``Enum`` class, or
            a function ``(op, dev, start) -> tuple`` such as `widths`. None
            is not searched.
        repr: It is in the operator's repr and device symbols.
        init: It is a constructor argument.

    Raises:
        TypeError: `domain` is none of those.
    """
    if not (
        domain is None
        or isinstance(domain, tuple)
        or (isinstance(domain, type) and issubclass(domain, Enum))
        or isinstance(domain, FunctionType)
    ):
        raise TypeError(
            f"tunable(domain={domain!r}): give a tuple, an Enum or a function "
            f"(op, dev, start) -> tuple"
        )
    return Param(
        default=default,
        derive=None,
        array=True,
        repr=repr,
        init=init,
        tunable=True,
        domain=domain,
    )


class Member:
    """Base of what an operator's body declares unannotated.

    On an instance it reads as what the operator bound to its name.
    """

    name: str = ""
    owner: type | None = None

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name
        self.owner = owner

    def __get__(self, op, owner=None):
        if op is None:
            return self
        try:
            return op._bound[self.name]
        except (AttributeError, KeyError):
            raise AttributeError(
                f"{type(op).__name__}.{self.name} is not bound yet"
            ) from None


class PerCall(Member):
    """What ``per_call()`` returns: a value that may change every call.

    Attributes:
        dtype: Its integer type.
        derive: A function of the operator computing it once per build, or
            None for one every call gives.
        bounds: The parameter it bounds: it reads as that parameter until a
            call gives a smaller value, the build sized for the full one.
    """

    def __init__(self, dtype, derive, bounds):
        self.dtype = dtype
        self.derive = derive
        self.bounds = bounds

    def __repr__(self) -> str:
        return f"per_call({np.dtype(self.dtype).name})"


def per_call(
    dtype: Any = np.int32,
    *,
    derive: Callable[[Any], Any] | None = None,
    bounds: Param | None = None,
) -> Any:
    """Declare a value the cores read that may change every call without a rebuild.

    A value no build fixes (no `derive`, no `bounds`) is given every call:
    an operator with a dispatch library takes it as a dispatch scalar,
    which rebuilds the instruction stream around it.

    Args:
        dtype: An integer type.
        derive: A function of the operator computing it once per build.
        bounds: A ``param()`` this value bounds per call: ``valid =
            per_call(bounds=M)``.

    Raises:
        TypeError: `dtype` is not an integer type, or both `derive` and
            `bounds` are given.
    """
    if np.dtype(dtype).kind not in "iu":
        raise TypeError(
            f"per_call({np.dtype(dtype).name}): a per-call value is an integer, "
            f"since the scratchpad encoding zeroes the top two bits of a word"
        )
    if derive is not None and bounds is not None:
        raise TypeError("per_call(derive=, bounds=): give one, not both")
    return PerCall(dtype, derive, bounds)


@dataclass(frozen=True)
class Select:
    """A shape chosen by a flag the caller gives or defaults, never inferred.

    ``Select(b_col_maj, (N, K), (K, N))``.
    """

    flag: Param
    when_true: tuple
    when_false: tuple

    def branch(self, op) -> tuple:
        """Return the dimensions the flag picks on ``op``."""
        flag = vars(op)[self.flag.name]
        if flag is None:
            raise ValueError(
                f"{self.flag.name} is None; a Select() on it needs a resolved operator"
            )
        return self.when_true if flag else self.when_false


def widths(op, dev, start: int) -> tuple[int, ...]:
    """Return the widths a count of columns, channels or cores may take.

    They are ``start`` and every power of two up to ``dev``'s columns, widest
    first: ``tunable(domain=widths)``.
    """
    powers = {1 << i for i in range(dev.cols.bit_length())}
    return tuple(sorted({start, *powers}, reverse=True))


def divisors(
    start: int,
    of: int,
    step: int = 1,
    cap: int | None = None,
    span: int | None = None,
) -> tuple[int, ...]:
    """Return ``start`` times each power of two that fits, largest first.

    Each divides `of`, is a multiple of `step`, is at most `cap` and is at
    most `span` octaves from ``start``. A domain function returns it:
    ``tunable(domain=lambda op, dev, t: divisors(t, of=op.K, step=64))``.
    """
    found = [start]
    value, octave = start * 2, 1
    while (
        of % value == 0
        and (cap is None or value <= cap)
        and (span is None or octave <= span)
    ):
        found.insert(0, value)
        value, octave = value * 2, octave + 1
    value, octave = start, 1
    while (
        value % 2 == 0 and (value // 2) % step == 0 and (span is None or octave <= span)
    ):
        value, octave = value // 2, octave + 1
        if of % value == 0 and (cap is None or value <= cap):
            found.append(value)
    return tuple(found)
