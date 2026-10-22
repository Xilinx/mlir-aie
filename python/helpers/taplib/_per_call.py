# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Integers known only at dispatch: expressions in one per-call value.

Plain Python, so they need no MLIR context: the host evaluates one per call,
names a device symbol after it, and ``stage`` lowers it to ``arith`` through
eudsl's ``ScalarValue`` where a pattern needs it on the device.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from ...dialects import (  # pyright: ignore[reportMissingImports]
    arith,  # pyright: ignore[reportAttributeAccessIssue]
)
from ...extras import types as T  # pyright: ignore[reportMissingImports]
from ...extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    ScalarValue,
    constant,
)


class PerCall:
    """An integer scalar given per call.

    Integer arithmetic on one makes an ``Affine``. ``marker`` says how a call
    gives it, and only its library reads it: ``"aie.dispatch_time"`` for a
    ``DispatchTime`` parameter. ``ssa`` is its staged value once a runtime
    sequence binds it.
    """

    __slots__ = ("name", "marker", "scalar_type", "ssa")

    def __init__(self, name: str, marker: Any, scalar_type: type):
        self.name, self.marker, self.scalar_type = name, marker, scalar_type
        self.ssa: Any = None

    def affine(self) -> Affine:
        return Affine(self)

    def __add__(self, k: int) -> Affine | int:
        return self.affine() + k

    def __sub__(self, k: int) -> Affine | int:
        return self.affine() - k

    def __mul__(self, k: int) -> Affine:
        return self.affine() * k

    __radd__, __rmul__ = __add__, __mul__

    def __rsub__(self, k: int) -> Affine | int:
        return k - self.affine()

    def __neg__(self) -> Affine:
        return -self.affine()

    def __floordiv__(self, d: int | Affine | PerCall) -> Affine | int:
        return self.affine() // d

    def __mod__(self, m: int) -> Affine:
        return self.affine() % m

    def __eq__(self, other) -> bool:
        if isinstance(other, (PerCall, Affine)):
            return self is other
        if isinstance(other, (int, float, np.integer, np.floating)):
            raise TypeError(f"{self} has no value to compare: it is given per call")
        return NotImplemented

    __hash__ = object.__hash__

    def __bool__(self):
        raise TypeError(f"{self} has no truth value: it is given per call")

    def __index__(self):
        raise TypeError(f"{self} is not a number: it is given per call")

    __int__ = __index__

    def __repr__(self) -> str:
        return f"PerCall({self.name!r}, {np.dtype(self.scalar_type).name})"


@dataclasses.dataclass(frozen=True)
class Affine:
    """An integer expression in one per-call value ``v``.

    ```text
    clamp(((v * scale + bias) >> down) * mul + add, lo, hi) % mod + post
    ```

    ``>>`` floors, a bound of None is none, and ``mod`` is a power of two
    (a ring's row) or None. ``post`` is added after the wrap, so a run of a
    ring written twice over, ``ring[a : a + R]``, has its end. A linear one
    has ``down`` 0 and keeps its terms in ``scale`` and ``bias``:
    ``(p + 1) * 64`` is ``Affine(p, 64, 64)``.

    Sums and products with integers, floor divisions by a power of two (so
    ``ceildiv``), sums of two linear ones in one value, and the quotient of
    two that are multiples of one another (``(4 * n) // n`` is 4) are
    expressions too. A clamp is carried through a sum, a product or a
    floor division by an integer. Anything else (a comparison, another
    divisor, two values) raises ``TypeError``.
    """

    value: PerCall
    scale: int = 1
    bias: int = 0
    down: int = 0
    mul: int = 1
    add: int = 0
    lo: int | None = None
    hi: int | None = None
    mod: int | None = None
    post: int = 0

    @property
    def clamped(self) -> bool:
        return self.lo is not None or self.hi is not None

    @property
    def linear(self) -> bool:
        return not (self.down or self.clamped or self.mod is not None)

    def affine(self) -> Affine:
        """Return this expression where only a linear one will do.

        Raises:
            TypeError: It floors, is clamped or wraps.
        """
        if not self.linear:
            raise TypeError(f"{self} is not linear in {self.value.name}")
        return self

    def clamp(self, lo: int | None = None, hi: int | None = None) -> Affine | int:
        """Return this expression held to ``[lo, hi]`` too, a bound of None none.

        Held to a range it never reaches, it is that range's nearer bound.

        Raises:
            TypeError: The expression wraps.
            ValueError: ``lo`` exceeds ``hi``.
        """
        if self.mod is not None:
            raise TypeError(f"no expression clamps ({self}) after it wraps")
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"{self} held to [{lo}, {hi}] holds no value")
        if lo is not None and self.hi is not None and self.hi < lo:
            return lo
        if hi is not None and self.lo is not None and self.lo > hi:
            return hi
        if lo is None or self.lo is not None and self.lo > lo:
            lo = self.lo
        if hi is None or self.hi is not None and self.hi < hi:
            hi = self.hi
        return dataclasses.replace(self, lo=lo, hi=hi)

    def _reclamped(self, op, bound, flip: bool = False) -> Affine:
        """``op`` of this expression unclamped, clamped to its bounds under ``bound``."""
        lo, hi = (None if b is None else bound(b) for b in (self.lo, self.hi))
        if flip:
            lo, hi = hi, lo
        return op(dataclasses.replace(self, lo=None, hi=None)).clamp(lo, hi)

    def __add__(self, k: int | Affine) -> Affine | int:
        if isinstance(k, Affine):
            if k.value is not self.value or not (self.linear and k.linear):
                return NotImplemented
            scale = self.scale + k.scale
            if not scale:
                return self.bias + k.bias
            return Affine(self.value, scale, self.bias + k.bias)
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            return NotImplemented
        c = int(k)
        if self.mod is not None:
            return dataclasses.replace(self, post=self.post + c)
        if self.clamped:
            return self._reclamped(lambda e: e + c, lambda b: b + c)
        if self.down:
            return dataclasses.replace(self, add=self.add + c)
        return dataclasses.replace(self, bias=self.bias + c)

    def __sub__(self, k: int | Affine) -> Affine | int:
        return self + (-k)

    def __rsub__(self, k: int) -> Affine | int:
        return -self + k

    def __neg__(self) -> Affine:
        return self * -1

    def __mul__(self, k: int) -> Affine:
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
            return NotImplemented
        c = int(k)
        if self.mod is not None:
            # (x % m) * c is (x * c) % (m * c) for c > 0, a ring row's elements.
            if c <= 0 or c & (c - 1):
                raise TypeError(f"no expression for ({self}) * {c}")
            wrapped = dataclasses.replace(self, mod=None, post=0)
            return dataclasses.replace(wrapped * c % (self.mod * c), post=self.post * c)
        if self.clamped:
            return self._reclamped(lambda e: e * c, lambda b: b * c, flip=c < 0)
        if self.down:
            return dataclasses.replace(self, mul=self.mul * c, add=self.add * c)
        return dataclasses.replace(self, scale=self.scale * c, bias=self.bias * c)

    __radd__, __rmul__ = __add__, __mul__

    def __floordiv__(self, d: int | Affine | PerCall) -> Affine | int:
        if isinstance(d, PerCall):
            d = d.affine()
        c = (
            None
            if isinstance(d, bool) or not isinstance(d, (int, np.integer))
            else int(d)
        )
        if self.mod is not None or self.clamped and not c:
            raise TypeError(f"no expression for ({self}) // {d}")
        if c is not None and c < 0:
            return (-self) // -c
        if self.clamped:
            # A positive divisor keeps the bounds' order.
            return self._reclamped(lambda e: e // c, lambda b: b // c)
        if isinstance(d, Affine):
            # floor(q u / u) = q, wherever u is not 0
            if d.value is self.value and not self.down and d.linear and d.scale:
                q, r = divmod(self.scale, d.scale)
                if not r and self.bias == q * d.bias:
                    return q
            raise TypeError(f"no expression for ({self}) // ({d})")
        if c is None:
            return NotImplemented
        if c == 0 or c & (c - 1):
            raise TypeError(f"no expression for a floor division by {c}")
        if self.mul != 1:
            if self.mul % c == 0:  # the floored term divides exactly
                return dataclasses.replace(self, mul=self.mul // c, add=self.add // c)
            raise TypeError(f"no expression for ({self}) // {c}")
        # floor((floor(u / 2^a) + b) / 2^k) = floor((u + b 2^a) / 2^(a+k))
        scale, bias = self.scale, self.bias + (self.add << self.down)
        k = self.down + c.bit_length() - 1
        while k and scale % 2 == 0 and bias % 2 == 0:
            scale, bias, k = scale // 2, bias // 2, k - 1
        if scale % (1 << k) == 0:  # floor(s v / 2^k + b / 2^k) is linear
            return Affine(self.value, scale >> k, bias >> k)
        return Affine(self.value, scale, bias, k)

    def __mod__(self, m: int) -> Affine:
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)):
            return NotImplemented
        c = int(m)
        if c <= 0 or c & (c - 1):
            raise TypeError(f"no expression for ({self}) % {c}: a ring is 2^k rows")
        if self.mod is not None and (self.mod % c or self.post):
            raise TypeError(f"no expression for ({self}) % {c}")
        return dataclasses.replace(self, mod=c)

    def __eq__(self, other) -> bool:
        if isinstance(other, Affine):
            # The value by identity: a PerCall has no value to compare.
            return self.value is other.value and all(
                getattr(self, f.name) == getattr(other, f.name)
                for f in dataclasses.fields(self)[1:]
            )
        if isinstance(other, (int, float, np.integer, np.floating)):
            raise TypeError(f"{self} has no value to compare: it is computed per call")
        return NotImplemented

    def __hash__(self) -> int:
        # The value by identity, as __eq__ compares it.
        terms = tuple(getattr(self, f.name) for f in dataclasses.fields(self)[1:])
        return hash((id(self.value), terms))

    def __bool__(self):
        raise TypeError(f"{self} has no truth value: it is computed per call")

    def __index__(self):
        raise TypeError(f"{self} is not a number: it is computed per call")

    __int__ = __index__

    @property
    def scalar_type(self) -> type:
        return self.value.scalar_type

    @property
    def name(self) -> str:
        """An identifier for the device symbol: ``p``, ``p_x64``, ``p_x64_p64``, ``p_m1``."""
        # (tag, k): a tag names a factor or bound, ``_x64``, ``_lom3``; None an
        # offset, ``_p64``, ``_m3``. A k of None is absent.
        terms = [("x", None if self.scale == 1 else self.scale), (None, self.bias)]
        if self.down:
            terms += [
                ("d", self.down),
                ("x", None if self.mul == 1 else self.mul),
                (None, self.add),
            ]
        terms += [("lo", self.lo), ("hi", self.hi), ("mod", self.mod)]
        terms += [(None, self.post)]
        text = self.value.name
        for tag, k in terms:
            if tag is not None and k is not None:
                text += f"_{tag}{'m' if k < 0 else ''}{abs(k)}"
            elif tag is None and k:
                text += f"_{'p' if k > 0 else 'm'}{abs(k)}"
        return text

    def __str__(self) -> str:
        """Return the expression as Python would write it: ``(p + 1) * 64`` is ``p * 64 + 64``."""
        x = self.value.name
        # (multiplier, addend): the linear part's, then the floored term's. The
        # multiplier's operand is a name or a parenthesized shift.
        for i, (mul, add) in enumerate(
            [(self.scale, self.bias)] + ([(self.mul, self.add)] if self.down else [])
        ):
            if i:
                x = f"{f'({x})' if ' ' in x else x} >> {self.down}"
                # + and * bind tighter than >>.
                if mul != 1 or add:
                    x = f"({x})"
            if mul == -1:
                x = f"-{x}"
            elif mul != 1:
                x = f"{x} * {mul}"
            if add:
                x = f"{x} {'+' if add > 0 else '-'} {abs(add)}"
        if self.lo is not None and self.hi is not None:
            x = f"clamp({x}, {self.lo}, {self.hi})"
        elif self.lo is not None:
            x = f"max({x}, {self.lo})"
        elif self.hi is not None:
            x = f"min({x}, {self.hi})"
        if self.mod is not None:
            x = f"{f'({x})' if ' ' in x else x} % {self.mod}"
            if self.post:
                x = f"{x} {'+' if self.post > 0 else '-'} {abs(self.post)}"
        return x

    def evaluate(self, values: Mapping[str, int]) -> int:
        """Its value where ``values`` gives its per-call value by name."""
        v = int(values[self.value.name])
        x = ((self.scale * v + self.bias) >> self.down) * self.mul + self.add
        if self.lo is not None:
            x = max(x, self.lo)
        if self.hi is not None:
            x = min(x, self.hi)
        return x if self.mod is None else x % self.mod + self.post

    def stage(self):
        """Stage this expression on its value's ``i64``, as eudsl's ``ScalarValue``.

        Raises:
            TypeError: No runtime sequence has bound the value.
        """
        # _symbolic imports this module.
        from ._symbolic import sint

        if self.value.ssa is None:
            raise TypeError(
                f"{self.value.name} has no staged value outside the runtime "
                "sequence it is bound to"
            )
        x = ScalarValue(sint(self.value.ssa))
        x = x * self.scale + self.bias
        if self.down:
            x = x // (1 << self.down)
        x = x * self.mul + self.add
        # ScalarValue has no min or max.
        if self.lo is not None:
            x = ScalarValue(arith.maxsi(x, constant(self.lo, T.i64())))
        if self.hi is not None:
            x = ScalarValue(arith.minsi(x, constant(self.hi, T.i64())))
        if self.mod is not None:  # a power of two: the floored remainder
            x = (x & (self.mod - 1)) + self.post
        return x
