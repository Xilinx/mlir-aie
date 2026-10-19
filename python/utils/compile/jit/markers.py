# markers.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Argument markers for JIT design parameters.

Five annotation categories are defined here (all exported from ``aie.iron``):

``CompileTime[T]``
    Marks a generator function parameter as compile-time.  Changing its value
    causes a recompile and a new cache entry.  Inspired by ``tl.constexpr`` in
    Triton.  Implemented as ``Annotated[T, ...]`` so pyright/mypy check callers
    against the wrapped type ``T`` itself (e.g. ``CompileTime[int]`` behaves
    like ``int`` for type-checking) while ``_introspect.py`` still recovers the
    marker at runtime via ``get_type_hints(..., include_extras=True)``.

``In``
    Marks a generator function parameter as a runtime *input* tensor.  Data is
    DMA-transferred from the host to the NPU on every kernel call.

``Out``
    Marks a generator function parameter as a runtime *output* tensor.  Data is
    DMA-transferred from the NPU to the host on every kernel call.

``InOut``
    Marks a generator function parameter as a runtime bidirectional tensor.
    Data is DMA-transferred in both directions on every kernel call.

A generator receives each tensor's type, ``np.ndarray[shape, np.dtype[T]]``,
never its data: the type of the tensor the design is called with, or one bound
ahead of time with ``specialize(name=np.ndarray[...])``. A different shape or
dtype is a different compiled design.

``DispatchTime[T]``
    Marks a keyword-only integer scalar that can vary per dispatch without
    recompiling. Explicit specialization instead fixes it at compile time.
    See ``DispatchTime`` below for generator binding and usage restrictions.

Any parameter without one of these annotations is rejected at ``@iron.jit``
decoration time when the parameter has a default value: an unannotated scalar
is bound at generation time, so the default would be baked into the compiled
kernel and per-call overrides silently ignored.  Annotate as
``CompileTime[T]`` (recompiles on change), ``DispatchTime[T]`` (runtime
scalar, one compile many values), or ``In``/``Out``/``InOut`` (DMA tensor)
instead.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Annotated, Any, NoReturn, TypeVar

import numpy as np
from ml_dtypes import bfloat16

from ....helpers.taplib._per_call import PerCall
from ... import bfp
from .params import Member, _text, resolve
from .params import Param as _Param

T = TypeVar("T")


CompileTime = Annotated[T, "aie.compile_time"]
"""Compile-time parameter annotation.

Use as a type annotation on generator function parameters that affect the
generated MLIR.  The value must be supplied at ``CompilableDesign``
construction time (or bound by ``@iron.jit(...)``).

Changing a ``CompileTime[T]``-annotated value → new cache key → recompile.
Required unless a default is given.

For example:

```python
from ml_dtypes import bfloat16

def gemm(a: In, b: In, c: Out,
         M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
         dtype: CompileTime[type] = bfloat16):
    ...
```
"""


class _Operand(Member):
    """A tensor declared in an operator's class body.

    On an instance, ``op.a`` is the operand the operator bound (`of`).

    Args:
        dims: The host shape: integers, ``param()``s and ``Select``s. A
            ``param()`` that is None on an instance is a dimension the shape
            leaves out there.
        dtype: The element type, or a ``param()`` holding it.
        obj_shape: What one ObjectFifo element holds. With one the tensor
            streams into the array; without one a hand-written sequence
            moves it.
        per: The ``param()`` (or several, multiplied) the stream is split
            over, one ObjectFifo each.
        depth: The ObjectFifo depth.
        replicate: Every ``per=`` lane gets the whole tensor, not a share.
        when: A bool ``param()``; the tensor exists only where it is true.
        like: An input whose dims these are; a caller gives this tensor the
            shape that input has on the call, so ``Out(like=x)`` of a flat
            ``x`` is ``(rows, cols)`` where ``x`` is.

    Raises:
        TypeError: ``replicate`` without ``per``, or both ``dims`` and
            ``like``.
    """

    fills = drains = False

    def __init__(
        self,
        *dims: Any,
        dtype: Any = bfloat16,
        obj_shape: Any = None,
        per: Any = None,
        depth: int = 2,
        replicate: bool = False,
        when: _Param | None = None,
        like: _Operand | None = None,
    ):
        if replicate and per is None:
            raise TypeError(
                "replicate=True needs per=<dim>: every lane receives the whole tensor"
            )
        if like is not None and dims:
            raise TypeError(f"give dims {dims} or like={like!r}, not both")
        self.like = like
        self.dims = dims if like is None else like.dims
        self.declared_dtype = dtype
        self.obj_dims = None if obj_shape is None else _dims(obj_shape)
        self.per = None if per is None else _dims(per)
        self.depth = depth
        self.replicate = replicate
        self.when = when

    @property
    def shape(self) -> tuple[int, ...]:
        return resolve(self.dims, self.op)

    @property
    def dtype(self):
        dtype = self.declared_dtype
        return getattr(self.op, dtype.name) if isinstance(dtype, _Param) else dtype

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def streamed(self) -> bool:
        return self.obj_dims is not None

    @property
    def obj_shape(self) -> tuple[int, ...]:
        if self.obj_dims is None:
            raise TypeError(f"{self.name} is declared without an obj_shape=: no stream")
        return resolve(self.obj_dims, self.op)

    @property
    def lanes(self) -> int:
        """The stream's ObjectFifos."""
        if self.obj_dims is None:
            raise TypeError(f"{self.name} is declared without an obj_shape=: no stream")
        return 1 if self.per is None else math.prod(resolve(self.per, self.op))

    def __repr__(self) -> str:
        role = type(self).__name__
        if self.op is None:
            if self.like is not None:
                return f"{role}(like={self.like.name})"
            return f"{role}({', '.join(map(_text, self.dims))})"
        return f"<{role} {self.name} {self.shape} {bfp.dtype_name(self.dtype)}>"


def _dims(spec) -> tuple:
    return tuple(spec) if isinstance(spec, (tuple, list)) else (spec,)


if TYPE_CHECKING:
    In = Out = InOut = type[np.ndarray]
else:

    class In(_Operand):
        """Runtime input tensor annotation (host → NPU, DMA each call)."""

        fills = True

    class Out(_Operand):
        """Runtime output tensor annotation (NPU → host, DMA each call)."""

        drains = True

    class InOut(_Operand):
        """Runtime bidirectional tensor annotation (DMA in both directions each call)."""

        fills = drains = True


DispatchTime = Annotated[T, "aie.dispatch_time"]
"""Runtime-scalar parameter annotation.

Per-call values rebuild instructions through a compiled host library without
changing the device-program cache key. An omitted value uses the signature
default, if any. Explicit binding with ``iron.jit(generator, name=value)`` or
``design.specialize(name=value)`` instead produces a typed NumPy constant and
includes it in the cache key; calls cannot override it.

``T`` must be a NumPy integer scalar type supported by ``Runtime``, such as
``np.int32`` or ``np.int64``. Built-in ``int``/``bool`` and floating-point
types are rejected. Parameters must be keyword-only, even when defaulted or
specialized.

The generator receives a ``PerCall`` for each unbound scalar. Forward it
exactly once as a direct ``Runtime(seq, fn_args=[...])`` entry, in any order.
Integer arithmetic on it (``n * 64``, ``(n + 1) // 2``) is an ``Affine``
expression, which a tensor access pattern built in the sequence body stages;
the callback's SSA argument serves runtime control flow. Truth tests,
comparisons, shapes, dtypes, and Worker arguments require ``CompileTime`` or
specialization instead.

The Python bridge supports one runtime sequence, rejects remaining load-PDI
operations, and cannot use ``full_elf=True`` while any parameters remain dynamic.

For example:

```python
import numpy as np

def scaled_copy(a: In, b: Out, *, scale: DispatchTime[np.int32]):
    ...
```
"""


class _DispatchParameter(PerCall):
    """A ``PerCall`` with no value until Runtime binds it to a sequence block argument."""

    __slots__ = ("position", "owner", "_binding")

    def __init__(self, name: str, scalar_type: type, position: int, owner: object):
        super().__init__(name, "aie.dispatch_time", scalar_type)
        self.position = position
        self.owner = owner
        self._binding = None

    @property
    def scalar_type(self) -> type:
        return self.dtype

    def __repr__(self) -> str:
        return f"DispatchTime({self.name!r}, {self.scalar_type.__name__})"

    def _misuse(self, *args, **kwargs) -> NoReturn:
        raise TypeError(
            f"DispatchTime parameter {self.name!r} has no generation-time value. "
            "+, -, *, // and % on it make an expression a tensor access pattern "
            "in the Runtime sequence stages; use the sequence callback's argument "
            "for runtime control flow. For truth tests, comparisons, shapes, "
            f"dtypes, or worker configuration, use CompileTime or specialize({self.name}=...)."
        )

    def _bind(self, binding: object, ssa) -> None:
        if self._binding is not None:
            raise TypeError(
                f"DispatchTime parameter {self.name!r} must be bound exactly once "
                "in one Runtime sequence."
            )
        self._binding = binding
        self.ssa = ssa

    # Python and NumPy must not silently interpret a symbolic parameter as a
    # truthy object, a dtype, an object-array element, or a generation-time
    # value. PerCall's integer arithmetic stays.
    __bool__ = __int__ = __index__ = __float__ = __complex__ = __hash__ = _misuse
    __array__ = __array_ufunc__ = __getattr__ = _misuse
    __call__ = __iter__ = __len__ = __getitem__ = __contains__ = _misuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _misuse
    __truediv__ = __rtruediv__ = __rfloordiv__ = _misuse
    __rmod__ = __pow__ = __rpow__ = __divmod__ = __rdivmod__ = _misuse
    __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = _misuse
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = _misuse
    __pos__ = __abs__ = __invert__ = __round__ = _misuse
