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

A generator receives each tensor as an operand of its marker's class, never its
data: an ``In``/``Out``/``InOut`` with the ``shape`` and ``dtype`` of the
tensor the design is called with, or of a type bound ahead of time with
``specialize(name=np.ndarray[...])``, as an operator's ``op.a`` is. Every API
that takes a tensor type also takes the operand; ``operand.ndarray_type`` is
its ``np.ndarray[shape, np.dtype[T]]``. A different shape or dtype is a
different compiled design.

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
import sys
from typing import Annotated, Any, NoReturn, TypeVar

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

import numpy as np
from ml_dtypes import bfloat16

from ....helpers.npdtypes import np_ndarray_type_get_dtype, np_ndarray_type_get_shape
from ....helpers.taplib import PerCall
from ... import bfp, coded
from .params import Member, OperatorParam, Select, _text, resolve

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
    ``In`` documents the arguments.
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
        when: Any = None,
        like: _Operand | None = None,
        coded: OperatorParam | bool = False,
        capacity: OperatorParam | int = 15,
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
        self.obj_dims = (
            None
            if obj_shape is None
            else (
                tuple(obj_shape)
                if isinstance(obj_shape, (tuple, list))
                else (obj_shape,)
            )
        )
        self.per = (
            None
            if per is None
            else tuple(per) if isinstance(per, (tuple, list)) else (per,)
        )
        self.depth = depth
        self.replicate = replicate
        self.when = when
        self.declared_coded = coded
        self.declared_capacity = capacity

    @property
    def coded(self) -> bool:
        """Whether the tensor moves as coded rows (``aie.utils.coded``)."""
        flag = self.declared_coded
        return bool(
            getattr(self.op, flag.name) if isinstance(flag, OperatorParam) else flag
        )

    @property
    def capacity(self) -> int:
        """The exceptions a coded row holds."""
        capacity = self.declared_capacity
        return int(
            getattr(self.op, capacity.name)
            if isinstance(capacity, OperatorParam)
            else capacity
        )

    @property
    def logical_shape(self) -> tuple[int, ...]:
        """The shape the declaration names.

        It is what a caller passes, and what inference, drawing and a
        reference see.
        """
        return resolve(self.dims, self.op)

    @property
    def logical_dtype(self):
        dtype = self.declared_dtype
        return (
            getattr(self.op, dtype.name) if isinstance(dtype, OperatorParam) else dtype
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape that moves.

        It is the logical shape, with a coded tensor's rows as their bytes.
        """
        shape = self.logical_shape
        if not self.coded:
            return shape
        return (*shape[:-1], coded.row_bytes(shape[-1], self.capacity))

    @property
    def dtype(self):
        """The element type that moves: the logical one, a coded tensor's bytes."""
        dtype = self.logical_dtype
        if not self.coded:
            return dtype
        if bfp.is_bfp(dtype) or np.dtype(dtype) != np.dtype(bfloat16):
            raise TypeError(
                f"{self.name}: coded rows hold bfloat16 weights, not "
                f"{bfp.dtype_name(dtype)}"
            )
        return np.uint8

    @property
    def elements(self) -> int:
        return math.prod(self.shape)

    @property
    def ndarray_type(self) -> type[np.ndarray]:
        """The tensor type this operand describes, ``np.ndarray[shape, np.dtype[T]]``."""
        return np.ndarray[self.shape, np.dtype[self.dtype]]

    @classmethod
    def for_tensor(cls, name: str, ndarray_type: type[np.ndarray]) -> Self:
        """Return the operand a generator's ``name`` parameter receives.

        Args:
            name: The generator parameter.
            ndarray_type: The tensor's type, ``np.ndarray[shape, np.dtype[T]]``.

        Returns:
            An operand of this class with that shape and dtype.
        """
        operand = cls(
            *np_ndarray_type_get_shape(ndarray_type),
            dtype=np_ndarray_type_get_dtype(ndarray_type),
        )
        operand.__set_name__(None, name)
        return operand

    @property
    def streamed(self) -> bool:
        """Return whether the tensor streams.

        On a class, whether it may; on an operator, also whether an
        obj_shape ``Select`` picks a stream there.
        """
        if self.obj_dims is None:
            return False
        return self.op is None or not any(
            isinstance(d, Select)
            and None in (d.when_true, d.when_false)
            and d.branch(self.op) is None
            for d in self.obj_dims
        )

    @property
    def obj_shape(self) -> tuple[int, ...]:
        """What one ObjectFifo element holds; coded, whole rows as their bytes."""
        if not self.streamed:
            raise TypeError(f"{self.name} has no obj_shape here: no stream")
        assert self.obj_dims is not None
        obj = resolve(self.obj_dims, self.op)
        if not self.coded:
            return obj
        # The row alone: a core's build may read no other extent.
        row = resolve(self.dims[-1:], self.op)[-1]
        if not obj or obj[-1] != row:
            raise ValueError(
                f"{self.name}: a coded stream moves whole rows of {row}, but its "
                f"obj_shape is {obj}"
            )
        return (*obj[:-1], coded.row_bytes(row, self.capacity))

    @property
    def lanes(self) -> int:
        """The stream's ObjectFifos."""
        if not self.streamed:
            raise TypeError(f"{self.name} has no obj_shape here: no stream")
        return 1 if self.per is None else math.prod(resolve(self.per, self.op))

    def __repr__(self) -> str:
        role = type(self).__name__
        if self.op is None:
            if self.like is not None:
                return f"{role}(like={self.like.name})"
            return f"{role}({', '.join(map(_text, self.dims))})"
        coded_rows = " coded" if self.coded else ""
        return f"<{role} {self.name} {self.shape} {bfp.dtype_name(self.dtype)}{coded_rows}>"


class In(_Operand):
    """Runtime input tensor (host → NPU, DMA each call).

    ``Out`` and ``InOut`` take the same arguments.

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
        coded: A bool, or a ``param()``/``tunable()`` holding one: the
            bfloat16 tensor moves as ``aie.utils.coded`` rows. Its dims stay
            the logical ``(..., M, K)`` (``logical_shape``, ``logical_dtype``),
            which inference, drawing and a reference see; ``shape``,
            ``dtype`` and ``obj_shape`` are what moves, ``(..., M, R)`` uint8,
            an ``obj_shape`` of whole rows mapped to their records.
        capacity: The exceptions a coded row holds, or a ``param()``.

    Raises:
        TypeError: ``replicate`` without ``per``, or both ``dims`` and
            ``like``.
    """

    fills = True


class Out(_Operand):
    """Runtime output tensor (NPU → host, DMA each call).

    Takes the arguments ``In`` documents.
    """

    drains = True


class InOut(_Operand):
    """Runtime bidirectional tensor (DMA in both directions each call).

    Takes the arguments ``In`` documents.
    """

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

    def __repr__(self) -> str:
        return f"DispatchTime({self.name!r}, {self.scalar_type.__name__})"

    def refuse(self, *args, **kwargs) -> NoReturn:
        """Raise the TypeError any generation-time use of this parameter gets.

        Raises:
            TypeError: Always.
        """
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
    # value. PerCall's integer arithmetic stays: None makes a NumPy scalar on
    # the left defer to the reflected operator, and a ufunc refuse.
    __array_ufunc__ = None
    __bool__ = __int__ = __index__ = __float__ = __complex__ = __hash__ = refuse
    __array__ = __getattr__ = refuse
    __call__ = __iter__ = __len__ = __getitem__ = __contains__ = refuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = refuse
    __truediv__ = __rtruediv__ = __rfloordiv__ = refuse
    __rmod__ = __pow__ = __rpow__ = __divmod__ = __rdivmod__ = refuse
    __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = refuse
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = refuse
    __pos__ = __abs__ = __invert__ = __round__ = refuse
