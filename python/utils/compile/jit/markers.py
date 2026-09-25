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

from typing import Annotated, NoReturn, TypeVar

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


class In:
    """Runtime input tensor annotation (host → NPU, DMA each call)."""


class Out:
    """Runtime output tensor annotation (NPU → host, DMA each call)."""


class InOut:
    """Runtime bidirectional tensor annotation (DMA in both directions each call)."""


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

The generator receives an identity-bearing symbolic parameter for each unbound
scalar. Forward it exactly once as a direct ``Runtime(seq, fn_args=[...])``
entry, in any order. Use the callback's SSA argument for runtime operations.
Generation-time arithmetic, truth tests, shapes, dtypes, and Worker arguments
require ``CompileTime`` or specialization instead.

The Python bridge supports one runtime sequence, rejects remaining load-PDI
operations, and cannot use ``full_elf=True`` while any parameters remain dynamic.

For example:

```python
import numpy as np

def scaled_copy(a: In, b: Out, *, scale: DispatchTime[np.int32]):
    ...
```
"""


class _DispatchParameter:
    """An opaque value until Runtime replaces it with a sequence block argument."""

    __slots__ = ("name", "scalar_type", "position", "owner", "_binding")

    def __init__(self, name: str, scalar_type: type, position: int, owner: object):
        self.name = name
        self.scalar_type = scalar_type
        self.position = position
        self.owner = owner
        self._binding = None

    def __repr__(self) -> str:
        return f"DispatchTime({self.name!r}, {self.scalar_type.__name__})"

    def _misuse(self, *args, **kwargs) -> NoReturn:
        raise TypeError(
            f"DispatchTime parameter {self.name!r} has no generation-time value. "
            "Forward it directly in Runtime(seq, fn_args=[...]) and use the "
            "sequence callback's argument for runtime arithmetic/control flow. "
            "For shapes, dtypes, or worker configuration, use CompileTime or "
            f"specialize({self.name}=...)."
        )

    def _bind(self, binding: object) -> None:
        if self._binding is not None:
            raise TypeError(
                f"DispatchTime parameter {self.name!r} must be bound exactly once "
                "in one Runtime sequence."
            )
        self._binding = binding

    # Python and NumPy must not silently interpret a symbolic parameter as a
    # truthy object, a dtype, an object-array element, or a generation-time value.
    __bool__ = __int__ = __index__ = __float__ = __complex__ = __hash__ = _misuse
    __array__ = __array_ufunc__ = __getattr__ = _misuse
    __call__ = __iter__ = __len__ = __getitem__ = __contains__ = _misuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _misuse
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _misuse
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = _misuse
    __mod__ = __rmod__ = __pow__ = __rpow__ = __divmod__ = __rdivmod__ = _misuse
    __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = _misuse
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = _misuse
    __neg__ = __pos__ = __abs__ = __invert__ = __round__ = _misuse
