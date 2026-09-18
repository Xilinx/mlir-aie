# markers.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Type-annotation markers for compile-time vs. runtime parameter classification.

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

from typing import Annotated, TypeVar

T = TypeVar("T")


class _ParameterTag:
    """Annotation metadata with a stable representation for JIT cache keys."""

    __slots__ = ()

    def __repr__(self) -> str:
        return type(self).__name__.removeprefix("_").removesuffix("Tag")


class _CompileTimeTag(_ParameterTag):
    """Mark a parameter whose value is bound when generating the design."""

    __slots__ = ()


_COMPILE_TIME_TAG = _CompileTimeTag()

CompileTime = Annotated[T, _COMPILE_TIME_TAG]
"""Compile-time parameter annotation.

Use as a type annotation on generator function parameters that affect the
generated MLIR.  The value must be supplied at ``CompilableDesign``
construction time (or bound by ``@iron.jit(...)``).

Changing a ``CompileTime[T]``-annotated value → new cache key → recompile.
Required unless a default is given.

Example::

    from ml_dtypes import bfloat16

    def gemm(a: In, b: In, c: Out,
             M: CompileTime[int], K: CompileTime[int], N: CompileTime[int],
             dtype: CompileTime[type] = bfloat16):
        ...
"""


class In:
    """Runtime input tensor annotation (host → NPU, DMA each call)."""


class Out:
    """Runtime output tensor annotation (NPU → host, DMA each call)."""


class InOut:
    """Runtime bidirectional tensor annotation (DMA in both directions each call)."""


class _DispatchTimeTag(_ParameterTag):
    """Mark a scalar whose value is supplied when dispatching the design."""

    __slots__ = ()


_DISPATCH_TIME_TAG = _DispatchTimeTag()

DispatchTime = Annotated[T, _DISPATCH_TIME_TAG]
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

Example::

    import numpy as np

    def scaled_copy(a: In, b: Out, *, scale: DispatchTime[np.int32]):
        ...
"""
