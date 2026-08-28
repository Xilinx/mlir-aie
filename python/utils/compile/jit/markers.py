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
    Marks a generator function parameter as a runtime *scalar*.  Unlike
    ``CompileTime[T]``, the value is not baked into the compiled kernel and
    does not affect the cache key — one compiled artifact is meant to serve
    many scalar values.  At generation time the generator receives the
    wrapped type ``T`` itself (e.g. ``np.int32``), not a concrete value, so it
    can forward it into ``Runtime(..., inputs=[...])`` and get back a runtime
    SSA block arg (the same scalar-type-in-``inputs`` duality ``Runtime``
    already implements).  **The host-side bridge that delivers a per-call
    value to that SSA arg is not wired up yet** (tracked as
    ROADMAP_iron_runtime_binding_time.md Phase 2.2/2.3) — today a
    ``DispatchTime[T]`` parameter classifies and generates correctly, but a
    per-call override does not yet reach the NPU dispatch.

Any parameter without one of these annotations is currently rejected at
``@iron.jit`` decoration time when the parameter has a default value — there
is no runtime-scalar host-bridge plumbing yet (tracked separately as future
work), so the default would be baked into the compiled kernel and per-call
overrides silently ignored.  Annotate as ``CompileTime[T]`` (recompiles on
change), ``DispatchTime[T]`` (runtime scalar, once the host bridge lands), or
``In``/``Out``/``InOut`` (DMA tensor) instead.
"""

from __future__ import annotations

from typing import Annotated, TypeVar

T = TypeVar("T")


class _CompileTimeTag:
    """Runtime tag embedded in ``Annotated[T, _CompileTimeTag()]``.

    Lets ``_introspect.py`` recognize a ``CompileTime[T]`` annotation without
    pyright treating the parameter's type as anything other than ``T``.
    """

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


class _DispatchTimeTag:
    """Runtime tag embedded in ``Annotated[T, _DispatchTimeTag()]``.

    Lets ``_introspect.py`` recognize a ``DispatchTime[T]`` annotation without
    pyright treating the parameter's type as anything other than ``T``.
    """

    __slots__ = ()


_DISPATCH_TIME_TAG = _DispatchTimeTag()

DispatchTime = Annotated[T, _DISPATCH_TIME_TAG]
"""Runtime-scalar parameter annotation.

Use as a type annotation on generator function parameters that are runtime
scalars: bound once per compiled artifact's *type* (not baked in by value),
re-suppliable per call without a recompile.

Unlike ``CompileTime[T]``, a ``DispatchTime[T]`` value is not part of the
cache key. Unlike ``In``/``Out``/``InOut``, no DMA is involved -- the scalar
reaches the device as a runtime sequence value (an ``npu.write32``/inline TXN
argument or an ``rt.inline_ops`` symbolic bind), not a buffer transfer.

Example::

    def scaled_copy(a: In, b: Out, scale: DispatchTime[int]):
        ...
"""
