# markers.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Argument markers: how a value reaches a design or a kernel.

``CompileTime`` / ``In`` / ``Out`` / ``InOut`` annotate ``@iron.jit`` generator
parameters. ``In`` / ``Out`` / ``InOut`` are also the vocabulary a
``KernelContract`` uses to name what each of a kernel's arguments is, alongside
``Param`` / ``Scalar`` / ``Count`` below, which have no design-parameter
meaning. One set of names for one question, whichever side is asking.

All are exported from ``aie.iron``.

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

``Param``
    A kernel argument held for the whole run rather than streamed per call
    (``scale``'s factor, ``filter2d``'s 3x3 window). A design gives it a core
    buffer with an initial value; it is not a design parameter.

``Scalar``
    A kernel argument that is a runtime scalar constant (``leaky_relu``'s
    alpha, ``axpy``'s a).

``Count``
    A kernel argument that is the trailing element count the C++ takes at
    runtime: the smaller of the ``In``/``Out`` element counts, so a
    channel-ratio conversion gets it from whichever side is 1:1
    (``rgba2hue``'s ``lineWidth`` counts hue pixels, a quarter of its RGBA
    input). A reduction's ``out_valid`` exempts it: the count is then the
    larger input's element count.

Any generator parameter without one of the first four annotations is currently
rejected at ``@iron.jit`` decoration time when the parameter has a default
value — there is no runtime-scalar plumbing yet (tracked separately as future
work), so the default would be baked into the compiled kernel and per-call
overrides silently ignored.  Annotate as ``CompileTime[T]`` (recompiles on
change) or ``In``/``Out``/``InOut`` (DMA tensor) instead.
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
    """Runtime bidirectional tensor annotation (DMA in both directions each call).

    As a kernel-contract role: the kernel accumulates into this argument
    (``mm``'s ``C += A * B``), so a design zeroes it before the first call.
    """


class Param:
    """Kernel argument held for the whole run, not streamed per call."""


class Scalar:
    """Kernel argument that is a runtime scalar constant."""


class Count:
    """Kernel argument that is the trailing element count, passed at runtime."""


#: Every marker a ``KernelContract`` may give an argument, in no order.
ROLES = (In, Out, InOut, Param, Scalar, Count)
