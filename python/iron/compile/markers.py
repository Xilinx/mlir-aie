# markers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Type-annotation markers for compile-time vs. runtime parameter classification.

Three annotation categories are defined here (all exported from ``aie.iron``):

``Compile[T]``
    Marks a generator function parameter as compile-time.  Changing its value
    causes a recompile and a new cache entry.  Inspired by ``tl.constexpr`` in
    Triton.  Standard ``Generic[T]``, fully compatible with mypy/pyright.

``In``
    Marks a generator function parameter as a runtime *input* tensor.  Data is
    DMA-transferred from the host to the NPU on every kernel call.

``Out``
    Marks a generator function parameter as a runtime *output* tensor.  Data is
    DMA-transferred from the NPU to the host on every kernel call.

``InOut``
    Marks a generator function parameter as a runtime bidirectional tensor.
    Data is DMA-transferred in both directions on every kernel call.

Any parameter without one of these four annotations (e.g. ``alpha: float``) is
treated as a runtime scalar: passed directly as a kernel argument each call,
no DMA transfer, no recompile.
"""

from __future__ import annotations

from typing import Generic, TypeVar

T = TypeVar("T")


class Compile(Generic[T]):
    """Compile-time parameter annotation.

    Use as a type annotation on generator function parameters that affect the
    generated MLIR.  The value must be supplied at ``CompilableDesign``
    construction time (or bound by ``@iron.jit(...)``).

    Changing a ``Compile[T]``-annotated value → new cache key → recompile.
    Required unless a default is given.

    Example::

        def gemm(a: In, b: In, c: Out,
                 M: Compile[int], K: Compile[int], N: Compile[int],
                 dtype: Compile[np.dtype] = np.bfloat16):
            ...
    """


class In:
    """Runtime input tensor annotation (host → NPU, DMA each call)."""


class Out:
    """Runtime output tensor annotation (NPU → host, DMA each call)."""


class InOut:
    """Runtime bidirectional tensor annotation (DMA in both directions each call)."""
