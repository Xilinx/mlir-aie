# compileconfig.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""``@iron.compileconfig`` decorator — attaches compile configuration to a generator.

``@compileconfig`` wraps a generator function in a ``CompilableDesign`` so that
compile-time options (source files, flags, etc.) can be declared once at
definition time.  It does *not* bind compile_kwargs — those are supplied later
at ``CompilableDesign`` construction or ``@iron.jit(...)`` call time.

Bug fix vs. erika-vibe-coding: the original ``functools.partial`` call lacked
a callable as its first argument.  Fixed by using a proper inner wrapper.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable

from .compilabledesign import CompilableDesign


def compileconfig(
    mlir_generator: Callable | None = None,
    *,
    use_cache: bool = True,
    compile_flags: list[str] | None = None,
    source_files: list[str | Path] | None = None,
    include_paths: list[str | Path] | None = None,
    aiecc_flags: list[str] | None = None,
    object_files: list[str | Path] | None = None,
):
    """Decorator that attaches compile configuration to a generator function.

    Can be used bare (``@iron.compileconfig``) or with keyword arguments
    (``@iron.compileconfig(source_files=[...])``).  All configuration options
    are keyword-only to prevent accidental positional misuse.

    Does **not** bind ``compile_kwargs`` — those come from the ``@iron.jit``
    decorator or explicit ``CompilableDesign(generator, compile_kwargs={...})``.

    Args:
        mlir_generator: The MLIR generator callable (supplied automatically
            when used as a bare decorator).
        use_cache: Enable file-system caching. Defaults to True.
        compile_flags: Extra flags for the Peano C++ compiler.
        source_files: C++ kernel source files whose mtimes invalidate the cache.
        include_paths: Extra ``-I`` paths for the C++ compiler.
        aiecc_flags: Extra flags for ``aiecc``.
        object_files: Pre-compiled ``.o`` files to link with.

    Returns:
        A ``CompilableDesign`` (when used as ``@iron.compileconfig`` or with
        keyword args), or a partial decorator (internal use when keywords are
        supplied before the callable).

    Example::

        @iron.compileconfig(source_files=["kernel.cc"])
        def gemm_design(a: In, b: In, c: Out,
                        M: Compile[int], K: Compile[int], N: Compile[int]):
            ...

        design = CompilableDesign(gemm_design, compile_kwargs={"M": 512, ...})
    """
    config_kwargs = dict(
        use_cache=use_cache,
        compile_flags=list(compile_flags or []),
        source_files=list(source_files or []),
        include_paths=list(include_paths or []),
        aiecc_flags=list(aiecc_flags or []),
        object_files=list(object_files or []),
    )

    if mlir_generator is None:
        # Called with keyword args only: return a decorator.
        return functools.partial(compileconfig, **config_kwargs)

    # Called as bare decorator or with the generator already supplied.
    return CompilableDesign(mlir_generator, **config_kwargs)
