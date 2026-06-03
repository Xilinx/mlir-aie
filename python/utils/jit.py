# jit.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""``@iron.jit`` decorator — Triton-style JIT compilation for the NPU.

``@iron.jit`` is a thin wrapper that creates a ``CallableDesign``.  Extra
kwargs that are not recognised as configuration keys become ``compile_kwargs``
(i.e. values for ``Compile[T]``-annotated generator parameters).

Three usage patterns are supported:

1. **Bare decorator** — no pre-bound compile params::

       @iron.jit
       def gemm(a: In, b: In, c: Out, *,
                M: Compile[int], K: Compile[int], N: Compile[int]):
           ...

       gemm(a, b, c, M=512, K=512, N=512)   # compile params at call time

2. **With configuration only** — source files, flags, etc., no compile params::

       @iron.jit(source_files=["kernel.cc"])
       def gemm(a: In, b: In, c: Out, *, M: Compile[int], ...):
           ...

3. **With pre-bound compile params** — Triton-style, params fixed at decoration::

       @iron.jit(M=512, K=512, N=512)
       def gemm(a: In, b: In, c: Out, *,
                M: Compile[int], K: Compile[int], N: Compile[int]):
           ...

       gemm(a, b, c)   # no compile params needed at call time

.. note::
   The decorated function body runs inside an implicit MLIR context
   (thread-local ``Location`` / ``InsertionPoint``).  See
   ``programming_guide/implicit_mlir_context.md`` for the model and
   for the most common gotchas (e.g. why ``@func`` pykernels must be
   decorated at module scope).
"""

from __future__ import annotations

import functools
import inspect as _inspect
from typing import Callable

from aie.utils.callabledesign import CallableDesign as _CallableDesign

# Derived from CallableDesign.__init__ so it stays in sync automatically.
# Excludes 'self', 'mlir_generator', and 'compile_kwargs' — those are
# positional/compile-param arguments, not config keys.
_JIT_CONFIG_KEYS = frozenset(
    p
    for p in _inspect.signature(_CallableDesign.__init__).parameters
    if p not in ("self", "mlir_generator", "compile_kwargs")
)


def jit(mlir_generator: Callable | None = None, **kwargs):
    """Decorator for JIT compilation and NPU execution.

    Standard configuration kwargs (``use_cache``, ``source_files``,
    ``aiecc_flags``, ``compile_flags``, ``include_paths``, ``object_files``,
    ``trace_config``) are forwarded to ``CallableDesign``.  All other kwargs
    become ``compile_kwargs`` (values for ``Compile[T]``-annotated parameters).

    Args:
        mlir_generator: The MLIR generator callable (supplied automatically
            when used as a bare decorator).
        **kwargs: Mix of config options and/or compile-time parameter values.

    Returns:
        A ``CallableDesign`` instance (or a partial decorator when called with
        kwargs before the generator is known).
    """
    if mlir_generator is None:
        # Called with kwargs only — return a partial so the generator can be
        # supplied when Python applies the decorator.
        return functools.partial(jit, **kwargs)

    config = {k: v for k, v in kwargs.items() if k in _JIT_CONFIG_KEYS}
    compile_kwargs = {k: v for k, v in kwargs.items() if k not in _JIT_CONFIG_KEYS}

    # --- Validate Compile[T] params when generator is callable ---
    if callable(mlir_generator):
        from aie.utils.compile.jit.compilabledesign import split_params

        compile_params, _, scalar_params = split_params(mlir_generator)

        # Guard 1-A: reject any compile kwarg that doesn't match a Compile[T]
        # param. Failing fast at decoration time catches typos like @jit(NN=...)
        # before they silently run a kernel with no value bound.
        if compile_kwargs:
            unknown = set(compile_kwargs.keys()) - set(compile_params)
            if unknown:
                raise TypeError(
                    f"@iron.jit received keyword argument(s) that do not match any "
                    f"Compile[T]-annotated parameter of {mlir_generator.__name__!r}: "
                    f"{sorted(unknown)}.\n"
                    f"  Valid Compile[T] params: {compile_params}.\n"
                    f"  Config keys: {sorted(_JIT_CONFIG_KEYS)}."
                )

        # Guard 1-C: reject unannotated non-tensor params with default values.
        # The framework has no plumbing for runtime scalar args yet (RTPs are
        # tracked as a follow-up — see project memory), so a default value
        # gets baked into the compiled MLIR at decoration time and any per-
        # call override is *silently* ignored.  That's the worst kind of bug:
        # the kernel runs successfully but with the wrong value.  Force the
        # author to be explicit instead.
        sig_for_defaults = _inspect.signature(mlir_generator)
        silent_default_scalars = [
            name
            for name in scalar_params
            if sig_for_defaults.parameters[name].default is not _inspect.Parameter.empty
        ]
        if silent_default_scalars:
            raise TypeError(
                f"@iron.jit: parameter(s) {silent_default_scalars!r} of "
                f"{mlir_generator.__name__!r} have default values but no "
                f"In / Out / InOut / Compile[T] annotation.  The framework has "
                f"no runtime-scalar plumbing yet, so the default would be "
                f"baked into the compiled kernel and per-call overrides "
                f"silently ignored.\n"
                f"  Fix options:\n"
                f"    * Use Compile[T] = default to keep the default and "
                f"recompile on per-call change.\n"
                f"    * Annotate as In / Out / InOut if it's a tensor."
            )

        # Guard: Compile[T] params must be keyword-only (unless pre-bound or
        # have a signature default).  Pre-bound and defaulted params are exempt
        # because callers can omit them at the call site, so positional-vs-
        # keyword ambiguity does not arise.
        sig = _inspect.signature(mlir_generator)
        non_kw_compile_params = [
            name
            for name in compile_params
            if sig.parameters[name].kind
            not in (
                _inspect.Parameter.KEYWORD_ONLY,
                _inspect.Parameter.VAR_KEYWORD,
            )
            and name not in compile_kwargs  # pre-bound params are exempt
            and sig.parameters[name].default
            is _inspect.Parameter.empty  # defaults exempt
        ]
        if non_kw_compile_params:
            raise TypeError(
                f"@iron.jit: Compile[T] parameter(s) {non_kw_compile_params!r} "
                f"in {mlir_generator.__name__!r} are not keyword-only.\n"
                f"Place a bare '*' before your Compile[T] parameters:\n\n"
                f"  # Before:\n"
                f"  def {mlir_generator.__name__}(a: In, b: Out, "
                + ", ".join(f"{n}: Compile[...]" for n in non_kw_compile_params)
                + "):\n"
                f"      ...\n\n"
                f"  # After:\n"
                f"  def {mlir_generator.__name__}(a: In, b: Out, *, "
                + ", ".join(f"{n}: Compile[...]" for n in non_kw_compile_params)
                + "):\n"
                f"      ..."
            )

    return _CallableDesign(
        mlir_generator,
        compile_kwargs=compile_kwargs if compile_kwargs else None,
        **config,
    )
