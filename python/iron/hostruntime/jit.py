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
"""

from __future__ import annotations

import functools
import inspect as _inspect
import warnings
from pathlib import Path
from typing import Callable

from .callabledesign import CallableDesign as _CallableDesign

# Derived from CallableDesign.__init__ so it stays in sync automatically.
# Excludes 'self', 'mlir_generator', and 'compile_kwargs' — those are
# positional/compile-param arguments, not config keys.
_JIT_CONFIG_KEYS = frozenset(
    p
    for p in _inspect.signature(_CallableDesign.__init__).parameters
    if p not in ("self", "mlir_generator", "compile_kwargs")
)

# Re-export CallableDesign under its original name for use in this module.
CallableDesign = _CallableDesign


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
        from aie.iron.compile.compilabledesign import _split_params

        compile_params, _, _ = _split_params(mlir_generator)

        # Guard 1-A: warn if any compile kwarg doesn't match a Compile[T] param.
        if compile_kwargs:
            unknown = set(compile_kwargs.keys()) - set(compile_params)
            if unknown:
                warnings.warn(
                    f"@iron.jit received keyword argument(s) that do not match any "
                    f"Compile[T]-annotated parameter of {mlir_generator.__name__!r}: "
                    f"{unknown}.\n"
                    f"  Valid Compile[T] params: {compile_params}.\n"
                    f"  Config keys: {sorted(_JIT_CONFIG_KEYS)}.",
                    stacklevel=2,
                )

        # Warn about captured globals that could cause stale cache hits.
        import types as _types
        import builtins as _builtins
        from aie.iron.compile.compilabledesign import _collect_co_names

        _builtin_names = set(dir(_builtins))
        all_names = _collect_co_names(mlir_generator.__code__)
        globs = mlir_generator.__globals__
        complex_globals = []
        for name in sorted(all_names):
            val = globs.get(name)
            if val is None:
                continue
            if isinstance(val, (_types.ModuleType, type)):
                continue
            if name in _builtin_names:
                continue
            if callable(val):
                continue
            if isinstance(val, (int, float, str, bool, bytes)):
                continue
            if isinstance(val, (tuple, list)) and all(
                isinstance(v, (int, float, str, bool)) for v in val
            ):
                continue
            complex_globals.append(name)
        if complex_globals:
            warnings.warn(
                f"@iron.jit: {mlir_generator.__name__!r} references module-level global(s) "
                f"{complex_globals!r} whose values cannot be reliably hashed for cache "
                f"invalidation. If these globals affect MLIR generation, use Compile[T] "
                f"parameters instead to ensure correct caching.",
                UserWarning,
                stacklevel=2,
            )

        # Guard: Compile[T] params must be keyword-only (unless pre-bound).
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

    return CallableDesign(
        mlir_generator,
        compile_kwargs=compile_kwargs if compile_kwargs else None,
        **config,
    )
