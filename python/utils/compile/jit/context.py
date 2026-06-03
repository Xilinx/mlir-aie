# context.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Compile-time context injection via contextvars.

``compile_context`` is a context manager that injects compile-time key/value
pairs into the current thread/task context so that any code called transitively
during MLIR generation can read them via ``get_compile_arg``.

This is an advanced/dynamic-case mechanism.  The primary API is the explicit
``Compile[T]``-annotated generator function signature — ``compile_context`` is
used internally by ``CompilableDesign.compile()`` and exposed as a public API
for composite/nested generator patterns.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Any

# Module-level ContextVar holding the active compile-time kwargs dict.
# A new copy is pushed on each nested compile_context entry so nesting is safe.
_compile_context_var: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "_compile_context_var", default={}
)


@contextmanager
def compile_context(**kwargs: Any):
    """Context manager that injects compile-time parameters into the active context.

    Any code called inside the ``with`` block can read the injected values via
    ``get_compile_arg``.  Contexts nest correctly: inner values shadow outer
    values for the duration of the inner block.

    Args:
        **kwargs: Compile-time parameter names and values to inject.

    Example::

        with compile_context(M=512, K=512, N=512):
            module = generate_mlir()   # can call get_compile_arg("M") etc.
    """
    # Merge with any outer context so inner callers still see outer values
    # for keys they don't override.
    outer = _compile_context_var.get()
    merged = {**outer, **kwargs}
    token = _compile_context_var.set(merged)
    try:
        yield merged
    finally:
        _compile_context_var.reset(token)


def get_compile_arg(key: str, default: Any = None) -> Any:
    """Read a compile-time parameter from the active ``CompileContext``.

    Returns ``default`` (``None``) when called outside any ``CompileContext``
    or when ``key`` was not injected.

    Args:
        key: Name of the compile-time parameter.
        default: Value returned when the key is absent.

    Returns:
        The value injected for ``key``, or ``default``.

    Example::

        M = get_compile_arg("M")   # returns 512 if CompileContext(M=512) is active
    """
    return _compile_context_var.get().get(key, default)
