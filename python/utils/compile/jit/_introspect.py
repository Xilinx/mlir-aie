# _introspect.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Annotation introspection helpers used to classify generator parameters
into ``CompileTime[T]`` / tensor / scalar buckets.

Carved out of ``compilabledesign.py`` to keep the main file focused on the
``CompilableDesign`` class itself.  The public surface is
:func:`split_params`; the rest is implementation detail behind
:func:`_introspect_generator`'s ``lru_cache``.
"""

from __future__ import annotations

import functools
import inspect
import logging
import typing
from typing import Callable, get_args, get_origin

from .markers import CompileTime, In, InOut, Out

logger = logging.getLogger(__name__)

# Annotation origins that represent runtime tensor directions.
_TENSOR_ANNOTATIONS = (In, Out, InOut)


def _is_compile_param(annotation) -> bool:
    """Return True for ``CompileTime[T]`` or ``Optional[CompileTime[T]]``."""
    if annotation is CompileTime:
        return True
    origin = get_origin(annotation)
    if origin is CompileTime:
        return True
    # get_type_hints rewrites `CompileTime[T] = None` defaults to Optional[...].
    if origin is typing.Union:
        return any(_is_compile_param(arg) for arg in get_args(annotation))
    return False


def _is_tensor_param(annotation) -> bool:
    """Return True if *annotation* is ``In``, ``Out``, or ``InOut``."""
    return annotation in _TENSOR_ANNOTATIONS


@functools.lru_cache(maxsize=None)
def _introspect_generator(generator: Callable):
    """Memoise ``(hints, signature, (compile, tensor, scalar))`` for a generator.

    All three derived values are pure functions of the generator's source —
    the same answer for every call of the same function object — but
    ``typing.get_type_hints`` and ``inspect.signature`` together cost
    ~40us per invocation.  Memoising shaves both ``split_params`` and
    ``CompilableDesign.split_runtime_args`` to a dict lookup on the JIT
    hot path.

    The returned param tuples are immutable; callers that need lists copy.
    """
    try:
        hints = typing.get_type_hints(generator)
    except Exception as exc:
        logger.debug("get_type_hints failed for %r: %s", generator, exc)
        hints = {}

    sig = inspect.signature(generator)
    compile_params: list[str] = []
    tensor_params: list[str] = []
    scalar_params: list[str] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            scalar_params.append(name)
        elif _is_compile_param(ann):
            compile_params.append(name)
        elif _is_tensor_param(ann):
            tensor_params.append(name)
        else:
            scalar_params.append(name)

    return (
        hints,
        sig,
        (
            tuple(compile_params),
            tuple(tensor_params),
            tuple(scalar_params),
        ),
    )


def split_params(generator: Callable) -> tuple[list[str], list[str], list[str]]:
    """Inspect *generator* and return ``(compile_params, tensor_params, scalar_params)``.

    * ``compile_params``  — names with ``CompileTime[T]`` annotation
    * ``tensor_params``   — names with ``In``/``Out``/``InOut`` annotation (in order)
    * ``scalar_params``   — names with any other annotation (runtime scalars)

    Uses ``typing.get_type_hints()`` so that stringified annotations (produced
    by ``from __future__ import annotations`` or PEP 563 mode) are evaluated
    correctly.  Falls back to ``inspect.signature`` annotations on any error
    (e.g. when the generator's globals are not resolvable at call time).
    """
    _, _, (cp, tp, sp) = _introspect_generator(generator)
    return list(cp), list(tp), list(sp)
