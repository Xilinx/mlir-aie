# _introspect.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Annotation introspection helpers for classifying generator parameters.

Classifies generator parameters into ``CompileTime[T]`` / tensor /
``DispatchTime[T]`` / scalar buckets. Carved out of ``compilabledesign.py`` to
keep the main file focused on the
``CompilableDesign`` class itself.  The public surface is
:func:`split_params`; the rest is implementation detail behind
:func:`_introspect_generator`'s ``lru_cache``.
"""

from __future__ import annotations

import functools
import inspect
import logging
import typing
from typing import Annotated, Callable, get_args, get_origin

from .markers import _COMPILE_TIME_TAG, _DISPATCH_TIME_TAG, In, InOut, Out

logger = logging.getLogger(__name__)

# Annotation origins that represent runtime tensor directions.
_TENSOR_ANNOTATIONS = (In, Out, InOut)


def _is_compile_param(annotation) -> bool:
    """Return True for ``CompileTime[T]`` or ``Optional[CompileTime[T]]``."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return any(arg is _COMPILE_TIME_TAG for arg in get_args(annotation)[1:])
    # get_type_hints rewrites `CompileTime[T] = None` defaults to Optional[...].
    if origin is typing.Union:
        return any(_is_compile_param(arg) for arg in get_args(annotation))
    return False


def _is_dispatch_param(annotation) -> bool:
    """Return True for ``DispatchTime[T]`` or ``Optional[DispatchTime[T]]``."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return any(arg is _DISPATCH_TIME_TAG for arg in get_args(annotation)[1:])
    # get_type_hints rewrites `DispatchTime[T] = None` defaults to Optional[...].
    if origin is typing.Union:
        return any(_is_dispatch_param(arg) for arg in get_args(annotation))
    return False


def _is_tensor_param(annotation) -> bool:
    """Return True if *annotation* is ``In``, ``Out``, or ``InOut``."""
    return annotation in _TENSOR_ANNOTATIONS


def _dispatch_param_type(annotation):
    """Return the wrapped type ``T`` for a ``DispatchTime[T]`` annotation.

    Mirrors ``_is_dispatch_param()``'s ``Annotated``/``Union`` unwrapping so
    callers recover the same ``T`` regardless of how ``get_type_hints``
    rewrote a defaulted param (``DispatchTime[T] = None`` becomes
    ``Optional[...]``).  Returns ``None`` if *annotation* is not a
    ``DispatchTime[T]`` annotation.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        if any(arg is _DISPATCH_TIME_TAG for arg in args[1:]):
            return args[0]
        return None
    if origin is typing.Union:
        for arg in get_args(annotation):
            wrapped = _dispatch_param_type(arg)
            if wrapped is not None:
                return wrapped
    return None


@functools.lru_cache(maxsize=None)
def _introspect_generator(generator: Callable):
    """Memoise ``(hints, signature, (compile, tensor, dispatch, scalar))``.

    All four derived values are pure functions of the generator's source —
    the same answer for every call of the same function object — but
    ``typing.get_type_hints`` and ``inspect.signature`` together cost
    ~40us per invocation.  Memoising shaves both ``split_params`` and
    ``CompilableDesign.split_runtime_args`` to a dict lookup on the JIT
    hot path.

    The returned param tuples are immutable; callers that need lists copy.
    """
    try:
        # include_extras=True: without it, get_type_hints strips Annotated
        # metadata and CompileTime[T] becomes indistinguishable from T.
        hints = typing.get_type_hints(generator, include_extras=True)
    except Exception as exc:
        logger.debug("get_type_hints failed for %r: %s", generator, exc)
        hints = {}

    sig = inspect.signature(generator)
    compile_params: list[str] = []
    tensor_params: list[str] = []
    dispatch_params: list[str] = []
    scalar_params: list[str] = []
    for name, param in sig.parameters.items():
        ann = hints.get(name, param.annotation)
        if ann is inspect.Parameter.empty:
            scalar_params.append(name)
        elif _is_compile_param(ann):
            compile_params.append(name)
        elif _is_tensor_param(ann):
            tensor_params.append(name)
        elif _is_dispatch_param(ann):
            dispatch_params.append(name)
        else:
            scalar_params.append(name)

    return (
        hints,
        sig,
        (
            tuple(compile_params),
            tuple(tensor_params),
            tuple(dispatch_params),
            tuple(scalar_params),
        ),
    )


def split_params(
    generator: Callable,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Inspect *generator*; return ``(compile, tensor, dispatch, scalar)`` params.

    * ``compile_params``  — names with ``CompileTime[T]`` annotation
    * ``tensor_params``   — names with ``In``/``Out``/``InOut`` annotation (in order)
    * ``dispatch_params``  — names with ``DispatchTime[T]`` annotation (runtime
      scalars, one compiled artifact serves many values)
    * ``scalar_params``   — names with any other annotation (unclassified;
      currently rejected at ``@iron.jit`` time if they carry a default)

    Uses ``typing.get_type_hints()`` so that stringified annotations (produced
    by ``from __future__ import annotations`` or PEP 563 mode) are evaluated
    correctly.  Falls back to ``inspect.signature`` annotations on any error
    (e.g. when the generator's globals are not resolvable at call time).
    """
    _, _, (cp, tp, dp, sp) = _introspect_generator(generator)
    return list(cp), list(tp), list(dp), list(sp)
