# _introspect.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The generator front end of a declaration.

`declare` reads a generator's signature into the same ``Declaration`` an
operator's class body makes, which ``CompilableDesign`` reads; `split_params`
lists its parameter names by kind.
"""

from __future__ import annotations

import functools
import inspect
import logging
import typing
from dataclasses import MISSING
from types import FunctionType
from typing import Annotated, Callable, get_args, get_origin

from .declaration import Declaration
from .markers import In, InOut, Out
from .params import Member, Param, PerCallMember, param

logger = logging.getLogger(__name__)

# Annotation origins that represent runtime tensor directions.
_TENSOR_ANNOTATIONS = (In, Out, InOut)


def _tagged_type(annotation, tag):
    """Return the wrapped ``T`` for an annotation carrying *tag*, else ``None``.

    One traversal for every marker: ``get_type_hints`` rewrites a defaulted
    ``Marker[T] = None`` to ``Optional[...]``, so both the direct
    ``Annotated`` form and the ``Union`` wrapper have to be unwrapped, and
    every marker needs it the same way.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        args = get_args(annotation)
        return (
            args[0]
            if any(isinstance(arg, str) and arg == tag for arg in args[1:])
            else None
        )
    if origin is typing.Union:
        for arg in get_args(annotation):
            wrapped = _tagged_type(arg, tag)
            if wrapped is not None:
                return wrapped
    return None


def _is_compile_param(annotation) -> bool:
    """Return True for ``CompileTime[T]`` or ``Optional[CompileTime[T]]``."""
    return _tagged_type(annotation, "aie.compile_time") is not None


def _is_tensor_param(annotation) -> bool:
    """Return True if *annotation* is ``In``, ``Out``, or ``InOut``."""
    return annotation in _TENSOR_ANNOTATIONS


def _dispatch_param_type(annotation):
    """Return the wrapped ``T`` of a ``DispatchTime[T]``, else ``None``."""
    return _tagged_type(annotation, "aie.dispatch_time")


@functools.lru_cache(maxsize=None)
def declare(generator: Callable) -> Declaration:
    """Return the declaration a generator's signature makes.

    ``CompileTime[T]`` parameters are its parameters, ``In``/``Out``/``InOut``
    its operands (their shapes come from a call's tensor types) and
    ``DispatchTime[T]`` its per-call values; any other is a scalar.
    Memoised: ``typing.get_type_hints`` and ``inspect.signature`` together
    cost ~40us, and the JIT hot path asks per call.
    """
    try:
        # include_extras=True: without it, get_type_hints strips Annotated
        # metadata and CompileTime[T] becomes indistinguishable from T.
        hints = typing.get_type_hints(generator, include_extras=True)
    except Exception as exc:
        logger.debug("get_type_hints failed for %r: %s", generator, exc)
        hints = {}

    sig = inspect.signature(generator)
    sig = sig.replace(
        parameters=[
            p.replace(annotation=hints.get(n, p.annotation))
            for n, p in sig.parameters.items()
        ]
    )
    params: dict[str, Param] = {}
    members: list[Member] = []
    scalars: list[str] = []
    for name, p in sig.parameters.items():
        ann = p.annotation
        if ann is inspect.Parameter.empty:
            scalars.append(name)
            continue
        if _is_compile_param(ann):
            default = MISSING if p.default is p.empty else p.default
            # A function default is the value, so the record computes it as
            # itself; so is a list, dict or set, which param() would reject as
            # shared between instances, but the function already shares it.
            member = param(
                default=(
                    (lambda op, f=default: f)
                    if isinstance(default, (FunctionType, list, dict, set))
                    else default
                )
            )
        elif _is_tensor_param(ann):
            member = ann()
        elif (dtype := _dispatch_param_type(ann)) is not None:
            member = PerCallMember(dtype, None, None)
        else:
            scalars.append(name)
            continue
        member.__set_name__(None, name)
        if isinstance(member, Param):
            params[name] = member
        else:
            members.append(member)
    return Declaration(
        generator.__name__ if isinstance(generator, FunctionType) else repr(generator),
        params,
        tuple(members),
        scalars=tuple(scalars),
        signature=sig,
    )


def split_params(
    generator: Callable,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return `generator`'s ``(compile, tensor, dispatch, scalar)`` parameter names.

    * ``compile_params``  — names with ``CompileTime[T]`` annotation
    * ``tensor_params``   — names with ``In``/``Out``/``InOut`` annotation (in
      order); a ``*tensors: In`` entry stands for however many tensors a call
      passes after the named ones
    * ``dispatch_params``  — names with ``DispatchTime[T]`` annotation (runtime
      scalars, one compiled artifact serves many values)
    * ``scalar_params``   — names with any other annotation (unclassified;
      currently rejected at ``@iron.jit`` time if they carry a default)

    They are read from ``declare(generator)``.
    """
    d = declare(generator)
    return (
        list(d.params),
        [m.name for m in d.operands],
        [m.name for m in d.per_calls],
        list(d.scalars),
    )
