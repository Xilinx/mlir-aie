# _introspect.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The generator front end of a declaration.

`declare` reads a generator's signature into the same ``Declaration`` an
operator's class body makes, which ``CompilableDesign`` reads.
"""

from __future__ import annotations

import functools
import inspect
import re
import typing
from dataclasses import MISSING
from types import FunctionType
from typing import Annotated, Callable, get_args, get_origin

from .declaration import Declaration
from .markers import In, InOut, Out
from .params import Member, OperatorParam, PerCallMember, param

# Annotation origins that represent runtime tensor directions.
_TENSOR_ANNOTATIONS = (In, Out, InOut)

_MARKER_NAME = re.compile(r"\b(?:In|Out|InOut|CompileTime|DispatchTime)\b")


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


@functools.lru_cache(maxsize=None)
def declare(generator: Callable) -> Declaration:
    """Return the declaration a generator's signature makes.

    ``CompileTime[T]`` parameters are its parameters, ``In``/``Out``/``InOut``
    its operands (their shapes come from a call's tensor types) and
    ``DispatchTime[T]`` its per-call values; any other is a scalar.
    Memoised: ``typing.get_type_hints`` and ``inspect.signature`` together
    cost ~40us, and the JIT hot path asks per call.
    """
    sig = inspect.signature(generator)
    try:
        # include_extras=True: without it, get_type_hints strips Annotated
        # metadata and CompileTime[T] becomes indistinguishable from T.
        hints = typing.get_type_hints(generator, include_extras=True)
    except Exception:
        # One unresolvable annotation (e.g. a TYPE_CHECKING-only import) must
        # not demote the others to scalars, so resolve each on its own.
        hints = {}
        namespace = generator.__globals__ if isinstance(generator, FunctionType) else {}
        for name, p in sig.parameters.items():
            if not isinstance(p.annotation, str):
                continue
            try:
                hints[name] = eval(p.annotation, namespace)
            except Exception as exc:
                if _MARKER_NAME.search(p.annotation):
                    raise TypeError(
                        f"Parameter {name!r} of {generator!r}: annotation "
                        f"{p.annotation!r} names an iron marker but does not "
                        f"resolve ({type(exc).__name__}: {exc})"
                    ) from exc

    sig = sig.replace(
        parameters=[
            p.replace(annotation=hints.get(n, p.annotation))
            for n, p in sig.parameters.items()
        ]
    )
    params: dict[str, OperatorParam] = {}
    members: list[Member] = []
    scalars: list[str] = []
    for name, p in sig.parameters.items():
        ann = p.annotation
        if ann is inspect.Parameter.empty:
            scalars.append(name)
            continue
        if _tagged_type(ann, "aie.compile_time") is not None:
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
        elif ann in _TENSOR_ANNOTATIONS:
            member = ann()
        elif isinstance(ann, _TENSOR_ANNOTATIONS):
            raise TypeError(
                f"Parameter {name!r} of {generator!r}: annotate a tensor with the "
                f"bare {type(ann).__name__}; its shape and dtype come from the "
                "call's tensor. A coded operand is an Operator's, whose caller "
                "uploads aie.utils.coded.encode_rows' rows"
            )
        elif (dtype := _tagged_type(ann, "aie.dispatch_time")) is not None:
            member = PerCallMember(dtype, None, None)
        else:
            scalars.append(name)
            continue
        member.__set_name__(None, name)
        if isinstance(member, OperatorParam):
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
