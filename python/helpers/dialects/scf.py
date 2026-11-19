# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import inspect
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

import numpy as np

from ...dialects.arith import (  # pyright: ignore[reportMissingImports]
    ConstantOp,
)
from ...dialects.scf import (  # pyright: ignore[reportMissingImports]
    ForOp,
    IfOp,
    WhileOp,
    condition,
    yield_,
)
from ...extras import types as T  # pyright: ignore[reportMissingImports]
from ...extras.dialects.arith import (  # pyright: ignore[reportMissingImports]
    constant,
    index_cast,
)
from ...ir import (  # pyright: ignore[reportMissingImports]
    IndexType,
    InsertionPoint,
    IntegerType,
    Value,
)
from ..sourceloc import user_code_location


def _normalize_bounds(start, stop, step):
    """Return ``start``, ``stop``, ``step`` defaulted and typed as ``_for`` itself expects them.

    A missing ``step`` defaults to 1; a missing ``stop`` makes ``start`` the
    stop and 0 the start. A ``np.integer`` bound becomes a Python ``int``;
    anything else (an MLIR ``Value``) passes through unchanged.
    """
    if step is None:
        step = 1
    if stop is None:
        stop = start
        start = 0
    return tuple(
        p.item() if isinstance(p, np.integer) else p for p in (start, stop, step)
    )


def _for_bounds(args: tuple, kwargs: dict) -> tuple:
    """Return the ``start``, ``stop``, ``step`` that calling ``_for(*args, **kwargs)`` would use.

    Binds against ``_for``'s own signature, so a positional or keyword
    ``start``/``stop``/``step`` is picked up the same way whichever form it
    was given in, and a change to ``_for``'s parameters can't drift this out
    of step with it.

    Args:
        args: The positional arguments a ``_for`` call was (or would be) made
            with.
        kwargs: Its keyword arguments.

    Returns:
        The bounds, defaulted and typed as `_normalize_bounds` returns them.
    """
    bound = inspect.signature(_for).bind_partial(*args, **kwargs).arguments
    return _normalize_bounds(bound.get("start"), bound.get("stop"), bound.get("step"))


def _for(
    start,
    stop=None,
    step=None,
    iter_args: Sequence[Value] | None = None,
    insert_yield: bool = True,
    *,
    loc=None,
    ip=None,
) -> Iterator[Any]:
    """Emit an scf.for loop, like the convenience wrapper in scf but with an added insert_yield parameter.

    The insert_yield parameter defaults to True; if left as True, the user no longer needs to manually insert
    yield operations (```yield_([])```). If the user wishes to specify yield directly (such as if there is
    a return value from the loop body), insert_yield should be set to False.

    Also handles numpy data types for the loop variable.
    """
    if not (
        isinstance(start, int)
        or isinstance(start, np.integer)
        or isinstance(start, Value)
    ):
        raise ValueError(
            f"MLIR SCF for loops can only use integer types, not {type(start)}"
        )

    start, stop, step = _normalize_bounds(start, stop, step)
    params = [start, stop, step]
    for i, p in enumerate(params):
        if isinstance(p, int):
            p = constant(p, index=True)
        elif not isinstance(p.type, IndexType):
            p = index_cast(p, to=T.index())
        params[i] = p

    start, stop, step = params

    for_op = ForOp(start, stop, step, iter_args, loc=loc, ip=ip)
    iv = for_op.induction_variable
    iter_args = tuple(for_op.inner_iter_args)
    with InsertionPoint(for_op.body):
        if len(iter_args) > 1:
            yield iv, iter_args, tuple(for_op.results)
        elif len(iter_args) == 1:
            yield iv, iter_args[0], for_op.results[0]
        else:
            yield iv
        if insert_yield:
            yield_([])


def forever_(*, loc=None, ip=None) -> Iterator[None]:
    """Emit an scf.while whose condition is always true, for a loop that never exits.

    Used like ``_for``: the body of a Python ``for`` over it is emitted once,
    into the loop body. Unlike an scf.for with a huge bound, the loop keeps no
    counter.
    """
    while_op = WhileOp([], [], loc=loc, ip=ip)
    with InsertionPoint(while_op.before.blocks.append()):
        condition(ConstantOp(IntegerType.get_signless(1), 1), [])
    with InsertionPoint(while_op.after.blocks.append()):
        yield
        yield_([])


@contextmanager
def if_(cond, hasElse=True, insert_yield=True, loc=None, ip=None):
    if loc is None:
        loc = user_code_location()
    if_op = IfOp(cond, has_else=hasElse, loc=loc, ip=ip)
    with InsertionPoint(if_op.thenRegion.blocks[0]):
        yield if_op
        if insert_yield:
            yield_([])
    if hasElse:
        with InsertionPoint(if_op.elseRegion.blocks[0]):
            if insert_yield:
                yield_([])


# This assumes that the parent if_ context manager passes down the else region.
@contextmanager
def else_(if_op, insert_yield=True):
    with InsertionPoint.at_block_terminator(if_op.elseRegion.blocks[0]):
        yield
