# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextvars import ContextVar

from aie.helpers.dialects.scf import (
    _for,
)
from aie.helpers.dialects.scf import (
    yield_ as _yield_,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.ir import InsertionPoint  # pyright: ignore[reportMissingImports]
from aie.iron.runtime.dmataskhandle import Task


class _YieldFrame:
    """One active ``range_`` loop with iter_args.

    Holds the loop body block and the values its terminating ``yield_``
    passed, if any.
    """

    def __init__(self, body):
        self.body = body
        self.values: list | None = None


# One frame per active range_ loop with iter_args. yield_ records its values
# only when it terminates the loop body itself, not a nested scf.if or loop.
# A ContextVar, like the active runtime sequence, so concurrent emitters never
# share frames.
_yield_frames: ContextVar[tuple[_YieldFrame, ...]] = ContextVar(
    "iron_yield_frames", default=()
)


def _unwrap(x):
    """Unwrap a Task to its SSA handle; pass everything else through unchanged.

    A Task carries its SSA handle across scf boundaries; plain Values are
    returned as-is.
    """
    return x.handle if isinstance(x, Task) else x


def range_(*args, iter_args=None, insert_yield=True, **kwargs):
    """``scf.for`` for IRON bodies, with ``Task`` support in ``iter_args``.

    See [`Task`][iron.runtime.dmataskhandle.Task].
    Identical to the low-level ``_for`` helper, except a ``Task`` passed as an
    ``iter_args`` entry is carried across iterations by its SSA handle: the loop
    body and the loop results receive it re-wrapped as a copy of the ``Task``
    passed in (so ``.start()``/``.free()``/``.await_()`` work; a loop result
    takes the type and state, e.g. freed or endpoint, of the ``Task`` the
    loop body's own ``yield_`` passed; a raw handle yielded there keeps the
    copy of the ``Task`` passed in), and
    [`yield_`][iron.controlflow.yield_] accepts ``Task`` entries too. This is
    what a hand-rolled software-pipelined DMA loop needs.
    """
    wrapped = {}
    if iter_args is not None:
        raw = []
        for i, a in enumerate(iter_args):
            if isinstance(a, Task):
                wrapped[i] = a
            raw.append(_unwrap(a))
        iter_args = raw

    def rewrap_args(a):
        # a is a single value, a tuple of iter_args, or absent (iv only).
        if isinstance(a, tuple):
            return tuple(
                wrapped[i]._with_handle(v) if i in wrapped else v
                for i, v in enumerate(a)
            )
        return wrapped[0]._with_handle(a) if 0 in wrapped else a

    for vals in _for(*args, iter_args=iter_args, insert_yield=insert_yield, **kwargs):
        if isinstance(vals, tuple) and len(vals) == 3:
            iv, a, results = vals
            results = rewrap_args(results)
            frame = _YieldFrame(InsertionPoint.current.block)
            outer = _yield_frames.get()
            _yield_frames.set(outer + (frame,))
            try:
                yield iv, rewrap_args(a), results
            finally:
                _yield_frames.set(outer)
            # A loop result is the Task the body yielded, so it takes that
            # Task's type and state (lifetime, endpoint), not the initial one's.
            result_tasks = results if isinstance(results, tuple) else (results,)
            for i, task in enumerate(frame.values or ()):
                if i in wrapped and isinstance(task, Task):
                    res = result_tasks[i]
                    handle = res.handle
                    res.__class__ = type(task)
                    res.__dict__ = {**vars(task), "_handle": handle}
        else:
            # iv-only (no iter_args) never has wrapped positions.
            yield vals


def yield_(values):
    """``scf.yield`` that accepts ``Task`` entries, yielding each Task's SSA handle.

    See [`Task`][iron.runtime.dmataskhandle.Task].
    """
    values = list(values)
    frames = _yield_frames.get()
    if frames and InsertionPoint.current.block == frames[-1].body:
        frames[-1].values = values
    _yield_([_unwrap(v) for v in values])
