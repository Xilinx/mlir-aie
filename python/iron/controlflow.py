# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from aie.helpers.dialects.scf import (
    _for,
)
from aie.helpers.dialects.scf import (
    yield_ as _yield_,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.iron.runtime.dmataskhandle import Task

# One frame per active range_ loop with iter_args; yield_ records the Tasks
# it yields into the innermost frame, by iter_args position.
_yielded_tasks: list[dict[int, Task]] = []


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
    keeps the freed state of the ``Task`` the body yielded), and
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
            _yielded_tasks.append({})
            try:
                yield iv, rewrap_args(a), results
            finally:
                yielded = _yielded_tasks.pop()
            # A loop result is the Task the body yielded, so it inherits that
            # Task's lifetime (e.g. freed in the body), not the initial one's.
            result_tasks = results if isinstance(results, tuple) else (results,)
            for i, task in yielded.items():
                if i in wrapped:
                    result_tasks[i]._freed = task._freed
        else:
            # iv-only (no iter_args) never has wrapped positions.
            yield vals


def yield_(values):
    """``scf.yield`` that accepts ``Task`` entries, yielding each Task's SSA handle.

    See [`Task`][iron.runtime.dmataskhandle.Task].
    """
    if _yielded_tasks:
        _yielded_tasks[-1].update(
            (i, v) for i, v in enumerate(values) if isinstance(v, Task)
        )
    _yield_([_unwrap(v) for v in values])
