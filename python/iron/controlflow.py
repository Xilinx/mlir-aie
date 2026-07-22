# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from aie.helpers.dialects.scf import _for
from aie.helpers.dialects.scf import yield_ as _yield_
from aie.iron.runtime.dmataskhandle import Task


def _unwrap(x):
    """A Task carries its SSA handle across scf boundaries; everything else
    (plain Values) passes through unchanged."""
    return x.handle if isinstance(x, Task) else x


def range_(*args, iter_args=None, insert_yield=True, **kwargs):
    """``scf.for`` for IRON bodies, with [`Task`][iron.runtime.dmataskhandle.Task]
    support in ``iter_args``.

    Identical to the low-level ``_for`` helper, except a ``Task`` passed as an
    ``iter_args`` entry is carried across iterations by its SSA handle: the loop
    body and the loop results receive it re-wrapped as a ``Task`` (so ``.free()``/
    ``.await_()`` work), and [`yield_`][iron.controlflow.yield_] accepts ``Task``
    entries too. This is what a hand-rolled software-pipelined DMA loop needs.
    """
    wrapped = {}
    if iter_args is not None:
        raw = []
        for i, a in enumerate(iter_args):
            if isinstance(a, Task):
                wrapped[i] = True
            raw.append(_unwrap(a))
        iter_args = raw

    def rewrap_args(a):
        # a is a single value, a tuple of iter_args, or absent (iv only).
        if isinstance(a, tuple):
            return tuple(Task(v) if wrapped.get(i) else v for i, v in enumerate(a))
        return Task(a) if wrapped.get(0) else a

    for vals in _for(*args, iter_args=iter_args, insert_yield=insert_yield, **kwargs):
        if not wrapped:
            yield vals
        elif isinstance(vals, tuple) and len(vals) == 3:
            iv, a, results = vals
            yield iv, rewrap_args(a), rewrap_args(results)
        else:
            # iv-only (no iter_args) never has wrapped positions.
            yield vals


def yield_(values):
    """``scf.yield`` that accepts [`Task`][iron.runtime.dmataskhandle.Task] entries,
    yielding each Task's SSA handle."""
    _yield_([_unwrap(v) for v in values])
