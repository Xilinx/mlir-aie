# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
from typing import Any, Iterator

from aie.extras.dialects.arith import constant  # pyright: ignore[reportMissingImports]
from aie.helpers.dialects.scf import (
    _for,
)
from aie.helpers.dialects.scf import (
    else_ as _else_,
)
from aie.helpers.dialects.scf import (
    if_ as _if_,
)
from aie.helpers.dialects.scf import (
    yield_ as _yield_,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.iron.runtime.dmataskhandle import Task
from aie.iron.runtime.taskgroup import TaskGroup

# Specs of the TaskGroups carried by the loops currently being emitted,
# innermost last, so yield_ can check a yielded group against them.
_carried_specs: list[list[tuple[bool, ...] | None]] = []


def _unwrap(x):
    """Unwrap a Task to its SSA handle; pass everything else through unchanged.

    A Task carries its SSA handle across scf boundaries; plain Values are
    returned as-is.
    """
    return x.handle if isinstance(x, Task) else x


def range_(*args, iter_args=None, insert_yield=True, **kwargs) -> Iterator[Any]:
    """``scf.for`` for IRON bodies, with ``Task`` and ``TaskGroup`` support in ``iter_args``.

    See [`Task`][iron.runtime.dmataskhandle.Task] and
    [`TaskGroup`][iron.runtime.taskgroup.TaskGroup].
    Identical to the low-level ``_for`` helper, except a ``Task`` passed as an
    ``iter_args`` entry is carried across iterations by its SSA handle: the loop
    body and the loop results receive it re-wrapped as a ``Task`` (so ``.free()``/
    ``.await_()`` work), and [`yield_`][iron.controlflow.yield_] accepts ``Task``
    entries too. A ``TaskGroup`` entry is carried as the handles of its
    transfers and comes back as a group ``finish()`` closes, which is what a
    software-pipelined DMA loop needs. Every group yielded back must have the
    same shape (transfer count and waited flags) as the one carried in.
    """
    # Each user-level iter_arg becomes one or more raw SSA iter_args. packers
    # records, per user entry, how to rebuild it: ("value", 1), ("task", 1) or
    # ("group", n, waited-flags).
    packers = []
    raw = []
    specs = []
    if iter_args is not None:
        for a in iter_args:
            if isinstance(a, TaskGroup):
                handles, waited = a._carry_out()
                packers.append(("group", len(handles), waited))
                raw.extend(handles)
                specs.append(waited)
            elif isinstance(a, Task):
                packers.append(("task", 1, None))
                raw.append(a.handle)
                specs.append(None)
            else:
                packers.append(("value", 1, None))
                raw.append(a)
                specs.append(None)
        iter_args = raw

    def rewrap(values):
        # values: the raw block args / results, as a tuple of Values.
        out = []
        pos = 0
        for kind, n, waited in packers:
            chunk = values[pos : pos + n]
            pos += n
            if kind == "group":
                out.append(TaskGroup._carry_in(list(chunk), waited))
            elif kind == "task":
                out.append(Task(chunk[0]))
            else:
                out.append(chunk[0])
        return out[0] if len(out) == 1 else tuple(out)

    if not packers:
        yield from _for(*args, iter_args=iter_args, insert_yield=insert_yield, **kwargs)
        return

    _carried_specs.append(specs)
    try:
        for vals in _for(
            *args, iter_args=iter_args, insert_yield=insert_yield, **kwargs
        ):
            iv, a, results = vals
            if len(raw) == 1:
                a, results = (a,), (results,)
            yield iv, rewrap(tuple(a)), rewrap(tuple(results))
    finally:
        _carried_specs.pop()


def yield_(values):
    """``scf.yield`` that accepts ``Task`` and ``TaskGroup`` entries.

    A ``Task`` yields its SSA handle; a ``TaskGroup`` yields the handles of its
    transfers (and is spent), checked against the shape of the group the
    enclosing ``range_`` carried in. See [`Task`][iron.runtime.dmataskhandle.Task].
    """
    specs = _carried_specs[-1] if _carried_specs else None
    raw = []
    for i, v in enumerate(values):
        if isinstance(v, TaskGroup):
            expected = specs[i] if specs is not None and i < len(specs) else None
            if expected is not None and v.spec != expected:
                raise ValueError(
                    f"yielded {v} has transfers waited {list(v.spec)} but the "
                    f"loop carries a group waited {list(expected)}; every "
                    "iteration must issue the same transfers in the same order"
                )
            handles, _ = v._carry_out()
            raw.extend(handles)
        else:
            raw.append(_unwrap(v))
    _yield_(raw)


@contextmanager
def if_(cond, has_else: bool = False):
    """Open an ``scf.if`` region in an IRON body as a ``with`` block.

    ``cond`` is an ``i1`` value (a comparison on a staged scalar) or a plain
    ``bool``. With ``has_else=True`` the op gets an else region, filled with
    [`else_`][iron.controlflow.else_]:

    ```python
    with if_(n_ragged > 0):
        ...
    with if_(n_ragged > 0, has_else=True) as branch:
        ...
    with else_(branch):
        ...
    ```
    """
    if isinstance(cond, bool):
        cond = constant(cond)
    with _if_(cond, hasElse=has_else) as op:
        yield op


@contextmanager
def else_(branch):
    """Open the else region of a ``with if_(cond, has_else=True) as branch`` block."""
    with _else_(branch):
        yield
