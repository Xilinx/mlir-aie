# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
from typing import Any, Iterator

from aie.extras.dialects.arith import constant  # pyright: ignore[reportMissingImports]
from aie.helpers.dialects.scf import (
    _for,
    _for_bounds,
)
from aie.helpers.dialects.scf import (
    else_ as _else_,
)
from aie.helpers.dialects.scf import (
    forever_ as _forever_,
)
from aie.helpers.dialects.scf import (
    if_ as _if_,
)
from aie.helpers.dialects.scf import (
    yield_ as _yield_,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.ir import (  # pyright: ignore[reportMissingImports]
    InsertionPoint,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.iron.runtime._context import frame, frames
from aie.iron.runtime.dmataskhandle import Task
from aie.iron.runtime.taskgroup import TaskGroup


def range_(*args, iter_args=None, **kwargs) -> Iterator[Any]:
    """``scf.for`` for IRON bodies, with ``Task`` and ``TaskGroup`` support in ``iter_args``.

    See [`Task`][aie.iron.runtime.dmataskhandle.Task] and
    [`TaskGroup`][aie.iron.runtime.taskgroup.TaskGroup].
    Without ``iter_args`` each iteration yields the induction variable. With
    them it yields ``(iv, args, results)``: the carried values as the body
    sees them and as the loop returns them, each a tuple with one entry per
    ``iter_args`` entry. The body must end with
    [`yield_`][aie.iron.controlflow.yield_] of the next iteration's values.

    A ``Task`` entry is carried across iterations by its SSA handle and comes
    back as a copy of the ``Task`` passed in (so ``.start()``/``.free()``/
    ``.await_()`` work); a loop result takes the lifetime of the ``Task`` the
    body's own ``yield_`` passed. A ``TaskGroup`` entry is carried as the
    handles of its transfers and comes back as a group ``finish()`` closes.
    Every group yielded back must have the same shape (transfer count and
    waited flags) as the one carried in.
    [`TaskGroup.pipelined`][aie.iron.runtime.taskgroup.TaskGroup.pipelined] builds
    the usual software pipeline out of this.
    """
    # Each user-level iter_arg becomes one or more raw SSA iter_args. packers
    # records, per user entry, how to rebuild it: ("value", 1, None),
    # ("task", 1, the Task passed in) or ("group", n, waited-flags).
    packers = []
    raw = []
    if iter_args is not None:
        for a in iter_args:
            if isinstance(a, TaskGroup):
                handles, waited = a._carry_out()
                packers.append(("group", len(handles), waited))
                raw.extend(handles)
            elif isinstance(a, Task):
                packers.append(("task", 1, a))
                raw.append(a.handle)
            else:
                packers.append(("value", 1, None))
                raw.append(a)

    def rewrap(values):
        # values: the raw block args / results, as a tuple of Values.
        out = []
        pos = 0
        for kind, n, extra in packers:
            chunk = values[pos : pos + n]
            pos += n
            if kind == "group":
                out.append(TaskGroup._carry_in(list(chunk), extra))
            elif kind == "task":
                out.append(extra._with_handle(chunk[0]))
            else:
                out.append(chunk[0])
        return tuple(out)

    # A loop without iter_args still shadows any enclosing loop's specs, so a
    # yield_ in its body is not checked against them.
    specs = (
        [extra if kind == "group" else None for kind, _, extra in packers]
        if packers
        else None
    )
    bounds = _for_bounds(args, kwargs)
    for vals in _for(*args, iter_args=raw, insert_yield=not packers, **kwargs):
        if not packers:
            with frame("for", bounds, vals.owner, specs):
                yield vals
            continue
        if not raw:
            iv, a, results = vals, (), ()
        else:
            iv, a, results = vals
            if len(raw) == 1:
                a, results = (a,), (results,)
        results = rewrap(tuple(results))
        with frame("for", bounds, iv.owner, specs) as loop:
            yield iv, rewrap(tuple(a)), results
        ops = iv.owner.operations
        if not len(ops) or ops[len(ops) - 1].name != "scf.yield":
            raise ValueError(
                "a range_ body with iter_args must end with yield_([...]), "
                f"one entry per iter_arg ({len(packers)} here)"
            )
        # A loop result is the Task the body yielded, so it takes that Task's
        # lifetime, not the initial one's.
        for i, (kind, _, _) in enumerate(packers):
            if kind == "task" and loop.values and isinstance(loop.values[i], Task):
                results[i]._carry(loop.values[i])


def forever_() -> Iterator[None]:
    """``scf.while(true)`` for a core body that runs until reconfiguration.

    The body of a Python ``for`` over it is emitted once, into a loop whose
    condition is always true, so the loop keeps no counter. A
    [`Worker`][aie.iron.Worker] already wraps its body in one unless
    ``while_true=False``; write it yourself when a body must do some work once
    before it loops forever.
    """
    for _ in _forever_():
        with frame("forever"):
            yield


def yield_(values):
    """``scf.yield`` that accepts ``Task`` and ``TaskGroup`` entries.

    A ``Task`` yields its SSA handle; a ``TaskGroup`` yields the handles of its
    transfers (and is spent), checked against the shape of the group the
    enclosing ``range_`` carried in. See [`Task`][aie.iron.runtime.dmataskhandle.Task].
    """
    values = list(values)
    loops = [f for f in frames() if f.kind == "for"]
    specs = loops[-1].specs if loops else None
    if loops and InsertionPoint.current.block == loops[-1].body:
        loops[-1].values = values
    if specs is not None and len(values) != len(specs):
        raise ValueError(
            f"yield_ got {len(values)} values but the loop carries {len(specs)}"
        )
    raw = []
    for i, v in enumerate(values):
        expected = specs[i] if specs is not None else None
        is_group = isinstance(v, TaskGroup)
        if specs is not None and is_group != (expected is not None):
            if is_group:
                raise ValueError(
                    f"yielded {v} in slot {i}, but the loop does not carry a "
                    "TaskGroup there"
                )
            raise ValueError(
                f"slot {i} of the loop carries a TaskGroup, so yield_ must "
                f"yield a TaskGroup there, not {v!r}"
            )
        if is_group:
            handles, waited = v._carry_out()
            if expected is not None and waited != expected:
                raise ValueError(
                    f"yielded {v} has transfers waited {list(waited)} but the "
                    f"loop carries a group waited {list(expected)}; every "
                    "iteration must issue the same transfers in the same order"
                )
            raw.extend(handles)
        else:
            raw.append(v.handle if isinstance(v, Task) else v)
    _yield_(raw)


@contextmanager
def if_(cond):
    """Open an ``scf.if`` region in an IRON body as a ``with`` block.

    ``cond`` is an ``i1`` value (a comparison on a staged scalar) or a plain
    ``bool``. The block's ``as`` target opens the else region with
    [`else_`][aie.iron.controlflow.else_]:

    ```python
    with if_(n_ragged > 0) as branch:
        ...
    with else_(branch):
        ...
    ```
    """
    if isinstance(cond, bool):
        cond = constant(cond)
    with _if_(cond, hasElse=False) as op, frame("if", cond):
        yield op


@contextmanager
def else_(branch):
    """Open the else region of a ``with if_(cond) as branch`` block."""
    blocks = branch.elseRegion.blocks
    if not len(blocks):
        with InsertionPoint(blocks.append()):
            _yield_([])
    with _else_(branch), frame("else"):
        yield
