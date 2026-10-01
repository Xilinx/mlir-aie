# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
from contextvars import ContextVar
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
from aie.ir import InsertionPoint  # pyright: ignore[reportMissingImports]
from aie.iron.runtime.dmataskhandle import Task
from aie.iron.runtime.taskgroup import TaskGroup

# Specs of the TaskGroups carried by the loops currently being emitted,
# innermost last, so yield_ can check a yielded group against them. A
# ContextVar (like the active runtime sequence) so concurrent threads or async
# tasks generating designs each see only their own loops.
_Specs = list[tuple[bool, ...] | None] | None
_carried_specs: ContextVar[tuple[_Specs, ...]] = ContextVar(
    "iron_carried_specs", default=()
)


def _push_specs(specs: _Specs) -> None:
    _carried_specs.set(_carried_specs.get() + (specs,))


def _pop_specs() -> None:
    _carried_specs.set(_carried_specs.get()[:-1])


def _unwrap(x):
    """Unwrap a Task to its SSA handle; pass everything else through unchanged.

    A Task carries its SSA handle across scf boundaries; plain Values are
    returned as-is.
    """
    return x.handle if isinstance(x, Task) else x


def range_(*args, iter_args=None, **kwargs) -> Iterator[Any]:
    """``scf.for`` for IRON bodies, with ``Task`` and ``TaskGroup`` support in ``iter_args``.

    See [`Task`][iron.runtime.dmataskhandle.Task] and
    [`TaskGroup`][iron.runtime.taskgroup.TaskGroup].
    Without ``iter_args`` each iteration yields the induction variable. With
    them it yields ``(iv, args, results)``: the carried values as the body
    sees them and as the loop returns them, each a tuple with one entry per
    ``iter_args`` entry. The body must end with
    [`yield_`][iron.controlflow.yield_] of the next iteration's values.

    A ``Task`` entry is carried across iterations by its SSA handle and comes
    back re-wrapped as a ``Task`` (so ``.free()``/``.await_()`` work). A
    ``TaskGroup`` entry is carried as the handles of its transfers and comes
    back as a group ``finish()`` closes. Every group yielded back must have the
    same shape (transfer count and waited flags) as the one carried in.
    [`TaskGroup.pipelined`][iron.runtime.taskgroup.TaskGroup.pipelined] builds
    the usual software pipeline out of this.
    """
    # Each user-level iter_arg becomes one or more raw SSA iter_args. packers
    # records, per user entry, how to rebuild it: ("value", 1), ("task", 1) or
    # ("group", n, waited-flags).
    packers = []
    raw = []
    if iter_args is not None:
        for a in iter_args:
            if isinstance(a, TaskGroup):
                handles, waited = a._carry_out()
                packers.append(("group", len(handles), waited))
                raw.extend(handles)
            elif isinstance(a, Task):
                packers.append(("task", 1, None))
                raw.append(a.handle)
            else:
                packers.append(("value", 1, None))
                raw.append(a)

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
        return tuple(out)

    # A loop without iter_args still shadows any enclosing loop's specs, so a
    # yield_ in its body is not checked against them.
    _push_specs([waited for _, _, waited in packers] if packers else None)
    try:
        for vals in _for(*args, iter_args=raw, insert_yield=not packers, **kwargs):
            if not packers:
                yield vals
                continue
            if not raw:
                iv, a, results = vals, (), ()
            else:
                iv, a, results = vals
                if len(raw) == 1:
                    a, results = (a,), (results,)
            yield iv, rewrap(tuple(a)), rewrap(tuple(results))
            body = iv.owner
            ops = body.operations
            if not len(ops) or ops[len(ops) - 1].name != "scf.yield":
                raise ValueError(
                    "a range_ body with iter_args must end with yield_([...]), "
                    f"one entry per iter_arg ({len(packers)} here)"
                )
    finally:
        _pop_specs()


def yield_(values):
    """``scf.yield`` that accepts ``Task`` and ``TaskGroup`` entries.

    A ``Task`` yields its SSA handle; a ``TaskGroup`` yields the handles of its
    transfers (and is spent), checked against the shape of the group the
    enclosing ``range_`` carried in. See [`Task`][iron.runtime.dmataskhandle.Task].
    """
    stack = _carried_specs.get()
    specs = stack[-1] if stack else None
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
            if expected is not None and v._waited() != expected:
                raise ValueError(
                    f"yielded {v} has transfers waited {list(v._waited())} but the "
                    f"loop carries a group waited {list(expected)}; every "
                    "iteration must issue the same transfers in the same order"
                )
            handles, _ = v._carry_out()
            raw.extend(handles)
        else:
            raw.append(_unwrap(v))
    _yield_(raw)


@contextmanager
def if_(cond):
    """Open an ``scf.if`` region in an IRON body as a ``with`` block.

    ``cond`` is an ``i1`` value (a comparison on a staged scalar) or a plain
    ``bool``. The block's ``as`` target opens the else region with
    [`else_`][iron.controlflow.else_]:

    ```python
    with if_(n_ragged > 0) as branch:
        ...
    with else_(branch):
        ...
    ```
    """
    if isinstance(cond, bool):
        cond = constant(cond)
    with _if_(cond, hasElse=False) as op:
        yield op


@contextmanager
def else_(branch):
    """Open the else region of a ``with if_(cond) as branch`` block."""
    blocks = branch.elseRegion.blocks
    if not len(blocks):
        with InsertionPoint(blocks.append()):
            _yield_([])
    with _else_(branch):
        yield
