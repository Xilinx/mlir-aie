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


class _LoopFrame:
    """One active ``range_`` loop.

    Holds the loop body block, the waited flags of each TaskGroup it carries
    (``None`` for other entries, and for a loop without iter_args), and the
    values its terminating ``yield_`` passed, if any.
    """

    def __init__(self, body, specs):
        self.body = body
        self.specs = specs
        self.values: list | None = None


# The loops currently being emitted, innermost last. yield_ checks a yielded
# group against the innermost one's specs, and records its values only when it
# terminates that loop's body itself, not a nested scf.if or loop. A ContextVar
# (like the active runtime sequence) so concurrent threads or async tasks
# generating designs each see only their own loops.
_loop_frames: ContextVar[tuple[_LoopFrame, ...]] = ContextVar(
    "iron_loop_frames", default=()
)


@contextmanager
def _loop_frame(body, specs):
    frame = _LoopFrame(body, specs)
    outer = _loop_frames.get()
    _loop_frames.set(outer + (frame,))
    try:
        yield frame
    finally:
        _loop_frames.set(outer)


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
    back as a copy of the ``Task`` passed in (so ``.start()``/``.free()``/
    ``.await_()`` work); a loop result takes the lifetime of the ``Task`` the
    body's own ``yield_`` passed. A ``TaskGroup`` entry is carried as the
    handles of its transfers and comes back as a group ``finish()`` closes.
    Every group yielded back must have the same shape (transfer count and
    waited flags) as the one carried in.
    [`TaskGroup.pipelined`][iron.runtime.taskgroup.TaskGroup.pipelined] builds
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
    for vals in _for(*args, iter_args=raw, insert_yield=not packers, **kwargs):
        if not packers:
            with _loop_frame(vals.owner, specs):
                yield vals
            continue
        if not raw:
            iv, a, results = vals, (), ()
        else:
            iv, a, results = vals
            if len(raw) == 1:
                a, results = (a,), (results,)
        results = rewrap(tuple(results))
        with _loop_frame(iv.owner, specs) as frame:
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
            if kind == "task" and frame.values and isinstance(frame.values[i], Task):
                results[i]._carry(frame.values[i])


def yield_(values):
    """``scf.yield`` that accepts ``Task`` and ``TaskGroup`` entries.

    A ``Task`` yields its SSA handle; a ``TaskGroup`` yields the handles of its
    transfers (and is spent), checked against the shape of the group the
    enclosing ``range_`` carried in. See [`Task`][iron.runtime.dmataskhandle.Task].
    """
    values = list(values)
    frames = _loop_frames.get()
    specs = frames[-1].specs if frames else None
    if frames and InsertionPoint.current.block == frames[-1].body:
        frames[-1].values = values
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
