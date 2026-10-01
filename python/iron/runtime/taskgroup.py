# taskgroup.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""TaskGroup: groups related runtime transfers so they are awaited/freed together."""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from ...dialects import arith
from ...dialects.aiex import dma_await_task, dma_free_task
from ...extras.dialects.arith import constant
from ...ir import Block, IndexType, InsertionPoint, Value


class TaskGroup:
    """A grouping of runtime data transfers awaited and freed together.

    Construct one inside a runtime sequence body and pass it as the ``group=``
    argument to ``fifo.fill(...)`` / ``fifo.drain(...)``. Call
    [`finish`][iron.runtime.taskgroup.TaskGroup.finish] to await the group's
    waited transfers and free the rest (waits are ordered before frees).

    ```python
    def seq(A, C):
        tg = TaskGroup()
        inA.prod().fill(A, group=tg)
        outC.cons().drain(C, wait=True, group=tg)
        tg.finish()
    ```

    A software pipeline keeps a step's transfers in flight while it issues
    the next step's: [`pipelined`][iron.runtime.taskgroup.TaskGroup.pipelined]
    hands out one group per step and finishes each a step later, with the
    same body whether the step count is a Python ``int`` or a dispatch-time
    value.
    """

    def __init__(self, id: int | None = None):
        """Construct a TaskGroup, registering it with the active runtime sequence.

        Args:
            id (int | None): Group id, unique within a Runtime. Defaults to the
                active sequence's next id. Passing an explicit id is only needed
                for the runtime's internal default group.
        """
        # Actions accumulated for this group: (dma_await_task | dma_free_task, [task]).
        self._actions: list = []
        # Lazy import to avoid a cycle (runtime -> taskgroup -> _context).
        from ._context import _active_sequence

        active = _active_sequence.get()
        if id is None:
            if active is None:
                raise RuntimeError(
                    "TaskGroup() must be constructed within the function passed "
                    "to Runtime(seq_fn, fn_args)."
                )
            id = next(active._runtime._task_group_index)
        self._group_id = id
        # Set once the group's transfers have been handed to a loop (as an
        # iter_arg or a yield); the object is then spent.
        self._carried = False
        if active is not None and id is not None:
            active.register_task_group(self)

    @property
    def group_id(self) -> int:
        """The id of the task group."""
        return self._group_id

    def finish(self) -> None:
        """Await this group's waited transfers, then free the rest."""
        from ._context import active_sequence

        if self._carried:
            raise RuntimeError(
                f"{self} was carried into a loop; finish the group the loop "
                "hands back (its body argument or result), not this one"
            )
        here = InsertionPoint.current.block
        for task, _ in self._transfers():
            self._check_reaches(_block_of(task), here)
        active_sequence().finish_task_group(self)

    def _check_reaches(self, issued: Block, here: Block) -> None:
        """Check a transfer issued in ``issued`` can be finished once in ``here``."""
        block = here
        while block != issued:
            parent = block.owner
            if parent is None or parent.operation.name == "aie.runtime_sequence":
                raise RuntimeError(
                    f"{self} has a transfer issued inside an if_/range_ body "
                    "that this finish() is outside of; finish it in that body, "
                    "or carry it out of a range_ as an iter_arg"
                )
            if parent.operation.name == "scf.for":
                raise RuntimeError(
                    f"{self} has a transfer issued before the range_ this "
                    "finish() is in, so it would be finished on every "
                    "iteration; finish it after the loop, or carry it in as "
                    "an iter_arg"
                )
            block = parent.operation.block

    @classmethod
    def pipelined(cls, n, depth: int = 2) -> Iterator[tuple[Any, TaskGroup]]:
        """Run ``n`` steps, each in a fresh group, keeping ``depth`` groups in flight.

        Each iteration yields ``(step, group)``. Put the step's transfers in
        ``group``; it is finished once the next ``depth - 1`` steps have been
        issued, and the last groups after the last step. ``depth=1`` finishes
        each step before the next starts.

        ```python
        for step, tg in TaskGroup.pipelined(n_steps):
            in_h.fill(a, tap=in_tiles[step], group=tg)
            out_h.drain(c, tap=out_tiles[step], group=tg, wait=True)
        ```

        With a Python ``int`` the steps unroll. With a staged ``n`` the
        sequence stays rolled: the first ``depth - 1`` steps run under
        ``if_(n > step)`` with ``step`` an ``int``, and the rest in one
        ``range_`` that carries the groups in flight, with ``step`` its
        index. So the body is traced ``depth`` times, and every step must
        issue the same transfers, waited the same way, in the same order.
        """
        if depth < 1:
            raise ValueError(f"pipelined depth must be at least 1, got {depth}")
        if isinstance(n, (int, np.integer)):
            in_flight: list[TaskGroup] = []
            for step in range(int(n)):
                tg = cls()
                yield step, tg
                in_flight.append(tg)
                if len(in_flight) == depth:
                    in_flight.pop(0).finish()
            for tg in in_flight:
                tg.finish()
            return
        if not isinstance(n.type, IndexType):
            n = arith.index_cast(IndexType.get(), n)
        yield from cls._pipelined(n, depth, [])

    @classmethod
    def _pipelined(cls, n, depth, in_flight):
        from ..controlflow import else_, if_, range_, yield_

        step = len(in_flight)
        if step == depth - 1:
            if not in_flight:
                for iv in range_(n):
                    tg = cls()
                    yield iv, tg
                    tg.finish()
                return
            for iv, carried, last in range_(step, n, iter_args=in_flight):
                tg = cls()
                yield iv, tg
                carried[0].finish()
                yield_([*carried[1:], tg])
            for tg in last:
                tg.finish()
            return
        # Both arms of the if_ see the groups in flight, so each rebuilds them.
        carried = [tg._carry_out() for tg in in_flight]
        more = arith.cmpi(arith.CmpIPredicate.sgt, n, constant(step, index=True))
        with if_(more) as branch:
            tg = cls()
            yield step, tg
            rebuilt = [cls._carry_in(*c) for c in carried]
            yield from cls._pipelined(n, depth, [*rebuilt, tg])
        if carried:
            with else_(branch):
                for c in carried:
                    cls._carry_in(*c).finish()

    # ------------------------------------------------ loop carrying (range_)

    def _carry_out(self) -> tuple[list, tuple[bool, ...]]:
        """Hand the group's transfers to a loop: (SSA handles, waited flags).

        Marks this object spent and drops it from the sequence's open groups;
        the loop re-creates a group over the handles on the other side.
        """
        from ._context import active_sequence

        if self._carried:
            raise RuntimeError(f"{self} has already been carried into a loop")
        transfers = self._transfers()
        # Actions hold the configure op (or a carried Value); the loop carries
        # the op's !index result.
        handles = [getattr(task, "result", task) for task, _ in transfers]
        self._carried = True
        self._actions = []
        active = active_sequence()
        if self in active._open_task_groups:
            active._open_task_groups.remove(self)
        return handles, tuple(w for _, w in transfers)

    @classmethod
    def _carry_in(cls, handles, waited: tuple[bool, ...]) -> "TaskGroup":
        """Rebuild a group over loop-carried handles (a body argument or result)."""
        if len(handles) != len(waited):
            raise ValueError("carried task group: handle count does not match its spec")
        tg = cls()
        for handle, w in zip(handles, waited):
            if w:
                tg._actions.append((dma_await_task, [handle]))
            tg._actions.append((dma_free_task, [handle]))
        return tg

    def _waited(self) -> tuple[bool, ...]:
        """Which of the group's transfers (in issue order) are waited."""
        return tuple(w for _, w in self._transfers())

    def _transfers(self) -> list[tuple[Any, bool]]:
        """Each transfer once, in issue order, with whether it is waited."""
        # Keyed by object identity: a Value's == emits an arith.cmpi rather
        # than comparing identities. An await and a free of one transfer
        # share the task object.
        seen: dict[int, list] = {}
        for fn, (task,) in self._actions:
            entry = seen.setdefault(id(task), [task, False])
            if fn == dma_await_task:
                entry[1] = True
        return [(task, waited) for task, waited in seen.values()]

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __str__(self):
        return f"TaskGroup({self.group_id})"


def _block_of(task) -> Block:
    """The block a transfer was issued in (its configure op, or a carried Value)."""
    owner = task.owner if isinstance(task, Value) else task
    if isinstance(owner, Block):
        return owner
    return owner.operation.block
