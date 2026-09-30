# taskgroup.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""TaskGroup: groups related runtime transfers so they are awaited/freed together."""

from __future__ import annotations


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

    A group can also ride a rolled loop: pass it as a ``range_`` ``iter_args``
    entry and ``yield_`` the group built in the body, and the next iteration
    (and the loop result) receives a group over the carried transfers that
    ``finish()`` closes as usual. This is how a software pipeline keeps one
    step in flight while issuing the next with a dispatch-time trip count:

    ```python
    prev = issue(0)                       # a TaskGroup
    for iv, prev, last in range_(1, n, iter_args=[prev], insert_yield=False):
        current = issue(iv)
        prev.finish()                     # the step issued one iteration ago
        yield_([current])
    last.finish()
    ```

    Every group yielded must hold the same number of transfers, with the same
    ones waited, as the group the loop started with (the carried SSA handles
    are positional). Once a group is carried, the Python object that was
    passed in is spent: finish the one the loop hands back instead.
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
        active_sequence().finish_task_group(self)

    # ------------------------------------------------ loop carrying (range_)

    def _carry_out(self) -> tuple[list, tuple[bool, ...]]:
        """Hand the group's transfers to a loop: (SSA handles, waited flags).

        Marks this object spent and drops it from the sequence's open groups;
        the loop re-creates a group over the handles on the other side.
        """
        from ...dialects.aiex import dma_await_task
        from ._context import active_sequence

        if self._carried:
            raise RuntimeError(f"{self} has already been carried into a loop")
        # Keyed by object identity: a Value's == emits an arith.cmpi rather
        # than comparing identities. An await and a free of one transfer
        # share the task object.
        seen: dict[int, int] = {}
        handles: list = []
        waited: list[bool] = []
        for fn, (task,) in self._actions:
            if id(task) not in seen:
                seen[id(task)] = len(handles)
                # Actions hold the configure op (or a carried Value); the loop
                # carries the op's !index result.
                handles.append(getattr(task, "result", task))
                waited.append(False)
            if fn == dma_await_task:
                waited[seen[id(task)]] = True
        self._carried = True
        self._actions = []
        active = active_sequence()
        if self in active._open_task_groups:
            active._open_task_groups.remove(self)
        return handles, tuple(waited)

    @classmethod
    def _carry_in(cls, handles, waited: tuple[bool, ...]) -> "TaskGroup":
        """Rebuild a group over loop-carried handles (a body argument or result)."""
        from ...dialects.aiex import dma_await_task, dma_free_task

        if len(handles) != len(waited):
            raise ValueError("carried task group: handle count does not match its spec")
        tg = cls()
        for handle, w in zip(handles, waited):
            if w:
                tg._actions.append((dma_await_task, [handle]))
            tg._actions.append((dma_free_task, [handle]))
        return tg

    @property
    def spec(self) -> tuple[bool, ...]:
        """Which of the group's transfers (in issue order) are waited."""
        from ...dialects.aiex import dma_await_task

        waited: dict[int, bool] = {}
        for fn, (task,) in self._actions:
            waited.setdefault(id(task), False)
            if fn == dma_await_task:
                waited[id(task)] = True
        return tuple(waited.values())

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __str__(self):
        return f"TaskGroup({self.group_id})"
