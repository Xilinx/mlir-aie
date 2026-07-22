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
    """

    def __init__(self, id: int | None = None):
        """Construct a TaskGroup, registering it with the active runtime sequence.

        Args:
            id (int | None): Group id, unique within a Runtime. Defaults to the
                active sequence's next id. Passing an explicit id is only needed
                for the runtime's internal default group.
        """
        # Lazy import to avoid a cycle (runtime -> taskgroup -> _context).
        from ._context import _active_sequence

        active = _active_sequence.get()
        if id is None:
            if active is None:
                raise RuntimeError(
                    "TaskGroup() must be constructed within the function passed "
                    "to Runtime.sequence()."
                )
            id = next(active._runtime._task_group_index)
        self._group_id = id
        if active is not None and id is not None:
            active.register_task_group(self)

    @property
    def group_id(self) -> int:
        """The id of the task group."""
        return self._group_id

    def finish(self) -> None:
        """Await this group's waited transfers, then free the rest."""
        from ._context import active_sequence

        active_sequence().finish_task_group(self)

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __str__(self):
        return f"TaskGroup({self.group_id})"
