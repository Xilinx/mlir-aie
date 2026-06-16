# taskgroup.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""TaskGroup: a Resolvable grouping of runtime DMA transfers.

A ``TaskGroup`` collects the data-movement tasks issued between its creation and
its :meth:`resolve` call. Resolving it emits the deferred completion handling for
those tasks (await the ones marked ``wait=True``; free the rest) and reclaims
their BD ids.

Because a ``TaskGroup`` is a free-standing :class:`Resolvable` rather than a
scope, groups may *overlap* in time — the software-pipelining idiom where the
previous group is finished only after the next group's transfers are issued::

    prev = None
    for ... :
        tg = TaskGroup()
        inA.prod().fill(A, ..., group=tg)
        outC.cons().drain(C, ..., group=tg, wait=True)
        if prev is not None:
            prev.resolve()      # finish the previous group → overlap
        prev = tg
    if prev is not None:
        prev.resolve()

Transfers issued without an explicit ``group=`` join the sequence's implicit
default group, which the runner resolves at end-of-sequence.
"""

from ... import ir  # type: ignore

from ..resolvable import Resolvable
from ._context import active_sequence


class TaskGroup(Resolvable):
    """A Resolvable grouping of runtime DMA transfers issued in a sequence body."""

    def __init__(self):
        """Create a TaskGroup and register it with the active runtime sequence.

        Must be called inside the function passed to :meth:`Runtime.sequence`.
        """
        seq = active_sequence()
        self._group_id = seq.next_task_group_id()
        self._tasks = []
        self._resolved = False
        seq.register_task_group(self)

    @property
    def group_id(self) -> int:
        """The id of the task group (unique within a sequence)."""
        return self._group_id

    @property
    def resolved(self) -> bool:
        """Whether this group has been resolved (its completions emitted)."""
        return self._resolved

    def _add(self, task) -> None:
        """Attach a DMA task to this group (called by fill/drain)."""
        if self._resolved:
            raise ValueError(
                f"Cannot add a transfer to {self}: it has already been resolved."
            )
        self._tasks.append(task)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Emit completion handling for this group's transfers.

        Awaits tasks marked ``wait=True`` and frees the rest, reclaiming each
        task's BD id. Idempotent.
        """
        if self._resolved:
            return
        self._resolved = True
        seq = active_sequence()
        # Await first, then free — matching the completion ordering the runtime
        # has always emitted (waited tasks produce the tokens that gate reuse).
        wait_tasks = [t for t in self._tasks if t.will_wait()]
        free_tasks = [t for t in self._tasks if not t.will_wait()]
        for task in wait_tasks:
            task.emit_wait()
            seq.reclaim_bd(task)
        for task in free_tasks:
            task.emit_free()
            seq.reclaim_bd(task)

    def __str__(self):
        return f"TaskGroup({self._group_id})"
