# taskgroup.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""TaskGroup: a grouping of runtime DMA transfers.

A ``TaskGroup`` collects the data-movement tasks issued between its creation and
its :meth:`finish` call. Finishing it emits the deferred completion handling for
those tasks (await the ones marked ``wait=True``; free the rest) and reclaims
their BD ids.

Because a ``TaskGroup`` is a free-standing object rather than a scope, groups may
*overlap* in time — the software-pipelining idiom where the previous group is
finished only after the next group's transfers are issued::

    prev = None
    for ... :
        tg = TaskGroup()
        inA.prod().fill(A, ..., group=tg)
        outC.cons().drain(C, ..., group=tg, wait=True)
        if prev is not None:
            prev.finish()       # finish the previous group → overlap
        prev = tg
    if prev is not None:
        prev.finish()

This interleaving is why ``finish`` is a method rather than a ``with`` context
manager (which could only express nested, not overlapping, lifetimes).

Transfers issued without an explicit ``group=`` join the sequence's implicit
default group, which the runner finishes at end-of-sequence.
"""

from ._context import active_sequence


class TaskGroup:
    """A grouping of runtime DMA transfers issued in a sequence body.

    Call :meth:`finish` to close the group: it awaits the group's ``wait=True``
    transfers, frees the rest, and reclaims their BD ids. Groups have explicit,
    possibly *interleaved* lifetimes — a double-buffering body opens the next
    group before finishing the previous one — so this is a plain method, not a
    ``with`` context manager (which could only express nested lifetimes).
    """

    def __init__(self):
        """Create a TaskGroup and register it with the active runtime sequence.

        Must be called inside the function passed to :class:`Program`'s
        ``sequence``.
        """
        seq = active_sequence()
        self._group_id = seq.next_task_group_id()
        self._tasks = []
        self._finished = False
        seq.register_task_group(self)

    @property
    def group_id(self) -> int:
        """The id of the task group (unique within a sequence)."""
        return self._group_id

    @property
    def finished(self) -> bool:
        """Whether this group has been finished (its completions emitted)."""
        return self._finished

    def _add(self, task) -> None:
        """Attach a DMA task to this group (called by fill/drain)."""
        if self._finished:
            raise ValueError(
                f"Cannot add a transfer to {self}: it has already been finished."
            )
        self._tasks.append(task)

    def finish(self) -> None:
        """Emit completion handling for this group's transfers.

        Awaits tasks marked ``wait=True`` and frees the rest, reclaiming each
        task's BD id. Idempotent. May be called in any order relative to other
        groups (e.g. finish the previous group after opening the next).
        """
        if self._finished:
            return
        self._finished = True
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
