# dmataskhandle.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Task: a handle to an in-flight shim DMA transfer.

Returned by ``fifo.fill``/``fifo.drain``. Its ``handle`` is the transfer's
``!index`` SSA value, so a ``Task`` can be carried across ``scf.for`` iterations
as a ``range_`` ``iter_args`` entry for software-pipelined transfers -- and it
carries ``.start()`` / ``.free()`` / ``.await_()`` verbs so the loop body does not
need to reach for the raw ``aiex.dma_start_task`` / ``aiex.dma_free_task`` /
``aiex.dma_await_task`` dialect ops.

A ``Task`` returned by an *unmanaged* transfer (``managed=False``) is not enrolled
in a ``TaskGroup``'s automatic await/free, so the caller owns its lifetime with
these verbs -- exactly what a hand-rolled ping-pong needs.
"""

from __future__ import annotations

import copy

from ...dialects.aiex import (  # pyright: ignore[reportMissingImports]
    dma_await_task,
    dma_free_task,
    dma_start_task,
)


class Task:
    """A handle to a submitted shim DMA transfer.

    Wraps the transfer's ``!index`` SSA value (``handle``). Pass a ``Task`` as a
    ``range_`` ``iter_args`` entry to carry an in-flight transfer across loop
    iterations; ``range_`` unwraps it to its ``handle`` for the ``scf.for``
    ``iter_arg`` and re-wraps the block argument as a ``Task`` for the body.
    """

    def __init__(self, handle):
        self._handle = handle
        self._freed = False

    @property
    def handle(self):
        """The transfer's ``!index`` SSA value (the ``scf`` iter_arg payload)."""
        return self._handle

    def _with_handle(self, handle) -> "Task":
        """Return this task carried to another SSA value, e.g. a loop's iter_arg."""
        task = copy.copy(self)
        task._handle = handle
        return task

    def start(self, repeat_count: int | None = None) -> "Task":
        """Push this task onto its channel's queue (``dma_start_task``).

        Its buffer descriptors are written when the task is configured, so each
        start costs one queue push. ``repeat_count`` replaces the task's
        configured count for this start only.

        A task carried through a ``range_`` ``iter_args`` entry can be started
        only if every value it can carry comes from the same configure (the
        loop yields it unchanged), since the push names one head buffer
        descriptor. A task reconfigured inside the loop is started where it is
        configured, before it is yielded; compilation otherwise fails.

        Returns:
            This task.

        Raises:
            RuntimeError: If this task was already freed, since its buffer
                descriptors may since describe another transfer.
        """
        if self._freed:
            raise RuntimeError(
                "Task.start() after Task.free(): the freed buffer descriptors "
                "may already describe another transfer. Free a task after its "
                "last start."
            )
        dma_start_task(self._handle, repeat_count=repeat_count)
        return self

    def free(self) -> None:
        """Return this transfer's buffer descriptor to the pool (``dma_free_task``).

        Raises:
            RuntimeError: If this task was already freed.
        """
        if self._freed:
            raise RuntimeError("Task.free() called twice on the same task.")
        dma_free_task(self._handle)
        self._freed = True

    def await_(self) -> None:
        """Block until this transfer completes (``dma_await_task``).

        The transfer must have been issued with ``wait=True`` so it carries a
        completion token.
        """
        dma_await_task(self._handle)
