# dmataskhandle.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Task: a handle to an in-flight shim DMA transfer.

Returned by ``fifo.fill``/``fifo.drain``. Its ``handle`` is the transfer's
``!index`` SSA value, so a ``Task`` can be carried across ``scf.for`` iterations
as a ``range_`` ``iter_args`` entry for software-pipelined transfers -- and it
carries ``.free()`` / ``.await_()`` verbs so the loop body does not need to reach
for the raw ``aiex.dma_free_task`` / ``aiex.dma_await_task`` dialect ops.

A ``Task`` returned by an *unmanaged* transfer (``managed=False``) is not enrolled
in a ``TaskGroup``'s automatic await/free, so the caller owns its lifetime with
these verbs -- exactly what a hand-rolled ping-pong needs.
"""

from __future__ import annotations

from ...dialects._aiex_ops_gen import (  # pyright: ignore[reportMissingImports]
    dma_await_task,
    dma_free_task,
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

    @property
    def handle(self):
        """The transfer's ``!index`` SSA value (the ``scf`` iter_arg payload)."""
        return self._handle

    def free(self) -> None:
        """Return this transfer's buffer descriptor to the pool (``dma_free_task``)."""
        dma_free_task(self._handle)

    def await_(self) -> None:
        """Block until this transfer completes (``dma_await_task``).

        The transfer must have been issued with ``wait=True`` so it carries a
        completion token.
        """
        dma_await_task(self._handle)
