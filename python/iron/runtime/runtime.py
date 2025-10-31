# runtime.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from __future__ import annotations
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
from typing import Callable

# import aie.utils.trace as trace_utils
from ...utils import trace as trace_utils

from ... import ir  # type: ignore

from ...dialects.aie import tile
from ...dialects.aiex import runtime_sequence
from ...dialects._aiex_ops_gen import dma_await_task, dma_free_task  # type: ignore
from ...helpers.taplib import TensorAccessPattern
from ..dataflow import ObjectFifoHandle
from ..device import PlacementTile, AnyShimTile
from ..resolvable import Resolvable
from ..worker import Worker, WorkerRuntimeBarrier, _BarrierSetOp
from .dmatask import DMATask
from .data import RuntimeData
from .endpoint import RuntimeEndpoint
from .taskgroup import RuntimeTaskGroup
from .task import (
    RuntimeTask,
    RuntimeStartTask,
    InlineOpRuntimeTask,
    FinishTaskGroupTask,
)


class Runtime(Resolvable):
    """A Runtime contains that operations and structure of all operations that
    need to be taken care of by the host/runtime in order to run a program.
    """

    """This is used to generate unique task group ids"""
    __task_group_index = 0

    def __init__(
        self,
    ) -> Runtime:
        """Initialize a runtime object."""
        self._rt_data = []
        self._tasks: list[RuntimeTask] = []
        self._fifos = set()
        self._workers = []
        self._open_task_groups = []
        self._trace_size = None
        self._trace_offset = None
        self._trace_workers = None
        self.ddr_id = None

    @contextmanager
    def sequence(self, *input_types: type[np.ndarray]):
        """A RuntimeSequence is a sequence of operations that are performed in
        support of a program. Common operations include input and output data movement.

        Raises:
            ValueError: Arguments are validated.
            ValueError: If task groups are not finished within the sequence() context, and error will be raised.

        Yields:
            _type_: Handles to the buffers matching the input types.
        """
        try:
            self._rt_data = list(map(RuntimeData, input_types))
            if len(self._rt_data) == 1:
                yield self._rt_data[0]
            else:
                yield tuple(self._rt_data.copy())
        finally:
            if len(self._open_task_groups) != 0:
                tgs_str = ", ".join([str(t) for t in self._open_task_groups])
                raise ValueError(f"Failed to close task groups: {tgs_str}")
            for of_handle in self._fifos:
                # It's very easy to accidentally generate multiple (identical)
                # consumers in the runtime. This bit of code prunes out duplicates.
                if not of_handle._is_prod:
                    fifo_obj = of_handle._object_fifo
                    runtime_cons = None
                    to_remove = []
                    for c in fifo_obj._cons:
                        if isinstance(c.endpoint, RuntimeEndpoint):
                            if not runtime_cons:
                                runtime_cons = c
                            else:
                                if (
                                    c.depth == runtime_cons.depth
                                    and c.dims_from_stream
                                    == runtime_cons.dims_from_stream
                                ):
                                    to_remove.append(c)
                                else:
                                    raise ValueError(
                                        f"Found two different RuntimeEndpoints for consumers of the same ObjectFifo: {fifo_obj}"
                                    )
                    for r in to_remove:
                        fifo_obj._cons.remove(r)

    def task_group(self) -> RuntimeTaskGroup:
        """Generate a handle to a RuntimeTaskGroup.
        This should be called within a Runtime.sequence() context.

        Returns:
            RuntimeTaskGroup: The new RuntimeTaskGroup
        """
        tg = RuntimeTaskGroup(self.__task_group_index)
        self._open_task_groups.append(tg)
        self.__task_group_index += 1
        return tg

    def finish_task_group(self, task_group: RuntimeTaskGroup):
        """Close out a RuntimeTaskGroup.
        This should be called within a Runtime.sequence() context.

        Args:
            task_group (RuntimeTaskGroup): _description_
        """
        self._open_task_groups.remove(task_group)
        self._tasks.append(FinishTaskGroupTask(task_group))

    def fill(
        self,
        in_fifo: ObjectFifoHandle,
        source: RuntimeData,
        tap: TensorAccessPattern | None = None,
        task_group: RuntimeTaskGroup | None = None,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    ) -> None:
        """Conceptually fill an ObjectFifoHandle (of type producer) with data from a runtime buffer.
        This should be called within a Runtime.sequence() context.

        Args:
            in_fifo (ObjectFifoHandle): The producer ObjectFifoHandle.
            source (RuntimeData): The input Runtime data buffer.
            tap (TensorAccessPattern | None, optional): A way of specifying how data in the buffer is accessed when sending it to the in_fifo.
                If None is given, this will default to a linear transfer containing all data in the source buffer. Defaults to None.
            task_group (RuntimeTaskGroup | None, optional): A TaskGroup to associate this task with. Defaults to None.
            wait (bool, optional): Whether this Task should be awaited on or not. If not, it will be freed when the task group is finished. Defaults to False.
            placement (PlacementTile, optional): The Shim tile to associate the data transfer with. Defaults to AnyShimTile.

        Raises:
            ValueError: Arguments are validated.
        """
        if source not in self._rt_data:
            raise ValueError(
                f"Source {source} is not a RuntimeData object generated by sequence()"
            )
        rt_endpoint = RuntimeEndpoint(placement)

        if tap is None:
            tap = source.default_tap()

        in_fifo.endpoint = rt_endpoint
        self._fifos.add(in_fifo)
        self._tasks.append(DMATask(in_fifo, source, tap, task_group, wait))

    def drain(
        self,
        out_fifo: ObjectFifoHandle,
        dest: RuntimeData,
        tap: TensorAccessPattern | None = None,
        task_group: RuntimeTaskGroup | None = None,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    ) -> None:
        """Conceptually fill an ObjectFifoHandle (of type consumer) of data and write that data to a runtime buffer.
        This should be called within a Runtime.sequence() context.

        Args:
            out_fifo (ObjectFifoHandle): The consumer ObjectFifoHandle.
            dest (RuntimeData): The output Runtime data buffer.
            tap (TensorAccessPattern | None, optional): A way of specifying how data in the buffer is accessed when sending it to the in_fifo.
                If None is given, this will default to a linear transfer containing all data in the source buffer. Defaults to None.
            task_group (RuntimeTaskGroup | None, optional):  A TaskGroup to associate this task with. Defaults to None. Defaults to None.
            wait (bool, optional): Whether this Task should be awaited on or not. If not, it will be freed when the task group is finished. Defaults to False.
            placement (PlacementTile, optional): The Shim tile to associate the data transfer with. Defaults to AnyShimTile.

        Raises:
            ValueError: Arguments are validated.
        """
        if dest not in self._rt_data:
            raise ValueError(
                f"Destination {dest} is not a RuntimeData object generated by sequence()"
            )
        rt_endpoint = RuntimeEndpoint(placement)
        if tap is None:
            tap = dest.default_tap()

        out_fifo.endpoint = rt_endpoint
        self._fifos.add(out_fifo)
        self._tasks.append(DMATask(out_fifo, dest, tap, task_group, wait))

    def start(self, *args: Worker):
        """A placeholder operation to indicate that one or more Worker should be started on the device.
        This should be called within a Runtime.sequence() context.

        Args:
            *args: One or more Workers. If more than one is given, they will be started in order.

        Raises:
            ValueError: Arguments are validated.
        """
        for worker in args:
            if not isinstance(worker, Worker):
                raise ValueError("Runtime can only start Worker objects")
            self._workers.append(worker)
            self._tasks.append(RuntimeStartTask(worker))

    def inline_ops(self, inline_func: Callable, inline_args: list):
        """Insert an InlineOpRuntimeTask into the runtime.
         This should be called within a Runtime.sequence() context.

        Args:
            inline_func (Callable): The function to execute within an MLIR context.
            inline_args (list): The state the function needs to execute.
        """
        # TODO: should filter args based on some criteria??
        self._tasks.append(InlineOpRuntimeTask(inline_func, inline_args))

    def enable_trace(
        self,
        trace_size: int = None,
        trace_offset: int = None,
        workers: [] = None,
        ddr_id: int = None,
        coretile_events: [] = None,
        memtile_events: [] = None,
        shimtile_events: [] = None,
    ):
        """Enable trace."""
        self._trace_size = trace_size
        self._trace_offset = trace_offset
        self._trace_workers = workers
        self._ddr_id = ddr_id
        self._coretile_events = coretile_events
        self._memtile_events = memtile_events
        self._shimtile_events = shimtile_events

    def set_barrier(self, barrier: WorkerRuntimeBarrier, value: int):
        """Set the value of a worker barrier.
        This should be called within a Runtime.sequence() context.

        Args:
            barrier (WorkerRuntimeBarrier): The WorkerRuntimeBarrier to set.
            value (int): The value to set the barrier to.
        """
        self._tasks.append(_BarrierSetOp(barrier, value))

    @property
    def workers(self) -> list[Worker]:
        """The workers associated with the Runtime by calls to start()"""
        return self._workers.copy()

    @property
    def fifos(self) -> list[ObjectFifoHandle]:
        """The ObjectFifoHandles associated with the Runtime by calls to fill() and drain()"""
        return self._fifos.copy()

    def get_first_cons_shimtile(self):
        """Find the first consumer side of an objfifo that is in the 0th row
        and uses it as the trace shim tile
        """
        for of_handle in self._fifos:
            if not of_handle._is_prod:
                endpoint_tile = of_handle._object_fifo._cons[0]._endpoint._tile
                if endpoint_tile.row == 0:
                    return endpoint_tile.op

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        rt_dtypes = [rt_data.arr_type for rt_data in self._rt_data]

        task_group_actions = defaultdict(list)

        @runtime_sequence(*rt_dtypes)
        def sequence(*args):

            if self._trace_size is not None:
                tiles_to_trace = []
                if self._trace_workers is not None:
                    for w in self._trace_workers:
                        tiles_to_trace.append(w.tile.op)
                else:
                    for w in self._workers:
                        if w.trace is not None:
                            tiles_to_trace.append(w.tile.op)

                trace_shim_tile = self.get_first_cons_shimtile()

                # print("config_trace")
                trace_utils.configure_packet_tracing_aie2(
                    # tiles_to_trace=[ tiles_to_trace[0] ],
                    tiles_to_trace=tiles_to_trace,
                    shim=trace_shim_tile,
                    trace_size=self._trace_size,
                    trace_offset=(
                        self._trace_offset if self._trace_offset is not None else 0
                    ),
                    ddr_id=self._ddr_id if self._ddr_id is not None else 4,
                    coretile_events=self._coretile_events,
                    memtile_events=self._memtile_events,
                    shimtile_events=self._shimtile_events,
                )

            for rt_data, rt_data_val in zip(self._rt_data, args):
                rt_data.op = rt_data_val

            no_waits = []
            for task in self._tasks:
                task.resolve()
                if isinstance(task, DMATask):
                    if task.will_wait():
                        if task.task_group:
                            task_group_actions[task.task_group].append(
                                (dma_await_task, [task.task])
                            )
                        else:
                            dma_await_task(task.task)
                            for t in no_waits:
                                dma_free_task(t.task)
                            no_waits = []
                    else:
                        if task.task_group:
                            task_group_actions[task.task_group].append(
                                (dma_free_task, [task.task])
                            )
                        else:
                            no_waits.append(task)
                if isinstance(task, FinishTaskGroupTask):
                    actions = task_group_actions[task.task_group]
                    for fn, args in actions:
                        if fn == dma_await_task:
                            fn(*args)
                    for fn, args in actions:
                        if fn != dma_await_task:
                            fn(*args)
                    task_group_actions[task.task_group] = None

            if self._trace_size is not None:
                trace_utils.gen_trace_done_aie2(trace_shim_tile)
