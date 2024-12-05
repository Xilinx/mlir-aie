# dmatask.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from ... import ir  # type: ignore

from ...dialects._aiex_ops_gen import dma_start_task, dma_await_task
from ...dialects.aiex import shim_dma_single_bd_task
from ..dataflow.objectfifo import ObjectFifoHandle
from .runtimedata import RuntimeData
from ...helpers.taplib import TensorAccessPattern
from .runtimetask import RuntimeTask
from .runtimetaskgroup import RuntimeTaskGroup


class DMATask(RuntimeTask):
    def __init__(
        self,
        object_fifo: ObjectFifoHandle,
        rt_data: RuntimeData,
        tap: TensorAccessPattern,
        task_group: RuntimeTaskGroup | None = None,
        wait: bool = False,
    ):
        self._object_fifo = object_fifo
        self._rt_data = rt_data
        self._tap = tap
        self._wait = wait
        self._task = None
        RuntimeTask.__init__(self, task_group)

    def will_wait(self) -> bool:
        return self._wait

    @property
    def fifo(self) -> ObjectFifoHandle:
        return self._object_fifo

    @property
    def task(self):
        assert self._task != None
        return self._task

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._task = shim_dma_single_bd_task(
            self._object_fifo.op,
            self._rt_data.op,
            tap=self._tap,
            issue_token=self._wait,
        )
        dma_start_task(self._task)
