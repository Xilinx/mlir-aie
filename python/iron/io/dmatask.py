from ... import ir  # type: ignore

from ...dialects._aiex_ops_gen import dma_start_task, dma_await_task
from ...dialects.aiex import shim_dma_single_bd_task
from ..dataflow.objectfifo import ObjectFifoHandle
from ..resolvable import Resolvable
from .inoutdata import InOutData
from ...helpers.taplib import TensorAccessPattern


class DMATask(Resolvable):
    def __init__(
        self,
        object_fifo: ObjectFifoHandle,
        inout_data: InOutData,
        tap: TensorAccessPattern,
        wait=False,
    ):
        self._object_fifo = object_fifo
        self._inout_data = inout_data
        self._tap = tap
        self._wait = wait
        self._task = None

    def will_wait(self):
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
            self._inout_data.op,
            tap=self._tap,
            issue_token=self._wait,
        )

        dma_start_task(self._task)
        if self._wait:
            dma_await_task(self._task)
