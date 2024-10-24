from ... import ir  # type: ignore

from ...dialects._aie_ops_gen import EndOp
from ...dialects._aiex_ops_gen import dma_start_task, dma_await_task
from ...dialects.aiex import dma_configure_task_for
from ...dialects.aie import bds, dma_bd
from ..dataflow.objectfifo import ObjectFifoHandle
from ..resolvable import Resolvable
from .inoutdata import InOutData
from ...helpers.tensortiler.tensortiler2D import TensorTile


class DMATask(Resolvable):
    def __init__(
        self,
        object_fifo: ObjectFifoHandle,
        inout_data: InOutData,
        data_tile: TensorTile,
        wait=False,
    ):
        self._object_fifo = object_fifo
        self._inout_data = inout_data
        self._tensor_tile = data_tile
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
        self._task = dma_configure_task_for(
            self._object_fifo.op, issue_token=self._wait
        )

        # TODO(erika) - fix issue w/ passthrough_kernel and remove this hack
        if (
            self._tensor_tile.transfer_len == self._tensor_tile.tensor_width
            and self._tensor_tile.tensor_height == 1
        ):
            dimensions = None
        else:
            dimensions = self._tensor_tile.dimensions

        with bds(self._task) as bd:
            with bd[0]:
                dma_bd(
                    self._inout_data.op,
                    len=self._tensor_tile.transfer_len,
                    dimensions=dimensions,
                )
                EndOp()
        dma_start_task(self._task)
        if self._wait:
            dma_await_task(self._task)
