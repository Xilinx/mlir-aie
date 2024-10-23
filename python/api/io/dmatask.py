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
        self.__object_fifo = object_fifo
        self.__inout_data = inout_data
        self.__tensor_tile = data_tile
        self.__wait = wait
        self.__task = None

    def will_wait(self):
        return self.__wait

    @property
    def task(self):
        assert self.__task != None
        return self.__task

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self.__task = dma_configure_task_for(
            self.__object_fifo.op, issue_token=self.__wait
        )

        # TODO(erika) - fix issue w/ passthrough_kernel and remove this hack
        if (
            self.__tensor_tile.transfer_len == self.__tensor_tile.tensor_width
            and self.__tensor_tile.tensor_height == 1
        ):
            dimensions = None
        else:
            dimensions = self.__tensor_tile.dimensions

        with bds(self.__task) as bd:
            with bd[0]:
                dma_bd(
                    self.__inout_data.op,
                    len=self.__tensor_tile.transfer_len,
                    dimensions=dimensions,
                )
                EndOp()
        dma_start_task(self.__task)
        if self.__wait:
            dma_await_task(self.__task)
