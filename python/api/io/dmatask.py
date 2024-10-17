from ... import ir  # type: ignore

from ...dialects._aie_ops_gen import EndOp
from ...dialects._aiex_ops_gen import dma_start_task, dma_await_task
from ...dialects.aiex import dma_configure_task_for
from ...dialects.aie import bds, dma_bd
from ..dataflow.objectfifo import ObjectFifoHandle
from ..resolvable import Resolvable
from .inoutdata import InOutData
from ...helpers.util import DataTileSpec


class DMATask(Resolvable):
    def __init__(
        self,
        object_fifo: ObjectFifoHandle,
        inout_data: InOutData,
        data_tile: DataTileSpec,
        wait=False,
    ):
        self.__object_fifo = object_fifo
        self.__inout_data = inout_data
        self.__data_tile_spec = data_tile
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
        with bds(self.__task) as bd:
            with bd[0]:
                dma_bd(
                    self.__inout_data.op,
                    len=self.__data_tile_spec.transfer_len,
                    dimensions=self.__data_tile_spec.dimensions,
                )
                EndOp()
        dma_start_task(self.__task)
        if self.__wait:
            dma_await_task(self.__task)
