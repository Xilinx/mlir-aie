# Address circular dependency between IOCoordinator and InOutData
from __future__ import annotations

import numpy as np
from ... import ir  # type: ignore

from ...dialects.aiex import runtime_sequence
from ...dialects._aiex_ops_gen import dma_free_task
from ...helpers.util import DataTiler, DataTileSpecifier
from ..dataflow.objectfifo import ObjectFifoHandle
from ..dataflow.endpoint import ObjectFifoEndpoint
from ..phys.tile import Tile
from ..resolvable import Resolvable
from .dmatask import DMATask
from .inoutdata import InOutData

from .shimpolicy import SingleShimPolicy, DistributeShimPolicy, ShimPlacementPolicy
from .syncpolicy import (
    SubtileLoopSyncPolicy,
    TileLoopSyncPolicy,
    NoSyncPolicy,
    SingleSyncPolicy,
    NSyncPolicy,
    DMASyncPolicy,
)


class IOEndpoint(ObjectFifoEndpoint):
    def __init__(self, column: int, row: int) -> IOEndpoint:
        self.__tile = Tile(column, row)

    @property
    def tile(self) -> Tile | None:
        return self.__tile

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IOEndpoint):
            return NotImplemented
        return self.tile() == other.tile()

    def __str__(self) -> str:
        return f"IOEndpoint({self.tile})"


class IOIterator:
    def __init__(
        self,
        io_coord: IOCoordinator,
        data_tile_iterator: DataTiler,
    ) -> IOIterator:
        self.io_coord = io_coord
        self.data_tile_iterator = data_tile_iterator

    def __iter__(self):
        return self

    def __next__(self):
        try:
            tile_next = next(self.data_tile_iterator)
            return tile_next
        except StopIteration:
            # self.io_coord._insert_sync("HI")
            raise StopIteration


class IOCoordinator(Resolvable):
    def __init__(
        self,
        shim_policy: ShimPlacementPolicy = SingleShimPolicy,
        sync_policy: DMASyncPolicy = SingleSyncPolicy,
    ) -> IOCoordinator:
        self.__shim_policy = shim_policy
        self.__sync_policy = sync_policy
        self.__inout_data = []
        self.__ops = []
        self.__fifos = set()

    def inout_data(self, input_type: type[np.ndarray]) -> InOutData:
        self.__inout_data.append(InOutData(input_type))
        return self.__inout_data[-1]

    def fill(
        self,
        in_fifo: ObjectFifoHandle,
        data_tile: DataTileSpecifier,
        source: InOutData,
        wait: bool = False,
        coords: tuple[int, int] | None = None,
    ) -> None:
        assert source in self.__inout_data
        if coords:
            column, row = coords
            io_endpoint = IOEndpoint(column, row)
            in_fifo.set_endpoint(io_endpoint)
        self.__fifos.add(in_fifo)
        self.__ops.append(DMATask(in_fifo, source, data_tile, wait))

    def drain(
        self,
        out_fifo: ObjectFifoHandle,
        data_tile: DataTileSpecifier,
        dest: InOutData,
        wait: bool = False,
        coords: tuple[int, int] | None = None,
    ) -> None:
        assert dest in self.__inout_data
        if coords:
            column, row = coords
            io_endpoint = IOEndpoint(column, row)
            out_fifo.set_endpoint(io_endpoint)
        self.__fifos.add(out_fifo)
        self.__ops.append(DMATask(out_fifo, dest, data_tile, wait))

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self.__fifos.copy()

    def tile_loop(self, iter: DataTiler) -> IOIterator:
        return IOIterator(self, iter)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        inout_types = [inout_data.inout_type for inout_data in self.__inout_data]

        @runtime_sequence(*inout_types)
        def sequence(*args):

            for io_data, io_data_val in zip(self.__inout_data, args):
                io_data.op = io_data_val

            no_waits = []
            for dma_task in self.__ops:
                dma_task.resolve()
                if dma_task.will_wait():
                    for t in no_waits:
                        dma_free_task(t.task)
                    no_waits = []
                else:
                    no_waits.append(dma_task)
