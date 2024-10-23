# Address circular dependency between IOCoordinator and InOutData
from __future__ import annotations

from contextlib import contextmanager
import numpy as np


from ... import ir  # type: ignore

from ...dialects.aiex import runtime_sequence
from ...dialects._aiex_ops_gen import dma_free_task
from ...helpers.tensortiler.tensortiler2D import TensorTile2DIter, TensorTile
from ..dataflow.objectfifo import ObjectFifoHandle
from ..resolvable import Resolvable
from .dmatask import DMATask
from .inoutdata import InOutData
from .ioendpoint import IOEndpoint

from .shimpolicy import SingleShimPolicy, DistributeShimPolicy, ShimPlacementPolicy
from .syncpolicy import (
    SubtileLoopSyncPolicy,
    TileLoopSyncPolicy,
    NoSyncPolicy,
    SingleSyncPolicy,
    NSyncPolicy,
    DMASyncPolicy,
)


class IOIterator:
    def __init__(
        self,
        io_coord: IOCoordinator,
        data_tile_iterator: TensorTile2DIter,
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

    @contextmanager
    def build_sequence(self, *input_types: type[np.ndarray]):
        self.__inout_data = list(map(InOutData, input_types))
        yield tuple(self.__inout_data)

    def fill(
        self,
        in_fifo: ObjectFifoHandle,
        data_tile: TensorTile,
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
        data_tile: TensorTile,
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

    def tile_loop(self, *args: TensorTile) -> IOIterator:
        if len(args) == 1:
            return IOIterator(self, *args)
        return IOIterator(self, zip(*args))

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
