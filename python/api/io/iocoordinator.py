# Address circular dependency between IOCoordinator and InOutData
from __future__ import annotations

import numpy as np

from ...extras.util import DataTiler
from ..dataflow.objectfifo import ObjectFifoHandle
from ..dataflow.endpoint import ObjectFifoEndpoint
from ..phys.tile import Tile

from .shimpolicy import SingleShimPolicy, DistributeShimPolicy, ShimPlacementPolicy
from .syncpolicy import (
    SubtileLoopSyncPolicy,
    TileLoopSyncPolicy,
    NoSyncPolicy,
    SingleSyncPolicy,
    NSyncPolicy,
    DMASyncPolicy,
)


class DataTileSpecifier:
    """TODO: move to utils, define what this is"""

    pass


class InOutData:
    def __init__(self, input_type: type[np.ndarray]):
        self.input_type = input_type


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
        return f"IOEndpoint({self.tile()})"


class IOIterator:
    def __init__(
        self,
        io_coord: IOCoordinator,
        data_tile_iterator: DataTiler,
        yield_final=False,
        yield_every=False,
    ) -> IOIterator:
        self.io_coord = io_coord
        self.data_tile_iterator = data_tile_iterator
        self.yield_final = yield_final
        self.yield_every = yield_every
        self.first = True

    def __iter__(self):
        return self

    def __next__(self):
        tile_next = next(self.data_tile_iterator)
        if not self.first:
            if self.yield_every:
                self.io_coord._insert_sync("every")
        else:
            self.first = False
        if tile_next == None:
            if self.yield_final:
                self.io_coord._insert_sync("final")
            raise StopIteration
        return tile_next


class IOCoordinator:
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
        coords: tuple[int, int] | None = None,
    ) -> None:
        assert source in self.__inout_data
        if coords:
            column, row = coords
            io_endpoint = IOEndpoint(column, row)
            in_fifo.set_endpoint(io_endpoint)
        self.__fifos.add(in_fifo)
        self.__ops.append("DMA fill({in_fifo}, {data_tile}, {source})")

    def drain(
        self,
        out_fifo: ObjectFifoHandle,
        data_tile: DataTileSpecifier,
        dest: InOutData,
        coords: tuple[int, int] | None = None,
    ) -> None:
        assert dest in self.__inout_data
        if coords:
            column, row = coords
            io_endpoint = IOEndpoint(column, row)
            out_fifo.set_endpoint(io_endpoint)
        self.__fifos.add(out_fifo)
        self.__ops.append("DMA drain({out_fifo}, {data_tile}, {source})")

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self.__fifos.copy()

    def tile_loop(self, iter: DataTiler) -> IOIterator:
        return IOIterator(self, iter)

    def _insert_sync(self, sync_str):
        self.__ops.append(f"Sync HERE({sync_str})")
