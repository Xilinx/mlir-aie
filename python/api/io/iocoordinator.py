import numpy as np

# Address circular dependency between IOCoordinator and ObjectFifoHandle
from __future__ import annotations

from ..dataflow.iofifo import IOObjectFifo
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


class InputData:
    def __init__(self, input_type: type[np.ndarray]):
        self.input_type = input_type


class OutputData:
    def __init__(self, output_type: type[np.ndarray]):
        self.output_type = output_type


class IOIterator:
    def __iter__(
        self,
        io_coord: IOCoordinator,
        data_tile_iterator: DataTiler,
        yield_final=False,
        yield_every=False,
    ):
        self.io_coord = io_coord
        self.data_tile_iterator = data_tile_iterator
        self.yield_final = yield_final
        self.yield_every = yield_every
        self.first = True

    def __next__(self):
        tile_next = next(self.data_tile_iterator)
        if not self.first:
            self.first = False
            if self.yield_every:
                self.io_coord._insert_sync("every")
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
    ):
        self.shim_policy = shim_policy
        self.sync_policy = sync_policy
        self.inputs = []
        self.outputs = []
        self.ops = []

    def input(self, input_type: type[np.ndarray]) -> InputData:
        self.inputs.append(InputData(input_type))
        return self.inputs[-1]

    def output(self, output_type: type[np.ndarray]) -> OutputData:
        self.outputs.append(InputData(output_type))
        return self.outputs[-1]

    def fill(
        self, in_fifo: IOObjectFifo, data_tile: DataTileSpecifier, source: InputData
    ) -> None:
        assert source in self.inputs
        self.ops.append("DMA fill({in_fifo}, {data_tile}, {source})")

    def drain(
        self, out_fifo: IOObjectFifo, data_tile: DataTileSpecifier, dest: OutputData
    ) -> None:
        assert dest in self.outputs
        self.ops.append("DMA drain({out_fifo}, {data_tile}, {source})")

    def tile_loop(self, iter: DataTiler) -> IOIterator:
        return IOIterator(self, iter)

    def _insert_sync(self, sync_str):
        self.ops.append(f"Sync HERE({sync_str})")
