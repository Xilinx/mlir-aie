# Address circular dependency between IOCoordinator and InOutData
from __future__ import annotations

from contextlib import contextmanager
import numpy as np


from ... import ir  # type: ignore

from ...dialects.aiex import runtime_sequence
from ...dialects._aiex_ops_gen import dma_free_task
from ...helpers.tensortiler.tensortiler2D import TensorTile2DIter, TensorTile
from ..dataflow.objectfifo import ObjectFifoHandle
from ..phys.tile import PlacementTile, AnyShimTile, Tile
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
        self._shim_policy = shim_policy
        self._sync_policy = sync_policy
        self._inout_data = []
        self._ops = []
        self._fifos = set()

    @contextmanager
    def runtime_sequence(self, *input_types: type[np.ndarray]):
        self._inout_data = list(map(InOutData, input_types))
        yield tuple(self._inout_data)

    def fill(
        self,
        in_fifo: ObjectFifoHandle,
        data_tile: TensorTile,
        source: InOutData,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    ) -> None:
        assert source in self._inout_data
        io_endpoint = IOEndpoint(placement)
        in_fifo.set_endpoint(io_endpoint)
        self._fifos.add(in_fifo)
        self._ops.append(DMATask(in_fifo, source, data_tile, wait))

    def drain(
        self,
        out_fifo: ObjectFifoHandle,
        data_tile: TensorTile,
        dest: InOutData,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    ) -> None:
        assert dest in self._inout_data
        io_endpoint = IOEndpoint(placement)
        out_fifo.set_endpoint(io_endpoint)
        self._fifos.add(out_fifo)
        self._ops.append(DMATask(out_fifo, dest, data_tile, wait))

    def get_fifos(self) -> list[ObjectFifoHandle]:
        return self._fifos.copy()

    def tile_loop(self, *args: TensorTile) -> IOIterator:
        if len(args) == 1:
            return IOIterator(self, *args)
        return IOIterator(self, zip(*args))

    def place_tasks(self, shim_tiles: list[Tile]) -> None:
        if self._shim_policy == SingleShimPolicy:
            for op in self._ops:
                ofe = op.fifo.get_endpoint()
                ofe = ofe[0]  # un-listify
                assert isinstance(
                    ofe, IOEndpoint
                ), f"Expected IOEndpoint, but found {type(ofe)}"
                if ofe.tile == AnyShimTile:
                    ofe.place(shim_tiles[0])

        elif isinstance(self._shim_policy, DistributeShimPolicy):
            placement_shim_tiles = shim_tiles[: self._shim_policy.num_shim_tiles]
            shim_idx = 0
            chunk_size = self._shim_policy.chunk_size
            chunk_idx = 0

            for op in self._ops:
                ofe = op.fifo.get_endpoint()
                ofe = ofe[0]  # un-listify
                assert isinstance(
                    ofe, IOEndpoint
                ), f"Expected IOEndpoint, but found {type(ofe)}"
                if ofe.tile == AnyShimTile:
                    ofe.place(placement_shim_tiles[shim_idx])
                    chunk_idx += 1
                    if chunk_idx == chunk_size:
                        chunk_idx = 0
                        shim_idx = (shim_idx + 1) % len(placement_shim_tiles)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        inout_types = [inout_data.inout_type for inout_data in self._inout_data]

        @runtime_sequence(*inout_types)
        def sequence(*args):

            for io_data, io_data_val in zip(self._inout_data, args):
                io_data.op = io_data_val

            no_waits = []
            for dma_task in self._ops:
                dma_task.resolve()
                if dma_task.will_wait():
                    for t in no_waits:
                        dma_free_task(t.task)
                    no_waits = []
                else:
                    no_waits.append(dma_task)
