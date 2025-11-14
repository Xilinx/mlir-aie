# placers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import Optional
import statistics

from .device import Device
from .runtime import Runtime
from .worker import Worker
from .device import AnyComputeTile, AnyMemTile, AnyShimTile, Tile
from .dataflow import ObjectFifoHandle, ObjectFifoEndpoint


class Placer(metaclass=ABCMeta):
    """Placer is an abstract class to define the interface between the Program
    and the Placer.
    """

    @abstractmethod
    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        """Assign placement informatio to a program.

        Args:
            device (Device): The device to use for placement.
            rt (Runtime): The runtime information for the program.
            workers (list[Worker]): The workers included in the program.
            object_fifos (list[ObjectFifoHandle]): The object fifos used by the program.
        """
        ...


class SequentialPlacer(Placer):
    """SequentialPlacer is a simple implementation of a placer. The SequentialPlacer is so named
    because it will sequentially place workers to Compute Tiles. After workers are placed, Memory Tiles and
    Shim Tiles are placed as close to the column of the given compute tile as possible.

    The SequentialPlacer only does validation of placement with respect to available DMA channels on the tiles.
    However, it can yield invalid placements that exceed other resource limits, such as memory, For complex or
    resource sensitive designs, a more complex placer or manual placement is required.

    The user may define a limited number of cores per column, which could help with issues in using packet-
    switched tracing. By limiting the number of cores per column, the placer will assign workers to compute
    tiles in a row-wise direction up to the defined limit then move to the next column for subsequent placement.
    """

    def __init__(self, cores_per_col: int | None = None):
        super().__init__()
        self.cores_per_col = cores_per_col

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):

        # Keep track of tiles available for placement
        shims = device.get_shim_tiles()
        mems = device.get_mem_tiles()
        computes = device.get_compute_tiles()

        # For shims and memtiles, we try to avoid overloading channels
        # by keeping tracks of [input_channels, output_channels] for each tile of that type.
        shim_endpoint_counts = defaultdict(list)
        for s in shims:
            shim_endpoint_counts[s] = [0, 0]
        mem_endpoint_counts = defaultdict(list)
        for m in mems:
            mem_endpoint_counts[m] = [0, 0]

        compute_idx = 0

        # If some workers are already taken, remove them from the available set
        for worker in workers:
            # This worker has already been placed
            if isinstance(worker.tile, Tile):
                if not worker.tile in computes:
                    raise ValueError(
                        f"Partial Placement Error: "
                        f"Tile {worker.tile} not available on "
                        f"device {device} or has already been used."
                    )
                computes.remove(worker.tile)

        # Shorten the list of compute tiles available if the cores per column value is set
        if self.cores_per_col is not None:
            unused_computes_at_col = {
                column: [tile for tile in computes if tile.col == column]
                for column in range(device.cols)
            }
            computes = []
            for col, tiles in unused_computes_at_col.items():
                if len(tiles) < self.cores_per_col:
                    raise ValueError(
                        f"Not enough compute tiles at column {col} to satisfy {self.cores_per_col} cores per column!"
                    )
                else:
                    computes.extend(tiles[: self.cores_per_col])

        # Place worker tiles
        for worker in workers:
            if worker.tile == AnyComputeTile:
                if compute_idx >= len(computes):
                    raise ValueError("Ran out of compute tiles for placement!")
                worker.place(computes[compute_idx])
                compute_idx += 1

            for buffer in worker.buffers:
                buffer.place(worker.tile)

        # Prepare to loop
        if len(computes) > 0:
            compute_idx = compute_idx % len(computes)

        for ofh in object_fifos:
            of_endpoints = ofh.all_of_endpoints()
            of_compute_endpoints_tiles = [
                ofe.tile for ofe in of_endpoints if ofe.tile in computes
            ]
            # Place "closest" to the compute endpoints
            common_col = self._get_common_col(of_compute_endpoints_tiles)

            for ofe in of_endpoints:
                if isinstance(ofe, Worker):
                    continue
                endpoint_type = 1 if ofe.is_prod() else 0

                if ofe.tile == AnyMemTile:
                    candidate_tiles = []
                    for m in mems:
                        # Only mark as candidate if there is a "connection" available
                        if mem_endpoint_counts[m][
                            endpoint_type
                        ] < device.get_num_connections(m, ofe.is_prod()):
                            candidate_tiles.append(m)
                elif ofe.tile == AnyShimTile:
                    candidate_tiles = []
                    for s in shims:
                        # Only mark as candidate if there is a "connection" available
                        if shim_endpoint_counts[s][
                            endpoint_type
                        ] < device.get_num_connections(s, ofe.is_prod()):
                            candidate_tiles.append(s)
                elif ofe.tile == AnyComputeTile:
                    candidate_tiles = computes
                else:
                    pass

                placement_tile = self._find_col_match(common_col, candidate_tiles)
                ofe.place(placement_tile)

                # Once placement tile is chosen, update the counts
                if ofe.tile == AnyMemTile:
                    mem_endpoint_counts[placement_tile][endpoint_type] += 1
                elif ofe.tile == AnyShimTile:
                    shim_endpoint_counts[placement_tile][endpoint_type] += 1

    def _get_common_col(self, tiles: list[Tile]) -> int:
        """
        A utility function that calculates a column that is "close" or "common"
        to a set of tiles. It is a simple heuristic using the average to represent "distance".
        """
        cols = [t.col for t in tiles if isinstance(t, Tile)]
        if len(cols) == 0:
            return 0
        avg_col = round(statistics.mean(cols))
        return avg_col

    def _find_col_match(self, col: int, tiles: list[Tile], device: Device) -> Tile:
        """
        A utility function that sequentially searches a list of tiles to find one with a matching column.
        The column is increased until a tile is found in the device, or an error is signaled.
        """
        new_col = col
        cols_checked = 0
        while cols_checked < device.cols:
            for t in tiles:
                if t.col == new_col:
                    return t
            new_col = (new_col + 1) % device.cols  # Loop around
            cols_checked += 1
        raise ValueError(
            f"Failed to find a tile matching column {col}: tried until column {new_col}. Try using a device with more columns."
        )
