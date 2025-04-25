# placers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import ABCMeta, abstractmethod
import statistics

from .device import Device
from .runtime import Runtime
from .worker import Worker
from .device import AnyComputeTile, AnyMemTile, AnyShimTile, Tile
from .dataflow import ObjectFifoHandle


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
    """SequentialPlacer is a simple implementation of a placer. The SequentialPlacer is to named
    because it will sequentially place workers to Compute Tiles. After workers are placed, Memory Tiles and
    Shim Tiles are placed so as to match the column of the given compute tile.

    The SequentialPlacer does not do any validation of placement and can often yield invalid placements
    that exceed resource limits for channels, memory, etc. For complex or resource sensitive designs,
    a more complex placer or manual placement is required.
    """

    def __init__(self):
        super().__init__()

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        shims_in = device.get_shim_tiles()
        shims_out = device.get_shim_tiles()
        shim_in_channels = {}
        shim_out_channels = {}

        mems_in = device.get_mem_tiles()
        mems_out = device.get_mem_tiles()
        mem_in_channels = {}
        mem_out_channels = {}

        computes = device.get_compute_tiles()
        computes_in = device.get_compute_tiles()
        computes_out = device.get_compute_tiles()
        compute_idx = 0
        compute_in_channels = {}
        compute_out_channels = {}

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

        for of in object_fifos:
            of_endpoints = of.all_of_endpoints()
            of_compute_endpoints = [
                of.tile for of in of_endpoints if of.tile in computes
            ]
            common_col = self._get_common_col(of_compute_endpoints)
            for ofe in of_endpoints:
                # Place "closest" to the compute endpoints
                if ofe.tile == AnyMemTile:
                    if of._is_prod:
                        memtile = self._find_col_match(common_col, mems_out, device)
                        ofe.place(memtile)
                        if not memtile in mem_out_channels:
                            mem_out_channels[memtile] = 0
                        mem_out_channels[memtile] += 1
                        max_memtile_out_channels = device.get_num_source_connections(
                            memtile
                        )
                        if mem_out_channels[memtile] >= max_memtile_out_channels:
                            mems_out.remove(memtile)
                    else:
                        memtile = self._find_col_match(common_col, mems_in, device)
                        ofe.place(memtile)
                        if not memtile in mem_in_channels:
                            mem_in_channels[memtile] = 0
                        mem_in_channels[memtile] += 1
                        max_memtile_in_channels = device.get_num_dest_connections(
                            memtile
                        )
                        if mem_in_channels[memtile] >= max_memtile_in_channels:
                            mems_in.remove(memtile)

                elif ofe.tile == AnyComputeTile:
                    computetile = self._find_col_match(common_col, computes, device)
                    ofe.place(computetile)
                    if not computetile in compute_in_channels:
                        compute_in_channels[computetile] = 0
                    compute_in_channels[computetile] += 1
                    if of._is_prod:
                        max_computetile_channels = device.get_num_source_connections(
                            computetile
                        )
                    else:
                        max_computetile_channels = device.get_num_dest_connections(
                            computetile
                        )
                    if compute_in_channels[computetile] >= max_computetile_channels:
                        computes.remove(computetile)

                elif ofe.tile == AnyShimTile:
                    if of._is_prod:
                        shimtile = self._find_col_match(common_col, shims_out, device)
                        ofe.place(shimtile)
                        if not shimtile in shim_out_channels:
                            shim_out_channels[shimtile] = 0
                        shim_out_channels[shimtile] += 1
                        max_shimtile_out_channels = 2
                        if shim_out_channels[shimtile] >= max_shimtile_out_channels:
                            shims_out.remove(shimtile)
                    else:
                        shimtile = self._find_col_match(common_col, shims_in, device)
                        ofe.place(shimtile)
                        if not shimtile in shim_in_channels:
                            shim_in_channels[shimtile] = 0
                        shim_in_channels[shimtile] += 1
                        max_shimtile_in_channels = 2
                        if shim_in_channels[shimtile] >= max_shimtile_in_channels:
                            shims_in.remove(shimtile)

    def _get_common_col(self, tiles: list[Tile]) -> int:
        """
        A utility function that calculates a column that is "close" or "common"
        to a set of tiles. It is a simple heuristic using the average to represent "distance"
        """
        cols = [t.col for t in tiles if isinstance(t, Tile)]
        if len(cols) == 0:
            return 0
        avg_col = round(statistics.mean(cols))
        return avg_col

    def _find_col_match(self, col: int, tiles: list[Tile], device: Device) -> Tile:
        """
        A utility function that sequentially searches a list of tiles to find one with a matching column.
        """
        new_col = col
        while new_col < device.cols:
            for t in tiles:
                if t.col == new_col:
                    return t
            new_col += 1
        raise ValueError(
            f"Failed to find a tile matching column {col}: tried until column {new_col}. Try using a device with more columns."
        )
