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

    @abstractmethod
    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ): ...


class SequentialPlacer(Placer):

    def __init__(self):
        super().__init__()

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        shims = device.get_shim_tiles()

        mems = device.get_mem_tiles()

        computes = device.get_compute_tiles()
        compute_idx = 0

        # If some workers are already taken, remove them from the available set
        for worker in workers:
            # This worker has already been placed
            if isinstance(worker.tile, Tile):
                if not worker.tile in computes:
                    raise ValueError(
                        f"Partial Placement Error: "
                        "Tile {worker.tile} not available on "
                        "device {device} or has already been used."
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
                    memtile = self._find_col_match(common_col, mems)
                    ofe.place(memtile)
                elif ofe.tile == AnyComputeTile:
                    computetile = self._find_col_match(common_col, computes)
                    ofe.place(computetile)
                elif ofe.tile == AnyShimTile:
                    shimtile = self._find_col_match(common_col, shims)
                    ofe.place(shimtile)

    def _get_common_col(self, tiles: list[Tile]) -> int:
        """
        This is a simplistic utility function that calculates a column that is "close" or "common"
        to a set of tiles. It is a simple heuristic using the average to represent "distance"
        """
        cols = [t.col for t in tiles if isinstance(t, Tile)]
        if len(cols) == 0:
            return 0
        avg_col = round(statistics.mean(cols))
        return avg_col

    def _find_col_match(self, col: int, tiles: list[Tile]) -> Tile:
        for t in tiles:
            if t.col == col:
                return t
        raise ValueError(f"Failed to find a tile matching column {col}")
