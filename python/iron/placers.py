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
from .dataflow import ObjectFifoHandle, ObjectFifoLink, ObjectFifoEndpoint


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

        mems_in = device.get_mem_tiles()
        mems_out = device.get_mem_tiles()

        computes = device.get_compute_tiles()
        computes_in = device.get_compute_tiles()
        computes_out = device.get_compute_tiles()
        compute_idx = 0

        channels_in: dict[Tile, (ObjectFifoEndpoint, int)] = {}
        channels_out: dict[Tile, (ObjectFifoEndpoint, int)] = {}

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
            of_handle_endpoints = of._object_fifo._get_endpoint(is_prod=of._is_prod)
            of_compute_endpoints = [
                of.tile for of in of_endpoints if of.tile in computes
            ]
            common_col = self._get_common_col(of_compute_endpoints)
            of_link_endpoints = [
                of for of in of_endpoints if isinstance(of, ObjectFifoLink)
            ]
            for ofe in of_handle_endpoints:
                # Place "closest" to the compute endpoints
                # TODO Worker: already placed, count the used channel
                if ofe.tile == AnyMemTile:
                    if of._is_prod:
                        self._place_endpoint(
                            ofe, mems_out, common_col, channels_out, device, output=True, link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe, mems_in, common_col, channels_in, device, link_channels=channels_out,
                        )

                elif ofe.tile == AnyComputeTile:
                    if of._is_prod:
                        self._place_endpoint(
                            ofe, computes_out, common_col, channels_out, device, output=True, link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe, computes_in, common_col, channels_in, device, link_channels=channels_out,
                        )

                elif ofe.tile == AnyShimTile:
                    if of._is_prod:
                        self._place_endpoint(
                            ofe, shims_out, common_col, channels_out, device, output=True
                        )
                    else:
                        self._place_endpoint(
                            ofe, shims_in, common_col, channels_in, device
                        )

            for ofe in of_link_endpoints:
                # Place "closest" to the compute endpoints
                # TODO Worker: already placed, count the used channel
                if ofe.tile == AnyMemTile:
                    if of._is_prod:
                        self._place_endpoint(
                            ofe, mems_out, common_col, channels_out, device, output=True, link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe, mems_in, common_col, channels_in, device, link_channels=channels_out,
                        )

                elif ofe.tile == AnyComputeTile:
                    if of._is_prod:
                        self._place_endpoint(
                            ofe, computes_out, common_col, channels_out, device, output=True, link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe, computes_in, common_col, channels_in, device, link_channels=channels_out,
                        )

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

    def _place_endpoint(
            self,
            ofe : ObjectFifoEndpoint,
            tiles: list[Tile],
            common_col: int,
            channels: dict[Tile, (ObjectFifoEndpoint, int)],
            device: Device,
            output = False,
            link_channels = [],
        ):
        """
        A utility function that sequentially searches a list of tiles to find one with a matching column.
        """
        num_required_channels = 1
        if isinstance(ofe, ObjectFifoLink):
            # Also account for channels on the other side of the DMA
            if output:
                num_required_channels = len(ofe._srcs)
                link_required_channels = len(ofe._dsts)
            else:
                num_required_channels = len(ofe._dsts)
                link_required_channels = len(ofe._srcs)

        # Check if placing is possible
        test_tiles = tiles.copy()
        while True:
            tile = self._find_col_match(common_col, test_tiles, device)
            total_channels = num_required_channels
            if tile in channels:
                for _, c in channels[tile]:
                    total_channels += c
            max_tile_channels = device.get_num_connections(tile, output)
            if (ofe.tile == AnyShimTile):
                max_tile_channels = 2
            if total_channels <= max_tile_channels:
                if isinstance(ofe, ObjectFifoLink):
                    if tile not in link_channels:
                        link_channels[tile] = []
                    total_link_channels = link_required_channels
                    for _, c in link_channels[tile]:
                        total_link_channels += c
                    if total_link_channels <= device.get_num_connections(tile, not output):
                        link_channels[tile].append((ofe, link_required_channels))
                        break
                else:
                    break
            test_tiles.remove(tile)
        
        ofe.place(tile)

        if not tile in channels:
            channels[tile] = []
        channels[tile].append((ofe, num_required_channels))
        used_channels = 0
        for _, c in channels[tile]:
            used_channels += c
        if used_channels >= max_tile_channels:
            tiles.remove(tile)
