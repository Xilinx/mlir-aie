# placers.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import ABCMeta, abstractmethod
from typing import Optional
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

    def __init__(self, cores_per_col: Optional[int] = None):
        super().__init__()
        self.cores_per_col = cores_per_col

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        # Keep track of tiles available for placement based
        # on number of available input / output DMA channels
        shims_in = device.get_shim_tiles()
        shims_out = device.get_shim_tiles()

        mems_in = device.get_mem_tiles()
        mems_out = device.get_mem_tiles()

        computes = device.get_compute_tiles()
        computes_in = device.get_compute_tiles()
        computes_out = device.get_compute_tiles()
        compute_idx = 0

        # For each tile keep track of how many input and output endpoints there are
        # Note: defaultdict(list) automatically assigns an empty list as the default value for
        # keys that don’t exist
        channels_in: dict[Tile, tuple[ObjectFifoEndpoint, int]] = {}
        channels_out: dict[Tile, tuple[ObjectFifoEndpoint, int]] = {}

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
                    raise ValueError(f"Not enough compute tiles at column {col}!")
                else:
                    computes.extend(tiles[: self.cores_per_col])

        for worker in workers:
            if worker.tile == AnyComputeTile:
                if compute_idx >= len(computes):
                    raise ValueError("Ran out of compute tiles for placement!")
                worker.place(computes[compute_idx])
                compute_idx += 1

            for buffer in worker.buffers:
                buffer.place(worker.tile)

            # Account for channels used by Workers, which are already placed
            prod_fifos = [of for of in worker.fifos if of._is_prod]
            cons_fifos = [of for of in worker.fifos if not of._is_prod]
            self._update_channels(
                worker,
                worker.tile,
                True,
                len(prod_fifos),
                channels_out,
                computes_out,
                device,
            )
            self._update_channels(
                worker,
                worker.tile,
                False,
                len(cons_fifos),
                channels_in,
                computes_in,
                device,
            )

        # Prepare to loop
        if len(computes) > 0:
            compute_idx = compute_idx % len(computes)

        for ofh in object_fifos:
            of_endpoints = ofh.all_of_endpoints()
            of_handle_endpoints = ofh._object_fifo._get_endpoint(is_prod=ofh._is_prod)
            of_compute_endpoints_tiles = [
                ofe.tile for ofe in of_endpoints if ofe.tile in computes
            ]
            common_col = self._get_common_col(of_compute_endpoints_tiles)
            of_link_endpoints = [
                ofe for ofe in of_endpoints if isinstance(ofe, ObjectFifoLink)
            ]
            # Place "closest" to the compute endpoints
            for ofe in of_handle_endpoints:
                if isinstance(ofe, Worker):
                    continue

                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            mems_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        self._place_endpoint(
                            ofe,
                            mems_in,
                            common_col,
                            channels_in,
                            device,
                        )

                elif ofe.tile == AnyShimTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            shims_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        self._place_endpoint(
                            ofe, shims_in, common_col, channels_in, device
                        )

            for ofe in of_link_endpoints:
                # When placing ObjectFifoLink endpoints account for both
                # input and output channel requirements
                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            mems_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                            link_tiles=mems_in,
                            link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe,
                            mems_in,
                            common_col,
                            channels_in,
                            device,
                            link_tiles=mems_out,
                            link_channels=channels_out,
                        )

                elif ofe.tile == AnyComputeTile:
                    if ofh._is_prod:
                        self._place_endpoint(
                            ofe,
                            computes_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                            link_tiles=computes_in,
                            link_channels=channels_in,
                        )
                    else:
                        self._place_endpoint(
                            ofe,
                            computes_in,
                            common_col,
                            channels_in,
                            device,
                            link_tiles=computes_out,
                            link_channels=channels_out,
                        )

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
        while new_col < device.cols:
            for t in tiles:
                if t.col == new_col:
                    return t
            new_col += 1
        raise ValueError(
            f"Failed to find a tile matching column {col}: tried until column {new_col}. Try using a device with more columns."
        )

    def _update_channels(
        self,
        ofe: ObjectFifoEndpoint,
        tile: Tile,
        output: bool,
        num_required_channels: int,
        channels: dict[Tile, tuple[ObjectFifoEndpoint, int]],
        tiles: list[Tile],
        device: Device,
    ):
        """
        A utility function that updates given channel and tile lists. It appends a new
        (endpoint, num_required_channels) entry to the channels dict for the given tile key, then
        verifies whether the total entries for that tile surpass the maximum number of available
        channels. If so, it removes the tile from the list of available tiles.
        """
        if num_required_channels == 0:
            return
        if tile not in channels:
            channels[tile] = []
        channels[tile].append((ofe, num_required_channels))
        used_channels = 0
        for _, c in channels[tile]:
            used_channels += c
        max_tile_channels = device.get_num_connections(tile, output)
        if used_channels >= max_tile_channels:
            tiles.remove(tile)

    def _place_endpoint(
        self,
        ofe: ObjectFifoEndpoint,
        tiles: list[Tile],
        common_col: int,
        channels: dict[Tile, tuple[ObjectFifoEndpoint, int]],
        device: Device,
        output=False,
        link_tiles=[],
        link_channels={},
    ):
        """
        A utility function that places a given endpoint based on available DMA channels. If the endpoint is a
        link, both input and output channels should be accounted for. Calls _update_channels() to update channel
        dictionaries and tile lists.
        """
        is_shim = False
        num_required_channels = 1
        if isinstance(ofe, ObjectFifoLink):
            # If endpoint is a link, account for both input and output DMA channels
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
            if total_channels <= max_tile_channels:
                if isinstance(ofe, ObjectFifoLink):
                    # Also check for channels in the other link direction
                    total_link_channels = link_required_channels
                    if tile in link_channels:
                        for _, c in link_channels[tile]:
                            total_link_channels += c
                    max_link_channels = device.get_num_connections(tile, not output)
                    if total_link_channels <= max_link_channels:
                        break
                else:
                    break
            test_tiles.remove(tile)

        # If no error was signaled by _find_col_match(), placement is possible
        ofe.place(tile)

        # Account for channels that were used by this placement
        self._update_channels(
            ofe,
            tile,
            output,
            num_required_channels,
            channels,
            tiles,
            device,
        )

        if isinstance(ofe, ObjectFifoLink):
            self._update_channels(
                ofe,
                tile,
                not output,
                link_required_channels,
                link_channels,
                link_tiles,
                device,
            )
