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


class _PlacerUtils:
    @staticmethod
    def get_common_col(tiles: list[Tile]) -> int:
        cols = [t.col for t in tiles if isinstance(t, Tile)]
        if len(cols) == 0:
            return 0
        avg_col = round(statistics.mean(cols))
        return avg_col

    @staticmethod
    def find_col_match(col: int, tiles: list[Tile], device: Device) -> Tile:
        new_col = col
        while new_col < device.cols:
            for t in tiles:
                if t.col == new_col:
                    return t
            new_col += 1
        raise ValueError(
            f"Failed to find a tile matching column {col}: tried until column {new_col}. Try using a device with more columns."
        )

    @staticmethod
    def update_channels(
        ofe: ObjectFifoEndpoint,
        tile: Tile,
        output: bool,
        num_required_channels: int,
        channels: dict,
        tiles: list[Tile],
        device: Device,
    ):
        if num_required_channels == 0:
            return
        if tile not in channels:
            channels[tile] = []
        channels[tile].append((ofe, num_required_channels))
        used_channels = sum(c for _, c in channels[tile])
        max_tile_channels = device.get_num_connections(tile, output)
        if used_channels >= max_tile_channels:
            if tile in tiles:
                tiles.remove(tile)

    @staticmethod
    def place_endpoint(
        ofe: ObjectFifoEndpoint,
        tiles: list[Tile],
        common_col: int,
        channels: dict,
        device: Device,
        output=False,
        link_tiles=[],
        link_channels={},
    ):
        num_required_channels = 1
        if isinstance(ofe, ObjectFifoLink):
            if output:
                num_required_channels = len(ofe._srcs)
                link_required_channels = len(ofe._dsts)
            else:
                num_required_channels = len(ofe._dsts)
                link_required_channels = len(ofe._srcs)
        test_tiles = tiles.copy()
        while True:
            tile = _PlacerUtils.find_col_match(common_col, test_tiles, device)
            total_channels = num_required_channels
            if tile in channels:
                total_channels += sum(c for _, c in channels[tile])
            max_tile_channels = device.get_num_connections(tile, output)
            if total_channels <= max_tile_channels:
                if isinstance(ofe, ObjectFifoLink):
                    total_link_channels = link_required_channels
                    if tile in link_channels:
                        total_link_channels += sum(c for _, c in link_channels[tile])
                    max_link_channels = device.get_num_connections(tile, not output)
                    if total_link_channels <= max_link_channels:
                        break
                else:
                    break
            test_tiles.remove(tile)
        ofe.place(tile)
        _PlacerUtils.update_channels(
            ofe,
            tile,
            output,
            num_required_channels,
            channels,
            tiles,
            device,
        )
        if isinstance(ofe, ObjectFifoLink):
            _PlacerUtils.update_channels(
                ofe,
                tile,
                not output,
                link_required_channels,
                link_channels,
                link_tiles,
                device,
            )


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
        shims_in = device.get_shim_tiles()
        shims_out = device.get_shim_tiles()
        mems_in = device.get_mem_tiles()
        mems_out = device.get_mem_tiles()
        computes = device.get_compute_tiles()
        computes_in = device.get_compute_tiles()
        computes_out = device.get_compute_tiles()
        compute_idx = 0

        channels_in: dict[Tile, list[tuple[ObjectFifoEndpoint, int]]] = {}
        channels_out: dict[Tile, list[tuple[ObjectFifoEndpoint, int]]] = {}

        for worker in workers:
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

            prod_fifos = [of for of in worker.fifos if of._is_prod]
            cons_fifos = [of for of in worker.fifos if not of._is_prod]
            _PlacerUtils.update_channels(
                worker,
                worker.tile,
                True,
                len(prod_fifos),
                channels_out,
                computes_out,
                device,
            )
            _PlacerUtils.update_channels(
                worker,
                worker.tile,
                False,
                len(cons_fifos),
                channels_in,
                computes_in,
                device,
            )

        if len(computes) > 0:
            compute_idx = compute_idx % len(computes)

        for ofh in object_fifos:
            of_endpoints = ofh.all_of_endpoints()
            of_handle_endpoints = ofh._object_fifo._get_endpoint(is_prod=ofh._is_prod)
            of_compute_endpoints_tiles = [
                ofe.tile for ofe in of_endpoints if ofe.tile in computes
            ]
            common_col = _PlacerUtils.get_common_col(of_compute_endpoints_tiles)
            of_link_endpoints = [
                ofe for ofe in of_endpoints if isinstance(ofe, ObjectFifoLink)
            ]
            for ofe in of_handle_endpoints:
                if isinstance(ofe, Worker):
                    continue
                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        _PlacerUtils.place_endpoint(
                            ofe,
                            mems_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        _PlacerUtils.place_endpoint(
                            ofe,
                            mems_in,
                            common_col,
                            channels_in,
                            device,
                        )
                elif ofe.tile == AnyShimTile:
                    if ofh._is_prod:
                        _PlacerUtils.place_endpoint(
                            ofe,
                            shims_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        _PlacerUtils.place_endpoint(
                            ofe, shims_in, common_col, channels_in, device
                        )
            for ofe in of_link_endpoints:
                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        _PlacerUtils.place_endpoint(
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
                        _PlacerUtils.place_endpoint(
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
                        _PlacerUtils.place_endpoint(
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
                        _PlacerUtils.place_endpoint(
                            ofe,
                            computes_in,
                            common_col,
                            channels_in,
                            device,
                            link_tiles=computes_out,
                            link_channels=channels_out,
                        )


class ColumnLimitedPlacer(Placer):
    def __init__(self, cores_per_col: int):
        super().__init__()
        self.cores_per_col = cores_per_col

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
        workers: list[Worker],
        object_fifos: list[ObjectFifoHandle],
    ):
        computes = device.get_compute_tiles()
        columns = sorted(set(tile.col for tile in computes))
        max_columns = len(columns)
        required_columns = (len(workers) + self.cores_per_col - 1) // self.cores_per_col

        if required_columns > max_columns:
            raise ValueError(
                f"Requested {required_columns} columns for {len(workers)} workers "
                f"with {self.cores_per_col} cores per column, but device only has {max_columns} columns."
            )

        col_to_tiles = {
            col: [tile for tile in computes if tile.col == col] for col in columns
        }

        worker_idx = 0
        for col in columns[:required_columns]:
            tiles = col_to_tiles[col]
            if len(tiles) < self.cores_per_col:
                raise ValueError(
                    f"Column {col} only has {len(tiles)} compute tiles, but {self.cores_per_col} requested."
                )
            for i in range(self.cores_per_col):
                if worker_idx >= len(workers):
                    break
                worker = workers[worker_idx]
                if worker.tile == AnyComputeTile:
                    worker.place(tiles[i])
                for buffer in worker.buffers:
                    buffer.place(worker.tile)
                worker_idx += 1
            if worker_idx >= len(workers):
                break

        shims_in = device.get_shim_tiles()
        shims_out = device.get_shim_tiles()
        mems_in = device.get_mem_tiles()
        mems_out = device.get_mem_tiles()
        computes_in = device.get_compute_tiles()
        computes_out = device.get_compute_tiles()
        channels_in: dict[Tile, list[tuple[ObjectFifoEndpoint, int]]] = {}
        channels_out: dict[Tile, list[tuple[ObjectFifoEndpoint, int]]] = {}

        for worker in workers:
            prod_fifos = [of for of in worker.fifos if of._is_prod]
            cons_fifos = [of for of in worker.fifos if not of._is_prod]
            _PlacerUtils.update_channels(
                worker,
                worker.tile,
                True,
                len(prod_fifos),
                channels_out,
                computes_out,
                device,
            )
            _PlacerUtils.update_channels(
                worker,
                worker.tile,
                False,
                len(cons_fifos),
                channels_in,
                computes_in,
                device,
            )

        for ofh in object_fifos:
            of_endpoints = ofh.all_of_endpoints()
            of_handle_endpoints = ofh._object_fifo._get_endpoint(is_prod=ofh._is_prod)
            of_compute_endpoints_tiles = [
                ofe.tile for ofe in of_endpoints if ofe.tile in computes
            ]
            common_col = _PlacerUtils.get_common_col(of_compute_endpoints_tiles)
            of_link_endpoints = [
                ofe for ofe in of_endpoints if isinstance(ofe, ObjectFifoLink)
            ]
            for ofe in of_handle_endpoints:
                if isinstance(ofe, Worker):
                    continue
                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        _PlacerUtils.place_endpoint(
                            ofe,
                            mems_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        _PlacerUtils.place_endpoint(
                            ofe,
                            mems_in,
                            common_col,
                            channels_in,
                            device,
                        )
                elif ofe.tile == AnyShimTile:
                    if ofh._is_prod:
                        _PlacerUtils.place_endpoint(
                            ofe,
                            shims_out,
                            common_col,
                            channels_out,
                            device,
                            output=True,
                        )
                    else:
                        _PlacerUtils.place_endpoint(
                            ofe, shims_in, common_col, channels_in, device
                        )
            for ofe in of_link_endpoints:
                if ofe.tile == AnyMemTile:
                    if ofh._is_prod:
                        _PlacerUtils.place_endpoint(
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
                        _PlacerUtils.place_endpoint(
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
                        _PlacerUtils.place_endpoint(
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
                        _PlacerUtils.place_endpoint(
                            ofe,
                            computes_in,
                            common_col,
                            channels_in,
                            device,
                            link_tiles=computes_out,
                            link_channels=channels_out,
                        )
