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
    ):
        """Assign placement informatio to a program.

        Args:
            device (Device): The device to use for placement.
            rt (Runtime): The runtime information for the program.
        """
        ...


class SequentialPlacer(Placer):
    """SequentialPlacer is a simple implementation of a placer. The SequentialPlacer is so named
    because it will sequentially place workers to Compute Tiles. After workers are placed, Memory Tiles and
    Shim Tiles are placed as close to the column of the given compute tile as possible.

    The SequentialPlacer only does validation of placement with respect to available DMA channels on the tiles.
    However, it can yield invalid placements that exceed other resource limits, such as memory, For complex or
    resource sensitive designs, a more complex placer or manual placement is required.
    """

    def __init__(self):
        super().__init__()

    def make_placement(
        self,
        device: Device,
        rt: Runtime,
    ):

        # Keep track of tiles available for placement
        shims = device.get_shim_tiles()
        mems = device.get_mem_tiles()
        computes = device.get_compute_tiles()

        compute_idx = 0

        # If some workers are already placed, remove them from the available set
        for worker in rt.workers:
            # This worker has already been placed
            if isinstance(worker.tile, Tile):
                if not worker.tile in computes:
                    raise ValueError(
                        f"Partial Placement Error: "
                        f"Tile {worker.tile} not available on "
                        f"device {device} or has already been used."
                    )
                computes.remove(worker.tile)

        # Place worker tiles
        for worker in rt.workers:
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

        shim_idx = 0
        mem_idx = 0
        for ofh in rt.fifohandles:
            if ofh.endpoint.tile == AnyComputeTile:
                ofh.endpoint.place(computes[compute_idx])
                compute_idx = (compute_idx + 1) % len(computes)
            elif ofh.endpoint.tile == AnyShimTile:
                ofh.endpoint.place(shims[shim_idx])
                shim_idx = (shim_idx + 1) % len(shims)
            elif ofh.endpoint.tile == AnyMemTile:
                ofh.endpoint.place(mems[mem_idx])
                mem_idx = (shim_idx + 1) % len(mems)
