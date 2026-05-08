# endpoint.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
"""ObjectFifoEndpoint: base class for placeable endpoints of an ObjectFIFO."""

from ..device import Tile


class ObjectFifoEndpoint:
    """The endpoint of an ObjectFifo. Each ObjectFifoHandle has one ObjectFifoEndpoint."""

    def __init__(self, tile: Tile | None):
        self._tile = tile

    @property
    def tile(self) -> Tile | None:
        """Return the tile of the endpoint.

        Returns:
            Tile | None: The current tile, or None if unplaced.
        """
        return self._tile
