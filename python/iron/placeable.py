# placeable.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from .device import Tile, PlacementTile


class Placeable:
    """Placeable is a base class of an object that might be Placed by a Placer."""

    def __init__(self, tile: PlacementTile | None):
        """Initialize a Placeable object.

        Args:
            tile (PlacementTile): A placeable object has a tile. This may be None during construction.
        """
        self._tile = tile

    def place(self, tile: Tile) -> None:
        """Place the object by assigning the object to a Tile.

        Args:
            tile (Tile): The placement tile.

        Raises:
            AlreadyPlacedError: If the object's tile is already set to a Tile object.
        """
        if isinstance(self._tile, Tile):
            raise AlreadyPlacedError(self.__class__, self._tile, tile)
        self._tile = tile

    @property
    def tile(self) -> PlacementTile | None:
        """Return the tile of the placeable object.

        Returns:
            PlacementTile: The current placement of the object.
        """
        return self._tile


class AlreadyPlacedError(Exception):
    """Placeable objects may raise this error if one attempts to assign them to a Tile more than once."""

    def __init__(self, cls, current_tile: Tile, new_tile: Tile):
        """Create an AlreadyPlacedError

        Args:
            current_tile (Tile): The current placement tile
            new_tile (Tile): The placement tile given for the second attempt to place the object.
        """
        self.message = (
            f"{cls} already placed at {current_tile}; cannot place at {new_tile}"
        )
        super().__init__(self.message)
