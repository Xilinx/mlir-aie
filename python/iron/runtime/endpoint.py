# endpoint.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

from __future__ import annotations

from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
)
from ..dataflow.endpoint import ObjectFifoEndpoint
from ..device import AnyShimTile, Tile


class RuntimeEndpoint(ObjectFifoEndpoint):
    """An ObjectFifo endpoint representing data transfer between the host and the device.

    Operates on a shim tile — either a ShimNOCTile (the default, used by NPU
    host-driven DMA) or a ShimPLTile (used by VCK5000 PLIO designs where the
    runtime DMA is wired to the PL side of the shim).
    """

    _SHIM_TILE_TYPES = (AIETileType.ShimNOCTile, AIETileType.ShimPLTile)

    def __init__(self, tile: "Tile | None" = AnyShimTile) -> None:
        if tile is None:
            tile = AnyShimTile
        if tile.tile_type is not None and tile.tile_type not in self._SHIM_TILE_TYPES:
            raise ValueError(
                f"RuntimeEndpoint requires a shim tile (ShimNOCTile or "
                f"ShimPLTile), but got tile_type={tile.tile_type}"
            )
        # An explicit typed tile is a shared placement constraint, just as it
        # is for Workers and ObjectFifo links. Clone only the default singleton
        # or a tile whose type still needs to be inferred.
        default_type = (
            tile.tile_type if tile.tile_type is not None else AIETileType.ShimNOCTile
        )
        endpoint_tile = (
            tile
            if tile is not AnyShimTile and tile.tile_type is not None
            else tile.with_type(default_type)
        )
        super().__init__(endpoint_tile)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeEndpoint):
            return NotImplemented
        # Two shim endpoints at the same (col, row) are the same endpoint.
        # Unplaced endpoints compare equal via (None, None) == (None, None),
        # which is intentional: fill()/drain() on the same unplaced handle
        # called repeatedly should not trip the "endpoint already set" guard.
        assert self.tile is not None and other.tile is not None
        return (self.tile.col, self.tile.row) == (other.tile.col, other.tile.row)

    def __str__(self) -> str:
        return f"RuntimeEndpoint({self.tile})"
