# endpoint.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from __future__ import annotations

from ..dataflow.endpoint import ObjectFifoEndpoint
from ..device import Tile, AnyShimTile
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIETileType,
)


class RuntimeEndpoint(ObjectFifoEndpoint):
    """An ObjectFifo endpoint representing data transfer between the host and the device.
    Operates on a shim tile — either a ShimNOCTile (the default, used by NPU
    host-driven DMA) or a ShimPLTile (used by VCK5000 PLIO designs where the
    runtime DMA is wired to the PL side of the shim).
    """

    _SHIM_TILE_TYPES = (AIETileType.ShimNOCTile, AIETileType.ShimPLTile)

    def __init__(self, tile: Tile = AnyShimTile) -> None:
        tile = tile.copy()
        if tile.tile_type is not None and tile.tile_type not in self._SHIM_TILE_TYPES:
            raise ValueError(
                f"RuntimeEndpoint requires a shim tile (ShimNOCTile or "
                f"ShimPLTile), but got tile_type={tile.tile_type}"
            )
        if tile.tile_type is None:
            tile.tile_type = AIETileType.ShimNOCTile
        super().__init__(tile)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeEndpoint):
            return NotImplemented
        assert self.tile is not None and other.tile is not None
        return (self.tile.col, self.tile.row) == (other.tile.col, other.tile.row)

    def __str__(self) -> str:
        return f"RuntimeEndpoint({self.tile})"
