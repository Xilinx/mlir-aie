# endpoint.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.

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

    def __init__(self, tile: Tile = AnyShimTile) -> None:
        if tile is None:
            tile = AnyShimTile
        # A ShimPLTile endpoint is preserved as-is; otherwise default/validate to
        # ShimNOCTile. with_type returns a fresh Tile, never mutating the input.
        if tile.tile_type == AIETileType.ShimPLTile:
            tile = tile.with_type(AIETileType.ShimPLTile)
        else:
            tile = tile.with_type(
                AIETileType.ShimNOCTile,
                mismatch_msg=(
                    "RuntimeEndpoint requires a shim tile (ShimNOCTile or "
                    f"ShimPLTile), but got tile_type={tile.tile_type}"
                ),
            )
        super().__init__(tile)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeEndpoint):
            return NotImplemented
        # Compare by coordinates, not Tile identity: two RuntimeEndpoints that
        # name the same shim location (or are both unplaced) are equivalent, so
        # the tiling-loop pattern of calling fill()/drain() repeatedly on one
        # handle doesn't trip the "endpoint already set" guard. Unplaced tiles
        # compare equal via (None, None) == (None, None).
        return (self.tile.col, self.tile.row) == (other.tile.col, other.tile.row)

    def __str__(self) -> str:
        return f"RuntimeEndpoint({self.tile})"
