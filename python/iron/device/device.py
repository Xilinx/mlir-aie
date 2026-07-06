# device.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIEArch,
    AIETileType,
)
from ...dialects.aie import (
    AIEDevice,  # pyright: ignore[reportAttributeAccessIssue]
    logical_tile,
    LogicalTileOp,
    get_target_model,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..resolvable import Resolvable
from .tile import Tile

import re


class Device(Resolvable):
    """A representation of a device of a specific type.

    Provides device metadata (column/row counts) and emits
    aie.logical_tile ops for Tile objects during resolve.
    """

    def __init__(self, device: AIEDevice) -> None:
        self._device = device
        self._tm = get_target_model(device)
        self._resolved_tiles: dict[int, LogicalTileOp] = {}

    @property
    def cols(self) -> int:
        """Number of columns in the device tile array."""
        return self._tm.columns()

    @property
    def rows(self) -> int:
        """Number of rows in the device tile array."""
        return self._tm.rows()

    @property
    def arch(self) -> AIEArch:
        """AIE architecture of the device (AIE1, AIE2, or AIE2p)."""
        return AIEArch(self._tm.get_target_arch())

    def _validate_coordinates(self, col, row):
        """Raise ValueError if coordinates are outside the device grid."""
        if col < 0 or col >= self._tm.columns() or row < 0 or row >= self._tm.rows():
            raise ValueError(
                f"Coordinates ({col}, {row}) are out of range for device "
                f"({self._tm.columns()} cols x {self._tm.rows()} rows)"
            )

    def get_tile_type(self, col, row) -> AIETileType:
        """Return the AIETileType for the given device coordinates."""
        self._validate_coordinates(col, row)
        return AIETileType(self._tm.get_tile_type(col, row))

    def resolve_tile(
        self,
        tile: Tile,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        tile_id = id(tile)
        if tile_id in self._resolved_tiles:
            tile.op = self._resolved_tiles[tile_id]
            return

        # Emit one aie.logical_tile per distinct Tile object — no merging by
        # coordinate. The compiler owns coordinate resolution: --aie-place-tiles
        # merges non-core logical tiles that share a coordinate onto one
        # physical tile, and the aie.device verifier rejects two cores landing
        # on the same coordinate. (Dedup above is by object identity only: the
        # SAME Tile referenced from multiple endpoints resolves once.)
        #
        # The logical_tile op requires a tile_type, so infer it from coordinates
        # when unset. Computed locally — the Tile object is never mutated. Bounds
        # and tile_type/coordinate-agreement are verified by LogicalTileOp::verify,
        # so no Python-side check is needed here.
        tile_type = tile.tile_type
        if tile_type is None:
            if tile.col is not None and tile.row is not None:
                tile_type = self.get_tile_type(tile.col, tile.row)
            else:
                raise ValueError(
                    f"Cannot resolve {tile}: tile_type must be set or inferred from coordinates."
                )

        op = logical_tile(
            tile_type,
            col=tile.col,
            row=tile.row,
            allocation_scheme=tile.allocation_scheme,
            loc=loc,
            ip=ip,
        )
        self._resolved_tiles[tile_id] = op
        tile.op = op


def create_class(class_name, device):

    def _device__init__(self) -> None:
        super(globals()[class_name], self).__init__(device=device)

    def _device_resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        return device

    globals()[class_name] = type(
        class_name,
        (Device,),
        {
            "__init__": _device__init__,
            "resolve": _device_resolve,
            "__doc__": f"A representation of a device that resolves to {device}",
        },
    )


for device in AIEDevice:
    class_name = re.sub(r"NPU(\d+)_(\d+)COL", r"NPU\1Col\2", device.name.upper())
    create_class(class_name, device)


def __getattr__(name: str) -> type[Device]:
    # The per-device subclasses (NPU1, NPU2Col4, XCVC1902, ...) are generated
    # from the AIEDevice enum by the loop above and live in module globals, so
    # this fallback only fires for names that were never generated. Raising
    # keeps real typos failing at import time; the annotation lets a static
    # type checker resolve the generated names as Device subclasses without a
    # hand-maintained list that would drift as devices are added.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
