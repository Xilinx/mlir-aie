# device.py -*- Python -*-
#
# Copyright (C) 2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

import re

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]
from ...dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    AIEArch,
    AIETileType,
)
from ...dialects.aie import (
    AIEDevice,  # pyright: ignore[reportAttributeAccessIssue]
    LogicalTileOp,
    WireBundle,  # pyright: ignore[reportAttributeAccessIssue]
    get_target_model,  # pyright: ignore[reportAttributeAccessIssue]
    logical_tile,
)
from ..resolvable import Resolvable
from .tile import Tile


class Device(Resolvable):
    """A representation of a device of a specific type.

    Provides device metadata (column/row counts) and emits
    aie.logical_tile ops for Tile objects during resolve.
    """

    def __init__(self, device: AIEDevice) -> None:
        self._device = device
        self._tm = get_target_model(device)
        self._resolved_tiles: dict[int, LogicalTileOp] = {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    @property
    def name(self) -> str:
        """The device's name in the dialect: ``"npu2"``, ``"npu1_1col"``, ...

        A name, not a family: a four-column NPU2 is ``"npu2_4col"``. Ask
        :attr:`arch` which architecture a device is.
        """
        return self._device.name

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

    @property
    def core_memory_bytes(self) -> int:
        """Data memory local to one compute tile."""
        return self._tm.get_local_memory_size()

    @property
    def default_core_stack_bytes(self) -> int:
        """Stack a Worker gets when nothing it calls declares a larger need."""
        return self._tm.get_default_core_stack_size()

    @property
    def core_rows(self) -> list[int]:
        """Rows of compute tiles, bottom to top."""
        return [
            r
            for r in range(self.rows)
            if self.get_tile_type(0, r) is AIETileType.CoreTile
        ]

    @property
    def core_dma_channels_in(self) -> int:
        """Input DMA channels a compute tile has, and so the most fifos one can be fed."""
        return self._tm.get_num_dest_switchbox_connections(
            0, self.core_rows[0], WireBundle.DMA
        )

    @property
    def core_dma_channels_out(self) -> int:
        """Output DMA channels available on a compute tile."""
        return self._tm.get_num_source_switchbox_connections(
            0, self.core_rows[0], WireBundle.DMA
        )

    @property
    def shim_dma_channels_in(self) -> int:
        """DMA channels the shim tiles feed the array through, summed over them:
        how many streams from the host the device carries at once."""
        return self._shim_dma(self._tm.get_num_source_shim_mux_connections)

    @property
    def shim_dma_channels_out(self) -> int:
        """DMA channels the shim tiles drain the array through, summed over them."""
        return self._shim_dma(self._tm.get_num_dest_shim_mux_connections)

    def _shim_dma(self, connections) -> int:
        return sum(
            connections(col, row, WireBundle.DMA)
            for col in range(self.cols)
            for row in range(self.rows)
            if self._tm.is_shim_noc_or_pl_tile(col, row)
        )

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

    def get_dma_bd_wrap_bits(self, col, row) -> int:
        """Wrap (size) field width, in bits."""
        self._validate_coordinates(col, row)
        return self._tm.get_dma_bd_wrap_bits(col, row)

    def get_dma_bd_step_bits(self, col, row) -> int:
        """Step (stride) field width, in bits. The field counts address granules."""
        self._validate_coordinates(col, row)
        return self._tm.get_dma_bd_step_bits(col, row)

    def get_dma_bd_iter_bits(self, col, row) -> int:
        """Return the iteration (repeat) field width, in bits."""
        self._validate_coordinates(col, row)
        return self._tm.get_dma_bd_iter_bits(col, row)

    @property
    def address_gen_granularity(self) -> int:
        """Address-generation granularity of the device, in bits."""
        return self._tm.get_address_gen_granularity()

    @property
    def max_lock_value(self) -> int:
        """Largest value a lock register holds."""
        return self._tm.get_max_lock_value()

    @property
    def max_repeat_count(self) -> int:
        """Largest repeat count one DMA task queue push carries (0 = none).

        A task started with a larger repeat count is issued as several pushes
        of the same task.
        """
        return self._tm.get_max_repeat_count()

    @property
    def dma_task_queue_depth(self) -> int:
        """How many tasks one DMA channel's task queue holds (0 = no queue)."""
        return self._tm.get_dma_task_queue_depth()

    def get_num_bds(self, tile_type: AIETileType) -> int:
        """Return how many DMA buffer descriptors (BDs) a tile of ``tile_type`` has.

        The BD budget is per tile TYPE, not per coordinate, and it is not
        uniform: on AIE2, a MemTile has 48 BDs while a CoreTile and a
        ShimNOCTile both have 16 — the same count for two different roles.
        Callers must name the tile type they mean rather than hardcode a
        BD count, since "16" is only right for two of the three types and
        silently wrong for the third.
        """
        for row in range(self.rows):
            for col in range(self.cols):
                if self.get_tile_type(col, row) == tile_type:
                    return self._tm.get_num_bds(col, row)
        raise ValueError(f"Device has no tile of type {tile_type!r}")

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
            loc=loc,
            ip=ip,
            packet_type=tile.packet_type,
            packet_id=tile.packet_id,
        )
        self._resolved_tiles[tile_id] = op
        tile.op = op


class NamedDevice(Device):
    """A device of one fixed type, constructed without arguments: ``NPU2()``.

    One subclass per :class:`AIEDevice` is generated below (``NPU1``,
    ``NPU2Col4``, ``XCVC1902``, ...).
    """

    aie_device: AIEDevice

    def __init__(self) -> None:
        super().__init__(device=self.aie_device)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> AIEDevice:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.aie_device


for device in AIEDevice:
    class_name = re.sub(r"NPU(\d+)_(\d+)COL", r"NPU\1Col\2", device.name.upper())
    globals()[class_name] = type(
        class_name,
        (NamedDevice,),
        {
            "aie_device": device,
            "__doc__": f"A representation of a device that resolves to {device}",
            "__module__": __name__,
        },
    )


def __getattr__(name: str) -> type[NamedDevice]:
    # The per-device subclasses (NPU1, NPU2Col4, XCVC1902, ...) are generated
    # from the AIEDevice enum by the loop above and live in module globals, so
    # this fallback only fires for names that were never generated. Raising
    # keeps real typos failing at import time; the annotation lets a static
    # type checker resolve the generated names as NamedDevice subclasses,
    # constructed without arguments, without a hand-maintained list that
    # would drift as devices are added.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
