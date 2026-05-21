# flow.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Iron-level Flow primitive — an explicit AXI-stream route between two tile ports.

Peer of :class:`ObjectFifo` in the dataflow namespace.  ObjectFifo wraps
*route + buffers + locks + DMA* into one circular-buffer abstraction; a
Flow is the lower-level "just declare the route" primitive, paired with
explicit :class:`TileDma` programs (and :class:`Buffer` / :class:`Lock`
shared state) for designs that need direct control.
"""

from ... import ir  # type: ignore
from ...dialects._aie_enum_gen import (  # type: ignore
    AIETileType,
    DMAChannelDir,
    WireBundle,
)
from ...dialects.aie import flow as _flow_op, shim_dma_allocation
from ..device import Tile  # noqa: F401  (re-exported via package)
from ..resolvable import NotResolvedError, Resolvable

_SHIM_TILE_TYPES = (AIETileType.ShimNOCTile, AIETileType.ShimPLTile)


class Flow(Resolvable):
    """An explicit AXI-stream route between (src_tile, src_port, src_channel) and
    (dst_tile, dst_port, dst_channel).

    Lowers to a single ``aie.flow`` op.  The user is responsible for
    arranging matching :class:`TileDma` channels on the producer and
    consumer ends.
    """

    def __init__(
        self,
        src,
        dst,
        *,
        src_port: WireBundle = WireBundle.DMA,
        src_channel: int = 0,
        dst_port: WireBundle = WireBundle.DMA,
        dst_channel: int = 0,
        shim_symbol: str | None = None,
    ):
        """Construct a Flow.

        Args:
            src (Tile): The source tile.
            dst (Tile): The destination tile.
            src_port (WireBundle): The source port bundle.  Defaults to DMA.
            src_channel (int): The source channel.  Defaults to 0.
            dst_port (WireBundle): The destination port bundle.  Defaults to DMA.
            dst_channel (int): The destination channel.  Defaults to 0.
            shim_symbol (str | None): When this Flow has a shim endpoint that
                will be driven from the runtime sequence (via
                ``shim_dma_single_bd_task("symbol", ...)``), provide the
                symbol name here and the Flow will emit a matching
                ``aie.shim_dma_allocation`` at the device level.  Direction
                is inferred: shim-as-source → MM2S, shim-as-dest → S2MM.
        """
        self._src = src
        self._dst = dst
        self._src_port = src_port
        self._src_channel = src_channel
        self._dst_port = dst_port
        self._dst_channel = dst_channel
        self._shim_symbol = shim_symbol
        self._op = None

    @property
    def src(self):
        return self._src

    @property
    def dst(self):
        return self._dst

    @property
    def op(self):
        if self._op is None:
            raise NotResolvedError()
        return self._op

    def all_tiles(self):
        """The tiles this Flow touches — Program uses this to resolve them."""
        return [self._src, self._dst]

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        if self._op is None:
            self._op = _flow_op(
                self._src.op,
                self._src_port,
                self._src_channel,
                self._dst.op,
                self._dst_port,
                self._dst_channel,
            )
            if self._shim_symbol is not None:
                if self._src.tile_type in _SHIM_TILE_TYPES:
                    shim_dma_allocation(
                        self._shim_symbol,
                        self._src.op,
                        DMAChannelDir.MM2S,
                        self._src_channel,
                    )
                elif self._dst.tile_type in _SHIM_TILE_TYPES:
                    shim_dma_allocation(
                        self._shim_symbol,
                        self._dst.op,
                        DMAChannelDir.S2MM,
                        self._dst_channel,
                    )
                else:
                    raise ValueError(
                        f"Flow.shim_symbol={self._shim_symbol!r} requires "
                        f"the Flow to have a shim endpoint, but neither "
                        f"src ({self._src}) nor dst ({self._dst}) is a "
                        f"shim tile."
                    )
