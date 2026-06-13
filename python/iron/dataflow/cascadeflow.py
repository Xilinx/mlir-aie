# cascadeflow.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""CascadeFlow: a directed cascade stream connection between two Workers."""

from ...dialects.aie import (
    cascade_flow as _cascade_flow_op,  # pyright: ignore[reportAttributeAccessIssue]
)
from ..resolvable import Resolvable


class CascadeFlow(Resolvable):
    """A directed cascade stream connection from one Worker to another.

    Construct one of these for each cascade edge in your design::

        CascadeFlow(producer_worker, consumer_worker)

    Lowers to ``aie.cascade_flow(producer.tile, consumer.tile)`` after both
    Workers are placed. The kernel functions are responsible for using the
    ``put_mcd`` / ``get_scd`` intrinsics to actually drive/read the cascade
    stream — this object only declares the directed topology edge.

    Hardware constraints (enforced by the underlying op verifier):

    * Source and destination tiles must be cardinal-adjacent.
    * Each compute tile has at most one cascade input (from N or W) and one
      cascade output (to S or E). Multiple cascade outputs from the same
      tile will fail at lowering, not at construction.
    * ShimTiles and MemTiles do not have cascade interfaces.

    Discovery: each newly-constructed CascadeFlow registers itself on its
    *source* Worker's ``_outgoing_cascades`` list. ``Program.resolve()``
    walks the runtime's workers and resolves each worker's outgoing
    cascades after placement — no global registry, no drain step.
    """

    def __init__(self, src, dst):
        """Construct a CascadeFlow.

        Args:
            src: Source ``Worker`` whose tile drives the cascade stream.
            dst: Destination ``Worker`` whose tile reads the cascade stream.
        """
        self._src = src
        self._dst = dst
        # Self-register on the source Worker so Program.resolve() can find
        # us by walking its workers (the same walk it already does).
        src._outgoing_cascades.append(self)

    def resolve(self, loc=None, ip=None) -> None:
        """Emit ``aie.cascade_flow(src.tile, dst.tile)``."""
        _cascade_flow_op(self._src.tile.op, self._dst.tile.op)
