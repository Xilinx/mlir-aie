# cascadeflow.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""CascadeFlow: represents a cascade stream connection between two Workers."""

from ...dialects.aie import cascade_flow as _cascade_flow_op
from ..resolvable import Resolvable


class CascadeFlow(Resolvable):
    """Represents a cascade stream connection between two Workers.

    After workers are placed (tiles assigned), calling resolve() emits
    the aie.cascade_flow op connecting the src worker's tile to the dst
    worker's tile.
    """

    def __init__(self, src, dst):
        """Construct a CascadeFlow.

        Args:
            src: Source Worker (must have .tile.op after placement)
            dst: Destination Worker (must have .tile.op after placement)
        """
        self._src = src
        self._dst = dst

    def resolve(self, loc=None, ip=None) -> None:
        """Emit the cascade_flow MLIR op connecting src.tile to dst.tile."""
        _cascade_flow_op(self._src.tile.op, self._dst.tile.op)
