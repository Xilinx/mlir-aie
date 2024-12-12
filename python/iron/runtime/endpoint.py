# endpoint.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from __future__ import annotations

from ..dataflow.endpoint import ObjectFifoEndpoint
from ..device import PlacementTile


class RuntimeEndpoint(ObjectFifoEndpoint):
    """The Endpoint of an ObjectFifo that is produced or consumed by the Runtime.
    The placement of this Endpoint should be a Shim Tile.
    """

    def __init__(self, placement: PlacementTile) -> RuntimeEndpoint:
        super().__init__(placement)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RuntimeEndpoint):
            return NotImplemented
        return self.tile == other.tile

    def __str__(self) -> str:
        return f"RuntimeEndpoint({self.tile})"
