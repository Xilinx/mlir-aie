# endpoint.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from abc import abstractmethod
from ..phys.tile import Tile, PlacementTile


class ObjectFifoEndpoint:
    @property
    @abstractmethod
    def tile(self) -> PlacementTile: ...

    @abstractmethod
    def place(self, tile: Tile) -> None: ...
