# __init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Device representations for supported AMD Ryzen AI NPU targets."""

from .device import (
    Device,
    NPU1,
    NPU1Col1,
    NPU1Col2,
    NPU1Col3,
    NPU2,
    NPU2Col1,
    NPU2Col2,
    NPU2Col3,
    NPU2Col4,
    NPU2Col5,
    NPU2Col6,
    NPU2Col7,
    XCVC1902,
)
from .tile import Tile, TileType
