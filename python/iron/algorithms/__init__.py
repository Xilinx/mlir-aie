# __init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""High-level algorithm templates built on IRON (transform, for_each, etc.)."""

from ._transform import (
    transform,
    transform_binary,
    transform_parallel,
    transform_parallel_binary,
)
from .conv_pipeline import (
    row_at_a_time,
    row_at_a_time_tiled,
    row_at_a_time_with_skip,
    sliding_3row,
)
from .for_each import for_each
from .reduce import reduce

__all__ = [
    "row_at_a_time",
    "row_at_a_time_tiled",
    "row_at_a_time_with_skip",
    "sliding_3row",
    "for_each",
    "reduce",
    "transform",
    "transform_binary",
    "transform_parallel",
    "transform_parallel_binary",
]
