# kernels/__init__.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Factory functions for AIE kernel ExternalFunctions.

Submodules:
- :mod:`.eltwise` — passthrough, scale, add, mul, relu
- :mod:`.reduce` — reduce_add, reduce_min, reduce_max
- :mod:`.vision` — rgba2hue, threshold, bitwise_or, bitwise_and, gray2rgba, rgba2gray, filter2d, add_weighted
- :mod:`.activation` — softmax, gelu, silu, swiglu, bf16_exp
- :mod:`.linalg` — mm, mm_zero, mv, cascade_mm
- :mod:`.conv` — conv2dk1, conv2dk3, conv2dk1_skip, conv2dk1_i8, conv2dk14, conv2dk1_skip_init, bn_*
"""

from .eltwise import passthrough, scale, add, mul, relu
from .reduce import reduce_add, reduce_min, reduce_max
from .vision import (
    rgba2hue,
    threshold,
    bitwise_or,
    bitwise_and,
    gray2rgba,
    rgba2gray,
    filter2d,
    add_weighted,
)
from .activation import softmax, gelu, silu, swiglu, bf16_exp
from .linalg import mm, mm_zero, mv, cascade_mm
from .conv import (
    conv2dk1,
    conv2dk3,
    conv2dk1_skip,
    conv2dk1_i8,
    conv2dk14,
    conv2dk1_skip_init,
    bn_conv2dk1_relu,
    bn_conv2dk3,
    bn_conv2dk1_i8,
    bn_conv2dk1_skip,
    bn_conv2dk3_dw,
)

__all__ = [
    "passthrough",
    "scale",
    "add",
    "mul",
    "reduce_add",
    "reduce_min",
    "reduce_max",
    "relu",
    "rgba2hue",
    "threshold",
    "bitwise_or",
    "bitwise_and",
    "gray2rgba",
    "rgba2gray",
    "filter2d",
    "add_weighted",
    "softmax",
    "gelu",
    "silu",
    "swiglu",
    "bf16_exp",
    "mm",
    "mm_zero",
    "mv",
    "cascade_mm",
    "conv2dk1",
    "conv2dk3",
    "conv2dk1_skip",
    "conv2dk1_i8",
    "conv2dk14",
    "conv2dk1_skip_init",
    "bn_conv2dk1_relu",
    "bn_conv2dk3",
    "bn_conv2dk1_i8",
    "bn_conv2dk1_skip",
    "bn_conv2dk3_dw",
]
