# kernels/__init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Factory functions for AIE kernel ExternalFunctions.

Submodules:
- `eltwise` — passthrough, scale, add, mul, relu
- `reduce` — reduce_add, reduce_min, reduce_max, compute_max, argmax, argmax_combine
- `vision` — rgba2hue, threshold, bitwise_or, bitwise_and, gray2rgba, rgba2gray, filter2d, add_weighted
- `activation`: softmax, gelu, silu, swiglu, bf16_exp, exp2f_vec
- `linalg` — mm, mv, cascade_mm  (mm/mv expose ``.zero`` for the companion zero-fill kernel)
- `conv` — conv2dk1, conv2dk3, conv2dk1_skip, conv2dk1_i8, conv2dk14, conv2dk1_skip_init, bn_*
"""

from .activation import (
    bf16_exp,
    bf16_exp_ref,
    exp2f_vec,
    exp2f_vec_ref,
    gelu,
    gelu_ref,
    relu_ref,
    silu,
    silu_ref,
    softmax,
    softmax_ref,
    swiglu,
)
from .conv import (
    bn_conv2dk1_i8,
    bn_conv2dk1_input_split_partial_put_ui8,
    bn_conv2dk1_input_split_partial_skip_get,
    bn_conv2dk1_partial_get_relu_i8,
    bn_conv2dk1_partial_put_i8,
    bn_conv2dk1_relu,
    bn_conv2dk1_relu_xy_pool_padded,
    bn_conv2dk1_skip,
    bn_conv2dk3,
    bn_conv2dk3_dw,
    bn_conv2dk3_dw_out_split,
    bn_fc_relu_ui16_pad,
    conv2dk1,
    conv2dk1_i8,
    conv2dk1_skip,
    conv2dk1_skip_init,
    conv2dk3,
    conv2dk14,
)
from .eltwise import add, mul, passthrough, relu, scale
from .linalg import cascade_mm, mm, mv
from .reduce import (
    argmax,
    argmax_combine,
    argmax_ref,
    compute_max,
    reduce_add,
    reduce_max,
    reduce_min,
)
from .vision import (
    add_weighted,
    bitwise_and,
    bitwise_or,
    filter2d,
    gray2rgba,
    rgba2gray,
    rgba2hue,
    threshold,
)

__all__ = [
    "passthrough",
    "scale",
    "add",
    "mul",
    "reduce_add",
    "reduce_min",
    "reduce_max",
    "compute_max",
    "argmax",
    "argmax_combine",
    "argmax_ref",
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
    "exp2f_vec",
    "relu_ref",
    "silu_ref",
    "gelu_ref",
    "bf16_exp_ref",
    "exp2f_vec_ref",
    "softmax_ref",
    "mm",
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
    "bn_conv2dk1_relu_xy_pool_padded",
    "bn_fc_relu_ui16_pad",
    "bn_conv2dk1_partial_put_i8",
    "bn_conv2dk1_partial_get_relu_i8",
    "bn_conv2dk3_dw_out_split",
    "bn_conv2dk1_input_split_partial_put_ui8",
    "bn_conv2dk1_input_split_partial_skip_get",
]
