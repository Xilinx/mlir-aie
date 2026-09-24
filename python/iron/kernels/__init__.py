# kernels/__init__.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Factory functions for AIE kernel ExternalFunctions.

Submodules:
- `eltwise` — passthrough, scale, add, mul, relu
- `datamovement` — axpy, convert_copy, expand, rope, transpose
- `core` — set_rounding (the core's rounding-mode register, named by a contract's `setup`)
- `reduce` — reduce_add, reduce_min, reduce_max, compute_max
- `vision` — rgba2hue, threshold, bitwise_or, bitwise_and, gray2rgba, rgba2gray, filter2d, add_weighted
- `activation` — softmax, gelu, silu, swiglu, bf16_exp, exp2f_vec, tanh, sigmoid, leaky_relu
- `norm` — rms_norm, rms_norm_eps, layer_norm
- `quant` — q4nx_dequant (AIE2P packed q4nx to bfp16ebs8)
- `transformer` — rms_norm, layer_norm, layer_norm_f32, layer_norm_affine_cast, rope, mm_activation_epilogue
- `linalg` — mm, mv, cascade_mm, mm_bfp (a ``MatrixKernel``: ``.mac_dims``
  and ``.stream_dims`` read the blocking and DMA transforms off the
  contract's operand layouts)
- `mm(...).zero` and `mv(...).zero` construct companion zero-fill kernels;
  `mm.mac_dims(...)` and `cascade_mm.mac_dims(...)` query micro-kernel geometry
  without constructing a kernel.
- `zero` — independent zero-fill kernel

Every factory attaches a [`KernelContract`][iron.kernels.KernelContract] as
``.contract``: the role of each argument (``In``, ``Out``, ``InOut``,
``Param``), a numpy reference, a tolerance and the dtype facts a signature
cannot say. It is what ``aie.iron.algorithms.kernel_design`` uses to build,
run and check any kernel, and the ``*_ref`` functions exported here are those
references. ``factories`` lists the factory names.
- `conv` — conv2dk1, conv2dk3, conv2dk1_skip, conv2dk1_i8, conv2dk14, conv2dk1_skip_init, bn_*
"""

import inspect
import sys
from typing import get_type_hints

from aie.iron.kernel import ExternalFunction

from ._common import (
    KernelContract,
    Param,
    TensorLayout,
)
from .activation import (
    bf16_exp,
    bf16_exp_lut_ref,
    bf16_exp_ref,
    exp2f_vec,
    exp2f_vec_ref,
    gelu,
    gelu_ref,
    gelu_sized,
    leaky_relu,
    leaky_relu_ref,
    relu_ref,
    sigmoid,
    sigmoid_lut_ref,
    sigmoid_ref,
    silu,
    silu_lut_ref,
    silu_ref,
    silu_sized,
    softmax,
    softmax_ref,
    swiglu,
    swiglu_lut_ref,
    swiglu_ref,
    tanh,
    tanh_lut_ref,
    tanh_ref,
)
from .conv import (
    DWCONV1D_TAIL,
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
    conv2dk1_i8_ref,
    conv2dk1_ref,
    conv2dk1_skip,
    conv2dk1_skip_init,
    conv2dk1_skip_init_ref,
    conv2dk1_skip_ref,
    conv2dk3,
    conv2dk3_ref,
    conv2dk14,
    conv2dk14_ref,
    dwconv1d,
    dwconv1d_channels_first,
    dwconv1d_channels_first_ref,
    dwconv1d_channels_last,
    dwconv1d_channels_last_ref,
    dwconv1d_ref,
)
from .core import RoundingMode, conv_even, set_rounding
from .datamovement import (
    axpy,
    axpy_ref,
    convert_copy,
    convert_copy_ref,
    expand,
    expand_ref,
    rope,
    rope_ref,
    transpose,
    transpose_ref,
)
from .eltwise import (
    add,
    add_ref,
    add_sized,
    mul,
    mul_add,
    mul_add_ref,
    mul_ref,
    mul_sized,
    passthrough,
    relu,
    relu_sized,
    scale,
    scale_ref,
)
from .fused import fused_mm
from .linalg import (
    MatrixKernel,
    cascade_mm,
    cascade_mm_put,
    mha,
    mm,
    mm_acc_dtype,
    mm_bfp,
    mm_bfp_mixed_ref,
    mm_bfp_ref,
    mm_bfp_shuffle,
    mm_bfp_shuffle_ref,
    mm_bfp_tile_ref,
    mm_ref,
    mm_stream_dims,
    mm_tile_ref,
    mv,
    mv_bf16_ref,
    mv_ref,
    mv_tile_ref,
    prefill_fv,
    prefill_fv_ref,
)
from .norm import layer_norm, layer_norm_ref, rms_norm, rms_norm_eps, rms_norm_ref
from .quant import q4nx_dequant, q4nx_dequant_ref
from .reduce import (
    compute_max,
    compute_max_ref,
    reduce_add,
    reduce_add_ref,
    reduce_max,
    reduce_max_ref,
    reduce_min,
    reduce_min_ref,
)
from .transformer import (
    layer_norm_affine_cast,
    layer_norm_affine_cast_ref,
    layer_norm_f32,
    layer_norm_f32_ref,
    mm_activation_epilogue,
    mm_activation_epilogue_ref,
)
from .vision import (
    add_weighted,
    add_weighted_ref,
    bitwise_and,
    bitwise_and_ref,
    bitwise_or,
    bitwise_or_ref,
    filter2d,
    filter2d_ref,
    gray2rgba,
    gray2rgba_ref,
    rgba2gray,
    rgba2gray_ref,
    rgba2hue,
    rgba2hue_ref,
    threshold,
    threshold_ref,
)
from .zero import zero

__all__ = [
    "KernelContract",
    "MatrixKernel",
    "TensorLayout",
    "Param",
    "RoundingMode",
    "conv_even",
    "set_rounding",
    "zero",
    "passthrough",
    "scale",
    "add",
    "add_sized",
    "mul",
    "mul_sized",
    "mul_add",
    "mul_add_ref",
    "rms_norm",
    "q4nx_dequant",
    "q4nx_dequant_ref",
    "rms_norm_ref",
    "layer_norm",
    "layer_norm_ref",
    "layer_norm_f32",
    "layer_norm_f32_ref",
    "layer_norm_affine_cast",
    "layer_norm_affine_cast_ref",
    "rope",
    "rope_ref",
    "mm_activation_epilogue",
    "mm_activation_epilogue_ref",
    "reduce_add",
    "reduce_min",
    "reduce_max",
    "compute_max",
    "compute_max_ref",
    "relu",
    "relu_sized",
    "rgba2hue",
    "rgba2hue_ref",
    "threshold",
    "threshold_ref",
    "bitwise_or",
    "bitwise_or_ref",
    "bitwise_and",
    "bitwise_and_ref",
    "gray2rgba",
    "gray2rgba_ref",
    "rgba2gray",
    "rgba2gray_ref",
    "filter2d",
    "filter2d_ref",
    "add_weighted",
    "add_weighted_ref",
    "softmax",
    "gelu",
    "gelu_sized",
    "silu",
    "silu_sized",
    "swiglu",
    "swiglu_ref",
    "bf16_exp",
    "exp2f_vec",
    "tanh",
    "sigmoid",
    "leaky_relu",
    "axpy",
    "convert_copy",
    "expand",
    "transpose",
    "add_ref",
    "mul_ref",
    "scale_ref",
    "reduce_add_ref",
    "reduce_min_ref",
    "reduce_max_ref",
    "axpy_ref",
    "convert_copy_ref",
    "expand_ref",
    "transpose_ref",
    "mm_ref",
    "mm_tile_ref",
    "mv_ref",
    "mv_tile_ref",
    "mv_bf16_ref",
    "mm_stream_dims",
    "rms_norm_eps",
    "relu_ref",
    "silu_ref",
    "gelu_ref",
    "bf16_exp_lut_ref",
    "bf16_exp_ref",
    "exp2f_vec_ref",
    "softmax_ref",
    "sigmoid_lut_ref",
    "silu_lut_ref",
    "swiglu_lut_ref",
    "tanh_lut_ref",
    "tanh_ref",
    "sigmoid_ref",
    "leaky_relu_ref",
    "mm",
    "fused_mm",
    "mm_acc_dtype",
    "mha",
    "prefill_fv",
    "prefill_fv_ref",
    "mm_bfp",
    "mm_bfp_ref",
    "mm_bfp_mixed_ref",
    "mm_bfp_tile_ref",
    "mm_bfp_shuffle_ref",
    "mm_bfp_shuffle",
    "mv",
    "cascade_mm",
    "cascade_mm_put",
    "conv2dk1",
    "conv2dk1_ref",
    "conv2dk3",
    "dwconv1d",
    "dwconv1d_ref",
    "dwconv1d_channels_first",
    "dwconv1d_channels_first_ref",
    "dwconv1d_channels_last",
    "dwconv1d_channels_last_ref",
    "DWCONV1D_TAIL",
    "conv2dk3_ref",
    "conv2dk1_skip",
    "conv2dk1_skip_ref",
    "conv2dk1_i8",
    "conv2dk1_i8_ref",
    "conv2dk14",
    "conv2dk14_ref",
    "conv2dk1_skip_init",
    "conv2dk1_skip_init_ref",
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


def factories() -> list[str]:
    """Names of the exported kernel factories, in ``__all__`` order.

    A factory returns ``ExternalFunction`` or one of its subclasses. The
    ``*_ref`` references, query helpers such as ``mm_stream_dims`` and the
    contract classes are exported too, so anything that walks the library
    (the contract test, the static-check sweep) reads this rather than
    keeping its own list of names to skip.
    """
    module = sys.modules[__name__]

    def builds_a_kernel(f) -> bool:
        if not inspect.isfunction(f):
            return False
        declared = inspect.signature(f).return_annotation
        if isinstance(declared, str):
            declared = get_type_hints(f).get("return")
        return inspect.isclass(declared) and issubclass(declared, ExternalFunction)

    return [name for name in __all__ if builds_a_kernel(getattr(module, name))]
