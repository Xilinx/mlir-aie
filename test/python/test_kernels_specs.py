# test_kernels_specs.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Spec-table-driven unit tests for aie.iron.kernels factory functions.

Each kernel factory is described by a single row in KERNEL_SPECS. Generic
parametrized tests exercise the common surface (returns ExternalFunction,
source is locatable, _arg_types length, default _name, invalid-kwargs raise).
Per-kernel name and shape variants are listed alongside the spec.

Also covers the public arg_shape() / arg_dtype() introspection methods
that BaseKernel exposes for unwrapping parameterized np.ndarray arg types.

Sibling files:
  test_kernels_memoization.py  — memoization, independent zero, auto-prefix-on-collision
  test_kernels_chess.py        — use_chess + emulated bf16 plumbing

The shared _isolate_extern_state fixture lives in conftest.py at this
directory level.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pytest
from aie.iron import kernels
from aie.iron.device import NPU1Col1, NPU2Col1
from aie.iron.kernel import ExternalFunction
from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device
from ml_dtypes import bfloat16

# ---------------------------------------------------------------------------
# Spec table
# ---------------------------------------------------------------------------


@dataclass
class KernelSpec:
    """Declarative description of a kernel factory's expected surface."""

    name: str  # spec id (used as pytest parameter id)
    factory: Callable
    kwargs: dict  # baseline kwargs that should produce a valid kernel
    arg_count: int
    expected_name: str  # expected ef._name with baseline kwargs
    lut_source: str | None = None  # kernel selected by the AIE2 LUT translation unit
    # Additional (kwargs_overrides, expected_name) pairs
    name_variants: list[tuple[dict, str]] = field(default_factory=list)
    # (kwargs_overrides, error_pattern) pairs
    invalid_kwargs: list[tuple[dict, str]] = field(default_factory=list)
    # (kwargs_overrides, arg_index, expected_shape_tuple) — for shape sanity
    shape_checks: list[tuple[dict, int, tuple]] = field(default_factory=list)
    # (kwargs_overrides, expected_tile_size_at_arg_0)
    tile_size_checks: list[tuple[dict, int]] = field(default_factory=list)


KERNEL_SPECS: list[KernelSpec] = [
    KernelSpec(
        name="zero",
        factory=kernels.zero,
        kwargs=dict(tile_size=1024, dtype=np.int32),
        arg_count=1,
        expected_name="zero",
        invalid_kwargs=[(dict(tile_size=0), "positive integer or shape")],
        shape_checks=[(dict(tile_size=(8, 16)), 0, (8, 16))],
        tile_size_checks=[(dict(tile_size=192), 192)],
    ),
    # ----- eltwise -----
    KernelSpec(
        name="passthrough",
        factory=kernels.passthrough,
        kwargs=dict(tile_size=4096, dtype=np.int32),
        arg_count=3,
        expected_name="passThroughLine",
        shape_checks=[
            (dict(tile_size=192, dtype=np.int16), 0, (192,)),
        ],
        tile_size_checks=[
            (dict(tile_size=384, dtype=np.uint8), 384),
        ],
    ),
    KernelSpec(
        name="scale",
        factory=kernels.scale,
        kwargs=dict(tile_size=1024, dtype=np.int32),
        arg_count=4,
        expected_name="vector_scalar_mul_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=True),
                "vector_scalar_mul_vector",
            ),
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=False),
                "vector_scalar_mul_scalar",
            ),
        ],
        invalid_kwargs=[
            (
                dict(tile_size=1024, dtype=np.float32),
                "dtype must be np.int16 or np.int32",
            ),
        ],
    ),
    KernelSpec(
        name="add",
        factory=kernels.add,
        kwargs=dict(tile_size=1024, dtype=bfloat16),
        arg_count=3,
        expected_name="eltwise_add_bf16_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=True),
                "eltwise_add_bf16_vector",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "eltwise_add_bf16_scalar",
            ),
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=np.float32), "dtype must be bfloat16"),
        ],
    ),
    KernelSpec(
        name="mul",
        factory=kernels.mul,
        kwargs=dict(tile_size=1024, dtype=bfloat16),
        arg_count=3,
        expected_name="eltwise_mul_bf16_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=True),
                "eltwise_mul_bf16_vector",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "eltwise_mul_bf16_scalar",
            ),
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=np.float32), "dtype must be bfloat16"),
            (dict(tile_size=512), "tile_size must be 1024"),
        ],
    ),
    KernelSpec(
        name="add_sized",
        factory=kernels.add_sized,
        kwargs=dict(tile_size=1024),
        arg_count=4,  # a, b, c, size
        expected_name="eltwise_add_bf16_vector_size",
    ),
    KernelSpec(
        name="mul_sized",
        factory=kernels.mul_sized,
        kwargs=dict(tile_size=1024),
        arg_count=4,  # a, b, c, size
        expected_name="eltwise_mul_bf16_vector_size",
    ),
    KernelSpec(
        name="relu_sized",
        factory=kernels.relu_sized,
        kwargs=dict(tile_size=1024),
        arg_count=3,  # in, out, size
        expected_name="relu_bf16_size",
    ),
    # ----- reduce -----
    KernelSpec(
        name="reduce_add",
        factory=kernels.reduce_add,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="reduce_add_vector",
        name_variants=[
            (dict(tile_size=1024, vectorized=True), "reduce_add_vector"),
            (dict(tile_size=1024, vectorized=False), "reduce_add_scalar"),
            (dict(tile_size=512, dtype=np.int32), "reduce_add_vector"),
            (dict(tile_size=1024, dtype=bfloat16), "reduce_add_vector_bfloat16"),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "reduce_add_scalar_bfloat16",
            ),
        ],
        invalid_kwargs=[
            (
                dict(tile_size=1024, dtype=np.float32),
                "dtype must be np.int32 or bfloat16",
            ),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            # int32 output: 1 element = 4 bytes → already DMA-aligned.
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
            # bfloat16: out is padded to 2 elements (4 bytes) for DMA alignment.
            (dict(tile_size=1024, dtype=bfloat16), 1, (2,)),
        ],
        tile_size_checks=[(dict(tile_size=2048, dtype=np.int32), 2048)],
    ),
    KernelSpec(
        name="reduce_min",
        factory=kernels.reduce_min,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="reduce_min_vector",
        name_variants=[
            (dict(tile_size=1024, vectorized=True), "reduce_min_vector"),
            (dict(tile_size=1024, vectorized=False), "reduce_min_scalar"),
            (dict(tile_size=512, dtype=np.int32), "reduce_min_vector"),
            (dict(tile_size=1024, dtype=bfloat16), "reduce_min_vector_bfloat16"),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "reduce_min_scalar_bfloat16",
            ),
        ],
        invalid_kwargs=[
            (
                dict(tile_size=1024, dtype=np.float32),
                "dtype must be np.int32 or bfloat16",
            ),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
            # bfloat16: out is padded to 2 elements (4 bytes) for DMA alignment.
            (dict(tile_size=1024, dtype=bfloat16), 1, (2,)),
        ],
        tile_size_checks=[(dict(tile_size=2048, dtype=np.int32), 2048)],
    ),
    KernelSpec(
        name="reduce_max",
        factory=kernels.reduce_max,
        kwargs=dict(tile_size=1024, dtype=np.int32),
        arg_count=3,
        expected_name="reduce_max_vector",
        name_variants=[
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=True),
                "reduce_max_vector",
            ),
            (
                dict(tile_size=1024, dtype=np.int32, vectorized=False),
                "reduce_max_scalar",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=True),
                "reduce_max_vector_bfloat16",
            ),
            (
                dict(tile_size=1024, dtype=bfloat16, vectorized=False),
                "reduce_max_scalar_bfloat16",
            ),
            (dict(tile_size=1024, dtype=bfloat16), "reduce_max_vector_bfloat16"),
        ],
        invalid_kwargs=[
            (
                dict(tile_size=1024, dtype=np.float32),
                "dtype must be np.int32 or bfloat16",
            ),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            # int32: out is 4-byte aligned with a single element.
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
            # bfloat16: out must be padded to 2 elements (4 bytes) for DMA alignment.
            (dict(tile_size=1024, dtype=bfloat16), 1, (2,)),
        ],
    ),
    # ----- activation -----
    KernelSpec(
        name="relu",
        factory=kernels.relu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="bf16_relu",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="softmax",
        factory=kernels.softmax,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="softmax_bf16",
        lut_source="softmax.cc",
        invalid_kwargs=[(dict(tile_size=1000), "not a multiple")],
    ),
    KernelSpec(
        name="gelu",
        factory=kernels.gelu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="gelu_bf16",
        lut_source="gelu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="silu",
        factory=kernels.silu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="silu_bf16",
        lut_source="silu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="silu_sized",
        factory=kernels.silu_sized,
        kwargs=dict(tile_size=1024),
        arg_count=3,  # in, out, size
        expected_name="silu_bf16_size",
        lut_source="silu.cc",
    ),
    KernelSpec(
        name="gelu_sized",
        factory=kernels.gelu_sized,
        kwargs=dict(tile_size=1024),
        arg_count=3,  # in, out, size
        expected_name="gelu_bf16_size",
        lut_source="gelu.cc",
    ),
    KernelSpec(
        name="swiglu",
        factory=kernels.swiglu,
        kwargs=dict(tile_size=1024),
        arg_count=4,
        expected_name="swiglu_bf16",
        lut_source="swiglu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="bf16_exp",
        factory=kernels.bf16_exp,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="exp_bf16_1024",
        lut_source="bf16_exp.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="tanh",
        factory=kernels.tanh,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="tanh_bf16",
        lut_source="tanh.cc",
        invalid_kwargs=[
            (dict(tile_size=0), "must be positive"),
            (dict(tile_size=1000), "not a multiple of"),
        ],
        tile_size_checks=[(dict(tile_size=2048), 2048)],
    ),
    KernelSpec(
        name="sigmoid",
        factory=kernels.sigmoid,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="sigmoid_bf16",
        lut_source="sigmoid.cc",
        invalid_kwargs=[
            (dict(tile_size=0), "must be positive"),
            (dict(tile_size=1000), "not a multiple of"),
        ],
        tile_size_checks=[(dict(tile_size=2048), 2048)],
    ),
    KernelSpec(
        name="leaky_relu",
        factory=kernels.leaky_relu,
        kwargs=dict(tile_size=1024),
        arg_count=4,  # in, out, size (int32), alpha (bfloat16)
        expected_name="leaky_relu_bf16",
        lut_source="leaky_relu.cc",
        invalid_kwargs=[
            (dict(tile_size=size), "multiple of 32 and at least 64")
            for size in (-32, 0, 32, 33, 63, 65, 1000)
        ],
        tile_size_checks=[
            (dict(tile_size=64), 64),
            (dict(tile_size=96), 96),
            (dict(tile_size=512), 512),
            (dict(tile_size=2048), 2048),
        ],
    ),
    KernelSpec(
        name="exp2f_vec",
        factory=kernels.exp2f_vec,
        kwargs=dict(tile_size=1024),
        arg_count=3,
        expected_name="exp2f_vec_f32",
        invalid_kwargs=[
            (dict(tile_size=1000), "multiple of 16"),
            (dict(tile_size=1024, min_x=-127.0), "min_x must be >= -126"),
        ],
        shape_checks=[
            (dict(tile_size=32), 0, (32,)),
        ],
    ),
    # ----- vision -----
    KernelSpec(
        name="rgba2hue",
        factory=kernels.rgba2hue,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="rgba2hueLine",
        invalid_kwargs=[(dict(line_width=1000), "not a multiple")],
        shape_checks=[
            (dict(line_width=640), 0, (640 * 4,)),
            (dict(line_width=640), 1, (640,)),
        ],
    ),
    KernelSpec(
        name="rgba2gray",
        factory=kernels.rgba2gray,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="rgba2grayLine",
        shape_checks=[
            (dict(line_width=640), 0, (640 * 4,)),
            (dict(line_width=640), 1, (640,)),
        ],
    ),
    KernelSpec(
        name="gray2rgba",
        factory=kernels.gray2rgba,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="gray2rgbaLine",
        shape_checks=[
            (dict(line_width=640), 0, (640,)),
            (dict(line_width=640), 1, (640 * 4,)),
        ],
    ),
    KernelSpec(
        name="threshold",
        factory=kernels.threshold,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=6,
        expected_name="thresholdLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "thresholdLine"),
            (dict(line_width=1920, dtype=np.int32), "thresholdLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    KernelSpec(
        name="bitwise_or",
        factory=kernels.bitwise_or,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=4,
        expected_name="bitwiseORLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "bitwiseORLine"),
            (dict(line_width=1920, dtype=np.int32), "bitwiseORLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    KernelSpec(
        name="bitwise_and",
        factory=kernels.bitwise_and,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=4,
        expected_name="bitwiseANDLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "bitwiseANDLine"),
            (dict(line_width=1920, dtype=np.int32), "bitwiseANDLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    KernelSpec(
        name="filter2d",
        factory=kernels.filter2d,
        kwargs=dict(line_width=1920),
        arg_count=6,
        expected_name="filter2dLine",
        shape_checks=[(dict(line_width=640), 0, (640,))],
    ),
    KernelSpec(
        name="add_weighted",
        factory=kernels.add_weighted,
        kwargs=dict(line_width=1920, dtype=np.uint8),
        arg_count=7,
        expected_name="addWeightedLine",
        name_variants=[
            (dict(line_width=1920, dtype=np.int16), "addWeightedLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
            # addWeighted.cc's int32 branch has no int32 x int16 MAC and never compiled.
            (dict(line_width=1920, dtype=np.int32), "no int32 build"),
        ],
        shape_checks=[(dict(line_width=640, dtype=np.uint8), 0, (640,))],
    ),
    # ----- linalg -----
    KernelSpec(
        name="mm",
        factory=kernels.mm,
        kwargs=dict(),
        arg_count=3,
        expected_name="matmul_i16_i16",
        name_variants=[
            (
                dict(input_dtype=np.int16, output_dtype=np.int16, vectorized=True),
                "matmul_i16_i16",
            ),
            (
                dict(input_dtype=np.int16, output_dtype=np.int16, vectorized=False),
                "matmul_scalar_i16_i16",
            ),
            (dict(input_dtype=bfloat16, output_dtype=bfloat16), "matmul_bf16_bf16"),
            (dict(input_dtype=np.int8, output_dtype=np.int8), "matmul_i8_i8"),
            (dict(input_dtype=bfloat16, output_dtype=np.float32), "matmul_bf16_f32"),
        ],
        invalid_kwargs=[
            (dict(input_dtype=np.float64, output_dtype=np.float64), "unsupported"),
        ],
        shape_checks=[
            (dict(dim_m=32, dim_k=16, dim_n=48), 2, (32 * 48,)),
        ],
    ),
    KernelSpec(
        name="mv",
        factory=kernels.mv,
        kwargs=dict(),
        arg_count=3,
        expected_name="matvec_vectorized_i16_i32",
        name_variants=[
            (dict(vectorized=True), "matvec_vectorized_i16_i32"),
            (dict(vectorized=False), "matvec_scalar_i16_i32"),
        ],
        invalid_kwargs=[
            (dict(input_dtype=np.int8, output_dtype=np.int8), "only.*supported"),
        ],
        shape_checks=[
            (dict(dim_m=16, dim_k=64), 1, (64,)),
            (dict(dim_m=16, dim_k=64), 2, (16,)),
        ],
    ),
    KernelSpec(
        name="cascade_mm",
        factory=kernels.cascade_mm,
        kwargs=dict(),
        arg_count=3,
        expected_name="matmul_scalar_cascade_get_only_i16_i16",
        name_variants=[
            (
                dict(input_dtype=bfloat16, output_dtype=bfloat16),
                "matmul_scalar_cascade_get_only_bf16_bf16",
            ),
        ],
        invalid_kwargs=[
            (dict(input_dtype=np.int8, output_dtype=np.int8), "unsupported"),
        ],
    ),
    # ----- conv -----
    KernelSpec(
        name="conv2dk1",
        factory=kernels.conv2dk1,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk1_i8"),
            (dict(act_dtype=np.uint8), "conv2dk1_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="conv2dk3",
        factory=kernels.conv2dk3,
        kwargs=dict(),
        arg_count=13,
        expected_name="conv2dk3_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk3_i8"),
            (dict(act_dtype=np.uint8), "conv2dk3_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="conv2dk1_skip",
        factory=kernels.conv2dk1_skip,
        kwargs=dict(),
        arg_count=10,
        expected_name="conv2dk1_skip_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk1_skip_i8"),
            (dict(act_dtype=np.uint8), "conv2dk1_skip_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="conv2dk1_i8",
        factory=kernels.conv2dk1_i8,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_i8",
    ),
    KernelSpec(
        name="conv2dk14",
        factory=kernels.conv2dk14,
        kwargs=dict(),
        arg_count=8,
        expected_name="conv2dk14_i8",
    ),
    KernelSpec(
        name="conv2dk1_skip_init",
        factory=kernels.conv2dk1_skip_init,
        kwargs=dict(),
        arg_count=12,
        expected_name="conv2dk1_skip_init_i8",
        name_variants=[
            (dict(act_dtype=np.int8), "conv2dk1_skip_init_i8"),
            (dict(act_dtype=np.uint8), "conv2dk1_skip_init_ui8"),
        ],
        invalid_kwargs=[(dict(act_dtype=np.float32), "act_dtype")],
    ),
    KernelSpec(
        name="bn_conv2dk1_relu",
        factory=kernels.bn_conv2dk1_relu,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_relu_i8_ui8",
    ),
    KernelSpec(
        name="bn_conv2dk3",
        factory=kernels.bn_conv2dk3,
        kwargs=dict(),
        arg_count=13,
        expected_name="conv2dk3_stride2_i8",
    ),
    KernelSpec(
        name="bn_conv2dk1_i8",
        factory=kernels.bn_conv2dk1_i8,
        kwargs=dict(),
        arg_count=7,
        expected_name="conv2dk1_ui8_i8",
    ),
    KernelSpec(
        name="bn_conv2dk1_skip",
        factory=kernels.bn_conv2dk1_skip,
        kwargs=dict(),
        arg_count=9,
        expected_name="conv2dk1_skip_ui8_ui8_i8",
        name_variants=[
            (dict(skip_dtype=np.uint8), "conv2dk1_skip_ui8_ui8_i8"),
            (dict(skip_dtype=np.int8), "conv2dk1_skip_ui8_i8_i8"),
        ],
        invalid_kwargs=[(dict(skip_dtype=np.float32), "skip_dtype")],
    ),
    KernelSpec(
        name="bn_conv2dk3_dw",
        factory=kernels.bn_conv2dk3_dw,
        kwargs=dict(stride=2),
        arg_count=13,
        expected_name="conv2dk3_dw_stride2_relu_ui8_ui8",
        name_variants=[
            (dict(stride=1), "conv2dk3_dw_stride1_relu_ui8_ui8"),
            (dict(stride=2), "conv2dk3_dw_stride2_relu_ui8_ui8"),
        ],
        invalid_kwargs=[(dict(stride=3), "stride")],
    ),
    KernelSpec(
        name="bn_conv2dk1_relu_xy_pool_padded",
        factory=kernels.bn_conv2dk1_relu_xy_pool_padded,
        kwargs=dict(),
        arg_count=11,
        expected_name="conv2dk1_xy_pool_fused_relu_large_padded_i8_ui8",
    ),
    KernelSpec(
        name="bn_fc_relu_ui16_pad",
        factory=kernels.bn_fc_relu_ui16_pad,
        kwargs=dict(),
        arg_count=8,
        expected_name="post_L2_conv2dk1_relu_i16_ui16_pad",
    ),
    KernelSpec(
        name="bn_conv2dk1_partial_put_i8",
        factory=kernels.bn_conv2dk1_partial_put_i8,
        kwargs=dict(),
        arg_count=9,
        expected_name="bn13_1_conv2dk1_i8_ui8_partial_width_put_new",
        name_variants=[
            (dict(block_index=13), "bn13_1_conv2dk1_i8_ui8_partial_width_put_new"),
            (dict(block_index=14), "bn14_1_conv2dk1_i8_ui8_partial_width_put_new"),
        ],
        invalid_kwargs=[(dict(block_index=12), "block_index")],
    ),
    KernelSpec(
        name="bn_conv2dk1_partial_get_relu_i8",
        factory=kernels.bn_conv2dk1_partial_get_relu_i8,
        kwargs=dict(),
        arg_count=12,
        expected_name="bn13_1_conv2dk1_i8_ui8_partial_width_get_new",
        name_variants=[
            (dict(block_index=14), "bn14_1_conv2dk1_i8_ui8_partial_width_get_new"),
        ],
        invalid_kwargs=[(dict(block_index=12), "block_index")],
    ),
    KernelSpec(
        name="bn_conv2dk3_dw_out_split",
        factory=kernels.bn_conv2dk3_dw_out_split,
        kwargs=dict(),
        arg_count=14,
        expected_name="bn13_conv2dk3_ui8_out_split",
        name_variants=[
            (dict(block_index=14), "bn14_conv2dk3_ui8_out_split"),
        ],
        invalid_kwargs=[(dict(block_index=12), "block_index")],
    ),
    KernelSpec(
        name="bn_conv2dk1_input_split_partial_put_ui8",
        factory=kernels.bn_conv2dk1_input_split_partial_put_ui8,
        kwargs=dict(),
        arg_count=9,
        expected_name="bn13_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
        name_variants=[
            (
                dict(block_index=14),
                "bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new",
            ),
        ],
        invalid_kwargs=[(dict(block_index=12), "block_index")],
    ),
    KernelSpec(
        name="bn_conv2dk1_input_split_partial_skip_get",
        factory=kernels.bn_conv2dk1_input_split_partial_skip_get,
        kwargs=dict(),
        arg_count=14,
        expected_name="bn_13_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
        name_variants=[
            (
                dict(block_index=14),
                "bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new",
            ),
        ],
        invalid_kwargs=[(dict(block_index=12), "block_index")],
    ),
    # ----- data movement (kernels.datamovement) -----
    KernelSpec(
        name="axpy",
        factory=kernels.axpy,
        kwargs=dict(tile_size=1024),
        arg_count=5,  # x, y, a (scalar), z, size
        expected_name="saxpy",
        invalid_kwargs=[(dict(tile_size=1000), "multiple of 64")],
    ),
    KernelSpec(
        name="expand",
        factory=kernels.expand,
        kwargs=dict(tile_size=1024, group_size=32),
        arg_count=2,  # packed uint4 payload+scales, bf16 out
        expected_name="expand_uint4_to_bfloat16",
        invalid_kwargs=[(dict(group_size=48), "multiple of 32")],
    ),
    KernelSpec(
        name="transpose",
        factory=kernels.transpose,
        kwargs=dict(dim_m=32, dim_n=32, subtile=4),
        arg_count=2,
        expected_name="transpose_4x4",
        name_variants=[(dict(dim_m=32, dim_n=32, subtile=8), "transpose_8x8")],
        invalid_kwargs=[
            (dict(subtile=3), "subtile must be 4 or 8"),
            # dim_m=0 sails through the modulo checks below (0 % anything is
            # 0) and would otherwise reach the kernel's compile-time division.
            (dict(dim_m=0, dim_n=4, subtile=4, dtype=np.uint32), "must be positive"),
        ],
    ),
    KernelSpec(
        name="convert_copy",
        factory=kernels.convert_copy,
        kwargs=dict(tile_size=1024),
        arg_count=3,  # f32 in, bf16 out, size
        expected_name="cast_f32_bf16_row",
        # Binds datamovement/cast_f32_bf16.cc (upstream's cast, chosen over the dropped
        # IRON convert_copy.cc — see KERNEL_DEDUP_REPORT §4.1).
        invalid_kwargs=[(dict(tile_size=1000), "multiple of 16")],
    ),
    KernelSpec(
        name="rope",
        factory=kernels.rope,
        kwargs=dict(tile_size=1024),
        arg_count=4,  # in, lut, out, dims
        expected_name="rope",
        name_variants=[(dict(two_halves=True), "rope_two_halves")],
        shape_checks=[(dict(tile_size=96, two_halves=True), 0, (96,))],
    ),
    # ----- norm (kernels.norm) -----
    KernelSpec(
        name="rms_norm",
        factory=kernels.rms_norm,
        kwargs=dict(tile_size=1024),
        arg_count=3,  # in, out, cols
        expected_name="rms_norm",
    ),
    KernelSpec(
        name="rms_norm_eps",
        factory=kernels.rms_norm_eps,
        kwargs=dict(tile_size=1024),
        arg_count=4,  # in, out, cols, epsilon
        expected_name="rms_norm_eps",
    ),
    KernelSpec(
        name="layer_norm",
        factory=kernels.layer_norm,
        kwargs=dict(tile_size=1024),
        arg_count=3,  # in, out, cols
        expected_name="layer_norm",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(seq):
    return [s.name for s in seq]


def _flat(specs, attr):
    """Flatten (spec, *tuple_items) for parametrize tables."""
    out = []
    for s in specs:
        for entry in getattr(s, attr):
            out.append((s, *entry))
    return out


def _flat_ids(rows, label):
    return [f"{r[0].name}-{label}{i}" for i, r in enumerate(rows)]


def _call_factory(spec: KernelSpec, kwargs: dict):
    return spec.factory(**kwargs)


ARG_COUNT_OVERRIDES: list[tuple[KernelSpec, dict, int]] = []


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_returns_external_function(spec: KernelSpec):
    ef = _call_factory(spec, spec.kwargs)
    assert isinstance(ef, ExternalFunction)


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_source_locatable(spec: KernelSpec):
    ef = _call_factory(spec, spec.kwargs)
    assert ef.source_string is None
    assert ef.source_file is not None
    assert Path(ef.source_file).is_file()


@pytest.mark.parametrize("device,arch", [(NPU1Col1, "aie2"), (NPU2Col1, "aie2p")])
@pytest.mark.parametrize(
    "spec",
    [s for s in KERNEL_SPECS if s.lut_source],
    ids=lambda s: s.name,
)
def test_lut_source_selection(spec: KernelSpec, device, arch):
    previous = get_current_device(probe_runtime=False)
    set_current_device(device())
    try:
        ef = spec.factory(**spec.kwargs)
        assert ef.source_string is None
        assert Path(ef.source_file).is_file()
        selectors = [
            flag
            for flag in ef.compile_flags
            if flag.startswith("-DAIE_LUT_KERNEL_SOURCE=")
        ]
        if arch == "aie2":
            assert Path(ef.source_file).name == "lut_kernel.cc"
            assert len(selectors) == 1
            selected = Path(selectors[0].split("=", 1)[1].strip('"'))
            assert selected.is_file()
            assert selected.name == spec.lut_source
        else:
            assert not selectors
            assert Path(ef.source_file).name == spec.lut_source
    finally:
        set_current_device(previous)


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_arg_types_length(spec: KernelSpec):
    ef = _call_factory(spec, spec.kwargs)
    assert len(ef._arg_types) == spec.arg_count


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_default_function_name(spec: KernelSpec):
    ef = _call_factory(spec, spec.kwargs)
    # The logical (original) name is the stable identity; a parameterized
    # kernel's effective symbol is the original name with a deterministic
    # digest prefix (see _make_extern).  Assert on the order-independent
    # original name rather than the prefixed symbol.
    assert ef._original_name == spec.expected_name
    if ef._symbol_prefix is not None:
        assert ef._name == f"{ef._symbol_prefix}_{spec.expected_name}"
    else:
        assert ef._name == spec.expected_name


_NAME_VARIANTS = _flat(KERNEL_SPECS, "name_variants")


@pytest.mark.parametrize(
    "spec,kwargs,expected_name",
    _NAME_VARIANTS,
    ids=_flat_ids(_NAME_VARIANTS, "v"),
)
def test_name_variant(spec: KernelSpec, kwargs: dict, expected_name: str):
    ef = _call_factory(spec, kwargs)
    # See test_default_function_name: assert on the order-independent original
    # name; a parameterized variant carries a deterministic digest prefix.
    assert ef._original_name == expected_name
    if ef._symbol_prefix is not None:
        assert ef._name == f"{ef._symbol_prefix}_{expected_name}"
    else:
        assert ef._name == expected_name


_INVALID = _flat(KERNEL_SPECS, "invalid_kwargs")


@pytest.mark.parametrize(
    "spec,kwargs,pattern",
    _INVALID,
    ids=_flat_ids(_INVALID, "bad"),
)
def test_invalid_kwargs_raise(spec: KernelSpec, kwargs: dict, pattern: str):
    with pytest.raises(ValueError, match=pattern):
        _call_factory(spec, kwargs)


_SHAPES = _flat(KERNEL_SPECS, "shape_checks")


@pytest.mark.parametrize(
    "spec,kwargs,arg_idx,expected_shape",
    _SHAPES,
    ids=_flat_ids(_SHAPES, "shape"),
)
def test_arg_shape(spec: KernelSpec, kwargs: dict, arg_idx: int, expected_shape: tuple):
    ef = _call_factory(spec, kwargs)
    # Use the public arg_shape() method (which the whats-new notebook also
    # uses now) — exercises it across every shape_check spec entry.
    assert ef.arg_shape(arg_idx) == expected_shape


_TILE_SIZES = _flat(KERNEL_SPECS, "tile_size_checks")


@pytest.mark.parametrize(
    "spec,kwargs,expected_tile_size",
    _TILE_SIZES,
    ids=_flat_ids(_TILE_SIZES, "ts"),
)
def test_tile_size_at_arg_0(spec: KernelSpec, kwargs: dict, expected_tile_size: int):
    ef = _call_factory(spec, kwargs)
    assert ef.tile_size(0) == expected_tile_size


@pytest.mark.parametrize(
    "spec,kwargs,expected_arg_count",
    ARG_COUNT_OVERRIDES,
    ids=[f"{r[0].name}-argc{i}" for i, r in enumerate(ARG_COUNT_OVERRIDES)],
)
def test_arg_count_override(spec: KernelSpec, kwargs: dict, expected_arg_count: int):
    """Variant arg_counts (e.g. bn_conv2dk3_dw stride=1 has an extra arg)."""
    ef = _call_factory(spec, kwargs)
    assert len(ef._arg_types) == expected_arg_count


# ---------------------------------------------------------------------------
# Public arg_shape() / arg_dtype() introspection methods on BaseKernel.
# Replace the cryptic ``k.arg_types()[i].__args__[0][0]`` /
# ``k.arg_types()[i].__args__[1].__args__[0]`` idioms users had to write
# before.  The whats-new notebook cells 14/15 also use these.
# ---------------------------------------------------------------------------


def test_arg_dtype_returns_numpy_dtype():
    """arg_dtype() pulls the dtype from a parameterized np.ndarray arg type."""
    ef = kernels.passthrough(tile_size=4096, dtype=np.int32)
    assert ef.arg_dtype(0) == np.dtype(np.int32)


def test_arg_dtype_distinguishes_per_arg():
    """Different args can carry different dtypes — arg_dtype handles each."""
    ef = kernels.mm(
        dim_m=64, dim_k=64, dim_n=32, input_dtype=np.int8, output_dtype=np.int32
    )
    assert ef.arg_dtype(0) == np.dtype(np.int8)
    assert ef.arg_dtype(1) == np.dtype(np.int8)
    assert ef.arg_dtype(2) == np.dtype(np.int32)


def test_arg_shape_full_tuple():
    """arg_shape() returns the complete shape tuple (not just first dim)."""
    ef = kernels.passthrough(tile_size=2048, dtype=np.int16)
    assert ef.arg_shape(0) == (2048,)


def test_tile_size_equivalent_to_arg_shape_first_dim():
    """tile_size() is now a convenience wrapper over arg_shape(); confirm."""
    ef = kernels.passthrough(tile_size=4096, dtype=np.int32)
    assert ef.tile_size(0) == ef.arg_shape(0)[0]


def test_arg_shape_out_of_range_raises():
    """Out-of-range arg_index gets a clean error — same as tile_size()."""
    ef = kernels.passthrough(tile_size=96, dtype=np.int32)  # 3 args
    with pytest.raises(ValueError, match="out of range"):
        ef.arg_shape(99)


def test_arg_dtype_out_of_range_raises():
    ef = kernels.passthrough(tile_size=96, dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        ef.arg_dtype(99)


@pytest.fixture(params=["aie2", "aie2p"])
def kernel_arch(request):
    previous = get_current_device(probe_runtime=False)
    set_current_device(NPU1Col1() if request.param == "aie2" else NPU2Col1())
    try:
        yield request.param
    finally:
        set_current_device(previous)


@pytest.mark.parametrize("name", ["rms_norm", "rms_norm_eps", "layer_norm", "rope"])
def test_row_factory_aliases_and_arch_ports(name, kernel_arch):
    from aie.iron.kernels import datamovement, norm, transformer

    factory = getattr(kernels, name)
    canonical = datamovement if name == "rope" else norm
    assert factory is getattr(canonical, name)
    if name != "rms_norm_eps":
        assert factory is getattr(transformer, name)
    assert factory().arg_shape(0) == (1024,)
    fn = factory(tile_size=2048)
    assert fn == factory(cols=2048)
    assert fn.arg_shape(0) == (2048,)
    assert Path(fn._source_file).parent.name == canonical.__name__.split(".")[-1]
    assert len(fn.contract.roles) == len(fn.arg_types())
    if name != "rope":
        assert any(kernel_arch.upper() in flag for flag in fn._compile_flags)
    with pytest.raises(ValueError, match="must agree"):
        factory(tile_size=512, cols=2048)
    with pytest.raises(ValueError, match="positive"):
        factory(tile_size=0)


@pytest.mark.parametrize("name", ["rms_norm", "layer_norm"])
def test_norm_reference_eps_keyword(name):
    from aie.iron.kernels import norm, transformer

    reference = getattr(kernels, f"{name}_ref")
    assert reference is getattr(norm, f"{name}_ref")
    assert reference is getattr(transformer, f"{name}_ref")
    x = np.array([[1, 2, 4, 8]], dtype=bfloat16)
    xf = x.astype(np.float32)
    centered = xf if name == "rms_norm" else xf - xf.mean(axis=-1, keepdims=True)
    expected = centered / np.sqrt(
        (centered * centered).mean(axis=-1, keepdims=True) + 0.5
    )
    np.testing.assert_array_equal(reference(x, eps=0.5), expected.astype(bfloat16))


def test_norm_tail_and_vector_constraints(kernel_arch):
    assert kernels.rms_norm(tile_size=33).arg_shape(0) == (33,)
    fn = kernels.rms_norm_eps(cols=33)
    expected_setup = None if kernel_arch == "aie2" else kernels.conv_even
    assert fn.contract.setup is expected_setup
    assert kernels.rms_norm(cols=33).contract.setup is expected_setup
    x = np.ones((2, 33), dtype=bfloat16)
    np.testing.assert_array_equal(
        fn.contract.reference(x, 0.5), kernels.rms_norm_ref(x, eps=0.5)
    )
    for cols in (16, 208):
        assert kernels.layer_norm(cols=cols).arg_shape(0) == (cols,)
    with pytest.raises(ValueError, match="multiple of 16"):
        kernels.layer_norm(cols=17)


@pytest.mark.parametrize("two_halves", [False, True])
@pytest.mark.parametrize("tile_size", [96, 128])
def test_rope_layout_contract_and_vector_constraints(
    kernel_arch, two_halves, tile_size
):
    width = 32 if two_halves else 16
    fn = kernels.rope(cols=tile_size, two_halves=two_halves)
    assert fn._original_name == ("rope_two_halves" if two_halves else "rope")
    with pytest.raises(ValueError, match=f"multiple of {width}"):
        kernels.rope(cols=width - 2, two_halves=two_halves)
    x = np.arange(tile_size, dtype=np.float32).astype(bfloat16)
    lut = np.zeros(tile_size, dtype=bfloat16)
    lut[1::2] = 1
    expected = np.empty_like(x)
    if two_halves:
        expected[: tile_size // 2] = -x[tile_size // 2 :]
        expected[tile_size // 2 :] = x[: tile_size // 2]
    else:
        expected[0::2] = -x[1::2]
        expected[1::2] = x[0::2]
    np.testing.assert_array_equal(fn.contract.reference(x, lut), expected)


@pytest.mark.parametrize(
    "name", ["add_sized", "mul_sized", "relu_sized", "silu_sized", "gelu_sized"]
)
def test_sized_factory_contracts(name, kernel_arch):
    from aie.iron.kernels import Param
    from aie.utils.compile.jit.markers import In, Out

    fn = getattr(kernels, name)(tile_size=1024)
    binary = name in ("add_sized", "mul_sized")
    assert fn.contract.roles == ((In, In, Out, Param) if binary else (In, Out, Param))
    assert fn.contract.parameter_bindings == ((3 if binary else 2, 1024),)
    assert fn.contract.tolerance.note
    inputs = [np.ones(1024, dtype=bfloat16)] * (2 if binary else 1)
    reference_name = name.replace("_sized", "_ref")
    expected = getattr(kernels, reference_name)(*inputs)
    np.testing.assert_array_equal(fn.contract.reference(*inputs), expected)
    with pytest.raises(ValueError):
        getattr(kernels, name)(tile_size=0)
    if binary:
        assert getattr(kernels, name)(tile_size=1025).arg_shape(0) == (1025,)
    else:
        with pytest.raises(ValueError):
            getattr(kernels, name)(tile_size=1025)


@pytest.mark.parametrize(
    "name", ["add_sized", "mul_sized", "relu_sized", "silu_sized", "gelu_sized"]
)
def test_sized_factories_specialize_tiny_bounds(name, kernel_arch):
    factory = getattr(kernels, name)
    width = (
        1
        if name in ("add_sized", "mul_sized")
        else 32 if name == "relu_sized" or kernel_arch == "aie2p" else 16
    )
    macro = name.removesuffix("_sized").upper() + "_ELEMS"
    tiny = factory(tile_size=width)
    other = factory(tile_size=width * 4)
    assert tiny.arg_shape(0) == (width,)
    assert f"-D{macro}={width}" in tiny.compile_flags
    assert f"-D{macro}={width * 4}" in other.compile_flags
    assert tiny.object_file_name != other.object_file_name
    assert tiny._symbol_prefix != other._symbol_prefix


def test_relu_sized_requires_whole_vectors(kernel_arch):
    assert kernels.relu_sized(tile_size=32).arg_shape(0) == (32,)
    with pytest.raises(ValueError, match="32-element vector step"):
        kernels.relu_sized(tile_size=2049)


@pytest.mark.parametrize(
    "name", ["gelu", "silu", "relu", "swiglu", "tanh", "sigmoid", "leaky_relu"]
)
def test_fixed_activations_compile_their_count(name, kernel_arch):
    fn = getattr(kernels, name)()
    assert f"-D{name.upper()}_ELEMS=1024" in fn.compile_flags


@pytest.mark.parametrize(
    "name,tile_size",
    [
        ("tanh", 1056),
        ("tanh", 2048),
        ("sigmoid", 1056),
        ("sigmoid", 2048),
        ("leaky_relu", 64),
        ("leaky_relu", 96),
        ("leaky_relu", 2048),
    ],
)
def test_runtime_activation_sizes_preserve_contract(name, tile_size, kernel_arch):
    from aie.iron.kernels import Param

    fn = getattr(kernels, name)(tile_size=tile_size)
    assert fn.arg_shape(0) == fn.arg_shape(1) == (tile_size,)
    assert fn.contract.roles[2] is Param
    assert fn.contract.parameter_bindings == ((2, tile_size),)
    assert f"-D{name.upper()}_ELEMS={tile_size}" in fn.compile_flags
    if name == "leaky_relu":
        assert fn.contract.roles[3] is Param
        values = np.array([-2, 0, 2], dtype=bfloat16)
        np.testing.assert_array_equal(
            fn.contract.reference(values, 0.5), kernels.leaky_relu_ref(values, 0.5)
        )


@pytest.mark.parametrize(
    "arch,dtypes,expected",
    [
        ("aie2", (bfloat16, bfloat16), (4, 8, 4)),
        ("aie2p", (bfloat16, bfloat16), (4, 8, 8)),
        ("aie2", (np.int8, np.int8), (4, 8, 8)),
        ("aie2p", (np.int16, np.int16), (4, 4, 8)),
    ],
)
def test_mm_mac_dims_reads_the_table(arch, dtypes, expected):
    assert kernels.mm.mac_dims(*dtypes, arch=arch) == expected


def test_mm_mac_dims_follows_the_bf16_emulation_toggle():
    """The toggle moves the AIE2P micro-kernel to 8x8x8; nothing else moves."""
    assert kernels.mm.mac_dims(
        bfloat16, bfloat16, arch="aie2p", emulate_bf16_mmul_with_bfp16=True
    ) == (8, 8, 8)
    # Not on aie2, which has no emulation path.
    assert kernels.mm.mac_dims(
        bfloat16, bfloat16, arch="aie2", emulate_bf16_mmul_with_bfp16=True
    ) == (4, 8, 4)
    # Not for integer inputs.
    assert kernels.mm.mac_dims(
        np.int16, np.int16, arch="aie2p", emulate_bf16_mmul_with_bfp16=True
    ) == (4, 4, 8)


def test_mm_mac_dims_rejects_a_dtype_pair_with_no_kernel():
    with pytest.raises(ValueError, match="unsupported"):
        kernels.mm.mac_dims(np.float32, np.float32, arch="aie2")


@pytest.mark.parametrize(
    "factory,expected",
    [
        (kernels.mm, ((4, 8, 4), (4, 8, 8))),
        (kernels.cascade_mm, ((1, 1, 1), (1, 1, 1))),
    ],
)
def test_mac_dims_resolves_an_arch_from_a_device(factory, expected):
    from aie.iron.device import NPU1, NPU2

    assert factory.mac_dims(bfloat16, bfloat16, device=NPU1()) == expected[0]
    assert factory.mac_dims(bfloat16, bfloat16, device=NPU2()) == expected[1]


@pytest.mark.parametrize("factory", [kernels.mm, kernels.cascade_mm])
def test_the_factory_and_the_instance_agree(factory):
    built = factory(64, 64, 64, bfloat16, bfloat16)
    assert isinstance(built, ExternalFunction)
    assert built.mac_dims == factory.mac_dims(bfloat16, bfloat16)


def test_cascade_mm_carries_the_same_accessor():
    assert kernels.cascade_mm.mac_dims(bfloat16, bfloat16, arch="aie2") == (1, 1, 1)


@pytest.mark.parametrize("factory", [kernels.mm, kernels.cascade_mm])
@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_mac_dims_does_not_construct_or_register_a_kernel(factory, arch, monkeypatch):
    from aie.iron.kernels import linalg

    def unexpected_construction(*args, **kwargs):
        pytest.fail("Geometry queries must not construct kernels")

    monkeypatch.setattr(linalg, "_make_extern", unexpected_construction)
    instances = list(ExternalFunction._instances)
    assert callable(factory.mac_dims)
    assert len(factory.mac_dims(bfloat16, bfloat16, arch=arch)) == 3
    assert list(ExternalFunction._instances) == instances


@pytest.mark.parametrize("factory", [kernels.mm, kernels.cascade_mm])
def test_mac_dims_rejects_an_unsupported_arch(factory):
    with pytest.raises(ValueError, match="unsupported"):
        factory.mac_dims(bfloat16, bfloat16, arch="unsupported")


@pytest.mark.parametrize("arch", ["aie2", "aie2p"])
def test_cascade_mac_dims_rejects_an_unsupported_dtype_pair(arch):
    with pytest.raises(ValueError, match="unsupported"):
        kernels.cascade_mm.mac_dims(np.int8, np.int8, arch=arch)


@pytest.mark.parametrize("factory", [kernels.mm, kernels.cascade_mm])
def test_matrix_factory_preserves_discovery_metadata(factory):
    import inspect

    from aie.iron.kernels.linalg import MatrixKernel

    assert inspect.isfunction(factory)
    assert factory.__name__ in kernels.factories()
    assert issubclass(inspect.signature(factory).return_annotation, MatrixKernel)
    assert inspect.signature(factory).parameters["dim_m"].default == 64
    if factory is kernels.mm:
        assert len(factory.dtypes) == 7
        assert {"input_dtype": np.int16, "output_dtype": np.int16} in factory.dtypes


@pytest.mark.parametrize("kwargs", kernels.mm.dtypes)
@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("emulate", [False, True])
def test_mm_geometry_matches_contract_for_every_variant(
    kwargs, vectorized, emulate, kernel_arch
):
    from aie.iron.kernels.linalg import MatrixKernel

    options = dict(
        **kwargs,
        vectorized=vectorized,
        emulate_bf16_mmul_with_bfp16=emulate,
    )
    fn = kernels.mm(16, 16, 16, **options)
    assert isinstance(fn, MatrixKernel)
    assert fn.mac_dims == kernels.mm.mac_dims(**options, arch=kernel_arch)
    a, b, c = fn.contract.layouts
    r, s, t = fn.mac_dims
    assert (a.block, b.block, c.block) == ((r, s), (s, t), (r, t))
    if not vectorized:
        assert fn.mac_dims == (1, 1, 1)
        assert fn.stream_dims == (None, None, None)
    else:
        assert all(dims is not None for dims in fn.stream_dims)


@pytest.mark.parametrize("factory", [kernels.mm, kernels.cascade_mm])
def test_matrix_geometry_explicit_arch_overrides_device(factory):
    # An invalid device must never be consulted when arch is explicit.
    assert factory.mac_dims(bfloat16, bfloat16, arch="aie2", device=object()) == (
        factory.mac_dims(bfloat16, bfloat16, arch="aie2")
    )


@pytest.mark.parametrize(
    "factory", [kernels.mm, kernels.cascade_mm, kernels.cascade_mm_put]
)
def test_matrix_construction_rejects_unsupported_arch(factory, monkeypatch):
    from aie.iron.kernels import linalg

    monkeypatch.setattr(linalg, "_detect_arch", lambda: "unsupported")
    with pytest.raises(ValueError, match="unsupported"):
        factory()


@pytest.mark.parametrize("factory", [kernels.mm, kernels.mv, kernels.cascade_mm])
@pytest.mark.parametrize("use_chess", [False, True])
def test_matrix_zero_companion_uses_contract_initializer(
    factory, use_chess, kernel_arch
):
    fn = factory(use_chess=use_chess)
    # Constructing a product must not register an unused independent initializer.
    instances = list(ExternalFunction._instances)
    assert instances == [fn]
    initializer = fn.zero
    assert initializer == fn.contract.initializers[0][1](fn)
    assert initializer == fn.zero
    assert initializer.object_file is not fn.object_file
    assert initializer.use_chess == use_chess
    assert initializer.arg_types() == [fn.arg_types()[2]]
    assert Path(initializer.source_file).parts[-2:] == ("zero", "zero.cc")


def test_cascade_siblings_share_artifact_and_preserve_contract(kernel_arch):
    from aie.iron.kernels.linalg import MatrixKernel

    fn = kernels.cascade_mm()
    assert isinstance(fn, MatrixKernel)
    assert fn.get_only is fn
    assert fn.contract.unsupported
    assert fn.mac_dims == (1, 1, 1)
    assert fn.stream_dims == (None, None, None)
    for mode in ("put_only", "put_get"):
        sibling = getattr(fn, mode)
        assert sibling.object_file is fn.object_file
        assert sibling.arg_types() == fn.arg_types()
        assert sibling._name == fn.object_file.resolve_symbol(
            f"matmul_scalar_cascade_{mode}_i16_i16"
        )


def test_bf16_mv_retains_output_contract_without_accumulator(kernel_arch):
    from aie.utils.compile.jit.markers import Out

    fn = kernels.mv(input_dtype=bfloat16, output_dtype=bfloat16, dim_k=256)
    assert fn.contract.roles[-1] is Out
    assert fn.contract.initializers == ()
    assert not hasattr(fn, "zero")
