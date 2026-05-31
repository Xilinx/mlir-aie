# test_kernels_specs.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Spec-table-driven unit tests for aie.iron.kernels factory functions.

Each kernel factory is described by a single row in KERNEL_SPECS. Generic
parametrized tests exercise the common surface (returns ExternalFunction,
source is locatable, _arg_types length, default _name, invalid-kwargs raise).
Per-kernel name and shape variants are listed alongside the spec.

Also covers the public arg_shape() / arg_dtype() introspection methods
that BaseKernel exposes for unwrapping parameterized np.ndarray arg types.

Sibling files:
  test_kernels_memoization.py  — memoization, .zero, auto-prefix-on-collision
  test_kernels_chess.py        — use_chess + emulated bf16 plumbing

The shared _isolate_extern_state fixture lives in conftest.py at this
directory level.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.iron import kernels
from aie.iron.kernel import ExternalFunction


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
    # Source is either a real .cc file (source_substring=None checks _source_file)
    # or an embedded source_string containing a particular #include.
    source_kind: str = "file"  # "file" | "string_or_file"
    source_substring: str | None = None  # for "string_or_file": substring to find
    # Additional (kwargs_overrides, expected_name) pairs
    name_variants: list[tuple[dict, str]] = field(default_factory=list)
    # (kwargs_overrides, error_pattern) pairs
    invalid_kwargs: list[tuple[dict, str]] = field(default_factory=list)
    # (kwargs_overrides, arg_index, expected_shape_tuple) — for shape sanity
    shape_checks: list[tuple[dict, int, tuple]] = field(default_factory=list)
    # (kwargs_overrides, expected_tile_size_at_arg_0)
    tile_size_checks: list[tuple[dict, int]] = field(default_factory=list)


KERNEL_SPECS: list[KernelSpec] = [
    # ----- eltwise -----
    KernelSpec(
        name="passthrough",
        factory=kernels.passthrough,
        kwargs=dict(tile_size=4096, dtype=np.int32),
        arg_count=3,
        expected_name="passThroughLine",
        shape_checks=[
            (dict(tile_size=64, dtype=np.int16), 0, (64,)),
        ],
        tile_size_checks=[
            (dict(tile_size=256, dtype=np.uint8), 256),
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
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=bfloat16), "dtype must be np.int32"),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            # int32 output: 1 element = 4 bytes → already DMA-aligned.
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
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
        ],
        invalid_kwargs=[
            (dict(tile_size=1024, dtype=bfloat16), "dtype must be np.int32"),
        ],
        shape_checks=[
            (dict(tile_size=2048, dtype=np.int32), 0, (2048,)),
            (dict(tile_size=2048, dtype=np.int32), 1, (1,)),
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
        source_kind="string_or_file",
        source_substring="softmax.cc",
        invalid_kwargs=[(dict(tile_size=2048), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="gelu",
        factory=kernels.gelu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="gelu_bf16",
        source_kind="string_or_file",
        source_substring="gelu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="silu",
        factory=kernels.silu,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="silu_bf16",
        source_kind="string_or_file",
        source_substring="silu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="swiglu",
        factory=kernels.swiglu,
        kwargs=dict(tile_size=1024),
        arg_count=4,
        expected_name="swiglu_bf16",
        source_kind="string_or_file",
        source_substring="swiglu.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    KernelSpec(
        name="bf16_exp",
        factory=kernels.bf16_exp,
        kwargs=dict(tile_size=1024),
        arg_count=2,
        expected_name="exp_bf16_1024",
        source_kind="string_or_file",
        source_substring="bf16_exp.cc",
        invalid_kwargs=[(dict(tile_size=512), "tile_size must be 1024")],
    ),
    # ----- vision -----
    KernelSpec(
        name="rgba2hue",
        factory=kernels.rgba2hue,
        kwargs=dict(line_width=1920),
        arg_count=3,
        expected_name="rgba2hueLine",
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
            (dict(line_width=1920, dtype=np.int32), "addWeightedLine"),
        ],
        invalid_kwargs=[
            (dict(line_width=1920, dtype=np.float32), "unsupported dtype"),
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
            (dict(block_index=14), "bn14_1_conv2dk1_ui8_ui8_input_split_partial_width_put_new"),
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
            (dict(block_index=14), "bn_14_2_conv2dk1_ui8_i8_i8_scalar_input_split_partial_width_get_new"),
        ],
        invalid_kwargs=[(dict(block_index=12), "block_index")],
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


ARG_COUNT_OVERRIDES: list[tuple[KernelSpec, dict, int]] = []


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_returns_external_function(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    assert isinstance(ef, ExternalFunction)


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_source_locatable(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    if spec.source_kind == "file":
        src = ef._source_file
        assert src is not None
        assert Path(src).exists(), f"Source file not found: {src}"
    else:
        # source_string OR source_file must be set; if string, must reference the .cc
        assert ef._source_string is not None or ef._source_file is not None
        if ef._source_string is not None and spec.source_substring is not None:
            assert spec.source_substring in ef._source_string


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_arg_types_length(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    assert len(ef._arg_types) == spec.arg_count


@pytest.mark.parametrize("spec", KERNEL_SPECS, ids=_ids(KERNEL_SPECS))
def test_default_function_name(spec: KernelSpec):
    ef = spec.factory(**spec.kwargs)
    assert ef._name == spec.expected_name


_NAME_VARIANTS = _flat(KERNEL_SPECS, "name_variants")


@pytest.mark.parametrize(
    "spec,kwargs,expected_name",
    _NAME_VARIANTS,
    ids=_flat_ids(_NAME_VARIANTS, "v"),
)
def test_name_variant(spec: KernelSpec, kwargs: dict, expected_name: str):
    ef = spec.factory(**kwargs)
    assert ef._name == expected_name


_INVALID = _flat(KERNEL_SPECS, "invalid_kwargs")


@pytest.mark.parametrize(
    "spec,kwargs,pattern",
    _INVALID,
    ids=_flat_ids(_INVALID, "bad"),
)
def test_invalid_kwargs_raise(spec: KernelSpec, kwargs: dict, pattern: str):
    with pytest.raises(ValueError, match=pattern):
        spec.factory(**kwargs)


_SHAPES = _flat(KERNEL_SPECS, "shape_checks")


@pytest.mark.parametrize(
    "spec,kwargs,arg_idx,expected_shape",
    _SHAPES,
    ids=_flat_ids(_SHAPES, "shape"),
)
def test_arg_shape(spec: KernelSpec, kwargs: dict, arg_idx: int, expected_shape: tuple):
    ef = spec.factory(**kwargs)
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
    ef = spec.factory(**kwargs)
    assert ef.tile_size(0) == expected_tile_size


@pytest.mark.parametrize(
    "spec,kwargs,expected_arg_count",
    ARG_COUNT_OVERRIDES,
    ids=[f"{r[0].name}-argc{i}" for i, r in enumerate(ARG_COUNT_OVERRIDES)],
)
def test_arg_count_override(spec: KernelSpec, kwargs: dict, expected_arg_count: int):
    """Variant arg_counts (e.g. bn_conv2dk3_dw stride=1 has an extra arg)."""
    ef = spec.factory(**kwargs)
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
    ef = kernels.passthrough(tile_size=64, dtype=np.int32)  # 3 args
    with pytest.raises(ValueError, match="out of range"):
        ef.arg_shape(99)


def test_arg_dtype_out_of_range_raises():
    ef = kernels.passthrough(tile_size=64, dtype=np.int32)
    with pytest.raises(ValueError, match="out of range"):
        ef.arg_dtype(99)
